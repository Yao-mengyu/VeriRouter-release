use crate::circuits::common::assert_gate_true;
use crate::util::ExactZip;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitData;

use crate::circuits::common::{
    add_merkle_path, add_score_table_row, assert_bool_implies_eq, assert_bool_implies_zero,
    commit_neighbor_visits, connect_hashes, constrain_node_id_with_mode, count_true,
    hash_score_table, verify_graph_openings, MerklePathTargets, NodeIdRangeMode,
    ScoreTableRowTargets,
};
use crate::commit_check::GraphMerkleData;
use crate::gadgets::dist_ref::{
    add_dist_ref_batch_targets, add_dist_ref_targets, add_dist_ref_vec_targets,
    assert_dist_refs_against_opened_table, fill_dist_ref_batch_witness, set_dist_ref_fields,
    set_dist_ref_witness, DistRefBatchTargets, DistRefTargets,
};
use crate::gadgets::monotone_stack::{
    add_prev_lower_trace_targets, add_prev_lower_trace_targets_with_active_capacity,
    assert_prev_lower_with_monotone_stack_enabled, fill_prev_lower_witness, PrevLowerTraceTargets,
};
use crate::gadgets::order::{
    assert_nonnegative_diff_if_bounded_operands, assert_positive_diff_if_bounded_operands,
};
use crate::gadgets::search::{
    add_pq_trace_targets, assert_final_topk_model_membership_from_sorted,
    assert_min_pq_trace_with_bounded_del_times, assert_ranked_state, is_new_via_permutation,
    PqTraceTargets,
};
use crate::util::{
    c_entry_position, c_neighbor_position, c_pop_position, c_position_limit, canonical_node_id,
    recursion_config, u32_to_f, u64_to_f, w_initial_position, w_pop_position, w_push_position,
    NodeId, C, D, F, GENERIC_MONOTONE_KEY_BITS, NODE_ID_SENTINEL, POSITION_BITS, SCORE_BITS,
};
use crate::witness::ProofInput;

#[derive(Clone)]
pub struct EfSearchTargets {
    pub score_table_hash_pi: plonky2::hash::hash_types::HashOutTarget,
    pub graph_root_pi: plonky2::hash::hash_types::HashOutTarget,
    pub nbr_commit_pi: plonky2::hash::hash_types::HashOutTarget,
    pub output_id_pi: Target,
    pub entry_pi: Target,
    pub upper_len_pi: Target,
    pub rounds_len_pi: Target,
    pub ef_search_pi: Target,
    pub top_k_pi: Target,
    pub node_ids: Vec<Target>,
    pub node_opens: Vec<ScoreTableRowTargets>,
    pub row_layers: Vec<Target>,
    pub row_srcs: Vec<Target>,
    pub row_neighbors: Vec<Vec<Target>>,
    pub row_paths: Vec<MerklePathTargets>,
    pub graph_static_enabled: Vec<bool>,
    pub row_src_refs: Vec<DistRefTargets>,
    pub row_neighbor_refs: Vec<Vec<DistRefTargets>>,
    pub expansion_pop_ids: Vec<Target>,
    pub visit_is_new: Vec<BoolTarget>,
    pub seen_sorted_ids: Vec<Target>,
    pub seen_sorted_times: Vec<Target>,
    pub seen_sorted_is_new: Vec<BoolTarget>,
    pub work_updates: Vec<WorkUpdateTargets>,
    pub c_pq: PqTraceTargets,
    pub w_pq: PqTraceTargets,
    pub c_prev_lower: PrevLowerTraceTargets,
    pub w_prev_lower: PrevLowerTraceTargets,
    pub final_topk_refs: Vec<DistRefTargets>,
    pub ref_batch: DistRefBatchTargets,
}

#[derive(Clone)]
pub struct WorkUpdateTargets {
    pub active: BoolTarget,
    pub item: DistRefTargets,
    pub assigned_c_position: Target,
    pub w_push_position: Target,
    pub w_pop_position: Target,
    pub admitted: BoolTarget,
}

#[derive(Clone, Copy, Debug)]
pub struct EfSearchBuildStats {
    pub pre_padding_gate_count: usize,
    pub score_table_relation_gate_count: usize,
    pub graph_relation_gate_count: usize,
    pub trace_target_gate_count: usize,
    pub visit_permutation_gate_count: usize,
    pub round_link_gate_count: usize,
    pub score_table_hash_gate_count: usize,
    pub graph_merkle_opening_gate_count: usize,
    pub graph_merkle_leaf_hash_gate_count: usize,
    pub graph_merkle_path_gate_count: usize,
    pub dist_ref_membership_gate_count: usize,
    pub workupd_relation_gate_count: usize,
    pub working_update_gate_count: usize,
    pub candidate_round_gate_count: usize,
    pub candidate_prev_lower_gate_count: usize,
    pub candidate_pq_gate_count: usize,
    pub working_prev_lower_gate_count: usize,
    pub working_pq_gate_count: usize,
    pub final_state_gate_count: usize,
}

fn ref_score_or_zero_if_sent(
    builder: &mut CircuitBuilder<F, D>,
    target_ref: &DistRefTargets,
    sent_id: Target,
) -> (Target, BoolTarget) {
    let zero = builder.zero();
    let is_sent = builder.is_equal(target_ref.node_id, sent_id);
    let is_real = builder.not(is_sent);
    let score_or_zero = builder.select(is_real, target_ref.score_enc, zero);
    (score_or_zero, is_real)
}

fn connect_dist_ref_to_sentinel(
    builder: &mut CircuitBuilder<F, D>,
    ref_t: &DistRefTargets,
    sent_id: Target,
) {
    let zero = builder.zero();
    builder.connect(ref_t.node_id, sent_id);
    builder.connect(ref_t.score_enc, zero);
    builder.connect(ref_t.model_id, zero);
}

fn constrain_work_update_schedule(
    builder: &mut CircuitBuilder<F, D>,
    update: &WorkUpdateTargets,
    assigned_c_position: u32,
    update_idx: usize,
    ef_search: usize,
    sent_id: Target,
) {
    let zero = builder.zero();
    let inactive = builder.not(update.active);
    let item_is_dummy = builder.is_equal(update.item.node_id, sent_id);
    let item_is_real = builder.not(item_is_dummy);
    assert_gate_true(builder, update.active, item_is_real);
    assert_bool_implies_eq(builder, inactive, update.item.node_id, sent_id);
    assert_bool_implies_zero(builder, inactive, update.item.score_enc);
    assert_bool_implies_zero(builder, inactive, update.item.model_id);
    assert_bool_implies_zero(builder, inactive, update.assigned_c_position);
    assert_bool_implies_zero(builder, inactive, update.admitted.target);

    let assigned_const = builder.constant(u32_to_f(assigned_c_position));
    let assigned_expected = builder.select(update.active, assigned_const, zero);
    builder.connect(update.assigned_c_position, assigned_expected);

    let push_const = builder.constant(u32_to_f(w_push_position(update_idx, ef_search)));
    let pop_const = builder.constant(u32_to_f(w_pop_position(update_idx, ef_search)));
    let push_expected = builder.select(update.active, push_const, zero);
    let pop_expected = builder.select(update.active, pop_const, zero);
    builder.connect(update.w_push_position, push_expected);
    builder.connect(update.w_pop_position, pop_expected);
}

fn bind_physical_work_update_source(
    builder: &mut CircuitBuilder<F, D>,
    update: &WorkUpdateTargets,
    source_ref: &DistRefTargets,
    source_is_new: BoolTarget,
) {
    builder.connect(update.active.target, source_is_new.target);
    assert_bool_implies_eq(
        builder,
        update.active,
        update.item.node_id,
        source_ref.node_id,
    );
    assert_bool_implies_eq(
        builder,
        update.active,
        update.item.score_enc,
        source_ref.score_enc,
    );
    assert_bool_implies_eq(
        builder,
        update.active,
        update.item.model_id,
        source_ref.model_id,
    );
}

fn c_packed_priority(
    builder: &mut CircuitBuilder<F, D>,
    score_enc: Target,
    tie_position: Target,
    score_max: Target,
    multiplier: Target,
) -> Target {
    let one = builder.one();
    let score_complement = builder.sub(score_max, score_enc);
    let zero_based_position = builder.sub(tie_position, one);
    builder.mul_add(score_complement, multiplier, zero_based_position)
}

fn w_packed_priority(
    builder: &mut CircuitBuilder<F, D>,
    target_ref: &DistRefTargets,
    dummy_ordinal: Target,
    tie_position: Target,
    multiplier: Target,
    position_max: Target,
    ef_cap: Target,
    sent_id: Target,
) -> (Target, BoolTarget) {
    let one = builder.one();
    let is_dummy = builder.is_equal(target_ref.node_id, sent_id);
    let is_real = builder.not(is_dummy);
    let dummy_key = builder.sub(dummy_ordinal, one);
    let score_term = builder.mul(target_ref.score_enc, multiplier);
    let reverse_position = builder.sub(position_max, tie_position);
    let real_without_cap = builder.add(score_term, reverse_position);
    let real_key = builder.add(ef_cap, real_without_cap);
    let key = builder.select(is_real, real_key, dummy_key);

    // A real queue item never carries a dummy ordinal. A dummy's ordinal is a
    // separately-accounted token field; source time remains temporal identity.
    let real_ordinal = builder.mul(is_real.target, dummy_ordinal);
    builder.assert_zero(real_ordinal);
    let dummy_position = builder.mul(is_dummy.target, tie_position);
    builder.assert_zero(dummy_position);
    (key, is_real)
}

fn packed_priority_parameters(
    rounds_len: usize,
    round_width: usize,
    ef_search: usize,
) -> (u32, u64) {
    let position_max = c_position_limit(rounds_len, round_width);
    let multiplier = u64::from(position_max);
    let score_space = 1u128 << SCORE_BITS;
    let c_max = score_space * u128::from(position_max) - 1;
    let w_max = (ef_search as u128) + c_max;
    let field_safe_limit = 1u128 << GENERIC_MONOTONE_KEY_BITS;
    assert!(
        c_max < field_safe_limit && w_max < field_safe_limit,
        "packed priority exceeds the {}-bit safe comparison domain: Cmax={} Wmax={}",
        GENERIC_MONOTONE_KEY_BITS,
        c_max,
        w_max
    );
    (position_max, multiplier)
}

fn assert_worst_to_best_w_state(
    builder: &mut CircuitBuilder<F, D>,
    ranked_refs: &[DistRefTargets],
    ranked_priorities: &[Target],
    sent_id: Target,
) -> (Vec<BoolTarget>, Vec<Target>, Target) {
    assert_eq!(ranked_refs.len(), ranked_priorities.len());
    let zero = builder.zero();
    let mut ranked_scores = Vec::with_capacity(ranked_refs.len());
    let mut ranked_is_real = Vec::with_capacity(ranked_refs.len());
    for ranked_ref in ranked_refs {
        let is_sent = builder.is_equal(ranked_ref.node_id, sent_id);
        let is_real = builder.not(is_sent);
        ranked_is_real.push(is_real);
        let score_or_zero = builder.select(is_real, ranked_ref.score_enc, zero);
        ranked_scores.push(score_or_zero);
    }
    for i in 0..ranked_refs.len().saturating_sub(1) {
        let left_is_sent = builder.not(ranked_is_real[i]);
        let right_is_sent = builder.not(ranked_is_real[i + 1]);
        assert_gate_true(builder, right_is_sent, left_is_sent);

        // W is exposed from worst to best. Packed priorities are unique:
        // dummy ordinals are distinct, and real items carry distinct assigned
        // C positions. Therefore each adjacent priority must strictly rise.
        let always = builder._true();
        assert_positive_diff_if_bounded_operands(
            builder,
            always,
            ranked_priorities[i + 1],
            ranked_priorities[i],
            GENERIC_MONOTONE_KEY_BITS,
        );
    }
    let ranked_count = count_true(builder, &ranked_is_real);
    (ranked_is_real, ranked_scores, ranked_count)
}

#[allow(clippy::too_many_arguments)]
pub fn build_ef_search_circuit(
    use_zk: bool,
    node_count: usize,
    row_neighbor_lens: &[usize],
    upper_len: usize,
    rounds_len: usize,
    ef_search: usize,
    k: usize,
    visit_len: usize,
    c_final_len: usize,
    w_final_len: usize,
    graph_tree_height: usize,
    ref_unused_len: usize,
) -> (CircuitData<F, C, D>, EfSearchTargets) {
    let graph_static_enabled = vec![true; row_neighbor_lens.len()];
    let (data, targets, _) = build_ef_search_circuit_with_options(
        use_zk,
        node_count,
        row_neighbor_lens,
        upper_len,
        rounds_len,
        ef_search,
        k,
        visit_len,
        c_final_len,
        w_final_len,
        graph_tree_height,
        ref_unused_len,
        &graph_static_enabled,
        NodeIdRangeMode::Wide56,
    );
    (data, targets)
}

#[allow(clippy::too_many_arguments)]
pub fn build_ef_search_circuit_with_options(
    use_zk: bool,
    node_count: usize,
    row_neighbor_lens: &[usize],
    upper_len: usize,
    rounds_len: usize,
    ef_search: usize,
    k: usize,
    visit_len: usize,
    c_final_len: usize,
    w_final_len: usize,
    graph_tree_height: usize,
    ref_unused_len: usize,
    graph_static_enabled: &[bool],
    node_id_mode: NodeIdRangeMode,
) -> (CircuitData<F, C, D>, EfSearchTargets, EfSearchBuildStats) {
    assert!(
        node_count > 0,
        "ef_search requires at least one opened node"
    );
    assert!(ef_search > 0, "ef_search must be positive");
    assert!(k > 0, "ef_search k must be positive");
    assert!(k <= ef_search, "ef_search requires k <= ef_search");
    assert_eq!(
        w_final_len, ef_search,
        "W final witness must expose exactly EfCap slots"
    );
    assert_eq!(
        graph_static_enabled.len(),
        row_neighbor_lens.len(),
        "graph static mask length must match graph row count"
    );

    let config = recursion_config(use_zk);
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let round_width = row_neighbor_lens
        .iter()
        .skip(upper_len)
        .take(rounds_len)
        .copied()
        .max()
        .unwrap_or(0);
    let step_count = rounds_len * round_width;
    let (position_max, priority_multiplier) =
        packed_priority_parameters(rounds_len, round_width, ef_search);

    let score_table_hash_pi = builder.add_virtual_hash_public_input();
    let graph_root_pi = builder.add_virtual_hash_public_input();
    let nbr_commit_pi = builder.add_virtual_hash_public_input();
    let output_id_pi = builder.add_virtual_public_input();
    let entry_pi = builder.add_virtual_public_input();
    constrain_node_id_with_mode(&mut builder, entry_pi, node_id_mode);
    let upper_len_pi = builder.add_virtual_public_input();
    let upper_len_const = builder.constant(u32_to_f(upper_len as u32));
    builder.connect(upper_len_pi, upper_len_const);
    let rounds_len_pi = builder.add_virtual_public_input();
    let rounds_len_const = builder.constant(u32_to_f(rounds_len as u32));
    builder.connect(rounds_len_pi, rounds_len_const);
    let ef_search_pi = builder.add_virtual_public_input();
    let ef_search_const = builder.constant(u32_to_f(ef_search as u32));
    builder.connect(ef_search_pi, ef_search_const);
    let top_k_pi = builder.add_virtual_public_input();
    let top_k_const = builder.constant(u32_to_f(k as u32));
    builder.connect(top_k_pi, top_k_const);

    let score_table_relation_start = builder.num_gates();
    let node_ids = builder.add_virtual_targets(node_count);
    let mut node_opens = Vec::with_capacity(node_count);
    let sent_id = builder.constant(u64_to_f(NODE_ID_SENTINEL));
    let zero = builder.zero();
    for _ in 0..node_count {
        node_opens.push(add_score_table_row(&mut builder));
    }
    for i in 0..node_count {
        builder.range_check(node_opens[i].score_enc, SCORE_BITS);
        let is_dummy = constrain_node_id_with_mode(&mut builder, node_ids[i], node_id_mode);
        assert_bool_implies_zero(&mut builder, is_dummy, node_opens[i].score_enc);
        assert_bool_implies_zero(&mut builder, is_dummy, node_opens[i].model_id);
    }
    let score_encs = node_opens
        .iter()
        .map(|open| open.score_enc)
        .collect::<Vec<_>>();
    let model_ids = node_opens
        .iter()
        .map(|open| open.model_id)
        .collect::<Vec<_>>();
    let score_table_hash_start = builder.num_gates();
    let score_table_hash_calc = hash_score_table(&mut builder, &node_ids, &score_encs, &model_ids);
    connect_hashes(&mut builder, &score_table_hash_pi, &score_table_hash_calc);
    let score_table_hash_gate_count = builder.num_gates() - score_table_hash_start;
    let score_table_relation_gate_count = builder.num_gates() - score_table_relation_start;

    let graph_relation_start = builder.num_gates();
    let mut row_layers = Vec::with_capacity(row_neighbor_lens.len());
    let mut row_srcs = Vec::with_capacity(row_neighbor_lens.len());
    let mut row_neighbors = Vec::with_capacity(row_neighbor_lens.len());
    let mut row_paths = Vec::with_capacity(row_neighbor_lens.len());
    for &len in row_neighbor_lens {
        row_layers.push(builder.add_virtual_target());
        let row_src = builder.add_virtual_target();
        row_srcs.push(row_src);
        let row_nbs = builder.add_virtual_targets(len);
        row_neighbors.push(row_nbs);
        let row_idx = row_paths.len();
        let path_height = if graph_static_enabled[row_idx] {
            graph_tree_height
        } else {
            0
        };
        row_paths.push(add_merkle_path(&mut builder, path_height));
    }
    let row_src_refs = (0..row_neighbor_lens.len())
        .map(|_| add_dist_ref_targets(&mut builder))
        .collect::<Vec<_>>();
    let row_neighbor_refs = row_neighbor_lens
        .iter()
        .map(|&len| add_dist_ref_vec_targets(&mut builder, len))
        .collect::<Vec<_>>();
    for row_idx in 0..row_neighbor_lens.len() {
        builder.connect(row_srcs[row_idx], row_src_refs[row_idx].node_id);
        for (row_nb, nb_ref) in row_neighbors[row_idx]
            .iter()
            .zip(row_neighbor_refs[row_idx].iter())
        {
            builder.connect(*row_nb, nb_ref.node_id);
        }
    }
    let graph_enabled = row_src_refs
        .iter()
        .map(|row_src_ref| {
            let is_sent = builder.is_equal(row_src_ref.node_id, sent_id);
            builder.not(is_sent)
        })
        .collect::<Vec<_>>();
    let mut graph_open_layers = Vec::new();
    let mut graph_open_srcs = Vec::new();
    let mut graph_open_neighbors = Vec::new();
    let mut graph_open_paths = Vec::new();
    let mut graph_open_enabled = Vec::new();
    for row_idx in 0..row_neighbor_lens.len() {
        if graph_static_enabled[row_idx] {
            graph_open_layers.push(row_layers[row_idx]);
            graph_open_srcs.push(row_srcs[row_idx]);
            graph_open_neighbors.push(row_neighbors[row_idx].clone());
            graph_open_paths.push(row_paths[row_idx].clone());
            graph_open_enabled.push(graph_enabled[row_idx]);
        } else {
            builder.connect(row_srcs[row_idx], sent_id);
            connect_dist_ref_to_sentinel(&mut builder, &row_src_refs[row_idx], sent_id);
            for (&row_nb, nb_ref) in row_neighbors[row_idx]
                .iter()
                .zip(row_neighbor_refs[row_idx].iter())
            {
                builder.connect(row_nb, sent_id);
                connect_dist_ref_to_sentinel(&mut builder, nb_ref, sent_id);
            }
        }
    }
    let graph_merkle_start = builder.num_gates();
    let (graph_merkle_leaf_hash_gate_count, graph_merkle_path_gate_count) =
        if graph_open_layers.is_empty() {
            (0, 0)
        } else {
            let stats = verify_graph_openings(
                &mut builder,
                &graph_root_pi,
                &graph_open_layers,
                &graph_open_srcs,
                &graph_open_neighbors,
                &graph_open_paths,
                &graph_open_enabled,
            );
            (stats.leaf_hash_gate_count, stats.path_gate_count)
        };
    let graph_merkle_opening_gate_count = builder.num_gates() - graph_merkle_start;
    let graph_relation_gate_count = builder.num_gates() - graph_relation_start;

    let trace_target_start = builder.num_gates();
    let mut visit_is_new = Vec::with_capacity(visit_len);
    for _ in 0..visit_len {
        visit_is_new.push(builder.add_virtual_bool_target_safe());
    }
    // Each pop id is connected to its row's source DistRef node id (a table
    // id or the sentinel), or the row is statically the sentinel.
    let expansion_pop_ids = builder.add_virtual_targets(rounds_len);

    let work_update_capacity = 1 + step_count;
    let work_updates = (0..work_update_capacity)
        .map(|_| WorkUpdateTargets {
            active: builder.add_virtual_bool_target_safe(),
            item: add_dist_ref_targets(&mut builder),
            assigned_c_position: builder.add_virtual_target(),
            w_push_position: builder.add_virtual_target(),
            w_pop_position: builder.add_virtual_target(),
            admitted: builder.add_virtual_bool_target_safe(),
        })
        .collect::<Vec<_>>();

    let c_pq = add_pq_trace_targets(&mut builder, rounds_len, c_final_len);
    let w_pq = add_pq_trace_targets(&mut builder, work_update_capacity, w_final_len);
    let c_prev_lower = add_prev_lower_trace_targets(&mut builder, rounds_len + c_final_len);
    let w_output_active_capacity = usize::min(work_update_capacity, node_count) + w_final_len;
    let w_prev_lower = add_prev_lower_trace_targets_with_active_capacity(
        &mut builder,
        work_update_capacity + w_final_len,
        w_output_active_capacity,
    );
    let final_topk_refs = add_dist_ref_vec_targets(&mut builder, k);
    let graph_ref_len = row_neighbor_lens
        .iter()
        .enumerate()
        .filter(|(idx, _)| graph_static_enabled[*idx])
        .map(|(_, len)| 1usize + *len)
        .sum::<usize>();
    // Graph openings already authenticate every raw level-0 neighbor. Only
    // WorkUpd row zero (the entry payload) needs an independent table lookup.
    let raw_ref_len = graph_ref_len + 1 + c_final_len + work_update_capacity + w_final_len;
    let ref_batch = add_dist_ref_batch_targets(&mut builder, raw_ref_len, ref_unused_len);
    let trace_target_gate_count = builder.num_gates() - trace_target_start;

    let visit_permutation_start = builder.num_gates();
    let mut flat_visit_ids = Vec::with_capacity(visit_len);
    for row_idx in upper_len..(upper_len + rounds_len) {
        flat_visit_ids.extend(row_neighbors[row_idx].iter().copied());
    }
    assert_eq!(
        flat_visit_ids.len(),
        visit_len,
        "visit_len mismatch vs ef rows"
    );
    let nbr_commit_calc = commit_neighbor_visits(&mut builder, &flat_visit_ids, &visit_is_new);
    connect_hashes(&mut builder, &nbr_commit_pi, &nbr_commit_calc);
    let seen_challenge_inputs = [
        entry_pi,
        rounds_len_pi,
        nbr_commit_pi.elements[0],
        nbr_commit_pi.elements[1],
        nbr_commit_pi.elements[2],
        nbr_commit_pi.elements[3],
    ];
    // Every visit id is a level-0 neighbor DistRef node id (table id or
    // sentinel), or statically the sentinel.
    let (seen_sorted_ids, seen_sorted_times, seen_sorted_is_new) = is_new_via_permutation(
        &mut builder,
        &flat_visit_ids,
        &visit_is_new,
        entry_pi,
        NODE_ID_SENTINEL,
        (visit_len as u64).saturating_add(1),
        &seen_challenge_inputs,
    );
    let _ = &seen_sorted_ids;
    let visit_permutation_gate_count = builder.num_gates() - visit_permutation_start;

    let ef_t = builder.constant(u32_to_f(ef_search as u32));
    let k_t = builder.constant(u32_to_f(k as u32));
    let position_max_t = builder.constant(u32_to_f(position_max));
    let priority_multiplier_t = builder.constant(u64_to_f(priority_multiplier));
    let score_max_t = builder.constant(u64_to_f((1u64 << SCORE_BITS) - 1));
    let round_link_start = builder.num_gates();
    let mut all_refs = Vec::new();
    for row_idx in 0..row_neighbor_lens.len() {
        if graph_static_enabled[row_idx] {
            all_refs.push(row_src_refs[row_idx].clone());
            all_refs.extend(row_neighbor_refs[row_idx].iter().cloned());
        }
    }
    all_refs.push(work_updates[0].item.clone());
    for r in 0..rounds_len {
        let row_idx = upper_len + r;
        let round_active = graph_enabled[row_idx];
        builder.connect(expansion_pop_ids[r], row_src_refs[row_idx].node_id);
        assert_bool_implies_eq(
            &mut builder,
            round_active,
            c_pq.del_refs[r].node_id,
            row_src_refs[row_idx].node_id,
        );
        assert_bool_implies_eq(
            &mut builder,
            round_active,
            c_pq.del_refs[r].score_enc,
            row_src_refs[row_idx].score_enc,
        );
        assert_bool_implies_eq(
            &mut builder,
            round_active,
            c_pq.del_refs[r].model_id,
            row_src_refs[row_idx].model_id,
        );
    }
    all_refs.extend(c_pq.final_refs.iter().cloned());
    all_refs.extend(w_pq.del_refs.iter().cloned());
    all_refs.extend(w_pq.final_refs.iter().cloned());
    let round_link_gate_count = builder.num_gates() - round_link_start;
    let dist_ref_start = builder.num_gates();
    assert_dist_refs_against_opened_table(
        &mut builder,
        &all_refs,
        &ref_batch,
        &score_table_hash_pi,
        &node_ids,
        &node_opens,
        sent_id,
        entry_pi,
        output_id_pi,
        5002,
        true,
    );
    let dist_ref_membership_gate_count = builder.num_gates() - dist_ref_start;

    let workupd_relation_start = builder.num_gates();
    let mut step_refs = Vec::with_capacity(step_count);
    let mut step_is_new = Vec::with_capacity(step_count);
    let mut visit_cursor = 0usize;
    for r in 0..rounds_len {
        let row_idx = upper_len + r;
        let round_active = graph_enabled[row_idx];
        if r > 0 {
            assert_gate_true(&mut builder, round_active, graph_enabled[upper_len + r - 1]);
        }
        let round_inactive = builder.not(round_active);
        let is_l0 = builder.is_equal(row_layers[row_idx], zero);
        builder.assert_one(is_l0.target);
        for p in 0..round_width {
            if p < row_neighbors[row_idx].len() {
                let neighbor_ref = &row_neighbor_refs[row_idx][p];
                assert_bool_implies_eq(&mut builder, round_inactive, neighbor_ref.node_id, sent_id);
                assert_bool_implies_zero(&mut builder, round_inactive, neighbor_ref.score_enc);
                assert_bool_implies_zero(&mut builder, round_inactive, neighbor_ref.model_id);
                assert_gate_true(&mut builder, visit_is_new[visit_cursor], round_active);
                step_refs.push(row_neighbor_refs[row_idx][p].clone());
                step_is_new.push(visit_is_new[visit_cursor]);
                visit_cursor += 1;
            } else {
                step_refs.push(DistRefTargets {
                    node_id: sent_id,
                    score_enc: zero,
                    model_id: zero,
                });
                step_is_new.push(builder._false());
            }
        }
    }
    assert_eq!(visit_cursor, visit_len, "step padding mismatch");

    // WorkUpd is physically aligned with X=(entry, authenticated level-0
    // neighbor slots): row zero is entry and row 1+t is raw slot t. Repeated,
    // padding, and inactive-round slots remain canonical holes instead of
    // being compacted into a dense first-visit subsequence.
    builder.assert_one(work_updates[0].active.target);
    builder.connect(work_updates[0].item.node_id, entry_pi);

    for i in 0..work_update_capacity {
        let assigned_position = if i == 0 {
            c_entry_position()
        } else {
            let t = i - 1;
            c_neighbor_position(t / round_width, t % round_width, round_width)
        };
        constrain_work_update_schedule(
            &mut builder,
            &work_updates[i],
            assigned_position,
            i,
            ef_search,
            sent_id,
        );
    }

    for t in 0..step_count {
        let update = &work_updates[1 + t];
        bind_physical_work_update_source(&mut builder, update, &step_refs[t], step_is_new[t]);
        let step_is_dummy = builder.is_equal(step_refs[t].node_id, sent_id);
        let step_is_real = builder.not(step_is_dummy);
        assert_gate_true(&mut builder, step_is_new[t], step_is_real);
    }
    let workupd_relation_gate_count = builder.num_gates() - workupd_relation_start;

    let mut c_insert_src_times = Vec::with_capacity(work_update_capacity);
    let mut c_insert_node_ids = Vec::with_capacity(work_update_capacity);
    let mut c_insert_dummy_ordinals = Vec::with_capacity(work_update_capacity);
    let mut c_insert_tie_positions = Vec::with_capacity(work_update_capacity);
    let mut c_insert_enabled = Vec::with_capacity(work_update_capacity);
    for update in &work_updates {
        c_insert_src_times.push(update.assigned_c_position);
        c_insert_node_ids.push(update.item.node_id);
        c_insert_dummy_ordinals.push(zero);
        c_insert_tie_positions.push(update.assigned_c_position);
        c_insert_enabled.push(update.admitted);
    }

    let mut w_insert_src_times = Vec::with_capacity(ef_search + work_update_capacity);
    let mut w_insert_node_ids = Vec::with_capacity(ef_search + work_update_capacity);
    let mut w_insert_dummy_ordinals = Vec::with_capacity(ef_search + work_update_capacity);
    let mut w_insert_tie_positions = Vec::with_capacity(ef_search + work_update_capacity);
    let mut w_insert_enabled = Vec::with_capacity(ef_search + work_update_capacity);
    for init_slot in 0..ef_search {
        let ordinal = (init_slot + 1) as u32;
        w_insert_src_times.push(builder.constant(u32_to_f(w_initial_position(init_slot))));
        w_insert_node_ids.push(sent_id);
        w_insert_dummy_ordinals.push(builder.constant(u32_to_f(ordinal)));
        w_insert_tie_positions.push(zero);
        w_insert_enabled.push(builder._true());
    }
    for update in &work_updates {
        w_insert_src_times.push(update.w_push_position);
        w_insert_node_ids.push(update.item.node_id);
        w_insert_dummy_ordinals.push(zero);
        w_insert_tie_positions.push(update.assigned_c_position);
        w_insert_enabled.push(update.active);
    }

    let working_update_start = builder.num_gates();
    let mut w_del_times = Vec::with_capacity(work_update_capacity);
    let mut w_del_keys = Vec::with_capacity(work_update_capacity);
    for i in 0..work_update_capacity {
        let update = &work_updates[i];
        let inactive = builder.not(update.active);
        assert_bool_implies_zero(&mut builder, inactive, w_pq.del_src_times[i]);
        assert_bool_implies_eq(&mut builder, inactive, w_pq.del_refs[i].node_id, sent_id);
        assert_bool_implies_zero(&mut builder, inactive, w_pq.del_refs[i].score_enc);
        assert_bool_implies_zero(&mut builder, inactive, w_pq.del_refs[i].model_id);
        assert_bool_implies_zero(&mut builder, inactive, w_pq.del_dummy_ordinals[i]);
        assert_bool_implies_zero(&mut builder, inactive, w_pq.del_tie_positions[i]);

        let self_delete = builder.is_equal(w_pq.del_src_times[i], update.w_push_position);
        let not_self_delete = builder.not(self_delete);
        let expected_admitted = builder.and(update.active, not_self_delete);
        builder.connect(update.admitted.target, expected_admitted.target);

        w_del_times.push(update.w_pop_position);
        let (key, _) = w_packed_priority(
            &mut builder,
            &w_pq.del_refs[i],
            w_pq.del_dummy_ordinals[i],
            w_pq.del_tie_positions[i],
            priority_multiplier_t,
            position_max_t,
            ef_t,
            sent_id,
        );
        let enabled_key = builder.select(update.active, key, zero);
        w_del_keys.push(enabled_key);
    }
    let working_update_gate_count = builder.num_gates() - working_update_start;

    let candidate_round_start = builder.num_gates();
    let mut c_del_times = Vec::with_capacity(rounds_len);
    let mut c_del_out_times = Vec::with_capacity(rounds_len);
    let mut c_del_keys = Vec::with_capacity(rounds_len);
    let mut c_del_enabled = Vec::with_capacity(rounds_len);
    for r in 0..rounds_len {
        let del_time = builder.constant(u32_to_f(c_pop_position(r, round_width)));
        c_del_times.push(del_time);
        let row_idx = upper_len + r;
        let del_enabled = graph_enabled[row_idx];
        c_del_enabled.push(del_enabled);
        let (score, is_real) = ref_score_or_zero_if_sent(&mut builder, &c_pq.del_refs[r], sent_id);
        assert_gate_true(&mut builder, del_enabled, is_real);
        let key = c_packed_priority(
            &mut builder,
            score,
            c_pq.del_tie_positions[r],
            score_max_t,
            priority_multiplier_t,
        );
        c_del_keys.push(builder.select(del_enabled, key, zero));
        c_del_out_times.push(builder.select(del_enabled, del_time, zero));
        builder.connect(c_pq.del_dummy_ordinals[r], zero);
        let inactive = builder.not(del_enabled);
        assert_bool_implies_zero(&mut builder, inactive, c_pq.del_src_times[r]);
        assert_bool_implies_zero(&mut builder, inactive, c_pq.del_tie_positions[r]);
        assert_bool_implies_eq(&mut builder, inactive, c_pq.del_refs[r].node_id, sent_id);
        assert_bool_implies_zero(&mut builder, inactive, c_pq.del_refs[r].score_enc);
        assert_bool_implies_zero(&mut builder, inactive, c_pq.del_refs[r].model_id);
    }

    // A real W eviction certifies saturation. Select the first admitted
    // update's eviction in each round, then perform one threshold comparison.
    // This is equivalent to gating a comparison at every neighbor slot, but
    // avoids paying for `round_width` copies of the same range check. Rounds
    // with no admission (or whose first eviction is a dummy) need no threshold
    // check and are permissible output-preserving stutters in the relation.
    for r in 0..rounds_len {
        let mut no_prior_admit = builder._true();
        let mut selected_evicted_node_id = sent_id;
        let mut selected_evicted_score = zero;
        let mut selected_evicted_tie_position = zero;
        for p in 0..round_width {
            let t = r * round_width + p;
            let update_idx = 1 + t;
            let update = &work_updates[update_idx];
            let first_admit = builder.and(no_prior_admit, update.admitted);
            selected_evicted_node_id = builder.select(
                first_admit,
                w_pq.del_refs[update_idx].node_id,
                selected_evicted_node_id,
            );
            selected_evicted_score = builder.select(
                first_admit,
                w_pq.del_refs[update_idx].score_enc,
                selected_evicted_score,
            );
            selected_evicted_tie_position = builder.select(
                first_admit,
                w_pq.del_tie_positions[update_idx],
                selected_evicted_tie_position,
            );
            let not_admitted = builder.not(update.admitted);
            no_prior_admit = builder.and(no_prior_admit, not_admitted);
        }
        let evicted_is_dummy = builder.is_equal(selected_evicted_node_id, sent_id);
        let threshold_gate = builder.not(evicted_is_dummy);
        let evicted_priority = c_packed_priority(
            &mut builder,
            selected_evicted_score,
            selected_evicted_tie_position,
            score_max_t,
            priority_multiplier_t,
        );
        // Both are packed C priorities of bounded scores and PQ-member tie
        // positions (< 2^62 by packed_priority_parameters).
        assert_nonnegative_diff_if_bounded_operands(
            &mut builder,
            threshold_gate,
            evicted_priority,
            c_del_keys[r],
            GENERIC_MONOTONE_KEY_BITS,
        );
    }
    let candidate_round_gate_count = builder.num_gates() - candidate_round_start;

    let mut c_final_keys = Vec::with_capacity(c_final_len);
    let mut c_final_enabled = Vec::with_capacity(c_final_len);
    for i in 0..c_final_len {
        let final_ref = &c_pq.final_refs[i];
        let (score, is_real) = ref_score_or_zero_if_sent(&mut builder, final_ref, sent_id);
        let key = c_packed_priority(
            &mut builder,
            score,
            c_pq.final_tie_positions[i],
            score_max_t,
            priority_multiplier_t,
        );
        c_final_keys.push(builder.select(is_real, key, zero));
        c_final_enabled.push(is_real);
        builder.connect(c_pq.final_dummy_ordinals[i], zero);
        let is_dummy = builder.not(is_real);
        assert_bool_implies_zero(&mut builder, is_dummy, c_pq.final_src_times[i]);
        assert_bool_implies_zero(&mut builder, is_dummy, c_pq.final_tie_positions[i]);
    }
    let mut c_full_keys = c_del_keys.clone();
    c_full_keys.extend(c_final_keys.iter().copied());
    let mut c_full_times = c_del_out_times.clone();
    c_full_times.extend(c_pq.final_src_times.iter().copied());
    let mut c_full_enabled = c_del_enabled.clone();
    c_full_enabled.extend(c_final_enabled.iter().copied());
    let candidate_prev_lower_start = builder.num_gates();
    assert_prev_lower_with_monotone_stack_enabled(
        &mut builder,
        &c_full_keys,
        &c_full_times,
        &c_full_enabled,
        &c_prev_lower,
        entry_pi,
        output_id_pi,
        1001,
    );
    let candidate_prev_lower_gate_count = builder.num_gates() - candidate_prev_lower_start;
    let candidate_pq_start = builder.num_gates();
    // Every C deletion time is the fixed c_pop_position for its round, so the
    // PQ gadget can reuse that schedule bound while still checking each
    // witness-supplied source and parent time once.
    assert_min_pq_trace_with_bounded_del_times(
        &mut builder,
        &c_insert_src_times,
        &c_insert_node_ids,
        &c_insert_dummy_ordinals,
        &c_insert_tie_positions,
        &c_insert_enabled,
        &c_pq,
        &c_del_times,
        &c_del_enabled,
        &c_final_enabled,
        &c_prev_lower.prev_lower,
        &c_prev_lower.prev_lower_times,
        entry_pi,
        output_id_pi,
        3001,
    );
    let candidate_pq_gate_count = builder.num_gates() - candidate_pq_start;

    let working_prev_lower_start = builder.num_gates();
    let w_final_enabled = vec![builder._true(); w_final_len];
    let mut w_final_keys = Vec::with_capacity(w_final_len);
    for i in 0..w_final_len {
        let (key, _) = w_packed_priority(
            &mut builder,
            &w_pq.final_refs[i],
            w_pq.final_dummy_ordinals[i],
            w_pq.final_tie_positions[i],
            priority_multiplier_t,
            position_max_t,
            ef_t,
            sent_id,
        );
        w_final_keys.push(key);
    }
    let mut w_full_keys = w_del_keys.clone();
    w_full_keys.extend(w_final_keys.iter().copied());
    let mut w_full_times = w_del_times.clone();
    w_full_times.extend(w_pq.final_src_times.iter().copied());
    let w_del_enabled = work_updates
        .iter()
        .map(|update| update.active)
        .collect::<Vec<_>>();
    let mut w_full_enabled = w_del_enabled.clone();
    w_full_enabled.extend(w_final_enabled.iter().copied());
    assert_prev_lower_with_monotone_stack_enabled(
        &mut builder,
        &w_full_keys,
        &w_full_times,
        &w_full_enabled,
        &w_prev_lower,
        entry_pi,
        output_id_pi,
        2001,
    );
    let working_prev_lower_gate_count = builder.num_gates() - working_prev_lower_start;
    let working_pq_start = builder.num_gates();
    // constrain_work_update_schedule connects every W deletion time to its
    // fixed active pop position (or zero when inactive), satisfying the
    // bounded-deletion-time entry point's proof obligation.
    assert_min_pq_trace_with_bounded_del_times(
        &mut builder,
        &w_insert_src_times,
        &w_insert_node_ids,
        &w_insert_dummy_ordinals,
        &w_insert_tie_positions,
        &w_insert_enabled,
        &w_pq,
        &w_del_times,
        &w_del_enabled,
        &w_final_enabled,
        &w_prev_lower.prev_lower,
        &w_prev_lower.prev_lower_times,
        entry_pi,
        output_id_pi,
        4001,
    );
    let working_pq_gate_count = builder.num_gates() - working_pq_start;

    let final_state_start = builder.num_gates();
    let (w_final_real, w_final_scores, w_final_count) =
        assert_worst_to_best_w_state(&mut builder, &w_pq.final_refs, &w_final_keys, sent_id);
    let always = builder._true();
    // w_final_count is a sum of w_final_len Booleans and k a constant.
    assert_nonnegative_diff_if_bounded_operands(
        &mut builder,
        always,
        w_final_count,
        k_t,
        POSITION_BITS,
    );
    let mut topk_expected_refs = Vec::with_capacity(k);
    let mut topk_expected_real = Vec::with_capacity(k);
    for offset in 0..k {
        let idx = w_final_len - 1 - offset;
        topk_expected_refs.push(w_pq.final_refs[idx].clone());
        topk_expected_real.push(w_final_real[idx]);
    }
    assert_final_topk_model_membership_from_sorted(
        &mut builder,
        &topk_expected_refs,
        &topk_expected_real,
        &final_topk_refs,
        output_id_pi,
    );

    if c_final_len > 0 {
        let (c_final_real, _, _) =
            assert_ranked_state(&mut builder, &c_pq.final_refs, &c_final_keys, sent_id);
        let final_c_nonempty = c_final_real[0];
        let w_full = builder.is_equal(w_final_count, ef_t);
        assert_gate_true(&mut builder, final_c_nonempty, w_full);
        let w_worst_c_priority = c_packed_priority(
            &mut builder,
            w_final_scores[0],
            w_pq.final_tie_positions[0],
            score_max_t,
            priority_multiplier_t,
        );
        assert_positive_diff_if_bounded_operands(
            &mut builder,
            final_c_nonempty,
            c_final_keys[0],
            w_worst_c_priority,
            GENERIC_MONOTONE_KEY_BITS,
        );
    }
    let final_state_gate_count = builder.num_gates() - final_state_start;

    let stats = EfSearchBuildStats {
        pre_padding_gate_count: builder.num_gates(),
        score_table_relation_gate_count,
        graph_relation_gate_count,
        trace_target_gate_count,
        visit_permutation_gate_count,
        round_link_gate_count,
        score_table_hash_gate_count,
        graph_merkle_opening_gate_count,
        graph_merkle_leaf_hash_gate_count,
        graph_merkle_path_gate_count,
        dist_ref_membership_gate_count,
        workupd_relation_gate_count,
        working_update_gate_count,
        candidate_round_gate_count,
        candidate_prev_lower_gate_count,
        candidate_pq_gate_count,
        working_prev_lower_gate_count,
        working_pq_gate_count,
        final_state_gate_count,
    };
    let data = builder.build::<C>();
    let targets = EfSearchTargets {
        score_table_hash_pi,
        graph_root_pi,
        nbr_commit_pi,
        output_id_pi,
        entry_pi,
        upper_len_pi,
        rounds_len_pi,
        ef_search_pi,
        top_k_pi,
        node_ids,
        node_opens,
        row_layers,
        row_srcs,
        row_neighbors,
        row_paths,
        row_src_refs,
        row_neighbor_refs,
        expansion_pop_ids,
        visit_is_new,
        seen_sorted_ids,
        seen_sorted_times,
        seen_sorted_is_new,
        work_updates,
        c_pq,
        w_pq,
        c_prev_lower,
        w_prev_lower,
        final_topk_refs,
        ref_batch,
        graph_static_enabled: graph_static_enabled.to_vec(),
    };
    (data, targets, stats)
}

#[allow(clippy::too_many_arguments)]
pub fn fill_ef_search_witness(
    pw: &mut PartialWitness<F>,
    targets: &EfSearchTargets,
    input: &ProofInput,
    entry: NodeId,
    output_model_id: u32,
    score_table_hash: &[u64; 4],
    graph_root: &[u64; 4],
    nbr_commit: &[u64; 4],
    graph_merkle: &GraphMerkleData,
) {
    pw.set_target(targets.output_id_pi, u32_to_f(output_model_id))
        .expect("set ef output");
    pw.set_target(targets.entry_pi, u64_to_f(entry))
        .expect("set ef entry");
    pw.set_target(
        targets.upper_len_pi,
        u32_to_f(input.upper_trace.len() as u32),
    )
    .expect("set ef upper_len");
    pw.set_target(
        targets.rounds_len_pi,
        u32_to_f(input.ef_search.expansions.len() as u32),
    )
    .expect("set ef rounds_len");
    pw.set_target(targets.ef_search_pi, u32_to_f(input.ef_search.ef_search))
        .expect("set ef search");
    pw.set_target(targets.top_k_pi, u32_to_f(input.ef_search.k))
        .expect("set ef top k");
    for (node_t, node) in targets.node_ids.iter().zip_exact(input.nodes.iter()) {
        pw.set_target(*node_t, u64_to_f(canonical_node_id(node.node_id)))
            .expect("set ef node id");
    }
    for (open_t, node) in targets.node_opens.iter().zip_exact(input.nodes.iter()) {
        let node_id = canonical_node_id(node.node_id);
        let (score_enc, model_id) = if node_id == NODE_ID_SENTINEL {
            (0, 0)
        } else {
            (node.score_enc, node.model_id)
        };
        pw.set_target(open_t.score_enc, u64_to_f(score_enc))
            .expect("set ef node score");
        pw.set_target(open_t.model_id, u32_to_f(model_id))
            .expect("set ef node model");
    }

    for (i, row) in (0..targets.row_srcs.len()).zip_exact(&input.expand_rows) {
        if canonical_node_id(row.src.node_id) != NODE_ID_SENTINEL {
            let graph_path = graph_merkle
                .open_by_key(canonical_node_id(row.src.node_id), row.layer)
                .expect("ef_search graph opening");
            for (bit_t, &bit) in targets.row_paths[i]
                .index_bits
                .iter()
                .zip_exact(graph_path.index_bits.iter())
            {
                pw.set_bool_target(*bit_t, bit).expect("set ef row bit");
            }
            for (sib_t, sib) in targets.row_paths[i]
                .siblings
                .iter()
                .zip_exact(graph_path.siblings.iter())
            {
                pw.set_hash_target(*sib_t, crate::commitment::digest_to_hashout(*sib))
                    .expect("set ef row sibling");
            }
        } else {
            for bit_t in &targets.row_paths[i].index_bits {
                pw.set_bool_target(*bit_t, false)
                    .expect("set ef dummy row bit");
            }
            for sib_t in &targets.row_paths[i].siblings {
                pw.set_hash_target(*sib_t, crate::commitment::digest_to_hashout([0, 0, 0, 0]))
                    .expect("set ef dummy row sibling");
            }
        }
        pw.set_target(targets.row_layers[i], u32_to_f(row.layer))
            .expect("set ef row layer");
        pw.set_target(
            targets.row_srcs[i],
            u64_to_f(canonical_node_id(row.src.node_id)),
        )
        .expect("set ef row src");
        for (target, nb_ref) in targets.row_neighbors[i]
            .iter()
            .zip_exact(row.neighbors.iter())
        {
            pw.set_target(*target, u64_to_f(canonical_node_id(nb_ref.node_id)))
                .expect("set ef row neighbor");
        }
        set_dist_ref_witness(pw, &targets.row_src_refs[i], &row.src, "ef row src");
        for (target, nb_ref) in targets.row_neighbor_refs[i]
            .iter()
            .zip_exact(row.neighbors.iter())
        {
            set_dist_ref_witness(pw, target, nb_ref, "ef row neighbor");
        }
    }

    for (flag_t, visit) in targets
        .visit_is_new
        .iter()
        .zip_exact(input.ef_search.neighbor_visits.iter())
    {
        pw.set_bool_target(*flag_t, visit.is_new)
            .expect("set ef visit flag");
    }
    for (target, expansion) in targets
        .expansion_pop_ids
        .iter()
        .zip_exact(input.ef_search.expansions.iter())
    {
        pw.set_target(*target, u64_to_f(canonical_node_id(expansion.pop_id)))
            .expect("set ef expansion pop id");
    }
    let mut seen_sorted = input
        .ef_search
        .neighbor_visits
        .iter()
        .enumerate()
        .map(|(time, visit)| (canonical_node_id(visit.node_id), time as u32, visit.is_new))
        .collect::<Vec<_>>();
    seen_sorted.sort_by_key(|(node_id, time, _)| (*node_id, *time));
    for ((id_t, time_t), (new_t, (node_id, time, is_new))) in targets
        .seen_sorted_ids
        .iter()
        .zip_exact(targets.seen_sorted_times.iter())
        .zip_exact(
            targets
                .seen_sorted_is_new
                .iter()
                .zip_exact(seen_sorted.iter()),
        )
    {
        pw.set_target(*id_t, u64_to_f(*node_id))
            .expect("set seen sorted id");
        pw.set_target(*time_t, u32_to_f(*time))
            .expect("set seen sorted time");
        pw.set_bool_target(*new_t, *is_new)
            .expect("set seen sorted is_new");
    }
    for (target, update) in targets
        .work_updates
        .iter()
        .zip_exact(input.ef_search.work_updates.iter())
    {
        pw.set_bool_target(target.active, update.active)
            .expect("set WorkUpd active");
        set_dist_ref_witness(pw, &target.item, &update.item, "WorkUpd item");
        pw.set_target(
            target.assigned_c_position,
            u32_to_f(update.assigned_c_position),
        )
        .expect("set WorkUpd assigned C position");
        pw.set_target(target.w_push_position, u32_to_f(update.w_push_position))
            .expect("set WorkUpd W push position");
        pw.set_target(target.w_pop_position, u32_to_f(update.w_pop_position))
            .expect("set WorkUpd W pop position");
        pw.set_bool_target(target.admitted, update.admitted)
            .expect("set WorkUpd admitted");
    }
    for (targets, trace) in [
        (&targets.c_pq, &input.ef_search.c_pq),
        (&targets.w_pq, &input.ef_search.w_pq),
    ] {
        for (target_values, values) in [
            (&targets.del_src_times, &trace.del_src_times),
            (&targets.del_dummy_ordinals, &trace.del_dummy_ordinals),
            (&targets.del_tie_positions, &trace.del_tie_positions),
            (&targets.final_src_times, &trace.final_src_times),
            (&targets.final_dummy_ordinals, &trace.final_dummy_ordinals),
            (&targets.final_tie_positions, &trace.final_tie_positions),
        ] {
            for (&target, &value) in target_values.iter().zip_exact(values.iter()) {
                pw.set_target(target, u32_to_f(value))
                    .expect("set queue position");
            }
        }
        for (target_values, values) in [
            (&targets.del_refs, &trace.del_refs),
            (&targets.final_refs, &trace.final_refs),
        ] {
            for (target, value) in target_values.iter().zip_exact(values.iter()) {
                set_dist_ref_witness(pw, target, value, "queue ref");
            }
        }
    }
    fill_prev_lower_witness(pw, &targets.c_prev_lower, &input.ef_search.c_prev_lower);
    fill_prev_lower_witness(pw, &targets.w_prev_lower, &input.ef_search.w_prev_lower);

    for (target_ref, topk) in targets
        .final_topk_refs
        .iter()
        .zip_exact(input.ef_search.final_topk.iter())
    {
        set_dist_ref_fields(
            pw,
            target_ref,
            topk.node_id,
            topk.score_enc,
            topk.model_id,
            "final topk ref",
        );
    }
    let mut raw_refs = Vec::new();
    for (row_idx, row) in input.expand_rows.iter().enumerate() {
        if targets.graph_static_enabled[row_idx] {
            raw_refs.push(row.src.clone());
            raw_refs.extend(row.neighbors.iter().cloned());
        }
    }
    raw_refs.push(input.ef_search.work_updates[0].item.clone());
    raw_refs.extend(input.ef_search.c_pq.final_refs.iter().cloned());
    raw_refs.extend(input.ef_search.w_pq.del_refs.iter().cloned());
    raw_refs.extend(input.ef_search.w_pq.final_refs.iter().cloned());
    fill_dist_ref_batch_witness(
        pw,
        &targets.ref_batch,
        &raw_refs,
        &input.nodes,
        NODE_ID_SENTINEL,
        "ef batch ref",
    );

    pw.set_hash_target(
        targets.score_table_hash_pi,
        crate::commitment::digest_to_hashout(*score_table_hash),
    )
    .expect("set ef score table hash");
    pw.set_hash_target(
        targets.graph_root_pi,
        crate::commitment::digest_to_hashout(*graph_root),
    )
    .expect("set ef graph root");
    pw.set_hash_target(
        targets.nbr_commit_pi,
        crate::commitment::digest_to_hashout(*nbr_commit),
    )
    .expect("set ef nbr commit");
}
