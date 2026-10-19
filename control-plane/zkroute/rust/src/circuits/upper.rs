use crate::util::ExactZip;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitData;

use crate::circuits::common::{
    add_merkle_path, add_score_table_row, assert_bool_implies_eq, assert_bool_implies_zero,
    assert_first_max_in_source_order_merged_if, assert_gate_true, connect_hashes,
    constrain_node_id_with_mode, hash_score_table, or_many, verify_graph_openings,
    MerklePathTargets, NodeIdRangeMode, ScoreTableRowTargets,
};
use crate::commit_check::GraphMerkleData;
use crate::gadgets::dist_ref::{
    add_dist_ref_batch_targets, add_dist_ref_targets, add_dist_ref_vec_targets,
    assert_dist_refs_against_opened_table_padded, fill_dist_ref_batch_witness,
    set_dist_ref_witness, DistRefBatchTargets, DistRefTargets,
};
use crate::gadgets::order::assert_positive_diff_if_bounded_operands;
use crate::util::{
    canonical_node_id, recursion_config, u32_to_f, u64_to_f, NodeId, C, D, F, NODE_ID_SENTINEL,
    SCORE_BITS,
};
use crate::witness::ProofInput;

fn assert_gate_zero(builder: &mut CircuitBuilder<F, D>, gate: plonky2::iop::target::BoolTarget) {
    builder.assert_zero(gate.target);
}

#[derive(Clone)]
pub struct UpperTargets {
    pub score_table_hash_pi: plonky2::hash::hash_types::HashOutTarget,
    pub graph_root_pi: plonky2::hash::hash_types::HashOutTarget,
    pub entry_pi: Target,
    pub end_pi: Target,
    pub upper_len_pi: Target,
    pub entry_layer_pi: Target,
    pub node_ids: Vec<Target>,
    pub node_opens: Vec<ScoreTableRowTargets>,
    pub row_layers: Vec<Target>,
    pub row_srcs: Vec<Target>,
    pub row_neighbors: Vec<Vec<Target>>,
    pub row_paths: Vec<MerklePathTargets>,
    pub row_src_refs: Vec<DistRefTargets>,
    pub row_neighbor_refs: Vec<Vec<DistRefTargets>>,
    pub selecteds: Vec<DistRefTargets>,
    pub ref_batch: DistRefBatchTargets,
}

#[derive(Clone, Copy, Debug)]
pub struct UpperBuildStats {
    pub pre_padding_gate_count: usize,
    pub score_table_hash_gate_count: usize,
    pub graph_merkle_opening_gate_count: usize,
    pub graph_merkle_leaf_hash_gate_count: usize,
    pub graph_merkle_path_gate_count: usize,
    pub dist_ref_membership_gate_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn build_upper_circuit(
    use_zk: bool,
    node_count: usize,
    max_upper_steps: usize,
    upper_neighbor_cap: usize,
    graph_tree_height: usize,
    ref_unused_cap: usize,
    node_id_mode: NodeIdRangeMode,
) -> (CircuitData<F, C, D>, UpperTargets, UpperBuildStats) {
    let config = recursion_config(use_zk);
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let score_table_hash_pi = builder.add_virtual_hash_public_input();
    let graph_root_pi = builder.add_virtual_hash_public_input();
    let entry_pi = builder.add_virtual_public_input();
    let end_pi = builder.add_virtual_public_input();
    constrain_node_id_with_mode(&mut builder, entry_pi, node_id_mode);
    constrain_node_id_with_mode(&mut builder, end_pi, node_id_mode);
    let upper_len_pi = builder.add_virtual_public_input();
    let upper_len_const = builder.constant(u32_to_f(max_upper_steps as u32));
    builder.connect(upper_len_pi, upper_len_const);
    let entry_layer_pi = builder.add_virtual_public_input();

    let node_ids = builder.add_virtual_targets(node_count);
    let mut node_opens = Vec::with_capacity(node_count);
    let sent_id = builder.constant(u64_to_f(NODE_ID_SENTINEL));
    for _ in 0..node_count {
        node_opens.push(add_score_table_row(&mut builder));
    }
    for i in 0..node_count {
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

    let mut row_layers = Vec::with_capacity(max_upper_steps);
    let mut row_srcs = Vec::with_capacity(max_upper_steps);
    let mut row_neighbors = Vec::with_capacity(max_upper_steps);
    let mut row_paths = Vec::with_capacity(max_upper_steps);
    for _ in 0..max_upper_steps {
        row_layers.push(builder.add_virtual_target());
        // Row ids need no range checks: they are connected to DistRef node
        // ids (table id or sentinel).
        row_srcs.push(builder.add_virtual_target());
        row_neighbors.push(builder.add_virtual_targets(upper_neighbor_cap));
        row_paths.push(add_merkle_path(&mut builder, graph_tree_height));
    }

    let row_src_refs = (0..max_upper_steps)
        .map(|_| add_dist_ref_targets(&mut builder))
        .collect::<Vec<_>>();
    let row_neighbor_refs = (0..max_upper_steps)
        .map(|_| add_dist_ref_vec_targets(&mut builder, upper_neighbor_cap))
        .collect::<Vec<_>>();
    let selecteds = (0..max_upper_steps)
        .map(|_| add_dist_ref_targets(&mut builder))
        .collect::<Vec<_>>();
    let raw_ref_len = max_upper_steps * (2 + upper_neighbor_cap);
    let ref_batch = add_dist_ref_batch_targets(&mut builder, raw_ref_len, ref_unused_cap);

    for t in 0..max_upper_steps {
        builder.connect(row_srcs[t], row_src_refs[t].node_id);
        for (row_nb, nb_ref) in row_neighbors[t].iter().zip(row_neighbor_refs[t].iter()) {
            builder.connect(*row_nb, nb_ref.node_id);
        }
    }

    let mut step_active = Vec::with_capacity(max_upper_steps);
    for selected in &selecteds {
        let is_sent = builder.is_equal(selected.node_id, sent_id);
        step_active.push(builder.not(is_sent));
    }

    let graph_merkle_start = builder.num_gates();
    let graph_merkle_stats = verify_graph_openings(
        &mut builder,
        &graph_root_pi,
        &row_layers,
        &row_srcs,
        &row_neighbors,
        &row_paths,
        &step_active,
    );
    let graph_merkle_opening_gate_count = builder.num_gates() - graph_merkle_start;

    let mut all_refs = Vec::new();
    all_refs.extend(row_src_refs.iter().cloned());
    for refs in &row_neighbor_refs {
        all_refs.extend(refs.iter().cloned());
    }
    all_refs.extend(selecteds.iter().cloned());
    let dist_ref_start = builder.num_gates();
    assert_dist_refs_against_opened_table_padded(
        &mut builder,
        &all_refs,
        &ref_batch,
        &score_table_hash_pi,
        &node_ids,
        &node_opens,
        sent_id,
        entry_pi,
        end_pi,
        5101,
        true,
    );
    let dist_ref_membership_gate_count = builder.num_gates() - dist_ref_start;

    let zero = builder.zero();
    if max_upper_steps == 0 {
        builder.connect(entry_layer_pi, zero);
    } else {
        let first_active = step_active[0];
        assert_bool_implies_eq(&mut builder, first_active, row_layers[0], entry_layer_pi);
        let first_inactive = builder.not(first_active);
        assert_bool_implies_zero(&mut builder, first_inactive, entry_layer_pi);
    }
    for t in 0..max_upper_steps {
        let active = step_active[t];
        let inactive = builder.not(active);
        assert_bool_implies_eq(&mut builder, inactive, row_src_refs[t].node_id, sent_id);
        for nb_ref in &row_neighbor_refs[t] {
            assert_bool_implies_eq(&mut builder, inactive, nb_ref.node_id, sent_id);
        }

        if t == 0 {
            assert_bool_implies_eq(&mut builder, active, row_srcs[t], entry_pi);
        } else {
            assert_gate_true(&mut builder, active, step_active[t - 1]);
            assert_bool_implies_eq(&mut builder, active, row_srcs[t], selecteds[t - 1].node_id);

            let same_layer = builder.is_equal(row_layers[t], row_layers[t - 1]);
            let layer_drop = builder.sub(row_layers[t - 1], row_layers[t]);
            let one = builder.one();
            let drop_by_one = builder.is_equal(layer_drop, one);
            let valid_transition = builder.or(same_layer, drop_by_one);
            assert_gate_true(&mut builder, active, valid_transition);

            let prev_stop = builder.is_equal(selecteds[t - 1].node_id, row_srcs[t - 1]);
            let prev_move = builder.not(prev_stop);
            let same_and_stop = builder.and(same_layer, prev_stop);
            let bad_same_stop = builder.and(active, same_and_stop);
            assert_gate_zero(&mut builder, bad_same_stop);
            let drop_and_move = builder.and(drop_by_one, prev_move);
            let bad_drop_move = builder.and(active, drop_and_move);
            assert_gate_zero(&mut builder, bad_drop_move);
        }

        let is_l0 = builder.is_equal(row_layers[t], zero);
        let active_and_l0 = builder.and(active, is_l0);
        assert_gate_zero(&mut builder, active_and_l0);

        let mut candidates = Vec::with_capacity(1 + upper_neighbor_cap);
        candidates.push(row_srcs[t]);
        candidates.extend(row_neighbors[t].iter().copied());
        let mut candidate_scores = Vec::with_capacity(1 + upper_neighbor_cap);
        candidate_scores.push(row_src_refs[t].score_enc);
        candidate_scores.extend(row_neighbor_refs[t].iter().map(|r| r.score_enc));

        assert_first_max_in_source_order_merged_if(
            &mut builder,
            active,
            selecteds[t].node_id,
            selecteds[t].score_enc,
            &candidates,
            &candidate_scores,
        );
        let selected_is_src = builder.is_equal(selecteds[t].node_id, row_srcs[t]);
        let selected_not_src = builder.not(selected_is_src);
        let selected_moved = builder.and(active, selected_not_src);
        // Both scores were range-checked under `active` above, and
        // selected_moved implies active.
        assert_positive_diff_if_bounded_operands(
            &mut builder,
            selected_moved,
            selecteds[t].score_enc,
            row_src_refs[t].score_enc,
            SCORE_BITS,
        );
    }

    let any_active = or_many(&mut builder, &step_active);
    for t in 0..max_upper_steps {
        let is_last = if t + 1 == max_upper_steps {
            step_active[t]
        } else {
            let next_inactive = builder.not(step_active[t + 1]);
            builder.and(step_active[t], next_inactive)
        };
        assert_bool_implies_eq(&mut builder, is_last, selecteds[t].node_id, row_srcs[t]);
        assert_bool_implies_eq(&mut builder, is_last, end_pi, row_srcs[t]);
        let one = builder.one();
        assert_bool_implies_eq(&mut builder, is_last, row_layers[t], one);
    }
    let no_active = builder.not(any_active);
    assert_bool_implies_eq(&mut builder, no_active, end_pi, entry_pi);

    let stats = UpperBuildStats {
        pre_padding_gate_count: builder.num_gates(),
        score_table_hash_gate_count,
        graph_merkle_opening_gate_count,
        graph_merkle_leaf_hash_gate_count: graph_merkle_stats.leaf_hash_gate_count,
        graph_merkle_path_gate_count: graph_merkle_stats.path_gate_count,
        dist_ref_membership_gate_count,
    };
    let data = builder.build::<C>();
    let targets = UpperTargets {
        score_table_hash_pi,
        graph_root_pi,
        entry_pi,
        end_pi,
        upper_len_pi,
        entry_layer_pi,
        node_ids,
        node_opens,
        row_layers,
        row_srcs,
        row_neighbors,
        row_paths,
        row_src_refs,
        row_neighbor_refs,
        selecteds,
        ref_batch,
    };
    (data, targets, stats)
}

fn set_zero_hash_target(
    pw: &mut PartialWitness<F>,
    target: plonky2::hash::hash_types::HashOutTarget,
    label: &str,
) {
    pw.set_hash_target(target, crate::commitment::digest_to_hashout([0, 0, 0, 0]))
        .unwrap_or_else(|_| panic!("set {label} zero hash"));
}

pub fn fill_upper_witness(
    pw: &mut PartialWitness<F>,
    targets: &UpperTargets,
    input: &ProofInput,
    score_table_hash: &[u64; 4],
    graph_root: &[u64; 4],
    entry: NodeId,
    end: NodeId,
    entry_layer: u32,
    graph_merkle: &GraphMerkleData,
) {
    pw.set_target(targets.entry_pi, u64_to_f(canonical_node_id(entry)))
        .expect("set upper padded entry");
    pw.set_target(targets.end_pi, u64_to_f(canonical_node_id(end)))
        .expect("set upper padded end");
    pw.set_target(
        targets.upper_len_pi,
        u32_to_f(targets.selecteds.len() as u32),
    )
    .expect("set upper padded len");
    pw.set_target(targets.entry_layer_pi, u32_to_f(entry_layer))
        .expect("set upper padded entry layer");

    for (node_t, node) in targets.node_ids.iter().zip_exact(input.nodes.iter()) {
        pw.set_target(*node_t, u64_to_f(canonical_node_id(node.node_id)))
            .expect("set upper padded node id");
    }
    for (open_t, node) in targets.node_opens.iter().zip_exact(input.nodes.iter()) {
        let node_id = canonical_node_id(node.node_id);
        let (score_enc, model_id) = if node_id == NODE_ID_SENTINEL {
            (0, 0)
        } else {
            (node.score_enc, node.model_id)
        };
        pw.set_target(open_t.score_enc, u64_to_f(score_enc))
            .expect("set upper padded node score");
        pw.set_target(open_t.model_id, u32_to_f(model_id))
            .expect("set upper padded node model");
    }

    for i in 0..targets.row_layers.len() {
        let row = &input.expand_rows[i];
        if canonical_node_id(row.src.node_id) != NODE_ID_SENTINEL {
            let graph_path = graph_merkle
                .open_by_key(canonical_node_id(row.src.node_id), row.layer)
                .expect("upper padded graph opening");
            for (bit_t, &bit) in targets.row_paths[i]
                .index_bits
                .iter()
                .zip_exact(graph_path.index_bits.iter())
            {
                pw.set_bool_target(*bit_t, bit)
                    .expect("set upper padded row bit");
            }
            for (sib_t, sib) in targets.row_paths[i]
                .siblings
                .iter()
                .zip_exact(graph_path.siblings.iter())
            {
                pw.set_hash_target(*sib_t, crate::commitment::digest_to_hashout(*sib))
                    .expect("set upper padded row sibling");
            }
        } else {
            for bit_t in &targets.row_paths[i].index_bits {
                pw.set_bool_target(*bit_t, false)
                    .expect("set upper padded dummy row bit");
            }
            for sib_t in &targets.row_paths[i].siblings {
                set_zero_hash_target(pw, *sib_t, "upper padded dummy row sibling");
            }
        }
        pw.set_target(targets.row_layers[i], u32_to_f(row.layer))
            .expect("set upper padded row layer");
        pw.set_target(
            targets.row_srcs[i],
            u64_to_f(canonical_node_id(row.src.node_id)),
        )
        .expect("set upper padded row src");
        for (target, nb_ref) in targets.row_neighbors[i]
            .iter()
            .zip_exact(row.neighbors.iter())
        {
            pw.set_target(*target, u64_to_f(canonical_node_id(nb_ref.node_id)))
                .expect("set upper padded row neighbor");
        }
        set_dist_ref_witness(
            pw,
            &targets.row_src_refs[i],
            &row.src,
            "upper padded row src",
        );
        for (target, nb_ref) in targets.row_neighbor_refs[i]
            .iter()
            .zip_exact(row.neighbors.iter())
        {
            set_dist_ref_witness(pw, target, nb_ref, "upper padded nb");
        }
    }

    for (target, step) in targets.selecteds.iter().zip_exact(input.upper_trace.iter()) {
        set_dist_ref_witness(pw, target, &step.selected, "upper padded selected");
    }

    let mut raw_refs = Vec::new();
    raw_refs.extend(
        input
            .expand_rows
            .iter()
            .take(targets.selecteds.len())
            .map(|row| row.src.clone()),
    );
    for row in input.expand_rows.iter().take(targets.selecteds.len()) {
        raw_refs.extend(row.neighbors.iter().cloned());
    }
    raw_refs.extend(input.upper_trace.iter().map(|step| step.selected.clone()));
    fill_dist_ref_batch_witness(
        pw,
        &targets.ref_batch,
        &raw_refs,
        &input.nodes,
        NODE_ID_SENTINEL,
        "upper padded batch ref",
    );

    pw.set_hash_target(
        targets.score_table_hash_pi,
        crate::commitment::digest_to_hashout(*score_table_hash),
    )
    .expect("set upper padded score table hash");
    pw.set_hash_target(
        targets.graph_root_pi,
        crate::commitment::digest_to_hashout(*graph_root),
    )
    .expect("set upper padded graph_root");
}
