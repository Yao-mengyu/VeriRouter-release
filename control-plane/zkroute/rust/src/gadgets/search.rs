//! Queue accounting, ranked search state, and visited-set constraints.
use crate::circuits::common::{assert_gate_true, count_true};
use crate::gadgets::dist_ref::{add_dist_ref_vec_targets, DistRefTargets};
use crate::gadgets::merkle::poseidon_hash_no_pad;
use crate::gadgets::order::{assert_positive_diff_if_bounded_operands, constrain_bits_le};
use crate::util::{
    u32_to_f, u64_to_f, NodeId, D, F, GENERIC_MONOTONE_KEY_BITS, MULTISET_CHALLENGE_COUNT,
    NODE_ID_BITS, POSITION_BITS,
};

/// `MULTISET_CHALLENGE_COUNT` challenge pairs: two from the transcript hash and
/// further ones from fresh domain-separated queries on that output.
fn challenge_pairs(
    builder: &mut CircuitBuilder<F, D>,
    hash: plonky2::hash::hash_types::HashOutTarget,
    extra_family: u64,
) -> Vec<(Target, Target)> {
    let mut pairs = vec![
        (hash.elements[0], hash.elements[1]),
        (hash.elements[2], hash.elements[3]),
    ];
    let mut extra_input = hash;
    while pairs.len() < MULTISET_CHALLENGE_COUNT {
        extra_input = crate::gadgets::merkle::application_challenge(
            builder,
            extra_family,
            &extra_input.elements,
        );
        pairs.push((extra_input.elements[0], extra_input.elements[1]));
    }
    pairs.truncate(MULTISET_CHALLENGE_COUNT);
    pairs
}
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;

#[derive(Clone)]
pub struct PqTraceTargets {
    pub del_src_times: Vec<Target>,
    pub del_refs: Vec<DistRefTargets>,
    pub del_dummy_ordinals: Vec<Target>,
    pub del_tie_positions: Vec<Target>,
    pub final_src_times: Vec<Target>,
    pub final_refs: Vec<DistRefTargets>,
    pub final_dummy_ordinals: Vec<Target>,
    pub final_tie_positions: Vec<Target>,
}

fn compress4(
    builder: &mut CircuitBuilder<F, D>,
    a: Target,
    b: Target,
    c: Target,
    d: Target,
    r: Target,
    shift: Target,
) -> Target {
    let a_shift = builder.add(a, shift);
    let acc = builder.mul_add(r, b, a_shift);
    let r2 = builder.mul(r, r);
    let acc = builder.mul_add(r2, c, acc);
    let r3 = builder.mul(r2, r);
    builder.mul_add(r3, d, acc)
}

pub fn add_pq_trace_targets(
    builder: &mut CircuitBuilder<F, D>,
    del_len: usize,
    final_len: usize,
) -> PqTraceTargets {
    PqTraceTargets {
        del_src_times: builder.add_virtual_targets(del_len),
        del_refs: add_dist_ref_vec_targets(builder, del_len),
        del_dummy_ordinals: builder.add_virtual_targets(del_len),
        del_tie_positions: builder.add_virtual_targets(del_len),
        final_src_times: builder.add_virtual_targets(final_len),
        final_refs: add_dist_ref_vec_targets(builder, final_len),
        final_dummy_ordinals: builder.add_virtual_targets(final_len),
        final_tie_positions: builder.add_virtual_targets(final_len),
    }
}

fn product_of_tokens(
    builder: &mut CircuitBuilder<F, D>,
    src_times: &[Target],
    node_ids: &[Target],
    dummy_ordinals: &[Target],
    tie_positions: &[Target],
    enabled: &[BoolTarget],
    r: Target,
    shift: Target,
) -> Target {
    assert_eq!(src_times.len(), node_ids.len());
    assert_eq!(src_times.len(), dummy_ordinals.len());
    assert_eq!(src_times.len(), tie_positions.len());
    assert_eq!(src_times.len(), enabled.len());
    let mut acc = builder.one();
    for ((((&src_time, &node_id), &dummy_ordinal), &tie_position), &en) in src_times
        .iter()
        .zip(node_ids.iter())
        .zip(dummy_ordinals.iter())
        .zip(tie_positions.iter())
        .zip(enabled.iter())
    {
        let term = compress4(
            builder,
            src_time,
            node_id,
            dummy_ordinal,
            tie_position,
            r,
            shift,
        );
        let delta = builder.mul_sub(acc, term, acc);
        acc = builder.mul_add(en.target, delta, acc);
    }
    acc
}

fn product_of_ref_tokens(
    builder: &mut CircuitBuilder<F, D>,
    src_times: &[Target],
    refs: &[DistRefTargets],
    dummy_ordinals: &[Target],
    tie_positions: &[Target],
    enabled: &[BoolTarget],
    r: Target,
    shift: Target,
) -> Target {
    product_of_tokens(
        builder,
        src_times,
        &refs.iter().map(|r| r.node_id).collect::<Vec<_>>(),
        dummy_ordinals,
        tie_positions,
        enabled,
        r,
        shift,
    )
}

/// efSearch-only PQ entry point whose deletion times are already constrained
/// by its fixed public schedule. Caller guarantees: every enabled insert
/// source time is below `2^POSITION_BITS`, each enabled insert's source time,
/// dummy ordinal and tie position are fixed by its enable bit and circuit
/// constants, and `full_prev_lower_times` are the parent times of a monotone
/// stack over the same bounded output times.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_min_pq_trace_with_bounded_del_times(
    builder: &mut CircuitBuilder<F, D>,
    insert_src_times: &[Target],
    insert_node_ids: &[Target],
    insert_dummy_ordinals: &[Target],
    insert_tie_positions: &[Target],
    insert_enabled: &[BoolTarget],
    trace: &PqTraceTargets,
    del_times: &[Target],
    del_enabled: &[BoolTarget],
    final_enabled: &[BoolTarget],
    full_prev_lower: &[Target],
    full_prev_lower_times: &[Target],
    seed_a: Target,
    seed_b: Target,
    domain: u32,
) {
    assert_eq!(insert_src_times.len(), insert_node_ids.len());
    assert_eq!(insert_src_times.len(), insert_dummy_ordinals.len());
    assert_eq!(insert_src_times.len(), insert_tie_positions.len());
    assert_eq!(insert_src_times.len(), insert_enabled.len());
    assert_eq!(trace.del_src_times.len(), trace.del_refs.len());
    assert_eq!(trace.del_src_times.len(), trace.del_dummy_ordinals.len());
    assert_eq!(trace.del_src_times.len(), trace.del_tie_positions.len());
    assert_eq!(trace.del_src_times.len(), del_times.len());
    assert_eq!(trace.del_src_times.len(), del_enabled.len());
    assert_eq!(trace.final_src_times.len(), trace.final_refs.len());
    assert_eq!(
        trace.final_src_times.len(),
        trace.final_dummy_ordinals.len()
    );
    assert_eq!(trace.final_src_times.len(), trace.final_tie_positions.len());
    assert_eq!(trace.final_src_times.len(), final_enabled.len());
    assert_eq!(
        full_prev_lower.len(),
        trace.del_src_times.len() + trace.final_src_times.len()
    );
    assert_eq!(full_prev_lower_times.len(), full_prev_lower.len());

    let del_len = trace.del_src_times.len();
    let final_len = trace.final_src_times.len();
    let full_len = del_len + final_len;
    let sentinel = builder.constant(u32_to_f(full_len as u32));

    let domain_t = builder.constant(u32_to_f(domain));
    let mut challenge_inputs = vec![
        seed_a,
        seed_b,
        domain_t,
        builder.constant(u32_to_f(insert_src_times.len() as u32)),
        builder.constant(u32_to_f(del_len as u32)),
        builder.constant(u32_to_f(final_len as u32)),
    ];
    // An enabled insert's src/dummy/tie are fixed by its enable bit and
    // constants; a disabled insert does not enter the product.
    for (&node_id, &enabled) in insert_node_ids.iter().zip(insert_enabled.iter()) {
        for value in [node_id, enabled.target] {
            if builder.target_as_constant(value).is_none() {
                challenge_inputs.push(value);
            }
        }
    }
    for ((((&src_time, r), &dummy_ordinal), &tie_position), &enabled) in trace
        .del_src_times
        .iter()
        .zip(trace.del_refs.iter())
        .zip(trace.del_dummy_ordinals.iter())
        .zip(trace.del_tie_positions.iter())
        .zip(del_enabled.iter())
    {
        challenge_inputs.push(src_time);
        challenge_inputs.push(r.node_id);
        challenge_inputs.push(dummy_ordinal);
        challenge_inputs.push(tie_position);
        challenge_inputs.push(enabled.target);
    }
    for ((((&src_time, r), &dummy_ordinal), &tie_position), &enabled) in trace
        .final_src_times
        .iter()
        .zip(trace.final_refs.iter())
        .zip(trace.final_dummy_ordinals.iter())
        .zip(trace.final_tie_positions.iter())
        .zip(final_enabled.iter())
    {
        challenge_inputs.push(src_time);
        challenge_inputs.push(r.node_id);
        challenge_inputs.push(dummy_ordinal);
        challenge_inputs.push(tie_position);
        challenge_inputs.push(enabled.target);
    }
    let hash = crate::gadgets::merkle::application_challenge(builder, 1, &challenge_inputs);

    for (r, shift) in challenge_pairs(builder, hash, 7) {
        let lhs = product_of_tokens(
            builder,
            insert_src_times,
            insert_node_ids,
            insert_dummy_ordinals,
            insert_tie_positions,
            insert_enabled,
            r,
            shift,
        );
        let rhs_del = product_of_tokens(
            builder,
            &trace.del_src_times,
            &trace.del_refs.iter().map(|r| r.node_id).collect::<Vec<_>>(),
            &trace.del_dummy_ordinals,
            &trace.del_tie_positions,
            del_enabled,
            r,
            shift,
        );
        let rhs_final = product_of_ref_tokens(
            builder,
            &trace.final_src_times,
            &trace.final_refs,
            &trace.final_dummy_ordinals,
            &trace.final_tie_positions,
            final_enabled,
            r,
            shift,
        );
        let rhs = builder.mul(rhs_del, rhs_final);
        builder.connect(lhs, rhs);
    }

    // Every enabled deletion/final source time equals an enabled (bounded)
    // insert source time by the token multiset above, so it needs no range
    // check of its own.

    for ((&del_time, &src_time), &enabled) in del_times
        .iter()
        .zip(trace.del_src_times.iter())
        .zip(del_enabled.iter())
    {
        let gate = enabled;
        assert_positive_diff_if_bounded_operands(builder, gate, del_time, src_time, POSITION_BITS);
    }

    for (idx, (&pg_idx, &pg_time)) in full_prev_lower
        .iter()
        .zip(full_prev_lower_times.iter())
        .enumerate()
    {
        let src_time = if idx < del_len {
            trace.del_src_times[idx]
        } else {
            trace.final_src_times[idx - del_len]
        };
        let is_sentinel = builder.is_equal(pg_idx, sentinel);
        let sentinel_time = builder.mul(is_sentinel.target, pg_time);
        builder.assert_zero(sentinel_time);
        let has_parent = builder.not(is_sentinel);
        let output_enabled = if idx < del_len {
            del_enabled[idx]
        } else {
            final_enabled[idx - del_len]
        };
        let enabled_parent = builder.and(output_enabled, has_parent);
        // The stack carries the parent's (bounded) output time.
        assert_positive_diff_if_bounded_operands(
            builder,
            enabled_parent,
            src_time,
            pg_time,
            POSITION_BITS,
        );
    }
}

pub fn is_new_via_permutation(
    builder: &mut CircuitBuilder<F, D>,
    raw_u: &[Target],
    raw_is_new: &[BoolTarget],
    entry: Target,
    sent_id: NodeId,
    time_base: u64,
    challenge_inputs: &[Target],
) -> (Vec<Target>, Vec<Target>, Vec<BoolTarget>) {
    let t_len = raw_u.len();
    assert_eq!(raw_u.len(), raw_is_new.len());

    let sorted_u: Vec<Target> = builder.add_virtual_targets(t_len);
    let sorted_time: Vec<Target> = builder.add_virtual_targets(t_len);
    let mut sorted_is_new = Vec::with_capacity(t_len);
    for _ in 0..t_len {
        sorted_is_new.push(builder.add_virtual_bool_target_safe());
    }

    let mut sorted_commit_inputs = vec![
        entry,
        builder.constant(u64_to_f(sent_id)),
        builder.constant(u64_to_f(time_base)),
        builder.constant(u32_to_f(t_len as u32)),
    ];
    for ((&sorted_id, &sorted_t), &sorted_new) in sorted_u
        .iter()
        .zip(sorted_time.iter())
        .zip(sorted_is_new.iter())
    {
        sorted_commit_inputs.push(sorted_id);
        sorted_commit_inputs.push(sorted_t);
        sorted_commit_inputs.push(sorted_new.target);
    }
    let sorted_commit = poseidon_hash_no_pad(builder, &sorted_commit_inputs);

    let mut fs_inputs = challenge_inputs.to_vec();
    for (&id, &is_new) in raw_u.iter().zip(raw_is_new) {
        fs_inputs.extend([id, is_new.target]);
    }
    fs_inputs.push(entry);
    fs_inputs.push(builder.constant(u64_to_f(sent_id)));
    fs_inputs.push(builder.constant(u64_to_f(time_base)));
    fs_inputs.push(builder.constant(u32_to_f(t_len as u32)));
    for element in sorted_commit.elements {
        fs_inputs.push(element);
    }
    let fs_hash = crate::gadgets::merkle::application_challenge(builder, 4, &fs_inputs);
    for (r, shift) in challenge_pairs(builder, fs_hash, 8) {
        let mut prod_raw = builder.one();
        let mut prod_sorted = builder.one();
        let mut prod_time_const = builder.one();
        let mut prod_time_sorted = builder.one();

        for i in 0..t_len {
            let time_const = builder.constant(u32_to_f(i as u32));
            let raw_acc = builder.mul_add(raw_is_new[i].target, r, time_const);
            let raw_acc = builder.mul_add(raw_acc, r, raw_u[i]);
            let term_raw = builder.add(raw_acc, shift);
            prod_raw = builder.mul(prod_raw, term_raw);

            let sorted_acc = builder.mul_add(sorted_is_new[i].target, r, sorted_time[i]);
            let sorted_acc = builder.mul_add(sorted_acc, r, sorted_u[i]);
            let term_sorted = builder.add(sorted_acc, shift);
            prod_sorted = builder.mul(prod_sorted, term_sorted);

            let term_const = builder.add(time_const, shift);
            prod_time_const = builder.mul(prod_time_const, term_const);
            let term_time_sorted = builder.add(sorted_time[i], shift);
            prod_time_sorted = builder.mul(prod_time_sorted, term_time_sorted);
        }

        builder.connect(prod_raw, prod_sorted);
        builder.connect(prod_time_const, prod_time_sorted);
    }

    // Every raw id is below `2^NODE_ID_BITS`, so the sorted ids (a
    // permutation of the raw ids) and sorted times (a permutation of
    // `0..t_len`) are bounded, and the two exclusive lexicographic checks
    // collapse into one range check of the selected step minus one: a
    // non-increasing pair makes it at least `p - 2^NODE_ID_BITS`.
    for i in 0..t_len.saturating_sub(1) {
        let same_id = builder.is_equal(sorted_u[i], sorted_u[i + 1]);
        let id_step = builder.sub(sorted_u[i + 1], sorted_u[i]);
        let time_step = builder.sub(sorted_time[i + 1], sorted_time[i]);
        let step = builder.select(same_id, time_step, id_step);
        let one = builder.one();
        let step_minus_one = builder.sub(step, one);
        constrain_bits_le(builder, step_minus_one, NODE_ID_BITS);
    }

    let sent_t = builder.constant(u64_to_f(sent_id));
    for i in 0..t_len {
        let u_i = sorted_u[i];
        let u_is_sent = builder.is_equal(u_i, sent_t);
        let u_is_entry = builder.is_equal(u_i, entry);
        let not_sent = builder.not(u_is_sent);
        let not_entry = builder.not(u_is_entry);
        let active_u = builder.and(not_sent, not_entry);
        let expected_is_new = if i == 0 {
            active_u
        } else {
            let u_prev = sorted_u[i - 1];
            let eq_prev = builder.is_equal(u_i, u_prev);
            let changed = builder.not(eq_prev);
            builder.and(active_u, changed)
        };
        builder.connect(sorted_is_new[i].target, expected_is_new.target);
    }

    (sorted_u, sorted_time, sorted_is_new)
}

pub fn assert_ranked_state(
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
        ranked_scores.push(builder.select(is_real, ranked_ref.score_enc, zero));
    }
    for i in 0..ranked_refs.len().saturating_sub(1) {
        assert_gate_true(builder, ranked_is_real[i + 1], ranked_is_real[i]);
        let both_real = builder.and(ranked_is_real[i], ranked_is_real[i + 1]);
        // C is exposed from best to worst, hence its packed min-priority
        // strictly increases. Assigned positions make real priorities unique;
        // the caller guarantees they are below `2^GENERIC_MONOTONE_KEY_BITS`.
        assert_positive_diff_if_bounded_operands(
            builder,
            both_real,
            ranked_priorities[i + 1],
            ranked_priorities[i],
            GENERIC_MONOTONE_KEY_BITS,
        );
    }
    let ranked_count = count_true(builder, &ranked_is_real);
    (ranked_is_real, ranked_scores, ranked_count)
}

pub fn assert_final_topk_model_membership_from_sorted(
    builder: &mut CircuitBuilder<F, D>,
    ranked_refs: &[DistRefTargets],
    ranked_is_real: &[BoolTarget],
    final_topk_refs: &[DistRefTargets],
    output_model_id: Target,
) {
    assert_eq!(ranked_refs.len(), ranked_is_real.len());
    assert_eq!(ranked_refs.len(), final_topk_refs.len());
    let mut in_topk_models = builder._false();
    for (i, final_topk_ref) in final_topk_refs.iter().enumerate() {
        builder.assert_one(ranked_is_real[i].target);
        builder.connect(final_topk_ref.node_id, ranked_refs[i].node_id);
        builder.connect(final_topk_ref.score_enc, ranked_refs[i].score_enc);
        builder.connect(final_topk_ref.model_id, ranked_refs[i].model_id);
        let model_match = builder.is_equal(final_topk_ref.model_id, output_model_id);
        in_topk_models = builder.or(in_topk_models, model_match);
    }
    builder.assert_one(in_topk_models.target);
}
