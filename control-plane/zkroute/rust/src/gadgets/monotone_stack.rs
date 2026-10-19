use crate::circuits::common::assert_gate_true;
use crate::util::{u64_to_f, ExactZip};
use crate::witness::PrevLowerTrace;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::gadgets::order::{
    assert_nonnegative_diff_if_bounded_operands,
    assert_nonnegative_diff_with_offset_if_bounded_operands,
    assert_positive_diff_if_bounded_operands,
};
use crate::util::{u32_to_f, D, F, GENERIC_MONOTONE_KEY_BITS, MULTISET_CHALLENGE_COUNT, POSITION_BITS};

#[derive(Clone)]
pub struct PrevLowerEventTargets {
    pub active: BoolTarget,
    pub is_push: BoolTarget,
    pub cur_index: Target,
    pub cur_key: Target,
    pub cur_time: Target,
    pub top_before_index: Target,
    pub top_before_key: Target,
    pub top_before_time: Target,
    pub top_after_index: Target,
    pub top_after_key: Target,
    pub top_after_time: Target,
}

#[derive(Clone)]
pub struct PrevLowerFinalNodeTargets {
    pub node_index: Target,
    pub node_key: Target,
    pub node_time: Target,
    pub parent_index: Target,
    pub parent_key: Target,
    pub parent_time: Target,
}

#[derive(Clone)]
pub struct PrevLowerTraceTargets {
    pub active_capacity: usize,
    pub prev_lower: Vec<Target>,
    pub prev_lower_times: Vec<Target>,
    pub events: Vec<PrevLowerEventTargets>,
    pub final_nodes: Vec<PrevLowerFinalNodeTargets>,
}

fn compress5(
    builder: &mut CircuitBuilder<F, D>,
    a: Target,
    b: Target,
    c: Target,
    d: Target,
    e: Target,
    beta: Target,
    gamma: Target,
    delta: Target,
    epsilon: Target,
    zeta: Target,
) -> Target {
    let acc = builder.add(a, zeta);
    let acc = builder.mul_add(beta, b, acc);
    let acc = builder.mul_add(gamma, c, acc);
    let acc = builder.mul_add(delta, d, acc);
    builder.mul_add(epsilon, e, acc)
}

fn compress6(
    builder: &mut CircuitBuilder<F, D>,
    a: Target,
    b: Target,
    c: Target,
    d: Target,
    e: Target,
    f: Target,
    beta: Target,
    gamma: Target,
    delta: Target,
    epsilon: Target,
    zeta: Target,
    eta: Target,
) -> Target {
    let acc = builder.add(a, eta);
    let acc = builder.mul_add(beta, b, acc);
    let acc = builder.mul_add(gamma, c, acc);
    let acc = builder.mul_add(delta, d, acc);
    let acc = builder.mul_add(epsilon, e, acc);
    builder.mul_add(zeta, f, acc)
}

fn compress_fields_horner(
    builder: &mut CircuitBuilder<F, D>,
    fields: &[Target],
    r: Target,
    shift: Target,
) -> Target {
    let mut acc = *fields.last().expect("nonempty tuple");
    for &field in fields[..fields.len() - 1].iter().rev() {
        acc = builder.mul_add(acc, r, field);
    }
    builder.add(acc, shift)
}

fn mul_if(
    builder: &mut CircuitBuilder<F, D>,
    acc: Target,
    gate: BoolTarget,
    term: Target,
) -> Target {
    let delta = builder.mul_sub(acc, term, acc);
    builder.mul_add(gate.target, delta, acc)
}

#[allow(clippy::too_many_arguments)]
fn assert_monotone_products_with_challenge(
    builder: &mut CircuitBuilder<F, D>,
    keys: &[Target],
    times: &[Target],
    enabled: &[BoolTarget],
    trace: &PrevLowerTraceTargets,
    r_prev: Target,
    shift_prev: Target,
    r_edge: Target,
    shift_edge: Target,
) {
    let one = builder.one();
    let mut prev_prod = one;
    for idx in 0..keys.len() {
        let index = builder.constant(u32_to_f(idx as u32));
        let term = compress_fields_horner(
            builder,
            &[
                index,
                keys[idx],
                trace.prev_lower[idx],
                trace.prev_lower_times[idx],
                times[idx],
            ],
            r_prev,
            shift_prev,
        );
        prev_prod = mul_if(builder, prev_prod, enabled[idx], term);
    }

    let mut push_prod = one;
    let mut push_edge_prod = one;
    let mut pop_edge_prod = one;
    for event in &trace.events {
        let active_push = builder.and(event.active, event.is_push);
        let is_pop = builder.not(event.is_push);
        let active_pop = builder.and(event.active, is_pop);
        let prev_term = compress_fields_horner(
            builder,
            &[
                event.cur_index,
                event.cur_key,
                event.top_before_index,
                event.top_before_time,
                event.cur_time,
            ],
            r_prev,
            shift_prev,
        );
        push_prod = mul_if(builder, push_prod, active_push, prev_term);

        let push_edge = compress_fields_horner(
            builder,
            &[
                event.cur_index,
                event.cur_key,
                event.cur_time,
                event.top_before_index,
                event.top_before_key,
                event.top_before_time,
            ],
            r_edge,
            shift_edge,
        );
        push_edge_prod = mul_if(builder, push_edge_prod, active_push, push_edge);

        let pop_edge = compress_fields_horner(
            builder,
            &[
                event.top_before_index,
                event.top_before_key,
                event.top_before_time,
                event.top_after_index,
                event.top_after_key,
                event.top_after_time,
            ],
            r_edge,
            shift_edge,
        );
        pop_edge_prod = mul_if(builder, pop_edge_prod, active_pop, pop_edge);
    }
    builder.connect(prev_prod, push_prod);

    let sentinel_index = builder.constant(u32_to_f(keys.len() as u32));
    let mut final_edge_prod = one;
    for node in &trace.final_nodes {
        let is_sentinel = builder.is_equal(node.node_index, sentinel_index);
        let is_active = builder.not(is_sentinel);
        let edge = compress_fields_horner(
            builder,
            &[
                node.node_index,
                node.node_key,
                node.node_time,
                node.parent_index,
                node.parent_key,
                node.parent_time,
            ],
            r_edge,
            shift_edge,
        );
        final_edge_prod = mul_if(builder, final_edge_prod, is_active, edge);
    }
    let rhs_edges = builder.mul(pop_edge_prod, final_edge_prod);
    builder.connect(push_edge_prod, rhs_edges);
}

pub fn add_prev_lower_trace_targets(
    builder: &mut CircuitBuilder<F, D>,
    key_len: usize,
) -> PrevLowerTraceTargets {
    add_prev_lower_trace_targets_with_active_capacity(builder, key_len, key_len)
}

pub fn add_prev_lower_trace_targets_with_active_capacity(
    builder: &mut CircuitBuilder<F, D>,
    key_len: usize,
    active_capacity: usize,
) -> PrevLowerTraceTargets {
    assert!(
        active_capacity <= key_len,
        "active prev-lower capacity cannot exceed the physical key length"
    );
    let event_len = 2 * usize::max(1, active_capacity);
    let final_len = usize::max(1, active_capacity);
    let mut prev_lower = Vec::with_capacity(key_len);
    let mut prev_lower_times = Vec::with_capacity(key_len);
    for _ in 0..key_len {
        prev_lower.push(builder.add_virtual_target());
        prev_lower_times.push(builder.add_virtual_target());
    }
    let mut events = Vec::with_capacity(event_len);
    for _ in 0..event_len {
        events.push(PrevLowerEventTargets {
            active: builder.add_virtual_bool_target_safe(),
            is_push: builder.add_virtual_bool_target_safe(),
            cur_index: builder.add_virtual_target(),
            cur_key: builder.add_virtual_target(),
            cur_time: builder.add_virtual_target(),
            top_before_index: builder.add_virtual_target(),
            top_before_key: builder.add_virtual_target(),
            top_before_time: builder.add_virtual_target(),
            top_after_index: builder.add_virtual_target(),
            top_after_key: builder.add_virtual_target(),
            top_after_time: builder.add_virtual_target(),
        });
    }
    let mut final_nodes = Vec::with_capacity(final_len);
    for _ in 0..final_len {
        final_nodes.push(PrevLowerFinalNodeTargets {
            node_index: builder.add_virtual_target(),
            node_key: builder.add_virtual_target(),
            node_time: builder.add_virtual_target(),
            parent_index: builder.add_virtual_target(),
            parent_key: builder.add_virtual_target(),
            parent_time: builder.add_virtual_target(),
        });
    }
    PrevLowerTraceTargets {
        active_capacity,
        prev_lower,
        prev_lower_times,
        events,
        final_nodes,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn assert_prev_lower_with_monotone_stack_enabled(
    builder: &mut CircuitBuilder<F, D>,
    keys: &[Target],
    times: &[Target],
    enabled: &[BoolTarget],
    trace: &PrevLowerTraceTargets,
    seed_a: Target,
    seed_b: Target,
    domain: u32,
) {
    let zero = builder.zero();
    let count = keys.len();
    assert_eq!(times.len(), count);
    assert_eq!(enabled.len(), count);

    assert_eq!(trace.prev_lower.len(), count);
    assert_eq!(trace.prev_lower_times.len(), count);
    let sentinel_index = builder.constant(u32_to_f(count as u32));

    let mut challenge_inputs = vec![
        seed_a,
        seed_b,
        builder.constant(u32_to_f(domain)),
        builder.constant(u32_to_f(count as u32)),
        builder.constant(u32_to_f(trace.events.len() as u32)),
        builder.constant(u32_to_f(trace.final_nodes.len() as u32)),
    ];
    for (idx, ((&key, &time), (&pg, &pg_time))) in keys
        .iter()
        .zip(times.iter())
        .zip(trace.prev_lower.iter().zip(trace.prev_lower_times.iter()))
        .enumerate()
    {
        challenge_inputs.push(enabled[idx].target);
        challenge_inputs.push(key);

        challenge_inputs.push(time);
        challenge_inputs.push(pg);
        challenge_inputs.push(pg_time);
    }
    for event in &trace.events {
        challenge_inputs.push(event.active.target);
        challenge_inputs.push(event.is_push.target);
        challenge_inputs.push(event.cur_index);
        challenge_inputs.push(event.cur_key);

        challenge_inputs.push(event.cur_time);
        // top_before_* is connected (when active) to the running top, a
        // function of earlier hashed event fields; inactive events do not
        // enter any challenge-dependent product.
        challenge_inputs.push(event.top_after_index);
        challenge_inputs.push(event.top_after_key);

        challenge_inputs.push(event.top_after_time);
    }
    for node in &trace.final_nodes {
        // An active final node's key/time are connected to the final top or
        // to the previous node's hashed parent fields; node_index stays
        // hashed because it fixes which nodes are active.
        challenge_inputs.push(node.node_index);
        challenge_inputs.push(node.parent_index);
        challenge_inputs.push(node.parent_key);

        challenge_inputs.push(node.parent_time);
    }
    let h = crate::gadgets::merkle::application_challenge(builder, 2, &challenge_inputs);
    let beta_prev = h.elements[0];
    let shift_prev = h.elements[1];
    let beta_edge = h.elements[2];
    let shift_edge = h.elements[3];
    let beta_prev2 = builder.mul(beta_prev, beta_prev);
    let beta_prev3 = builder.mul(beta_prev2, beta_prev);
    let beta_prev4 = builder.mul(beta_prev3, beta_prev);
    let beta_edge2 = builder.mul(beta_edge, beta_edge);
    let beta_edge3 = builder.mul(beta_edge2, beta_edge);
    let beta_edge4 = builder.mul(beta_edge3, beta_edge);
    let beta_edge5 = builder.mul(beta_edge4, beta_edge);

    let mut prev_prod = builder.one();
    for (idx, (((&key, &time), &pg), &pg_time)) in keys
        .iter()
        .zip(times.iter())
        .zip(trace.prev_lower.iter())
        .zip(trace.prev_lower_times.iter())
        .enumerate()
    {
        let idx_t = builder.constant(u32_to_f(idx as u32));
        let term = compress5(
            builder, idx_t, key, pg, pg_time, time, beta_prev, beta_prev2, beta_prev3, beta_prev4,
            shift_prev,
        );
        prev_prod = mul_if(builder, prev_prod, enabled[idx], term);

        // Disabled physical slots are not members of OutSeq. Keeping their
        // parent fields canonical prevents them from being smuggled into the
        // removal-position lookup through unconstrained padding.
        let disabled = builder.not(enabled[idx]);
        let pg_diff = builder.sub(pg, sentinel_index);
        let pg_gated = builder.mul(pg_diff, disabled.target);
        builder.assert_zero(pg_gated);
        let pg_time_gated = builder.mul(pg_time, disabled.target);
        builder.assert_zero(pg_time_gated);
    }

    let mut push_prod = builder.one();
    let mut push_edge_prod = builder.one();
    let mut pop_edge_prod = builder.one();
    let mut final_edge_prod = builder.one();

    let mut enabled_count = builder.zero();
    for flag in enabled {
        enabled_count = builder.add(enabled_count, flag.target);
    }
    // The caller may allocate fewer event/final-chain records than physical
    // key slots when it has an independently proved active-row upper bound.
    // Enforce that bound explicitly; event count alone is not sufficient
    // because pushes and pops can partially cancel.
    let active_capacity = builder.constant(u32_to_f(trace.active_capacity as u32));
    let always = builder._true();
    // active_capacity is a constant and enabled_count a sum of at most
    // `count < 2^32` Booleans, so both operands are already in range.
    assert_nonnegative_diff_if_bounded_operands(
        builder,
        always,
        active_capacity,
        enabled_count,
        POSITION_BITS,
    );
    let mut pushed = builder.zero();
    let mut top_index = sentinel_index;
    let mut top_key = zero;
    let mut top_time = zero;
    let mut block_key = zero;
    let mut block_time = zero;
    let mut expect_new_block = builder._true();
    let mut have_block = builder._false();
    let mut previous_block_index = zero;
    let mut seen_event_end = builder._false();

    for event in &trace.events {
        let active = event.active;
        let active_after_end = builder.mul(active.target, seen_event_end.target);
        builder.assert_zero(active_after_end);
        let active_push = builder.and(active, event.is_push);
        let not_is_push = builder.not(event.is_push);
        let active_pop = builder.and(active, not_is_push);

        // `cur_index < count` is implied: an active event's cur_index equals
        // that of the push closing its block, an enabled constant index by
        // the prev-row multiset.

        let before_index_diff = builder.sub(event.top_before_index, top_index);
        let before_index_gated = builder.mul(before_index_diff, active.target);
        builder.assert_zero(before_index_gated);
        let before_key_diff = builder.sub(event.top_before_key, top_key);
        let before_key_gated = builder.mul(before_key_diff, active.target);
        builder.assert_zero(before_key_gated);

        let before_time_diff = builder.sub(event.top_before_time, top_time);
        let before_time_gated = builder.mul(before_time_diff, active.target);
        builder.assert_zero(before_time_gated);

        let new_block = builder.and(active, expect_new_block);
        let not_expect_new_block = builder.not(expect_new_block);
        let continuing_block = builder.and(active, not_expect_new_block);
        let ordered_block = builder.and(new_block, have_block);
        // Both operands are cur_indexes of block-closing pushes, i.e.
        // enabled constant indexes below count < 2^32.
        assert_positive_diff_if_bounded_operands(
            builder,
            ordered_block,
            event.cur_index,
            previous_block_index,
            POSITION_BITS,
        );
        previous_block_index = builder.select(new_block, event.cur_index, previous_block_index);
        have_block = builder.or(have_block, new_block);

        let block_index_diff = builder.sub(event.cur_index, previous_block_index);
        let block_index_gated = builder.mul(block_index_diff, continuing_block.target);
        builder.assert_zero(block_index_gated);
        let block_key_diff = builder.sub(event.cur_key, block_key);
        let block_key_gated = builder.mul(block_key_diff, continuing_block.target);
        builder.assert_zero(block_key_gated);

        let block_time_diff = builder.sub(event.cur_time, block_time);
        let block_time_gated = builder.mul(block_time_diff, continuing_block.target);
        builder.assert_zero(block_time_gated);
        block_key = builder.select(new_block, event.cur_key, block_key);

        block_time = builder.select(new_block, event.cur_time, block_time);

        let top_is_sentinel = builder.is_equal(top_index, sentinel_index);
        let top_is_real = builder.not(top_is_sentinel);
        assert_gate_true(builder, active_pop, top_is_real);

        // Push and pop are mutually exclusive. Select the comparison
        // direction first so a packed-key stack pays for one checked
        // range comparison per active event: pop requires cur >= top,
        // while a push onto a nonempty stack requires top >= cur + 1.
        let needs_order = builder.and(active, top_is_real);
        let order_lhs = builder.select(event.is_push, event.top_before_key, event.cur_key);
        let order_rhs = builder.select(event.is_push, event.cur_key, event.top_before_key);
        // cur_key is keys[cur_index] and the running top is the sentinel
        // (key 0) or a pushed tuple, so both operands are enabled keys.
        assert_nonnegative_diff_with_offset_if_bounded_operands(
            builder,
            needs_order,
            order_lhs,
            order_rhs,
            event.is_push,
            GENERIC_MONOTONE_KEY_BITS,
        );

        let push_after_index_diff = builder.sub(event.top_after_index, event.cur_index);
        let push_after_index_gated = builder.mul(push_after_index_diff, active_push.target);
        builder.assert_zero(push_after_index_gated);
        let push_after_key_diff = builder.sub(event.top_after_key, event.cur_key);
        let push_after_key_gated = builder.mul(push_after_key_diff, active_push.target);
        builder.assert_zero(push_after_key_gated);

        let push_after_time_diff = builder.sub(event.top_after_time, event.cur_time);
        let push_after_time_gated = builder.mul(push_after_time_diff, active_push.target);
        builder.assert_zero(push_after_time_gated);

        let inactive = builder.not(active);
        seen_event_end = builder.or(seen_event_end, inactive);
        let inactive_push = builder.mul(inactive.target, event.is_push.target);
        builder.assert_zero(inactive_push);

        let prev_term = compress5(
            builder,
            event.cur_index,
            event.cur_key,
            event.top_before_index,
            event.top_before_time,
            event.cur_time,
            beta_prev,
            beta_prev2,
            beta_prev3,
            beta_prev4,
            shift_prev,
        );
        push_prod = mul_if(builder, push_prod, active_push, prev_term);

        let push_edge = compress6(
            builder,
            event.cur_index,
            event.cur_key,
            event.cur_time,
            event.top_before_index,
            event.top_before_key,
            event.top_before_time,
            beta_edge,
            beta_edge2,
            beta_edge3,
            beta_edge4,
            beta_edge5,
            shift_edge,
        );
        push_edge_prod = mul_if(builder, push_edge_prod, active_push, push_edge);

        let pop_edge = compress6(
            builder,
            event.top_before_index,
            event.top_before_key,
            event.top_before_time,
            event.top_after_index,
            event.top_after_key,
            event.top_after_time,
            beta_edge,
            beta_edge2,
            beta_edge3,
            beta_edge4,
            beta_edge5,
            shift_edge,
        );
        pop_edge_prod = mul_if(builder, pop_edge_prod, active_pop, pop_edge);

        pushed = builder.add(pushed, active_push.target);

        let next_top_index_active =
            builder.select(event.is_push, event.cur_index, event.top_after_index);
        let next_top_key_active = builder.select(event.is_push, event.cur_key, event.top_after_key);
        let next_top_time_active =
            builder.select(event.is_push, event.cur_time, event.top_after_time);
        top_index = builder.select(active, next_top_index_active, top_index);
        top_key = builder.select(active, next_top_key_active, top_key);

        top_time = builder.select(active, next_top_time_active, top_time);

        let next_expect = builder.select(active, event.is_push.target, expect_new_block.target);
        expect_new_block = BoolTarget::new_unsafe(next_expect);
    }

    // Every active physical OutSeq element contributes a complete block of
    // zero or more stack pops followed by exactly one push. A pop-only suffix
    // would otherwise satisfy the edge products while omitting the last push.
    builder.assert_one(expect_new_block.target);

    builder.connect(prev_prod, push_prod);
    builder.connect(pushed, enabled_count);

    let mut seen_end = builder._false();
    let mut prev_active = builder._false();
    let mut prev_parent_is_sentinel = builder._false();
    let mut prev_parent_index = sentinel_index;
    let mut prev_parent_key = zero;
    let mut prev_parent_time = zero;
    let mut first_active = builder._false();

    for node in &trace.final_nodes {
        let node_is_sentinel = builder.is_equal(node.node_index, sentinel_index);
        let is_active = builder.not(node_is_sentinel);

        let edge = compress6(
            builder,
            node.node_index,
            node.node_key,
            node.node_time,
            node.parent_index,
            node.parent_key,
            node.parent_time,
            beta_edge,
            beta_edge2,
            beta_edge3,
            beta_edge4,
            beta_edge5,
            shift_edge,
        );
        final_edge_prod = mul_if(builder, final_edge_prod, is_active, edge);

        let end_index_diff = builder.sub(node.node_index, sentinel_index);
        let end_index_gated = builder.mul(end_index_diff, seen_end.target);
        builder.assert_zero(end_index_gated);
        let end_key_gated = builder.mul(node.node_key, seen_end.target);
        builder.assert_zero(end_key_gated);

        let end_time_gated = builder.mul(node.node_time, seen_end.target);
        builder.assert_zero(end_time_gated);
        let end_parent_diff = builder.sub(node.parent_index, sentinel_index);
        let end_parent_gated = builder.mul(end_parent_diff, seen_end.target);
        builder.assert_zero(end_parent_gated);
        let end_parent_key_gated = builder.mul(node.parent_key, seen_end.target);
        builder.assert_zero(end_parent_key_gated);

        let end_parent_time_gated = builder.mul(node.parent_time, seen_end.target);
        builder.assert_zero(end_parent_time_gated);

        let not_first_active = builder.not(first_active);
        let is_first = builder.and(is_active, not_first_active);
        let first_index_diff = builder.sub(node.node_index, top_index);
        let first_index_gated = builder.mul(first_index_diff, is_first.target);
        builder.assert_zero(first_index_gated);
        let first_key_diff = builder.sub(node.node_key, top_key);
        let first_key_gated = builder.mul(first_key_diff, is_first.target);
        builder.assert_zero(first_key_gated);

        let first_time_diff = builder.sub(node.node_time, top_time);
        let first_time_gated = builder.mul(first_time_diff, is_first.target);
        builder.assert_zero(first_time_gated);

        let not_prev_parent_is_sentinel = builder.not(prev_parent_is_sentinel);
        let need_link = builder.and(prev_active, not_prev_parent_is_sentinel);
        assert_gate_true(builder, need_link, is_active);
        let link_index_diff = builder.sub(node.node_index, prev_parent_index);
        let link_index_gated = builder.mul(link_index_diff, need_link.target);
        builder.assert_zero(link_index_gated);
        let link_key_diff = builder.sub(node.node_key, prev_parent_key);
        let link_key_gated = builder.mul(link_key_diff, need_link.target);
        builder.assert_zero(link_key_gated);

        let link_time_diff = builder.sub(node.node_time, prev_parent_time);
        let link_time_gated = builder.mul(link_time_diff, need_link.target);
        builder.assert_zero(link_time_gated);

        let stop_here = builder.and(prev_active, prev_parent_is_sentinel);
        let inactive_after_stop = builder.not(is_active);
        assert_gate_true(builder, stop_here, inactive_after_stop);

        let parent_is_sentinel = builder.is_equal(node.parent_index, sentinel_index);
        let parent_key_gated = builder.mul(node.parent_key, parent_is_sentinel.target);
        builder.assert_zero(parent_key_gated);

        let parent_time_gated = builder.mul(node.parent_time, parent_is_sentinel.target);
        builder.assert_zero(parent_time_gated);

        seen_end = builder.or(seen_end, stop_here);
        first_active = builder.or(first_active, is_active);
        prev_active = is_active;
        prev_parent_is_sentinel = parent_is_sentinel;
        prev_parent_index = node.parent_index;
        prev_parent_key = node.parent_key;

        prev_parent_time = node.parent_time;
    }

    let mut any_enabled = builder._false();
    for &flag in enabled {
        any_enabled = builder.or(any_enabled, flag);
    }
    builder.connect(first_active.target, any_enabled.target);

    let rhs_edges = builder.mul(pop_edge_prod, final_edge_prod);
    builder.connect(push_edge_prod, rhs_edges);

    // Independent repeat for the same accounting equations; the first
    // challenge already binds the whole transcript, so a fresh oracle query
    // on it yields an independent second challenge.
    // Repeats 2..MULTISET_CHALLENGE_COUNT each query the oracle on the previous
    // output.
    let mut prev_hash = h;
    for _ in 1..MULTISET_CHALLENGE_COUNT {
        let next_hash =
            crate::gadgets::merkle::application_challenge(builder, 5, &prev_hash.elements);
        assert_monotone_products_with_challenge(
            builder,
            keys,
            times,
            enabled,
            trace,
            next_hash.elements[0],
            next_hash.elements[1],
            next_hash.elements[2],
            next_hash.elements[3],
        );
        prev_hash = next_hash;
    }
}

pub fn fill_prev_lower_witness(
    pw: &mut PartialWitness<F>,
    targets: &PrevLowerTraceTargets,
    trace: &PrevLowerTrace,
) {
    for (target_values, values) in [
        (&targets.prev_lower, &trace.prev_lower),
        (&targets.prev_lower_times, &trace.prev_lower_times),
    ] {
        for (&target, &value) in target_values.iter().zip_exact(values.iter()) {
            pw.set_target(target, u32_to_f(value))
                .expect("set prev-lower index/time");
        }
    }
    for (target, value) in targets.events.iter().zip_exact(trace.events.iter()) {
        pw.set_bool_target(target.active, value.active)
            .expect("set stack event active");
        pw.set_bool_target(target.is_push, value.is_push)
            .expect("set stack event push");
        for (target, value) in [
            (target.cur_index, u64::from(value.cur_index)),
            (target.cur_key, value.cur_key),
            (target.cur_time, u64::from(value.cur_time)),
            (target.top_before_index, u64::from(value.top_before_index)),
            (target.top_before_key, value.top_before_key),
            (target.top_before_time, u64::from(value.top_before_time)),
            (target.top_after_index, u64::from(value.top_after_index)),
            (target.top_after_key, value.top_after_key),
            (target.top_after_time, u64::from(value.top_after_time)),
        ] {
            pw.set_target(target, u64_to_f(value))
                .expect("set stack event field");
        }
    }
    for (target, value) in targets
        .final_nodes
        .iter()
        .zip_exact(trace.final_nodes.iter())
    {
        for (target, value) in [
            (target.node_index, u64::from(value.node_index)),
            (target.node_key, value.node_key),
            (target.node_time, u64::from(value.node_time)),
            (target.parent_index, u64::from(value.parent_index)),
            (target.parent_key, value.parent_key),
            (target.parent_time, u64::from(value.parent_time)),
        ] {
            pw.set_target(target, u64_to_f(value))
                .expect("set final stack node");
        }
    }
}
