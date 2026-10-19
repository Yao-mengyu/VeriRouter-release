use crate::circuits::common::assert_gate_true;
use crate::util::ExactZip;
use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use std::collections::BTreeSet;

use crate::circuits::common::{
    assert_bool_implies_eq, assert_bool_implies_zero, count_true, ScoreTableRowTargets,
};
use crate::gadgets::order::{
    assert_nonnegative_diff_if, assert_nonnegative_diff_if_bounded_operands,
    assert_positive_diff_if, assert_positive_diff_if_bounded_operands,
};
use crate::util::{
    canonical_node_id, u32_to_f, u64_to_f, NodeId, D, F, MULTISET_CHALLENGE_COUNT, NODE_ID_BITS,
};
use crate::witness::{DistRef, ProofNode};

#[derive(Clone)]
pub struct DistRefTargets {
    pub node_id: Target,
    pub score_enc: Target,
    pub model_id: Target,
}

#[derive(Clone)]
pub struct DistRefBatchTargets {
    pub sorted_refs: Vec<DistRefTargets>,
    pub unused_refs: Vec<DistRefTargets>,
}

pub fn add_dist_ref_targets(builder: &mut CircuitBuilder<F, D>) -> DistRefTargets {
    DistRefTargets {
        node_id: builder.add_virtual_target(),
        score_enc: builder.add_virtual_target(),
        model_id: builder.add_virtual_target(),
    }
}

pub fn add_dist_ref_vec_targets(
    builder: &mut CircuitBuilder<F, D>,
    len: usize,
) -> Vec<DistRefTargets> {
    (0..len).map(|_| add_dist_ref_targets(builder)).collect()
}

pub fn add_dist_ref_batch_targets(
    builder: &mut CircuitBuilder<F, D>,
    raw_len: usize,
    unused_len: usize,
) -> DistRefBatchTargets {
    DistRefBatchTargets {
        sorted_refs: add_dist_ref_vec_targets(builder, raw_len),
        unused_refs: add_dist_ref_vec_targets(builder, unused_len),
    }
}

#[derive(Clone, Copy)]
struct TupleChallenge {
    r: Target,
    shift: Target,
}

fn compress3(
    builder: &mut CircuitBuilder<F, D>,
    a: Target,
    b: Target,
    c: Target,
    challenge: TupleChallenge,
) -> Target {
    let acc = builder.mul_add(c, challenge.r, b);
    let acc = builder.mul_add(acc, challenge.r, a);
    builder.add(acc, challenge.shift)
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

fn push_ref_fields(out: &mut Vec<Target>, r: &DistRefTargets) {
    out.push(r.node_id);
    out.push(r.score_enc);
    out.push(r.model_id);
}

#[allow(clippy::too_many_arguments)]
fn dist_ref_challenge(
    builder: &mut CircuitBuilder<F, D>,
    refs: &[DistRefTargets],
    sorted_refs: &[DistRefTargets],
    unused_refs: &[DistRefTargets],
    table_hash: &HashOutTarget,
    sent_id: Target,
    seed_a: Target,
    seed_b: Target,
    domain: u32,
) -> HashOutTarget {
    let mut inputs = vec![
        seed_a,
        seed_b,
        builder.constant(u32_to_f(domain)),
        sent_id,
        builder.constant(u32_to_f(refs.len() as u32)),
        builder.constant(u32_to_f(sorted_refs.len() as u32)),
        builder.constant(u32_to_f(unused_refs.len() as u32)),
    ];
    inputs.extend_from_slice(&table_hash.elements);
    for r in refs {
        push_ref_fields(&mut inputs, r);
    }
    for r in sorted_refs {
        push_ref_fields(&mut inputs, r);
    }
    for r in unused_refs {
        push_ref_fields(&mut inputs, r);
    }
    crate::gadgets::merkle::application_challenge(builder, 3, &inputs)
}

fn assert_canonical_sentinel(
    builder: &mut CircuitBuilder<F, D>,
    ref_t: &DistRefTargets,
    sent_id: Target,
) -> BoolTarget {
    let is_sent = builder.is_equal(ref_t.node_id, sent_id);
    assert_bool_implies_zero(builder, is_sent, ref_t.score_enc);
    assert_bool_implies_zero(builder, is_sent, ref_t.model_id);
    is_sent
}

fn ref_term(
    builder: &mut CircuitBuilder<F, D>,
    ref_t: &DistRefTargets,
    challenge: TupleChallenge,
) -> Target {
    compress3(
        builder,
        ref_t.node_id,
        ref_t.score_enc,
        ref_t.model_id,
        challenge,
    )
}

fn table_term(
    builder: &mut CircuitBuilder<F, D>,
    node_id: Target,
    open: &ScoreTableRowTargets,
    challenge: TupleChallenge,
) -> Target {
    compress3(builder, node_id, open.score_enc, open.model_id, challenge)
}

pub fn set_dist_ref_fields(
    pw: &mut PartialWitness<F>,
    target: &DistRefTargets,
    node_id: NodeId,
    score_enc: u64,
    model_id: u32,
    label: &str,
) {
    let node_id = canonical_node_id(node_id);
    pw.set_target(target.node_id, u64_to_f(node_id))
        .unwrap_or_else(|_| panic!("set {label} id"));
    pw.set_target(target.score_enc, u64_to_f(score_enc))
        .unwrap_or_else(|_| panic!("set {label} score"));
    pw.set_target(target.model_id, u32_to_f(model_id))
        .unwrap_or_else(|_| panic!("set {label} model"));
}

pub fn set_dist_ref_witness(
    pw: &mut PartialWitness<F>,
    target: &DistRefTargets,
    value: &DistRef,
    label: &str,
) {
    set_dist_ref_fields(
        pw,
        target,
        value.node_id,
        value.score_enc,
        value.model_id,
        label,
    );
}

fn canonical_dist_ref(value: &DistRef, sent_id: NodeId) -> DistRef {
    let node_id = canonical_node_id(value.node_id);
    if node_id == sent_id {
        DistRef {
            node_id: sent_id,
            score_enc: 0,
            model_id: 0,
        }
    } else {
        DistRef {
            node_id,
            score_enc: value.score_enc,
            model_id: value.model_id,
        }
    }
}

fn sort_key(value: &DistRef, sent_id: NodeId) -> (u32, NodeId, u64, u32) {
    let node_id = canonical_node_id(value.node_id);
    if node_id == sent_id {
        (1, 0, 0, 0)
    } else {
        (0, node_id, value.score_enc, value.model_id)
    }
}

fn canonical_ref_from_node(node: &ProofNode, sent_id: NodeId) -> DistRef {
    canonical_dist_ref(
        &DistRef {
            node_id: node.node_id,
            score_enc: node.score_enc,
            model_id: node.model_id,
        },
        sent_id,
    )
}

pub fn build_dist_ref_batch_witness(
    raw_refs: &[DistRef],
    nodes: &[ProofNode],
    sent_id: NodeId,
) -> (Vec<DistRef>, Vec<DistRef>) {
    let canonical_refs = raw_refs
        .iter()
        .map(|value| canonical_dist_ref(value, sent_id))
        .collect::<Vec<_>>();
    let mut sorted_refs = canonical_refs.clone();
    sorted_refs.sort_by_key(|value| sort_key(value, sent_id));

    let used_node_ids = canonical_refs
        .iter()
        .filter(|value| value.node_id != sent_id)
        .map(|value| value.node_id)
        .collect::<BTreeSet<_>>();
    let unused_refs = nodes
        .iter()
        .filter(|node| {
            canonical_node_id(node.node_id) != sent_id
                && !used_node_ids.contains(&canonical_node_id(node.node_id))
        })
        .map(|node| canonical_ref_from_node(node, sent_id))
        .collect::<Vec<_>>();
    (sorted_refs, unused_refs)
}

pub fn fill_dist_ref_batch_witness(
    pw: &mut PartialWitness<F>,
    targets: &DistRefBatchTargets,
    raw_refs: &[DistRef],
    nodes: &[ProofNode],
    sent_id: NodeId,
    label: &str,
) {
    let (sorted_refs, mut unused_refs) = build_dist_ref_batch_witness(raw_refs, nodes, sent_id);
    assert!(
        unused_refs.len() <= targets.unused_refs.len(),
        "{label} unused ref len overflow: got {} > cap {}",
        unused_refs.len(),
        targets.unused_refs.len()
    );
    while unused_refs.len() < targets.unused_refs.len() {
        unused_refs.push(DistRef {
            node_id: sent_id,
            score_enc: 0,
            model_id: 0,
        });
    }
    for (target, value) in targets.sorted_refs.iter().zip_exact(sorted_refs.iter()) {
        set_dist_ref_witness(pw, target, value, label);
    }
    for (target, value) in targets.unused_refs.iter().zip_exact(unused_refs.iter()) {
        set_dist_ref_witness(pw, target, value, label);
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_dist_refs_against_opened_table_inner(
    builder: &mut CircuitBuilder<F, D>,
    refs: &[DistRefTargets],
    sorted_refs: &[DistRefTargets],
    unused_refs: &[DistRefTargets],
    table_hash: &HashOutTarget,
    node_ids: &[Target],
    node_opens: &[ScoreTableRowTargets],
    sent_id: Target,
    seed_a: Target,
    seed_b: Target,
    domain: u32,
    bounded_operands: bool,
) {
    assert_eq!(node_ids.len(), node_opens.len());
    assert_eq!(refs.len(), sorted_refs.len());

    let hash = dist_ref_challenge(
        builder,
        refs,
        sorted_refs,
        unused_refs,
        table_hash,
        sent_id,
        seed_a,
        seed_b,
        domain,
    );
    // Pairs 1-2 come from the transcript hash; further pairs come from fresh
    // domain-separated queries on that output.
    let mut challenges = vec![
        TupleChallenge {
            r: hash.elements[0],
            shift: hash.elements[1],
        },
        TupleChallenge {
            r: hash.elements[2],
            shift: hash.elements[3],
        },
    ];
    let mut extra_input = hash;
    while challenges.len() < MULTISET_CHALLENGE_COUNT {
        extra_input =
            crate::gadgets::merkle::application_challenge(builder, 6, &extra_input.elements);
        challenges.push(TupleChallenge {
            r: extra_input.elements[0],
            shift: extra_input.elements[1],
        });
    }
    challenges.truncate(MULTISET_CHALLENGE_COUNT);

    let one = builder.one();
    let mut raw_prods = [one; MULTISET_CHALLENGE_COUNT];
    let mut sorted_prods = [one; MULTISET_CHALLENGE_COUNT];
    let mut first_prods = [one; MULTISET_CHALLENGE_COUNT];

    let mut sorted_is_real = Vec::with_capacity(sorted_refs.len());
    let mut first_in_run_bits = Vec::with_capacity(sorted_refs.len());
    for ref_t in refs {
        assert_canonical_sentinel(builder, ref_t, sent_id);
        for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
            let term = ref_term(builder, ref_t, challenges[challenge_idx]);
            raw_prods[challenge_idx] = builder.mul(raw_prods[challenge_idx], term);
        }
    }

    for (idx, ref_t) in sorted_refs.iter().enumerate() {
        let is_sent = assert_canonical_sentinel(builder, ref_t, sent_id);
        let is_real = builder.not(is_sent);
        sorted_is_real.push(is_real);
        let terms: Vec<Target> = challenges
            .iter()
            .map(|&challenge| ref_term(builder, ref_t, challenge))
            .collect();
        for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
            sorted_prods[challenge_idx] =
                builder.mul(sorted_prods[challenge_idx], terms[challenge_idx]);
        }

        let first_in_run = if idx == 0 {
            is_real
        } else {
            let prev_is_real = sorted_is_real[idx - 1];
            assert_gate_true(builder, is_real, prev_is_real);

            let prev_ref = &sorted_refs[idx - 1];
            let both_real = builder.and(prev_is_real, is_real);
            // With bounded table ids, both real sorted refs are table tuples.
            if bounded_operands {
                assert_nonnegative_diff_if_bounded_operands(
                    builder,
                    both_real,
                    ref_t.node_id,
                    prev_ref.node_id,
                    NODE_ID_BITS,
                );
            } else {
                assert_nonnegative_diff_if(
                    builder,
                    both_real,
                    ref_t.node_id,
                    prev_ref.node_id,
                    NODE_ID_BITS,
                );
            }

            let same_node_id = builder.is_equal(ref_t.node_id, prev_ref.node_id);
            let same_node_id_real = builder.and(both_real, same_node_id);
            assert_bool_implies_eq(
                builder,
                same_node_id_real,
                ref_t.score_enc,
                prev_ref.score_enc,
            );
            assert_bool_implies_eq(
                builder,
                same_node_id_real,
                ref_t.model_id,
                prev_ref.model_id,
            );

            let node_id_changed = builder.not(same_node_id);
            builder.and(is_real, node_id_changed)
        };
        first_in_run_bits.push(first_in_run);
        for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
            first_prods[challenge_idx] = mul_if(
                builder,
                first_prods[challenge_idx],
                first_in_run,
                terms[challenge_idx],
            );
        }
    }
    for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
        builder.connect(raw_prods[challenge_idx], sorted_prods[challenge_idx]);
    }

    let mut unused_prods = [one; MULTISET_CHALLENGE_COUNT];
    let mut unused_is_real = Vec::with_capacity(unused_refs.len());
    for idx in 0..unused_refs.len() {
        let ref_t = &unused_refs[idx];
        let is_sent = assert_canonical_sentinel(builder, ref_t, sent_id);
        let is_real = builder.not(is_sent);
        unused_is_real.push(is_real);
        if idx > 0 {
            let prev_real = unused_is_real[idx - 1];
            assert_gate_true(builder, is_real, prev_real);
            let prev = &unused_refs[idx - 1];
            let both_real = builder.and(prev_real, is_real);
            // With bounded table ids, real unused refs are table tuples.
            if bounded_operands {
                assert_positive_diff_if_bounded_operands(
                    builder,
                    both_real,
                    ref_t.node_id,
                    prev.node_id,
                    NODE_ID_BITS,
                );
            } else {
                assert_positive_diff_if(
                    builder,
                    both_real,
                    ref_t.node_id,
                    prev.node_id,
                    NODE_ID_BITS,
                );
            }
        }
        for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
            let term = ref_term(builder, ref_t, challenges[challenge_idx]);
            unused_prods[challenge_idx] =
                mul_if(builder, unused_prods[challenge_idx], is_real, term);
        }
    }

    let mut table_prods = [one; MULTISET_CHALLENGE_COUNT];
    let mut table_active = Vec::with_capacity(node_ids.len());
    for (&node_id, open) in node_ids.iter().zip(node_opens.iter()) {
        let is_sent = builder.is_equal(node_id, sent_id);
        let is_active = builder.not(is_sent);
        assert_bool_implies_zero(builder, is_sent, open.score_enc);
        assert_bool_implies_zero(builder, is_sent, open.model_id);
        table_active.push(is_active);
        for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
            let term = table_term(builder, node_id, open, challenges[challenge_idx]);
            table_prods[challenge_idx] =
                mul_if(builder, table_prods[challenge_idx], is_active, term);
        }
    }
    for challenge_idx in 0..MULTISET_CHALLENGE_COUNT {
        let used_and_unused = builder.mul(first_prods[challenge_idx], unused_prods[challenge_idx]);
        builder.connect(used_and_unused, table_prods[challenge_idx]);
    }

    let first_real_count = count_true(builder, &first_in_run_bits);
    let unused_real_count = count_true(builder, &unused_is_real);
    let used_plus_unused = builder.add(first_real_count, unused_real_count);
    let active_count = count_true(builder, &table_active);
    builder.connect(used_plus_unused, active_count);
}

pub fn assert_dist_refs_against_opened_table(
    builder: &mut CircuitBuilder<F, D>,
    refs: &[DistRefTargets],
    batch: &DistRefBatchTargets,
    table_hash: &HashOutTarget,
    node_ids: &[Target],
    node_opens: &[ScoreTableRowTargets],
    sent_id: Target,
    seed_a: Target,
    seed_b: Target,
    domain: u32,
    bounded_operands: bool,
) {
    assert_dist_refs_against_opened_table_inner(
        builder,
        refs,
        &batch.sorted_refs,
        &batch.unused_refs,
        table_hash,
        node_ids,
        node_opens,
        sent_id,
        seed_a,
        seed_b,
        domain,
        bounded_operands,
    );
}

pub fn assert_dist_refs_against_opened_table_padded(
    builder: &mut CircuitBuilder<F, D>,
    refs: &[DistRefTargets],
    batch: &DistRefBatchTargets,
    table_hash: &HashOutTarget,
    node_ids: &[Target],
    node_opens: &[ScoreTableRowTargets],
    sent_id: Target,
    seed_a: Target,
    seed_b: Target,
    domain: u32,
    bounded_operands: bool,
) {
    assert_dist_refs_against_opened_table_inner(
        builder,
        refs,
        &batch.sorted_refs,
        &batch.unused_refs,
        table_hash,
        node_ids,
        node_opens,
        sent_id,
        seed_a,
        seed_b,
        domain,
        bounded_operands,
    );
}
