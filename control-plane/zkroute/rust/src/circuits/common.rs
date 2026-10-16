use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::gadgets::merkle::{poseidon_hash_no_pad, verify_merkle_multiproof_gated};
use crate::gadgets::order::{
    assert_nonnegative_diff_with_offset_if_bounded_operands, constrain_bits_if, constrain_bits_le,
};
use crate::util::{
    u32_to_f, u64_to_f, D, F, GRAPH_NEIGHBOR_LIST_HASH_DOMAIN, NODE_ID_BITS, NODE_ID_SENTINEL,
    NODE_VECTOR_HASH_DOMAIN, QUERY_HASH_DOMAIN, SCORE_BITS, SCORE_TABLE_HASH_DOMAIN, U32_BITS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeIdRangeMode {
    U32OrSentinel,
    Wide56,
}

pub fn constrain_node_id(builder: &mut CircuitBuilder<F, D>, node_id: Target) {
    constrain_bits_le(builder, node_id, NODE_ID_BITS);
}

pub fn constrain_u32_node_id_or_sentinel(
    builder: &mut CircuitBuilder<F, D>,
    node_id: Target,
) -> BoolTarget {
    let sentinel = builder.constant(u64_to_f(NODE_ID_SENTINEL));
    let is_sentinel = builder.is_equal(node_id, sentinel);
    let zero = builder.zero();
    let range_checked = builder.select(is_sentinel, zero, node_id);
    constrain_bits_le(builder, range_checked, U32_BITS);
    is_sentinel
}

pub fn constrain_node_id_with_mode(
    builder: &mut CircuitBuilder<F, D>,
    node_id: Target,
    mode: NodeIdRangeMode,
) -> BoolTarget {
    let sentinel = builder.constant(u64_to_f(NODE_ID_SENTINEL));
    match mode {
        NodeIdRangeMode::U32OrSentinel => constrain_u32_node_id_or_sentinel(builder, node_id),
        NodeIdRangeMode::Wide56 => {
            constrain_node_id(builder, node_id);
            builder.is_equal(node_id, sentinel)
        }
    }
}

#[derive(Clone)]
pub struct MerklePathTargets {
    pub index_bits: Vec<BoolTarget>,
    pub siblings: Vec<HashOutTarget>,
}

#[derive(Clone)]
pub struct ScoreTableRowTargets {
    pub score_enc: Target,
    pub model_id: Target,
}

#[derive(Clone, Copy, Debug)]
pub struct MerkleOpeningBuildStats {
    pub leaf_hash_gate_count: usize,
    pub path_gate_count: usize,
}

pub fn connect_hashes(builder: &mut CircuitBuilder<F, D>, a: &HashOutTarget, b: &HashOutTarget) {
    for i in 0..4 {
        builder.connect(a.elements[i], b.elements[i]);
    }
}

pub fn hash_query(
    builder: &mut CircuitBuilder<F, D>,
    zero_point: Target,
    query_u16: &[Target],
) -> HashOutTarget {
    let mut fields = Vec::with_capacity(3 + query_u16.len());
    fields.push(builder.constant(u64_to_f(QUERY_HASH_DOMAIN)));
    fields.push(builder.constant(u64_to_f(query_u16.len() as u64)));
    fields.push(zero_point);
    fields.extend_from_slice(query_u16);
    poseidon_hash_no_pad(builder, &fields)
}

fn hash_vector(builder: &mut CircuitBuilder<F, D>, vector_u16: &[Target]) -> HashOutTarget {
    let mut fields = Vec::with_capacity(2 + vector_u16.len());
    fields.push(builder.constant(u64_to_f(NODE_VECTOR_HASH_DOMAIN)));
    fields.push(builder.constant(u64_to_f(vector_u16.len() as u64)));
    fields.extend_from_slice(vector_u16);
    poseidon_hash_no_pad(builder, &fields)
}

pub fn hash_node_leaf(
    builder: &mut CircuitBuilder<F, D>,
    node_id: Target,
    model_id: Target,
    vector_u16: &[Target],
) -> HashOutTarget {
    let vec_hash = hash_vector(builder, vector_u16);
    let mut fields = vec![node_id, model_id];
    fields.extend_from_slice(&vec_hash.elements);
    poseidon_hash_no_pad(builder, &fields)
}

pub fn hash_neighbor_list(
    builder: &mut CircuitBuilder<F, D>,
    neighbors: &[Target],
) -> HashOutTarget {
    let mut fields = Vec::with_capacity(2 + neighbors.len());
    fields.push(builder.constant(u64_to_f(GRAPH_NEIGHBOR_LIST_HASH_DOMAIN)));
    fields.push(builder.constant(u64_to_f(neighbors.len() as u64)));
    fields.extend_from_slice(neighbors);
    poseidon_hash_no_pad(builder, &fields)
}

pub fn hash_graph_row_leaf(
    builder: &mut CircuitBuilder<F, D>,
    layer: Target,
    src: Target,
    neighbors: &[Target],
) -> HashOutTarget {
    let nbr_hash = hash_neighbor_list(builder, neighbors);
    let mut fields = vec![layer, src];
    fields.extend_from_slice(&nbr_hash.elements);
    poseidon_hash_no_pad(builder, &fields)
}

pub fn hash_score_table(
    builder: &mut CircuitBuilder<F, D>,
    ids: &[Target],
    score_encs: &[Target],
    model_ids: &[Target],
) -> HashOutTarget {
    assert_eq!(ids.len(), score_encs.len());
    assert_eq!(ids.len(), model_ids.len());

    let mut fields = Vec::with_capacity(2 + 3 * ids.len());
    fields.push(builder.constant(u64_to_f(SCORE_TABLE_HASH_DOMAIN)));
    fields.push(builder.constant(u64_to_f(ids.len() as u64)));
    for ((&node_id, &score_enc), &model_id) in
        ids.iter().zip(score_encs.iter()).zip(model_ids.iter())
    {
        fields.push(node_id);
        fields.push(score_enc);
        fields.push(model_id);
    }
    poseidon_hash_no_pad(builder, &fields)
}

pub fn commit_neighbor_visits(
    builder: &mut CircuitBuilder<F, D>,
    ids: &[Target],
    is_news: &[BoolTarget],
) -> HashOutTarget {
    assert_eq!(ids.len(), is_news.len());
    let mut fields = Vec::with_capacity(2 + 2 * ids.len());
    fields.push(builder.constant(u32_to_f(6101)));
    fields.push(builder.constant(u32_to_f(ids.len() as u32)));
    for (&node_id, &is_new) in ids.iter().zip(is_news.iter()) {
        fields.push(node_id);
        fields.push(is_new.target);
    }
    poseidon_hash_no_pad(builder, &fields)
}

pub fn add_score_table_row(builder: &mut CircuitBuilder<F, D>) -> ScoreTableRowTargets {
    ScoreTableRowTargets {
        score_enc: builder.add_virtual_target(),
        model_id: builder.add_virtual_target(),
    }
}

pub fn add_merkle_path(
    builder: &mut CircuitBuilder<F, D>,
    tree_height: usize,
) -> MerklePathTargets {
    let mut index_bits = Vec::with_capacity(tree_height);
    let mut siblings = Vec::with_capacity(tree_height);
    for _ in 0..tree_height {
        index_bits.push(builder.add_virtual_bool_target_safe());
        siblings.push(builder.add_virtual_hash());
    }
    MerklePathTargets {
        index_bits,
        siblings,
    }
}

pub fn verify_node_openings(
    builder: &mut CircuitBuilder<F, D>,
    node_root: &HashOutTarget,
    ids: &[Target],
    model_ids: &[Target],
    vectors: &[Vec<Target>],
    paths: &[MerklePathTargets],
    enabled: &[BoolTarget],
) -> MerkleOpeningBuildStats {
    assert_eq!(ids.len(), model_ids.len());
    assert_eq!(ids.len(), vectors.len());
    assert_eq!(ids.len(), paths.len());
    assert_eq!(ids.len(), enabled.len());
    let leaf_hash_start = builder.num_gates();
    let leaves = ids
        .iter()
        .zip(model_ids.iter())
        .zip(vectors.iter())
        .map(|((&node_id, &model_id), vector)| hash_node_leaf(builder, node_id, model_id, vector))
        .collect::<Vec<_>>();
    let leaf_hash_gate_count = builder.num_gates() - leaf_hash_start;
    let index_bits = paths
        .iter()
        .map(|path| path.index_bits.clone())
        .collect::<Vec<_>>();
    let siblings = paths
        .iter()
        .map(|path| path.siblings.clone())
        .collect::<Vec<_>>();
    let path_start = builder.num_gates();
    verify_merkle_multiproof_gated(builder, node_root, &leaves, &index_bits, &siblings, enabled);
    MerkleOpeningBuildStats {
        leaf_hash_gate_count,
        path_gate_count: builder.num_gates() - path_start,
    }
}

pub fn verify_graph_openings(
    builder: &mut CircuitBuilder<F, D>,
    graph_root: &HashOutTarget,
    layers: &[Target],
    srcs: &[Target],
    neighbors: &[Vec<Target>],
    paths: &[MerklePathTargets],
    enabled: &[BoolTarget],
) -> MerkleOpeningBuildStats {
    assert_eq!(layers.len(), srcs.len());
    assert_eq!(layers.len(), neighbors.len());
    assert_eq!(layers.len(), paths.len());
    assert_eq!(layers.len(), enabled.len());
    let leaf_hash_start = builder.num_gates();
    let leaves = layers
        .iter()
        .zip(srcs.iter())
        .zip(neighbors.iter())
        .map(|((&layer, &src), row_neighbors)| {
            hash_graph_row_leaf(builder, layer, src, row_neighbors)
        })
        .collect::<Vec<_>>();
    let leaf_hash_gate_count = builder.num_gates() - leaf_hash_start;
    let index_bits = paths
        .iter()
        .map(|path| path.index_bits.clone())
        .collect::<Vec<_>>();
    let siblings = paths
        .iter()
        .map(|path| path.siblings.clone())
        .collect::<Vec<_>>();
    let path_start = builder.num_gates();
    verify_merkle_multiproof_gated(
        builder,
        graph_root,
        &leaves,
        &index_bits,
        &siblings,
        enabled,
    );
    MerkleOpeningBuildStats {
        leaf_hash_gate_count,
        path_gate_count: builder.num_gates() - path_start,
    }
}

pub fn id_in_candidates_if_enabled(
    builder: &mut CircuitBuilder<F, D>,
    target_id: Target,
    candidates: &[Target],
    enabled: &[BoolTarget],
) -> BoolTarget {
    assert_eq!(candidates.len(), enabled.len());
    let mut found = builder._false();
    for i in 0..candidates.len() {
        let eq = builder.is_equal(target_id, candidates[i]);
        let active_eq = builder.and(enabled[i], eq);
        found = builder.or(found, active_eq);
    }
    found
}

/// Selects the first maximum-score occurrence in source order.
///
/// A HNSW row starts from the source node and scans neighbors left to right.
/// The native replay replaces the incumbent only on a strict improvement, so
/// a selected candidate must strictly beat every earlier candidate and may tie
/// only with later candidates. `gate` disables the relation for padded rows.
///
/// Each score is range-checked once (gated) with one checked comparison per
/// candidate: `selected >= candidate + strict`, where `strict` marks candidates
/// that precede the first occurrence of the selected node.
pub fn assert_first_max_in_source_order_merged_if(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    selected_id: Target,
    selected_score: Target,
    candidates: &[Target],
    candidate_scores: &[Target],
) {
    assert_eq!(candidates.len(), candidate_scores.len());

    let mut selected_exists = builder._false();
    let mut no_prior_match = builder._true();
    let mut first_matches = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        let selected_matches = builder.is_equal(selected_id, candidate);
        selected_exists = builder.or(selected_exists, selected_matches);
        first_matches.push(builder.and(no_prior_match, selected_matches));
        let not_selected_matches = builder.not(selected_matches);
        no_prior_match = builder.and(no_prior_match, not_selected_matches);
    }
    let selected_is_absent = builder.not(selected_exists);
    assert_bool_implies_zero(builder, gate, selected_is_absent.target);

    constrain_bits_if(builder, gate, selected_score, SCORE_BITS);
    for &candidate_score in candidate_scores {
        constrain_bits_if(builder, gate, candidate_score, SCORE_BITS);
    }
    let mut strict = vec![builder._false(); candidates.len()];
    let mut first_match_after = builder._false();
    for index in (0..candidates.len()).rev() {
        strict[index] = builder.and(gate, first_match_after);
        first_match_after = builder.or(first_match_after, first_matches[index]);
    }
    for (&candidate_score, &strict) in candidate_scores.iter().zip(&strict) {
        assert_nonnegative_diff_with_offset_if_bounded_operands(
            builder,
            gate,
            selected_score,
            candidate_score,
            strict,
            SCORE_BITS,
        );
    }
}

pub fn count_true(builder: &mut CircuitBuilder<F, D>, bits: &[BoolTarget]) -> Target {
    let mut acc = builder.zero();
    for bit in bits {
        acc = builder.add(acc, bit.target);
    }
    acc
}

pub fn or_many(builder: &mut CircuitBuilder<F, D>, bits: &[BoolTarget]) -> BoolTarget {
    let mut acc = builder._false();
    for bit in bits {
        acc = builder.or(acc, *bit);
    }
    acc
}

pub fn assert_bool_implies_eq(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    a: Target,
    b: Target,
) {
    let diff = builder.sub(a, b);
    let gated = builder.mul(diff, gate.target);
    builder.assert_zero(gated);
}

pub fn assert_bool_implies_zero(builder: &mut CircuitBuilder<F, D>, gate: BoolTarget, a: Target) {
    let gated = builder.mul(a, gate.target);
    builder.assert_zero(gated);
}

pub fn assert_gate_true(builder: &mut CircuitBuilder<F, D>, gate: BoolTarget, cond: BoolTarget) {
    let gated = builder.mul_sub(gate.target, cond.target, gate.target);
    builder.assert_zero(gated);
}
