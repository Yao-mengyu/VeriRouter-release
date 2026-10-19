//! Top-level hiding commitments shared with the VeriRouter data plane.

use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::Hasher;

use crate::gadgets::merkle::poseidon_hash_no_pad;
use crate::util::{
    canonical_node_id, u32_to_f, u64_to_f, D, DS_COMMIT, F, NODE_ID_SENTINEL, QUERY_HASH_DOMAIN,
    SCORE_TABLE_HASH_DOMAIN,
};
use crate::witness::ProofInput;

pub type Digest = [u64; 4];

/// `Poseidon(DS_COMMIT || raw_digest[4] || randomness[4])`.
pub fn commit_digest(raw_digest: Digest, randomness: Digest) -> Digest {
    let mut inputs = Vec::with_capacity(9);
    inputs.push(u64_to_f(DS_COMMIT));
    inputs.extend(raw_digest.map(u64_to_f));
    inputs.extend(randomness.map(u64_to_f));
    let hash = PoseidonHash::hash_no_pad(&inputs);
    hash.elements.map(|element| element.to_canonical_u64())
}

/// Circuit form of [`commit_digest`]. Both the raw digest and randomness may
/// remain private even when the returned commitment is registered as public.
pub fn commit_digest_target(
    builder: &mut CircuitBuilder<F, D>,
    raw_digest: HashOutTarget,
    randomness: HashOutTarget,
) -> HashOutTarget {
    let domain = builder.constant(u64_to_f(DS_COMMIT));
    let mut inputs = Vec::with_capacity(9);
    inputs.push(domain);
    inputs.extend_from_slice(&raw_digest.elements);
    inputs.extend_from_slice(&randomness.elements);
    poseidon_hash_no_pad(builder, &inputs)
}

pub fn digest_to_hashout(digest: Digest) -> HashOut<F> {
    HashOut {
        elements: digest.map(u64_to_f),
    }
}

fn hash_fields(fields: &[F]) -> HashOut<F> {
    PoseidonHash::hash_no_pad(fields)
}

pub fn hash_query(input: &ProofInput) -> [u64; 4] {
    let mut fields = Vec::with_capacity(3 + input.query.vector_u16.len());
    fields.push(u64_to_f(QUERY_HASH_DOMAIN));
    fields.push(u64_to_f(input.query.vector_u16.len() as u64));
    fields.push(u32_to_f(input.quantizer.zero_point));
    fields.extend(
        input
            .query
            .vector_u16
            .iter()
            .map(|&v| u32_to_f(u32::from(v))),
    );
    hashout_to_u64s(hash_fields(&fields))
}

fn hashout_to_u64s(hash: HashOut<F>) -> [u64; 4] {
    [
        hash.elements[0].to_canonical_u64(),
        hash.elements[1].to_canonical_u64(),
        hash.elements[2].to_canonical_u64(),
        hash.elements[3].to_canonical_u64(),
    ]
}

/// Canonical key-unique flat score-table hash used by the routing relation.
pub fn score_table_hash(input: &ProofInput) -> [u64; 4] {
    let mut fields = Vec::with_capacity(2 + 3 * input.nodes.len());
    fields.push(u64_to_f(SCORE_TABLE_HASH_DOMAIN));
    fields.push(u64_to_f(input.nodes.len() as u64));
    for node in &input.nodes {
        let node_id = canonical_node_id(node.node_id);
        fields.push(u64_to_f(node_id));
        if node_id == NODE_ID_SENTINEL {
            fields.push(u64_to_f(0));
            fields.push(u64_to_f(0));
        } else {
            fields.push(u64_to_f(node.score_enc));
            fields.push(u32_to_f(node.model_id));
        }
    }
    hashout_to_u64s(hash_fields(&fields))
}

pub fn compute_nbr_commit(input: &ProofInput) -> [u64; 4] {
    let mut fields = Vec::with_capacity(2 + 2 * input.ef_search.neighbor_visits.len());
    fields.push(u32_to_f(6101));
    fields.push(u32_to_f(input.ef_search.neighbor_visits.len() as u32));
    for visit in &input.ef_search.neighbor_visits {
        fields.push(u64_to_f(canonical_node_id(visit.node_id)));
        fields.push(u32_to_f(if visit.is_new { 1 } else { 0 }));
    }
    hashout_to_u64s(hash_fields(&fields))
}
