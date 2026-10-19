//! Native commitment hashes and their circuit forms.
use crate::util::{digest_to_hashout, f_u64, hashout_to_digest, Digest, D, DIGEST_WIDTH, F};
use anyhow::Result;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::poseidon::{PoseidonHash, PoseidonPermutation};
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::Hasher;

pub const DS_NODE: u64 = 2;
pub(crate) const DS_BYTES_INIT: u64 = 1;
pub(crate) const DS_MSG: u64 = 2;
pub(crate) const DS_BYTES_STEP: u64 = 5;
pub(crate) const DS_TRANSCRIPT_INIT: u64 = 6;
pub(crate) const DS_TRANSCRIPT_STEP: u64 = 7;

/// Finalize the ordered transcript with its exact message count.
pub const DS_REQ_FINAL: u64 = 0x5652_5f52_4551_0008;
/// Commit to a digest with a four-limb secret randomizer.
pub const DS_COMMIT: u64 = 0x5652_5f43_4f4d_0009;

fn poseidon_digest(inputs: &[F]) -> HashOut<F> {
    PoseidonHash::hash_no_pad(inputs)
}

fn compress_digests(domain: u64, left: HashOut<F>, right: HashOut<F>) -> HashOut<F> {
    let mut permutation = PoseidonPermutation::new(core::iter::repeat(F::ZERO));
    permutation.set_from_slice(&left.elements, 0);
    permutation.set_from_slice(&right.elements, DIGEST_WIDTH);
    permutation.set_elt(f_u64(domain), 8);
    permutation.permute();
    HashOut {
        elements: std::array::from_fn(|index| permutation.squeeze()[index]),
    }
}

fn hash_node(left: HashOut<F>, right: HashOut<F>) -> HashOut<F> {
    compress_digests(DS_NODE, left, right)
}

pub fn hash_node_digest(left: Digest, right: Digest) -> Digest {
    hashout_to_digest(hash_node(digest_to_hashout(left), digest_to_hashout(right)))
}

pub(crate) fn merkle_root(
    leaf: HashOut<F>,
    path: &[HashOut<F>],
    index_bits: &[bool],
) -> HashOut<F> {
    assert_eq!(path.len(), index_bits.len(), "Merkle path length mismatch");
    let mut cur = leaf;
    for (sib, bit) in path.iter().zip(index_bits.iter()) {
        let (l, r) = if *bit { (*sib, cur) } else { (cur, *sib) };
        cur = hash_node(l, r);
    }
    cur
}

pub(crate) fn hash_bytes_init(total_len: F) -> HashOut<F> {
    poseidon_digest(&[f_u64(DS_BYTES_INIT), total_len, F::ZERO])
}

pub fn hash_bytes_init_digest(total_len: u64) -> Digest {
    hashout_to_digest(hash_bytes_init(f_u64(total_len)))
}

fn hash_bytes_step(state: HashOut<F>, len: F, packed: F) -> HashOut<F> {
    let mut inputs = Vec::with_capacity(7);
    inputs.push(f_u64(DS_BYTES_STEP));
    inputs.extend(state.elements);
    inputs.push(len);
    inputs.push(packed);
    poseidon_digest(&inputs)
}

pub fn hash_bytes_step_digest(state: Digest, len: u64, packed: u64) -> Digest {
    hashout_to_digest(hash_bytes_step(
        digest_to_hashout(state),
        f_u64(len),
        f_u64(packed),
    ))
}

pub(crate) fn absorb_chunk_bytes(
    mut state: HashOut<F>,
    len: u64,
    packed_le: u64,
    chunk_bytes: usize,
) -> Result<HashOut<F>> {
    anyhow::ensure!(len <= chunk_bytes as u64, "chunk len too large");
    for byte_idx in 0..(len as usize) {
        let byte = (packed_le >> (8 * byte_idx)) & 0xff;
        state = hash_bytes_step(state, f_u64(1), f_u64(byte));
    }
    Ok(state)
}

pub(crate) fn hash_msg(role_id: F, total_len: F, content_hash: HashOut<F>) -> HashOut<F> {
    let mut inputs = Vec::with_capacity(7);
    inputs.push(f_u64(DS_MSG));
    inputs.push(role_id);
    inputs.push(total_len);
    inputs.extend(content_hash.elements);
    poseidon_digest(&inputs)
}

pub fn hash_msg_digest(role_id: u64, total_len: u64, content_hash: Digest) -> Digest {
    assert!(role_id < F::ORDER, "noncanonical role ID");
    hashout_to_digest(hash_msg(
        f_u64(role_id),
        f_u64(total_len),
        digest_to_hashout(content_hash),
    ))
}

pub fn transcript_init() -> Digest {
    hashout_to_digest(poseidon_digest(&[f_u64(DS_TRANSCRIPT_INIT), F::ZERO]))
}

pub(crate) fn transcript_step(acc: HashOut<F>, leaf: HashOut<F>) -> HashOut<F> {
    compress_digests(DS_TRANSCRIPT_STEP, acc, leaf)
}

pub fn transcript_step_digest(acc: Digest, leaf: Digest) -> Digest {
    hashout_to_digest(transcript_step(
        digest_to_hashout(acc),
        digest_to_hashout(leaf),
    ))
}

pub fn request_hash_digest(message_count: u64, final_fold: Digest) -> Digest {
    let mut inputs = Vec::with_capacity(2 + DIGEST_WIDTH);
    inputs.push(f_u64(DS_REQ_FINAL));
    inputs.push(f_u64(message_count));
    inputs.extend(digest_to_hashout(final_fold).elements);
    hashout_to_digest(poseidon_digest(&inputs))
}

pub fn commit_digest(digest: Digest, randomness: Digest) -> Digest {
    let mut inputs = Vec::with_capacity(1 + 2 * DIGEST_WIDTH);
    inputs.push(f_u64(DS_COMMIT));
    inputs.extend(digest_to_hashout(digest).elements);
    inputs.extend(digest_to_hashout(randomness).elements);
    hashout_to_digest(poseidon_digest(&inputs))
}

pub fn request_randomness_binding_digest(randomness: Digest) -> Digest {
    commit_digest([0; DIGEST_WIDTH], randomness)
}

pub(crate) fn request_hash_target(
    builder: &mut CircuitBuilder<F, D>,
    message_count: Target,
    final_fold: HashOutTarget,
) -> HashOutTarget {
    let domain = builder.constant(f_u64(DS_REQ_FINAL));
    poseidon_target(
        builder,
        hash_inputs(&[domain, message_count], &[final_fold]),
    )
}

pub(crate) fn commit_digest_target(
    builder: &mut CircuitBuilder<F, D>,
    digest: HashOutTarget,
    randomness: HashOutTarget,
) -> HashOutTarget {
    let domain = builder.constant(f_u64(DS_COMMIT));
    poseidon_target(builder, hash_inputs(&[domain], &[digest, randomness]))
}

pub(crate) fn request_randomness_binding_target(
    builder: &mut CircuitBuilder<F, D>,
    randomness: HashOutTarget,
) -> HashOutTarget {
    let zero = builder.zero();
    let zero_digest = HashOutTarget {
        elements: [zero; DIGEST_WIDTH],
    };
    commit_digest_target(builder, zero_digest, randomness)
}

fn poseidon_target(builder: &mut CircuitBuilder<F, D>, inputs: Vec<Target>) -> HashOutTarget {
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(inputs)
}

fn hash_inputs(prefix: &[Target], hashes: &[HashOutTarget]) -> Vec<Target> {
    let mut inputs = prefix.to_vec();
    for hash in hashes {
        inputs.extend_from_slice(&hash.elements);
    }
    inputs
}

pub fn hash_bytes_digest(bytes: &[u8]) -> Digest {
    let mut state = hash_bytes_init_digest(bytes.len() as u64);
    for &byte in bytes {
        state = hash_bytes_step_digest(state, 1, u64::from(byte));
    }
    state
}
