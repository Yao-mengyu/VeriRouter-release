//! Shared field types, domain tags, and Merkle path compression.

use anyhow::Result;
use std::collections::HashMap;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::PrimeField64;
use plonky2::field::types::{Field, Field64};
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::config::PoseidonGoldilocksConfig;

pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;

pub type F = GoldilocksField;
pub const DIGEST_WIDTH: usize = 4;
pub type Digest = [u64; DIGEST_WIDTH];

#[inline]
pub fn hashout_to_digest(value: HashOut<F>) -> Digest {
    value.elements.map(|element| element.to_canonical_u64())
}

#[inline]
pub fn digest_to_hashout(value: Digest) -> HashOut<F> {
    HashOut {
        elements: value.map(f_u64),
    }
}

#[inline]
pub fn f_u64(x: u64) -> F {
    assert!(x < F::ORDER, "noncanonical Goldilocks field element");
    F::from_canonical_u64(x)
}

/// A deduplicated Merkle authentication "multiproof" in the sense of
/// de-duplicating sibling nodes across many membership proofs for the same root.
///
/// Each sibling node is identified by:
/// - `level`: distance from leaves (0 = sibling leaf, 1 = sibling of parent, ...)
/// - `index`: node index at that level (i.e. leaf_index >> level), then xor 1 for sibling
///
/// This is intentionally simple: it optimizes witness size/IO without changing the circuit shape.
#[derive(Clone, Debug)]
pub struct DedupMerkleNodeWitness {
    pub level: usize,
    pub index: u64,
    pub value: Digest,
}

#[derive(Clone, Debug)]
pub struct DedupMerkleProofWitness {
    pub height: usize,
    pub nodes: Vec<DedupMerkleNodeWitness>,
}

/// Convert LSB-first index bits into a leaf index.
pub fn index_from_bits_lsb(bits: &[bool]) -> u64 {
    let mut idx = 0u64;
    for (i, b) in bits.iter().enumerate() {
        if *b {
            idx |= 1u64 << i;
        }
    }
    idx
}

/// Build a deduplicated sibling-node map from many per-leaf Merkle paths.
///
/// Each input is `(path, index_bits)` where:
/// - `path.len() == height`
/// - `index_bits.len() == height` (LSB-first)
pub fn build_dedup_proof_from_paths(
    height: usize,
    paths: &[(Vec<Digest>, Vec<bool>)],
) -> Result<DedupMerkleProofWitness> {
    let mut map: HashMap<(usize, u64), Digest> = HashMap::new();
    for (path, bits) in paths {
        anyhow::ensure!(
            path.len() == height,
            "path len mismatch: got {}, expected {}",
            path.len(),
            height
        );
        anyhow::ensure!(
            bits.len() == height,
            "index_bits len mismatch: got {}, expected {}",
            bits.len(),
            height
        );
        let leaf_index = index_from_bits_lsb(bits);
        for lvl in 0..height {
            let node_index = leaf_index >> lvl;
            let sib_index = node_index ^ 1;
            let key = (lvl, sib_index);
            let v = path[lvl];
            if let Some(prev) = map.insert(key, v) {
                anyhow::ensure!(
                    prev == v,
                    "dedup sibling mismatch at level={}, index={}: {:?} != {:?}",
                    lvl,
                    sib_index,
                    prev,
                    v
                );
            }
        }
    }
    let mut nodes: Vec<DedupMerkleNodeWitness> = map
        .into_iter()
        .map(|((level, index), value)| DedupMerkleNodeWitness {
            level,
            index,
            value,
        })
        .collect();
    // deterministic ordering (nice for reproducibility)
    nodes.sort_by(|a, b| (a.level, a.index).cmp(&(b.level, b.index)));
    Ok(DedupMerkleProofWitness { height, nodes })
}

/// Build a lookup map for fast expansion.
pub fn dedup_proof_map(proof: &DedupMerkleProofWitness) -> HashMap<(usize, u64), Digest> {
    proof
        .nodes
        .iter()
        .map(|n| ((n.level, n.index), n.value))
        .collect()
}

/// Expand a per-leaf sibling path from a pre-built dedup map.
///
/// `index_bits` is LSB-first, length = `height`.
pub fn expand_path_from_dedup_map(
    height: usize,
    map: &HashMap<(usize, u64), Digest>,
    index_bits: &[bool],
) -> Result<Vec<Digest>> {
    anyhow::ensure!(
        index_bits.len() == height,
        "index_bits len mismatch: got {}, expected {}",
        index_bits.len(),
        height
    );
    let leaf_index = index_from_bits_lsb(index_bits);
    let mut out = Vec::with_capacity(height);
    for lvl in 0..height {
        let node_index = leaf_index >> lvl;
        let sib_index = node_index ^ 1;
        let key = (lvl, sib_index);
        let v = *map.get(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "missing dedup sibling at level={}, index={}",
                lvl,
                sib_index
            )
        })?;
        out.push(v);
    }
    Ok(out)
}
