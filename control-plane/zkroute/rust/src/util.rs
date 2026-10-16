use plonky2::field::types::{Field, Field64};
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use std::fs;
use std::io::Read;
use std::path::Path;

// Plonky2 extension degree used by the recursive circuit system.
// This is NOT the embedding dimension of the vectors.
pub const D: usize = 2;
pub type NodeId = u64;
pub const U32_BITS: usize = 32;
pub const POSITION_BITS: usize = U32_BITS;
pub const NODE_ID_BITS: usize = 56;
pub const NODE_ID_SENTINEL: NodeId = (1u64 << NODE_ID_BITS) - 1;
pub const LEGACY_NODE_ID_SENTINEL: NodeId = u32::MAX as NodeId;
pub const MAX_REAL_NODE_ID: NodeId = NODE_ID_SENTINEL - 1;
pub const SCORE_BITS: usize = 48;
// Width used by single-field monotone comparisons, including efSearch's
// packed stable C/W priority. Builders reject profiles whose maximum packed
// priority reaches this bound.
pub const GENERIC_MONOTONE_KEY_BITS: usize = 62;
pub const SCORE_BIAS: u64 = 1u64 << (SCORE_BITS - 1);
pub const SCORE_TABLE_HASH_DOMAIN: u64 = 0x5343_4f52_455f_5431;
/// Shared hiding-commitment domain tag. Keep this identical to the data plane.
pub const DS_COMMIT: u64 = 0x5652_5f43_4f4d_0009;
pub const QUERY_HASH_DOMAIN: u64 = 0x5155_4552_595f_5631;
pub const NODE_VECTOR_HASH_DOMAIN: u64 = 0x4e4f_4445_5f56_4331;
pub const GRAPH_NEIGHBOR_LIST_HASH_DOMAIN: u64 = 0x4752_4150_485f_4e31;
pub const MULTISET_CHALLENGE_COUNT: usize = 2;

pub const fn is_node_id_sentinel(value: NodeId) -> bool {
    value == NODE_ID_SENTINEL || value == LEGACY_NODE_ID_SENTINEL
}

pub const fn canonical_node_id(value: NodeId) -> NodeId {
    if is_node_id_sentinel(value) {
        NODE_ID_SENTINEL
    } else {
        value
    }
}

pub type C = PoseidonGoldilocksConfig;
pub type F = <C as GenericConfig<D>>::F;

pub fn u32_to_f(x: u32) -> F {
    F::from_canonical_u64(x as u64)
}

pub fn u64_to_f(x: u64) -> F {
    assert!(x < F::ORDER, "noncanonical Goldilocks field element");
    F::from_canonical_u64(x)
}

/// Pair fixed-shape witness columns without silently discarding either tail.
pub trait ExactZip: ExactSizeIterator + Sized {
    fn zip_exact<U>(self, other: U) -> std::iter::Zip<Self, U::IntoIter>
    where
        U: IntoIterator,
        U::IntoIter: ExactSizeIterator,
    {
        let other = other.into_iter();
        assert_eq!(self.len(), other.len(), "witness length mismatch");
        self.zip(other)
    }
}
impl<I: ExactSizeIterator> ExactZip for I {}

pub fn recursion_config(use_zk: bool) -> CircuitConfig {
    if use_zk {
        CircuitConfig::standard_recursion_zk_config()
    } else {
        CircuitConfig::standard_recursion_config()
    }
}

pub fn read_text_or_zlib(path: &Path) -> std::io::Result<String> {
    let data = fs::read(path)?;
    if path.extension().and_then(|value| value.to_str()) == Some("bin") {
        let mut decoder = flate2::read::ZlibDecoder::new(&data[..]);
        let mut out = String::new();
        decoder.read_to_string(&mut out)?;
        Ok(out)
    } else {
        String::from_utf8(data)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }
}

fn as_u32(value: usize, label: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{label} exceeds the 32-bit position domain"))
}

/// The entry-point insertion is the first candidate-queue event.
pub const fn c_entry_position() -> u32 {
    1
}

/// Candidate pop in zero-based round `round_idx`:
/// `2 + round_idx * (round_width + 1)`.
pub fn c_pop_position(round_idx: usize, round_width: usize) -> u32 {
    let value = round_idx
        .checked_mul(
            round_width
                .checked_add(1)
                .expect("candidate round width overflow"),
        )
        .and_then(|value| value.checked_add(2))
        .expect("candidate pop position overflow");
    as_u32(value, "candidate pop position")
}

/// Candidate insertion for zero-based neighbor `slot_idx` in a round:
/// `2 + round_idx * (round_width + 1) + (slot_idx + 1)`.
pub fn c_neighbor_position(round_idx: usize, slot_idx: usize, round_width: usize) -> u32 {
    assert!(
        slot_idx < round_width,
        "candidate neighbor slot must be inside the fixed round width"
    );
    let value = usize::try_from(c_pop_position(round_idx, round_width))
        .expect("u32 position fits usize")
        .checked_add(slot_idx)
        .and_then(|value| value.checked_add(1))
        .expect("candidate neighbor position overflow");
    as_u32(value, "candidate neighbor position")
}

/// Public upper bound for candidate-queue operation positions:
/// `1 + rounds * (round_width + 1)`.
pub fn c_position_limit(rounds: usize, round_width: usize) -> u32 {
    let value = rounds
        .checked_mul(
            round_width
                .checked_add(1)
                .expect("candidate round width overflow"),
        )
        .and_then(|value| value.checked_add(1))
        .expect("candidate position limit overflow");
    as_u32(value, "candidate position limit")
}

/// Initial working-queue slot, including the entry at zero-based slot 0.
pub fn w_initial_position(slot_idx: usize) -> u32 {
    as_u32(
        slot_idx
            .checked_add(1)
            .expect("working initial position overflow"),
        "working initial position",
    )
}

/// Working-queue update push for zero-based update `step_idx`:
/// `ef_search + 2 * step_idx + 1`.
pub fn w_push_position(step_idx: usize, ef_search: usize) -> u32 {
    let value = step_idx
        .checked_mul(2)
        .and_then(|value| value.checked_add(ef_search))
        .and_then(|value| value.checked_add(1))
        .expect("working push position overflow");
    as_u32(value, "working push position")
}

/// Working-queue update pop for zero-based update `step_idx`:
/// `ef_search + 2 * step_idx + 2`.
pub fn w_pop_position(step_idx: usize, ef_search: usize) -> u32 {
    w_push_position(step_idx, ef_search)
        .checked_add(1)
        .expect("working pop position overflow")
}

/// Public upper bound for working-queue operation positions.
pub fn w_position_limit(step_count: usize, ef_search: usize) -> u32 {
    if step_count == 0 {
        as_u32(ef_search, "working position limit")
    } else {
        w_pop_position(step_count - 1, ef_search)
    }
}
