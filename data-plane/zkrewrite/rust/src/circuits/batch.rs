use crate::commitment::{
    commit_digest_target, request_randomness_binding_target, DS_BYTES_INIT, DS_BYTES_STEP, DS_MSG,
    DS_NODE, DS_TRANSCRIPT_STEP,
};
use anyhow::Result;

use plonky2::hash::hash_types::HashOutTarget;
use plonky2::hash::hashing::PlonkyPermutation;
use plonky2::hash::poseidon::{PoseidonHash, PoseidonPermutation};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};

use crate::util::{
    dedup_proof_map, digest_to_hashout, expand_path_from_dedup_map, f_u64, Digest, DIGEST_WIDTH, F,
};
use crate::witness::RewriteTreeBatchWitness;
use std::collections::HashMap;

pub use crate::util::{C, D};

pub const PI_MSG_COMMITMENT: usize = 0;
pub const PI_FRAG_COMMITMENT: usize = PI_MSG_COMMITMENT + DIGEST_WIDTH;
pub const PI_REQUEST_RANDOMNESS_BINDING: usize = PI_FRAG_COMMITMENT + DIGEST_WIDTH;
pub const PI_START_IDX: usize = PI_REQUEST_RANDOMNESS_BINDING + DIGEST_WIDTH;
pub const PI_END_IDX: usize = PI_START_IDX + 1;
pub const PI_REWRITTEN_COUNT_IN: usize = PI_END_IDX + 1;
pub const PI_REWRITTEN_COUNT_OUT: usize = PI_REWRITTEN_COUNT_IN + 1;
pub const PI_PREV_PRESENT: usize = PI_REWRITTEN_COUNT_OUT + 1;
pub const PI_LAST_PRESENT: usize = PI_PREV_PRESENT + 1;
pub const PI_USER_ACC_IN: usize = PI_LAST_PRESENT + 1;
pub const PI_USER_ROOT: usize = PI_USER_ACC_IN + DIGEST_WIDTH;
pub const PI_MOD_ACC_IN: usize = PI_USER_ROOT + DIGEST_WIDTH;
pub const PI_MOD_ROOT: usize = PI_MOD_ACC_IN + DIGEST_WIDTH;
pub const BATCH_PUBLIC_INPUTS: usize = PI_MOD_ROOT + DIGEST_WIDTH;

/// Circuit parameters (must be fixed to build a circuit shape).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RewriteParams {
    pub batch: usize,
    pub chunk_bytes: usize,
    pub max_orig_chunks: usize,
    pub max_msg_chunks: usize,
    pub max_frag_chunks: usize,
    pub msg_bank_len: usize,
    pub frag_bank_len: usize,
    pub h_msg: usize,
    pub h_frag: usize,
    pub max_frags: usize,
    pub max_gap_msgs: usize,
    /// If true (default), use cheaper range-check constraints for `ch_len <= chunk_bytes`.
    /// If false, use the legacy bitwise comparator (useful for benchmarking / regression).
    pub fast_range_check: bool,
}

#[derive(Clone, Debug)]
pub struct RewriteTargets {
    // public inputs
    pub msg_commitment: HashOutTarget,
    pub frag_commitment: HashOutTarget,
    pub request_randomness_binding: HashOutTarget,
    pub start_idx: Target,
    pub end_idx: Target,
    pub rewritten_count_in: Target,
    pub rewritten_count_out: Target,
    pub prev_present: BoolTarget,
    pub last_present: BoolTarget,
    pub user_acc_in: HashOutTarget,
    pub user_root: HashOutTarget,
    pub mod_acc_in: HashOutTarget,
    pub mod_root: HashOutTarget,
}

#[derive(Clone, Debug)]
pub struct RewriteWitnessTargets {
    pub msg_root: HashOutTarget,
    pub frag_root: HashOutTarget,
    pub msg_randomness: HashOutTarget,
    pub frag_randomness: HashOutTarget,
    pub request_randomness: HashOutTarget,
    pub start_idx: Target,
    pub end_idx: Target,
    pub rewritten_count_in: Target,
    pub rewritten_count_out: Target,
    pub prev_present: BoolTarget,
    pub user_acc_in: HashOutTarget,
    pub mod_acc_in: HashOutTarget,

    pub orig_present: Vec<BoolTarget>,
    pub orig_role_id: Vec<Target>,
    pub orig_total_len: Vec<Target>,
    pub orig_chunks_enabled: Vec<Vec<BoolTarget>>,
    pub orig_chunks_len: Vec<Vec<Target>>,
    pub orig_chunks_packed: Vec<Vec<Target>>,

    pub gap_enabled: Vec<Vec<BoolTarget>>,
    pub gap_leaf: Vec<Vec<HashOutTarget>>,
    pub gap_path: Vec<Vec<Vec<HashOutTarget>>>,
    pub gap_index_bits: Vec<Vec<Vec<BoolTarget>>>,

    pub prefixes_enabled: Vec<Vec<BoolTarget>>,
    pub prefixes_total_len: Vec<Vec<Target>>,
    pub prefixes_chunks_enabled: Vec<Vec<Vec<BoolTarget>>>,
    pub prefixes_chunks_len: Vec<Vec<Vec<Target>>>,
    pub prefixes_chunks_packed: Vec<Vec<Vec<Target>>>,
    pub prefixes_path: Vec<Vec<Vec<HashOutTarget>>>,
    pub prefixes_index_bits: Vec<Vec<Vec<BoolTarget>>>,

    pub suffixes_enabled: Vec<Vec<BoolTarget>>,
    pub suffixes_total_len: Vec<Vec<Target>>,
    pub suffixes_chunks_enabled: Vec<Vec<Vec<BoolTarget>>>,
    pub suffixes_chunks_len: Vec<Vec<Vec<Target>>>,
    pub suffixes_chunks_packed: Vec<Vec<Vec<Target>>>,
    pub suffixes_path: Vec<Vec<Vec<HashOutTarget>>>,
    pub suffixes_index_bits: Vec<Vec<Vec<BoolTarget>>>,
}

fn poseidon_digest(builder: &mut CircuitBuilder<F, D>, inputs: Vec<Target>) -> HashOutTarget {
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(inputs)
}

fn compress_digests(
    builder: &mut CircuitBuilder<F, D>,
    domain: u64,
    left: HashOutTarget,
    right: HashOutTarget,
) -> HashOutTarget {
    // `Target::default()` is VirtualTarget(0), not the constant zero. Fill the
    // whole state explicitly before installing the two rate digests and the
    // domain separator in capacity.
    let zero = builder.zero();
    let mut permutation = PoseidonPermutation::new(core::iter::repeat(zero));
    permutation.set_from_slice(&left.elements, 0);
    permutation.set_from_slice(&right.elements, DIGEST_WIDTH);
    permutation.set_elt(builder.constant(f_u64(domain)), 8);
    let no_swap = builder._false();
    let output = PoseidonHash::permute_swapped(permutation, no_swap, builder);
    HashOutTarget {
        elements: std::array::from_fn(|index| output.squeeze()[index]),
    }
}

fn hash_inputs(prefix: &[Target], hashes: &[HashOutTarget]) -> Vec<Target> {
    let mut inputs = Vec::with_capacity(prefix.len() + hashes.len() * DIGEST_WIDTH);
    inputs.extend_from_slice(prefix);
    for hash in hashes {
        inputs.extend_from_slice(&hash.elements);
    }
    inputs
}

fn select_hash(
    builder: &mut CircuitBuilder<F, D>,
    condition: BoolTarget,
    when_true: HashOutTarget,
    when_false: HashOutTarget,
) -> HashOutTarget {
    HashOutTarget {
        elements: std::array::from_fn(|i| {
            builder.select(condition, when_true.elements[i], when_false.elements[i])
        }),
    }
}

fn assert_leq_bool(builder: &mut CircuitBuilder<F, D>, a: BoolTarget, b: BoolTarget) {
    // a * (1 - b) == 0
    let one = builder.one();
    let one_minus_b = builder.sub(one, b.target);
    let prod = builder.mul(a.target, one_minus_b);
    let zero = builder.zero();
    builder.connect(prod, zero);
}

fn assert_eq_if_enabled(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    x: Target,
    y: Target,
) {
    // enabled * (x-y) == 0
    let diff = builder.sub(x, y);
    let prod = builder.mul(enabled.target, diff);
    let zero = builder.zero();
    builder.connect(prod, zero);
}

fn assert_hash_eq_if_enabled(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    x: HashOutTarget,
    y: HashOutTarget,
) {
    for i in 0..DIGEST_WIDTH {
        assert_eq_if_enabled(builder, enabled, x.elements[i], y.elements[i]);
    }
}

fn assert_real_bank_index(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    bits: &[BoolTarget],
    count: usize,
) {
    if count == 0 {
        builder.assert_zero(enabled.target);
        return;
    }
    let index = builder.le_sum(bits.iter());
    let zero = builder.zero();
    let selected = builder.select(enabled, index, zero);
    range_check_u64_leq(builder, selected, (count - 1) as u64, true);
}

fn merkle_root_from_path(
    builder: &mut CircuitBuilder<F, D>,
    leaf: HashOutTarget,
    path: &[HashOutTarget],
    index_bits: &[BoolTarget],
) -> HashOutTarget {
    let mut cur = leaf;
    for (sib, bit) in path.iter().zip(index_bits.iter()) {
        // bit == 1 means current is right child.
        let left = select_hash(builder, *bit, *sib, cur);
        let right = select_hash(builder, *bit, cur, *sib);
        cur = compress_digests(builder, DS_NODE, left, right);
    }
    cur
}

/// Fast constraint for `x <= max_inclusive` when `max_inclusive <= 8`.
///
/// This is much cheaper than the generic bitwise comparator and is tailored to the
/// common `chunk_bytes <= 8` setting used by this project (u64 packing).
fn range_check_u64_leq_small(builder: &mut CircuitBuilder<F, D>, x: Target, max_inclusive: u64) {
    assert!(max_inclusive <= 8);
    let zero = builder.zero();
    match max_inclusive {
        0 => {
            builder.connect(x, zero);
        }
        1 => {
            // 1 bit is enough: x in {0,1}
            let _ = builder.split_le(x, 1);
        }
        2 => {
            // 2 bits covers 0..3; forbid 3 (11)
            let b = builder.split_le(x, 2);
            let t = builder.mul(b[0].target, b[1].target);
            builder.connect(t, zero);
        }
        3 => {
            // 2 bits covers exactly 0..3
            let _ = builder.split_le(x, 2);
        }
        4 => {
            // 3 bits covers 0..7; forbid 5..7 by requiring: if b2==1 then b1=b0=0
            // i.e. b2 * (b1 + b0) == 0
            let b = builder.split_le(x, 3);
            let s = builder.add(b[0].target, b[1].target);
            let t = builder.mul(b[2].target, s);
            builder.connect(t, zero);
        }
        5 => {
            // 3 bits covers 0..7; forbid 6,7 by requiring NOT(b2=1 AND b1=1)
            let b = builder.split_le(x, 3);
            let t = builder.mul(b[2].target, b[1].target);
            builder.connect(t, zero);
        }
        6 => {
            // 3 bits covers 0..7; forbid 7 (111)
            let b = builder.split_le(x, 3);
            let t01 = builder.mul(b[0].target, b[1].target);
            let t = builder.mul(t01, b[2].target);
            builder.connect(t, zero);
        }
        7 => {
            // 3 bits covers exactly 0..7
            let _ = builder.split_le(x, 3);
        }
        8 => {
            // 4 bits covers 0..15; allow 0..7 (b3=0) and 8 (1000).
            // Enforce: if b3==1 then b2=b1=b0=0 => b3 * (b0+b1+b2) == 0.
            let b = builder.split_le(x, 4);
            let s01 = builder.add(b[0].target, b[1].target);
            let s = builder.add(s01, b[2].target);
            let t = builder.mul(b[3].target, s);
            builder.connect(t, zero);
        }
        _ => unreachable!(),
    }
}

/// Legacy bitwise comparator: `x <= max_inclusive` by bit-decomposition + prefix compare.
fn range_check_u64_leq_bitwise(builder: &mut CircuitBuilder<F, D>, x: Target, max_inclusive: u64) {
    // Enforce x <= max via bit-decomposition comparator.
    // This assumes x is a small canonical u64.
    let bits = 64 - max_inclusive.leading_zeros() as usize;
    let num_bits = bits.max(1);
    // `split_le` constrains x to `num_bits` bits.
    let xb = builder.split_le(x, num_bits);
    let mut gt = builder._false(); // x > max
    let mut eq = builder._true(); // prefix equal
    for i in (0..num_bits).rev() {
        let a = xb[i];
        let b_const = ((max_inclusive >> i) & 1) == 1;
        let b = if b_const {
            builder._true()
        } else {
            builder._false()
        };
        // gt_i = eq & a & !b
        let not_b = builder.not(b);
        let a_and_not_b = builder.and(a, not_b);
        let gt_i = builder.and(eq, a_and_not_b);
        gt = builder.or(gt, gt_i);
        // eq = eq & (a==b)
        let a_eq_b = builder.is_equal(a.target, b.target);
        eq = builder.and(eq, a_eq_b);
    }
    let zero = builder.zero();
    builder.connect(gt.target, zero);
}

/// Cheaper range check: enforce existence of `y` such that `x + y == max_inclusive`,
/// with `x` and `y` constrained to `num_bits` bits.
///
/// For `num_bits < 64`, this excludes wrap-around in Goldilocks and is typically much cheaper
/// than a full bitwise comparator.
fn range_check_u64_leq_slack(builder: &mut CircuitBuilder<F, D>, x: Target, max_inclusive: u64) {
    let bits = 64 - max_inclusive.leading_zeros() as usize;
    let num_bits = bits.max(1);
    // Safety: when num_bits==64, wrap-around could slip through because the field modulus is < 2^64.
    // Fall back to the legacy comparator in that case.
    if num_bits == 64 {
        range_check_u64_leq_bitwise(builder, x, max_inclusive);
        return;
    }

    builder.range_check(x, num_bits);
    let y = builder.add_virtual_target();
    builder.range_check(y, num_bits);
    let max_t = builder.constant(f_u64(max_inclusive));
    let sum = builder.add(x, y);
    builder.connect(sum, max_t);
}

fn range_check_u64_leq(
    builder: &mut CircuitBuilder<F, D>,
    x: Target,
    max_inclusive: u64,
    fast: bool,
) {
    if !fast {
        range_check_u64_leq_bitwise(builder, x, max_inclusive);
        return;
    }
    if max_inclusive <= 8 {
        range_check_u64_leq_small(builder, x, max_inclusive);
        return;
    }
    range_check_u64_leq_slack(builder, x, max_inclusive);
}

fn canonical_padding(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    len: Target,
    packed: Target,
) {
    // (1-enabled) * len == 0, (1-enabled) * packed == 0
    let one = builder.one();
    let one_minus = builder.sub(one, enabled.target);
    let prod_len = builder.mul(one_minus, len);
    let zero = builder.zero();
    builder.connect(prod_len, zero);
    let prod_packed = builder.mul(one_minus, packed);
    let zero2 = builder.zero();
    builder.connect(prod_packed, zero2);
}

fn hash_msg(
    builder: &mut CircuitBuilder<F, D>,
    role_id: Target,
    total_len: Target,
    content_hash: HashOutTarget,
) -> HashOutTarget {
    let ds_msg = builder.constant(f_u64(DS_MSG));
    poseidon_digest(
        builder,
        hash_inputs(&[ds_msg, role_id, total_len], &[content_hash]),
    )
}

fn transcript_step(
    builder: &mut CircuitBuilder<F, D>,
    acc: HashOutTarget,
    leaf: HashOutTarget,
) -> HashOutTarget {
    compress_digests(builder, DS_TRANSCRIPT_STEP, acc, leaf)
}

fn hash_bytes_init(builder: &mut CircuitBuilder<F, D>, total_len: Target) -> HashOutTarget {
    let ds_bytes_init = builder.constant(f_u64(DS_BYTES_INIT));
    let zero = builder.zero();
    poseidon_digest(builder, vec![ds_bytes_init, total_len, zero])
}

fn hash_bytes_step(
    builder: &mut CircuitBuilder<F, D>,
    state: HashOutTarget,
    len: Target,
    packed: Target,
) -> HashOutTarget {
    let ds_bytes_step = builder.constant(f_u64(DS_BYTES_STEP));
    poseidon_digest(
        builder,
        hash_inputs(&[ds_bytes_step], &[state])
            .into_iter()
            .chain([len, packed])
            .collect(),
    )
}

fn packed_chunk_bytes(
    builder: &mut CircuitBuilder<F, D>,
    packed: Target,
    chunk_bytes: usize,
) -> Vec<Target> {
    let bits = builder.split_le(packed, chunk_bytes * 8);
    let mut out = Vec::with_capacity(chunk_bytes);
    for byte_idx in 0..chunk_bytes {
        let mut byte = builder.zero();
        for bit_idx in 0..8 {
            let coeff = builder.constant(f_u64(1u64 << bit_idx));
            let term = builder.mul(coeff, bits[8 * byte_idx + bit_idx].target);
            byte = builder.add(byte, term);
        }
        out.push(byte);
    }
    out
}

struct DecodedChunk {
    bytes: Vec<Target>,
    live: Vec<BoolTarget>,
}

fn decode_chunk(
    builder: &mut CircuitBuilder<F, D>,
    len: Target,
    packed: Target,
    chunk_bytes: usize,
    fast_range_check: bool,
) -> DecodedChunk {
    range_check_u64_leq(builder, len, chunk_bytes as u64, fast_range_check);
    let bytes = packed_chunk_bytes(builder, packed, chunk_bytes);

    // `len` is in 0..=chunk_bytes, so precisely one of these equality flags can
    // be true. Each suffix sum is therefore Boolean and says whether its byte is
    // within the declared chunk length.
    let mut len_equals = Vec::with_capacity(chunk_bytes);
    for value in 1..=chunk_bytes {
        let value_t = builder.constant(f_u64(value as u64));
        len_equals.push(builder.is_equal(len, value_t));
    }
    let mut suffix = builder.zero();
    let mut live_rev = Vec::with_capacity(chunk_bytes);
    for equal in len_equals.into_iter().rev() {
        suffix = builder.add(suffix, equal.target);
        live_rev.push(BoolTarget::new_unsafe(suffix));
    }
    live_rev.reverse();

    DecodedChunk {
        bytes,
        live: live_rev,
    }
}

fn absorb_decoded_chunk_bytes(
    builder: &mut CircuitBuilder<F, D>,
    state: HashOutTarget,
    enabled: BoolTarget,
    decoded: &DecodedChunk,
) -> HashOutTarget {
    let one = builder.one();
    let mut cur = state;
    for (&byte, &live) in decoded.bytes.iter().zip(decoded.live.iter()) {
        let absorb = builder.and(enabled, live);
        let next = hash_bytes_step(builder, cur, one, byte);
        cur = select_hash(builder, absorb, next, cur);
    }
    cur
}

/// Build the commit-orig variant of Rewrite tree-batch circuit.
///
/// Public inputs order:
/// `[message_bank_commitment[4], fragment_bank_commitment[4],
///   request_randomness_binding[4], start_idx, end_idx,
///   rewritten_count_in, rewritten_count_out, prev_present, last_present,
///   user_acc_in[4], user_root[4], mod_acc_in[4], mod_root[4]]`
pub fn build_rewrite_tree_batch_circuit_with_zk(
    params: &RewriteParams,
    zero_knowledge: bool,
) -> (CircuitData<F, C, D>, RewriteTargets, RewriteWitnessTargets)
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    build_rewrite_tree_batch_internal(params, false, zero_knowledge)
}

/// Build the public-orig variant (legacy-like): exposes original message fields as public inputs.
///
/// Public inputs order:
/// - base: `[message_bank_commitment[4], fragment_bank_commitment[4],
///   request_randomness_binding[4], start_idx, end_idx, rewritten_count_in,
///   rewritten_count_out, prev_present, last_present, user_acc_in[4],
///   user_root[4], mod_acc_in[4], mod_root[4]]`
/// - then for each message `i` in `[0..BATCH)`:
///   `role_id, present, orig_total_len, (chunk_enabled, chunk_len, chunk_packed)*MAX_CHUNKS`
pub fn build_rewrite_tree_batch_public_orig_circuit_with_zk(
    params: &RewriteParams,
    zero_knowledge: bool,
) -> (CircuitData<F, C, D>, RewriteTargets, RewriteWitnessTargets)
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    build_rewrite_tree_batch_internal(params, true, zero_knowledge)
}

fn build_rewrite_tree_batch_internal(
    params: &RewriteParams,
    public_orig: bool,
    zero_knowledge: bool,
) -> (CircuitData<F, C, D>, RewriteTargets, RewriteWitnessTargets)
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    assert!(
        params.h_msg <= 32 && params.h_frag <= 32,
        "bank heights must fit the bounded profile"
    );
    assert!(params.msg_bank_len as u64 <= (1u64 << params.h_msg));
    assert!(params.frag_bank_len as u64 <= (1u64 << params.h_frag));
    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = zero_knowledge;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Raw bank roots and all salts are private witness targets. Only top-level
    // commitments (plus an internal request-randomness binding) become leaf PIs.
    let msg_root = builder.add_virtual_hash();
    let frag_root = builder.add_virtual_hash();
    let msg_randomness = builder.add_virtual_hash();
    let frag_randomness = builder.add_virtual_hash();
    let request_randomness = builder.add_virtual_hash();
    let msg_commitment = commit_digest_target(&mut builder, msg_root, msg_randomness);
    let frag_commitment = commit_digest_target(&mut builder, frag_root, frag_randomness);
    let request_randomness_binding =
        request_randomness_binding_target(&mut builder, request_randomness);

    let start_idx = builder.add_virtual_target();
    let end_idx = builder.add_virtual_target();
    let rewritten_count_in = builder.add_virtual_target();
    let rewritten_count_out = builder.add_virtual_target();
    let prev_present = builder.add_virtual_bool_target_safe();
    let user_acc_in = builder.add_virtual_hash();
    let mod_acc_in = builder.add_virtual_hash();

    // Allocate all witness targets.
    let mut orig_present = vec![];
    let mut orig_role_id = vec![];
    let mut orig_total_len = vec![];
    let mut orig_chunks_enabled = vec![];
    let mut orig_chunks_len = vec![];
    let mut orig_chunks_packed = vec![];

    let mut gap_enabled = vec![];
    let mut gap_leaf = vec![];
    let mut gap_path = vec![];
    let mut gap_index_bits = vec![];

    let mut prefixes_enabled = vec![];
    let mut prefixes_total_len = vec![];
    let mut prefixes_chunks_enabled = vec![];
    let mut prefixes_chunks_len = vec![];
    let mut prefixes_chunks_packed = vec![];
    let mut prefixes_path = vec![];
    let mut prefixes_index_bits = vec![];

    let mut suffixes_enabled = vec![];
    let mut suffixes_total_len = vec![];
    let mut suffixes_chunks_enabled = vec![];
    let mut suffixes_chunks_len = vec![];
    let mut suffixes_chunks_packed = vec![];
    let mut suffixes_path = vec![];
    let mut suffixes_index_bits = vec![];

    for _i in 0..params.batch {
        orig_present.push(builder.add_virtual_bool_target_safe());
        orig_role_id.push(builder.add_virtual_target());
        orig_total_len.push(builder.add_virtual_target());

        let mut ce = vec![];
        let mut cl = vec![];
        let mut cp = vec![];
        for _k in 0..params.max_orig_chunks {
            ce.push(builder.add_virtual_bool_target_safe());
            cl.push(builder.add_virtual_target());
            cp.push(builder.add_virtual_target());
        }
        orig_chunks_enabled.push(ce);
        orig_chunks_len.push(cl);
        orig_chunks_packed.push(cp);

        let mut ge = vec![];
        let mut gl = vec![];
        let mut gp = vec![];
        let mut gb = vec![];
        for _j in 0..params.max_gap_msgs {
            ge.push(builder.add_virtual_bool_target_safe());
            gl.push(builder.add_virtual_hash());
            let mut p = vec![];
            let mut b = vec![];
            for _lvl in 0..params.h_msg {
                p.push(builder.add_virtual_hash());
                b.push(builder.add_virtual_bool_target_safe());
            }
            gp.push(p);
            gb.push(b);
        }
        gap_enabled.push(ge);
        gap_leaf.push(gl);
        gap_path.push(gp);
        gap_index_bits.push(gb);

        // prefixes
        let mut pe = vec![];
        let mut pt = vec![];
        let mut pce = vec![];
        let mut pcl = vec![];
        let mut pcp = vec![];
        let mut pp = vec![];
        let mut pb = vec![];
        for _f in 0..params.max_frags {
            pe.push(builder.add_virtual_bool_target_safe());
            pt.push(builder.add_virtual_target());
            let mut fce = vec![];
            let mut fcl = vec![];
            let mut fcp = vec![];
            for _k in 0..params.max_frag_chunks {
                fce.push(builder.add_virtual_bool_target_safe());
                fcl.push(builder.add_virtual_target());
                fcp.push(builder.add_virtual_target());
            }
            pce.push(fce);
            pcl.push(fcl);
            pcp.push(fcp);
            let mut ppath = vec![];
            let mut pbits = vec![];
            for _lvl in 0..params.h_frag {
                ppath.push(builder.add_virtual_hash());
                pbits.push(builder.add_virtual_bool_target_safe());
            }
            pp.push(ppath);
            pb.push(pbits);
        }
        prefixes_enabled.push(pe);
        prefixes_total_len.push(pt);
        prefixes_chunks_enabled.push(pce);
        prefixes_chunks_len.push(pcl);
        prefixes_chunks_packed.push(pcp);
        prefixes_path.push(pp);
        prefixes_index_bits.push(pb);

        // suffixes
        let mut se = vec![];
        let mut st = vec![];
        let mut sce = vec![];
        let mut scl = vec![];
        let mut scp = vec![];
        let mut sp = vec![];
        let mut sb = vec![];
        for _f in 0..params.max_frags {
            se.push(builder.add_virtual_bool_target_safe());
            st.push(builder.add_virtual_target());
            let mut fce = vec![];
            let mut fcl = vec![];
            let mut fcp = vec![];
            for _k in 0..params.max_frag_chunks {
                fce.push(builder.add_virtual_bool_target_safe());
                fcl.push(builder.add_virtual_target());
                fcp.push(builder.add_virtual_target());
            }
            sce.push(fce);
            scl.push(fcl);
            scp.push(fcp);
            let mut spath = vec![];
            let mut sbits = vec![];
            for _lvl in 0..params.h_frag {
                spath.push(builder.add_virtual_hash());
                sbits.push(builder.add_virtual_bool_target_safe());
            }
            sp.push(spath);
            sb.push(sbits);
        }
        suffixes_enabled.push(se);
        suffixes_total_len.push(st);
        suffixes_chunks_enabled.push(sce);
        suffixes_chunks_len.push(scl);
        suffixes_chunks_packed.push(scp);
        suffixes_path.push(sp);
        suffixes_index_bits.push(sb);
    }

    // --- Constraints (port of `prompt_mod_tree_batch.rs`) ---
    let zero = builder.zero();
    let mut prev = prev_present;
    let mut computed_last_present = prev_present;
    let mut present_count = zero;
    let mut rewritten_count = rewritten_count_in;
    let mut user_acc = user_acc_in;
    let mut mod_acc = mod_acc_in;

    // Public-orig fields to append (same order as the original repo).
    let mut public_orig_fields: Vec<Target> = vec![];

    for i in 0..params.batch {
        // gap insertions before msg i
        for j in 0..params.max_gap_msgs {
            let en = gap_enabled[i][j];
            assert_leq_bool(&mut builder, en, prev);
            let leaf = gap_leaf[i][j];
            let path = &gap_path[i][j];
            let bits = &gap_index_bits[i][j];
            assert_real_bank_index(&mut builder, en, bits, params.msg_bank_len);
            let root_out = merkle_root_from_path(&mut builder, leaf, path, bits);
            assert_hash_eq_if_enabled(&mut builder, en, root_out, msg_root);
            let next_mod_acc = transcript_step(&mut builder, mod_acc, leaf);
            mod_acc = select_hash(&mut builder, en, next_mod_acc, mod_acc);
            rewritten_count = builder.add(rewritten_count, en.target);
        }

        // original message i
        let present = orig_present[i];
        assert_leq_bool(&mut builder, present, prev);
        prev = present;
        computed_last_present = present;
        present_count = builder.add(present_count, present.target);
        rewritten_count = builder.add(rewritten_count, present.target);

        let role_id = orig_role_id[i];
        let orig_total_len_i = orig_total_len[i];

        // original body hash
        let mut body_state = hash_bytes_init(&mut builder, orig_total_len_i);
        let mut body_sum_len = zero;

        let mut rewritten_total = builder.mul(present.target, orig_total_len_i);
        for f in 0..params.max_frags {
            let frag_en = prefixes_enabled[i][f];
            let frag_total = prefixes_total_len[i][f];
            let eff_total = builder.mul(frag_en.target, frag_total);
            rewritten_total = builder.add(rewritten_total, eff_total);
        }
        for f in 0..params.max_frags {
            let frag_en = suffixes_enabled[i][f];
            let frag_total = suffixes_total_len[i][f];
            let eff_total = builder.mul(frag_en.target, frag_total);
            rewritten_total = builder.add(rewritten_total, eff_total);
        }

        // rewritten bytes stream
        let mut bytes_state = hash_bytes_init(&mut builder, rewritten_total);
        let mut sum_len = zero;

        // Prefix fragments
        for f in 0..params.max_frags {
            let frag_en = prefixes_enabled[i][f];
            assert_leq_bool(&mut builder, frag_en, present);
            let frag_total = prefixes_total_len[i][f];

            // hash fragment bytes stream and enforce membership in frag_root
            let mut frag_state = hash_bytes_init(&mut builder, frag_total);
            let mut frag_sum = zero;
            for k in 0..params.max_frag_chunks {
                let ch_en = prefixes_chunks_enabled[i][f][k];
                assert_leq_bool(&mut builder, ch_en, frag_en);
                let ch_len = prefixes_chunks_len[i][f][k];
                let ch_packed = prefixes_chunks_packed[i][f][k];
                canonical_padding(&mut builder, ch_en, ch_len, ch_packed);
                let decoded = decode_chunk(
                    &mut builder,
                    ch_len,
                    ch_packed,
                    params.chunk_bytes,
                    params.fast_range_check,
                );

                let eff_len = builder.mul(ch_en.target, ch_len);
                frag_sum = builder.add(frag_sum, eff_len);

                frag_state = absorb_decoded_chunk_bytes(&mut builder, frag_state, ch_en, &decoded);
                bytes_state =
                    absorb_decoded_chunk_bytes(&mut builder, bytes_state, ch_en, &decoded);
                sum_len = builder.add(sum_len, eff_len);
            }
            assert_eq_if_enabled(&mut builder, frag_en, frag_total, frag_sum);

            let ppath = &prefixes_path[i][f];
            let pbits = &prefixes_index_bits[i][f];
            assert_real_bank_index(&mut builder, frag_en, pbits, params.frag_bank_len);
            let root_out = merkle_root_from_path(&mut builder, frag_state, ppath, pbits);
            assert_hash_eq_if_enabled(&mut builder, frag_en, root_out, frag_root);
        }

        // absorb original chunks into both body_state and rewritten bytes_state
        for k in 0..params.max_orig_chunks {
            let ch_en = orig_chunks_enabled[i][k];
            assert_leq_bool(&mut builder, ch_en, present);
            let ch_len = orig_chunks_len[i][k];
            let ch_packed = orig_chunks_packed[i][k];
            canonical_padding(&mut builder, ch_en, ch_len, ch_packed);
            let decoded = decode_chunk(
                &mut builder,
                ch_len,
                ch_packed,
                params.chunk_bytes,
                params.fast_range_check,
            );

            let eff_len = builder.mul(ch_en.target, ch_len);
            body_sum_len = builder.add(body_sum_len, eff_len);
            sum_len = builder.add(sum_len, eff_len);

            body_state = absorb_decoded_chunk_bytes(&mut builder, body_state, ch_en, &decoded);
            bytes_state = absorb_decoded_chunk_bytes(&mut builder, bytes_state, ch_en, &decoded);
        }
        // The original transcript commitment must bind a canonical byte string too:
        // its declared length is exactly the number of original bytes absorbed.
        assert_eq_if_enabled(&mut builder, present, orig_total_len_i, body_sum_len);

        // Suffix fragments
        for f in 0..params.max_frags {
            let frag_en = suffixes_enabled[i][f];
            assert_leq_bool(&mut builder, frag_en, present);
            let frag_total = suffixes_total_len[i][f];

            let mut frag_state = hash_bytes_init(&mut builder, frag_total);
            let mut frag_sum = zero;
            for k in 0..params.max_frag_chunks {
                let ch_en = suffixes_chunks_enabled[i][f][k];
                assert_leq_bool(&mut builder, ch_en, frag_en);
                let ch_len = suffixes_chunks_len[i][f][k];
                let ch_packed = suffixes_chunks_packed[i][f][k];
                canonical_padding(&mut builder, ch_en, ch_len, ch_packed);
                let decoded = decode_chunk(
                    &mut builder,
                    ch_len,
                    ch_packed,
                    params.chunk_bytes,
                    params.fast_range_check,
                );

                let eff_len = builder.mul(ch_en.target, ch_len);
                frag_sum = builder.add(frag_sum, eff_len);

                frag_state = absorb_decoded_chunk_bytes(&mut builder, frag_state, ch_en, &decoded);
                bytes_state =
                    absorb_decoded_chunk_bytes(&mut builder, bytes_state, ch_en, &decoded);
                sum_len = builder.add(sum_len, eff_len);
            }
            assert_eq_if_enabled(&mut builder, frag_en, frag_total, frag_sum);

            let spath = &suffixes_path[i][f];
            let sbits = &suffixes_index_bits[i][f];
            assert_real_bank_index(&mut builder, frag_en, sbits, params.frag_bank_len);
            let root_out = merkle_root_from_path(&mut builder, frag_state, spath, sbits);
            assert_hash_eq_if_enabled(&mut builder, frag_en, root_out, frag_root);
        }

        // when present: rewritten_total == sum_len
        assert_eq_if_enabled(&mut builder, present, rewritten_total, sum_len);

        let orig_leaf = hash_msg(&mut builder, role_id, orig_total_len_i, body_state);
        let rewritten_leaf = hash_msg(&mut builder, role_id, rewritten_total, bytes_state);

        let next_user_acc = transcript_step(&mut builder, user_acc, orig_leaf);
        user_acc = select_hash(&mut builder, present, next_user_acc, user_acc);
        let next_mod_acc = transcript_step(&mut builder, mod_acc, rewritten_leaf);
        mod_acc = select_hash(&mut builder, present, next_mod_acc, mod_acc);

        if public_orig {
            public_orig_fields.push(role_id);
            public_orig_fields.push(present.target);
            public_orig_fields.push(orig_total_len_i);
            for k in 0..params.max_orig_chunks {
                public_orig_fields.push(orig_chunks_enabled[i][k].target);
                public_orig_fields.push(orig_chunks_len[i][k]);
                public_orig_fields.push(orig_chunks_packed[i][k]);
            }
        }
    }

    // Bind the public interval to this leaf's actual prefix of present messages.
    let expected_end_idx = builder.add(start_idx, present_count);
    builder.connect(end_idx, expected_end_idx);
    builder.connect(rewritten_count_out, rewritten_count);

    // Counts/capacities are canonical 32-bit integers. Together with the
    // equalities above this prevents field wraparound in both transitions.
    builder.range_check(start_idx, 32);
    builder.range_check(end_idx, 32);
    builder.range_check(rewritten_count_in, 32);
    builder.range_check(rewritten_count_out, 32);

    let computed_mod_root = mod_acc;

    builder.register_public_inputs(&msg_commitment.elements);
    builder.register_public_inputs(&frag_commitment.elements);
    builder.register_public_inputs(&request_randomness_binding.elements);
    builder.register_public_input(start_idx);
    builder.register_public_input(end_idx);
    builder.register_public_input(rewritten_count_in);
    builder.register_public_input(rewritten_count_out);
    builder.register_public_input(prev_present.target);
    builder.register_public_input(computed_last_present.target);
    builder.register_public_inputs(&user_acc_in.elements);
    builder.register_public_inputs(&user_acc.elements);
    builder.register_public_inputs(&mod_acc_in.elements);
    builder.register_public_inputs(&computed_mod_root.elements);
    if public_orig {
        for t in public_orig_fields {
            builder.register_public_input(t);
        }
    }

    let data = builder.build::<C>();
    (
        data,
        RewriteTargets {
            msg_commitment,
            frag_commitment,
            request_randomness_binding,
            start_idx,
            end_idx,
            rewritten_count_in,
            rewritten_count_out,
            prev_present,
            last_present: computed_last_present,
            user_acc_in,
            user_root: user_acc,
            mod_acc_in,
            mod_root: computed_mod_root,
        },
        RewriteWitnessTargets {
            msg_root,
            frag_root,
            msg_randomness,
            frag_randomness,
            request_randomness,
            start_idx,
            end_idx,
            rewritten_count_in,
            rewritten_count_out,
            prev_present,
            user_acc_in,
            mod_acc_in,
            orig_present,
            orig_role_id,
            orig_total_len,
            orig_chunks_enabled,
            orig_chunks_len,
            orig_chunks_packed,
            gap_enabled,
            gap_leaf,
            gap_path,
            gap_index_bits,
            prefixes_enabled,
            prefixes_total_len,
            prefixes_chunks_enabled,
            prefixes_chunks_len,
            prefixes_chunks_packed,
            prefixes_path,
            prefixes_index_bits,
            suffixes_enabled,
            suffixes_total_len,
            suffixes_chunks_enabled,
            suffixes_chunks_len,
            suffixes_chunks_packed,
            suffixes_path,
            suffixes_index_bits,
        },
    )
}

/// Fill one authentication path, expanding a shared proof only when needed.
fn fill_merkle_path(
    pw: &mut PartialWitness<F>,
    targets: &[HashOutTarget],
    bit_targets: &[BoolTarget],
    path: &[Digest],
    bits: &[bool],
    enabled: bool,
    dedup: Option<&(usize, HashMap<(usize, u64), Digest>)>,
) -> Result<()> {
    anyhow::ensure!(
        bits.len() == bit_targets.len(),
        "Merkle index length mismatch"
    );
    let expanded;
    let path = if path.is_empty() && !targets.is_empty() {
        expanded = if enabled {
            let (height, map) = dedup
                .ok_or_else(|| anyhow::anyhow!("missing Merkle path and deduplicated proof"))?;
            expand_path_from_dedup_map(*height, map, bits)?
        } else {
            vec![[0; DIGEST_WIDTH]; targets.len()]
        };
        &expanded
    } else {
        path
    };
    anyhow::ensure!(path.len() == targets.len(), "Merkle path length mismatch");
    for level in 0..path.len() {
        pw.set_hash_target(targets[level], digest_to_hashout(path[level]))?;
        pw.set_bool_target(bit_targets[level], bits[level])?;
    }
    Ok(())
}

/// Fill the private witness and generate a batch proof.
pub fn prove_rewrite_tree_batch(
    data: &CircuitData<F, C, D>,
    t: &RewriteWitnessTargets,
    witness: &RewriteTreeBatchWitness,
) -> Result<plonky2::plonk::proof::ProofWithPublicInputs<F, C, D>> {
    let mut pw = PartialWitness::<F>::new();
    pw.set_hash_target(t.msg_root, digest_to_hashout(witness.msg_root))?;
    pw.set_hash_target(t.frag_root, digest_to_hashout(witness.frag_root))?;
    pw.set_hash_target(t.msg_randomness, digest_to_hashout(witness.msg_randomness))?;
    pw.set_hash_target(
        t.frag_randomness,
        digest_to_hashout(witness.frag_randomness),
    )?;
    pw.set_hash_target(
        t.request_randomness,
        digest_to_hashout(witness.request_randomness),
    )?;
    pw.set_target(t.start_idx, f_u64(witness.start_idx))?;
    pw.set_target(t.end_idx, f_u64(witness.end_idx))?;
    pw.set_target(t.rewritten_count_in, f_u64(witness.rewritten_count_in))?;
    pw.set_target(t.rewritten_count_out, f_u64(witness.rewritten_count_out))?;
    pw.set_bool_target(t.prev_present, witness.prev_present)?;
    pw.set_hash_target(t.user_acc_in, digest_to_hashout(witness.user_acc_in))?;
    pw.set_hash_target(t.mod_acc_in, digest_to_hashout(witness.mod_acc_in))?;

    // Pre-build dedup maps if present (faster than rebuilding per-leaf).
    let msg_dedup = witness
        .msg_dedup_proof
        .as_ref()
        .map(|p| (p.height, dedup_proof_map(p)));
    let frag_dedup = witness
        .frag_dedup_proof
        .as_ref()
        .map(|p| (p.height, dedup_proof_map(p)));

    anyhow::ensure!(
        witness.orig.len() == t.orig_present.len(),
        "orig len mismatch"
    );
    for i in 0..t.orig_present.len() {
        let om = &witness.orig[i];
        pw.set_bool_target(t.orig_present[i], om.present)?;
        pw.set_target(t.orig_role_id[i], f_u64(om.role_id))?;
        pw.set_target(t.orig_total_len[i], f_u64(om.total_len))?;
        anyhow::ensure!(
            om.chunks.len() == t.orig_chunks_enabled[i].len(),
            "orig.chunks len mismatch"
        );
        for k in 0..om.chunks.len() {
            let ch = &om.chunks[k];
            pw.set_bool_target(t.orig_chunks_enabled[i][k], ch.enabled)?;
            pw.set_target(t.orig_chunks_len[i][k], f_u64(ch.len))?;
            pw.set_target(t.orig_chunks_packed[i][k], f_u64(ch.packed_le))?;
        }
    }

    anyhow::ensure!(
        witness.gap_insertions.len() == t.gap_enabled.len(),
        "gap len mismatch"
    );
    for i in 0..t.gap_enabled.len() {
        anyhow::ensure!(
            witness.gap_insertions[i].len() == t.gap_enabled[i].len(),
            "gap inner len mismatch"
        );
        for j in 0..t.gap_enabled[i].len() {
            let ins = &witness.gap_insertions[i][j];
            pw.set_bool_target(t.gap_enabled[i][j], ins.enabled)?;
            pw.set_hash_target(t.gap_leaf[i][j], digest_to_hashout(ins.leaf))?;
            fill_merkle_path(
                &mut pw,
                &t.gap_path[i][j],
                &t.gap_index_bits[i][j],
                &ins.path,
                &ins.index_bits,
                ins.enabled,
                msg_dedup.as_ref(),
            )?;
        }
    }

    for (
        fragments,
        enabled,
        total_len,
        chunks_enabled,
        chunks_len,
        chunks_packed,
        paths,
        index_bits,
    ) in [
        (
            &witness.prefixes,
            &t.prefixes_enabled,
            &t.prefixes_total_len,
            &t.prefixes_chunks_enabled,
            &t.prefixes_chunks_len,
            &t.prefixes_chunks_packed,
            &t.prefixes_path,
            &t.prefixes_index_bits,
        ),
        (
            &witness.suffixes,
            &t.suffixes_enabled,
            &t.suffixes_total_len,
            &t.suffixes_chunks_enabled,
            &t.suffixes_chunks_len,
            &t.suffixes_chunks_packed,
            &t.suffixes_path,
            &t.suffixes_index_bits,
        ),
    ] {
        anyhow::ensure!(
            fragments.len() == enabled.len(),
            "fragment batch length mismatch"
        );
        for i in 0..enabled.len() {
            anyhow::ensure!(
                fragments[i].len() == enabled[i].len(),
                "fragment slot length mismatch"
            );
            for (f, frag) in fragments[i].iter().enumerate() {
                pw.set_bool_target(enabled[i][f], frag.enabled)?;
                pw.set_target(total_len[i][f], f_u64(frag.total_len))?;
                anyhow::ensure!(
                    frag.chunks.len() == chunks_enabled[i][f].len(),
                    "fragment chunk length mismatch"
                );
                for (k, chunk) in frag.chunks.iter().enumerate() {
                    pw.set_bool_target(chunks_enabled[i][f][k], chunk.enabled)?;
                    pw.set_target(chunks_len[i][f][k], f_u64(chunk.len))?;
                    pw.set_target(chunks_packed[i][f][k], f_u64(chunk.packed_le))?;
                }
                fill_merkle_path(
                    &mut pw,
                    &paths[i][f],
                    &index_bits[i][f],
                    &frag.path,
                    &frag.index_bits,
                    frag.enabled,
                    frag_dedup.as_ref(),
                )?;
            }
        }
    }

    Ok(data.prove(pw)?)
}
