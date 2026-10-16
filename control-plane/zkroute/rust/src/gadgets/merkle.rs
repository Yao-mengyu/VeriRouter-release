use plonky2::hash::hash_types::HashOutTarget;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;

pub fn poseidon_hash_no_pad<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    inputs: &[Target],
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(inputs.to_vec())
}

pub fn hash_pair<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    left: &HashOutTarget,
    right: &HashOutTarget,
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    let mut inputs = Vec::with_capacity(8);
    inputs.extend_from_slice(&left.elements);
    inputs.extend_from_slice(&right.elements);
    builder.hash_n_to_hash_no_pad::<PoseidonHash>(inputs)
}

pub fn merkle_root_from_leaves<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    mut leaves: Vec<HashOutTarget>,
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    assert!(!leaves.is_empty(), "empty merkle tree");
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity((leaves.len() + 1) / 2);
        for pair in leaves.chunks(2) {
            let left = &pair[0];
            let right = if pair.len() == 2 { &pair[1] } else { &pair[0] };
            next.push(hash_pair(builder, left, right));
        }
        leaves = next;
    }
    leaves[0]
}

pub fn merkle_recompute_root<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    leaf: HashOutTarget,
    path_bits: &[Target],
    path_hashes: &[HashOutTarget],
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    assert_eq!(path_bits.len(), path_hashes.len());
    let mut cur = leaf;
    for (bit, sibling) in path_bits.iter().zip(path_hashes.iter()) {
        let is_right = BoolTarget::new_unsafe(*bit);
        let left = select_hash(builder, is_right, sibling, &cur);
        let right = select_hash(builder, is_right, &cur, sibling);
        cur = hash_pair(builder, &left, &right);
    }
    cur
}

pub fn merkle_recompute_root_from_bools<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    leaf: HashOutTarget,
    path_bits: &[BoolTarget],
    path_hashes: &[HashOutTarget],
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    let bits: Vec<Target> = path_bits.iter().map(|b| b.target).collect();
    merkle_recompute_root(builder, leaf, &bits, path_hashes)
}

pub fn verify_merkle_multiproof_gated<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    root: &HashOutTarget,
    leaves: &[HashOutTarget],
    index_bits: &[Vec<BoolTarget>],
    siblings: &[Vec<HashOutTarget>],
    enabled: &[BoolTarget],
) where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    assert_eq!(leaves.len(), index_bits.len());
    assert_eq!(leaves.len(), siblings.len());
    assert_eq!(leaves.len(), enabled.len());

    for i in 0..leaves.len() {
        let root_calc =
            merkle_recompute_root_from_bools(builder, leaves[i], &index_bits[i], &siblings[i]);
        let gate = enabled[i].target;
        for j in 0..4 {
            let diff = builder.sub(root_calc.elements[j], root.elements[j]);
            let gated = builder.mul(diff, gate);
            builder.assert_zero(gated);
        }
    }
}

fn select_hash<F: plonky2::field::types::Field, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    b: BoolTarget,
    h0: &HashOutTarget,
    h1: &HashOutTarget,
) -> HashOutTarget
where
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
{
    let mut elements = [builder.zero(); 4];
    for i in 0..4 {
        elements[i] = builder.select(b, h0.elements[i], h1.elements[i]);
    }
    HashOutTarget { elements }
}

/// Typed, length-bound application Fiat--Shamir transcript. The prefix
/// separates application families; joint ROM behavior with the backend and
/// untagged Poseidon/Merkle uses is an explicit instantiation assumption.
pub fn application_challenge<
    F: plonky2::hash::hash_types::RichField + plonky2::field::extension::Extendable<D>,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    family: u64,
    inputs: &[Target],
) -> HashOutTarget {
    let mut transcript = vec![
        builder.constant(F::from_canonical_u64(0x56525f46535f5632)),
        builder.constant(F::from_canonical_u64(family)),
        builder.constant(F::from_canonical_usize(inputs.len())),
    ];
    transcript.extend_from_slice(inputs);
    poseidon_hash_no_pad(builder, &transcript)
}
