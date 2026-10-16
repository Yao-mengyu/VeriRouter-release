use crate::commitment::{
    commit_digest_target, request_hash_target, request_randomness_binding_target, transcript_init,
};
use std::sync::Arc;

use anyhow::Result;
use rayon::prelude::*;

use crate::circuits::batch::{
    PI_END_IDX, PI_FRAG_COMMITMENT, PI_LAST_PRESENT, PI_MOD_ACC_IN, PI_MOD_ROOT, PI_MSG_COMMITMENT,
    PI_PREV_PRESENT, PI_REQUEST_RANDOMNESS_BINDING, PI_REWRITTEN_COUNT_IN, PI_REWRITTEN_COUNT_OUT,
    PI_START_IDX, PI_USER_ACC_IN, PI_USER_ROOT,
};
use crate::util::{digest_to_hashout, Digest, C, D, F};
use plonky2::field::types::Field;
use plonky2::hash::hash_types::{HashOut, HashOutTarget};
use plonky2::iop::target::BoolTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};

#[derive(Clone, Copy, Debug, Default)]
pub struct RewriteAggStats {
    pub agg_setup_s: f64,
    pub agg_prove_s: f64,
}

/// Cache of Rewrite recursive circuits:
/// - a growing list of aggregator levels (each verifies proofs of the previous level)
///
/// Intended usage: construct once, then reuse across many runs (e.g. benchmarking different counts)
/// so circuit build (setup) is amortized.
pub struct RewriteRecursionCache {
    fan_in: usize,
    zero_knowledge: bool,
    leaf_data: Arc<CircuitData<F, C, D>>,
    agg_levels: Vec<(Arc<CircuitData<F, C, D>>, RewriteAggTargets)>,
    finalizers: Vec<Option<(Arc<CircuitData<F, C, D>>, RewriteFinalTargets)>>,
    root_aggs: Vec<Option<(Arc<CircuitData<F, C, D>>, RewriteAggTargets)>>,
}

#[derive(Clone, Debug)]
pub struct LevelSizeInfo {
    /// "leaf" | "agg_0" | ... | "root"
    pub level: String,
    /// The polynomial degree bits (typically drives proof size thresholds).
    pub degree_bits: usize,
    /// Proof size in bytes. For levels with multiple proofs, this is the average.
    pub proof_bytes_avg: usize,
    /// Count of proofs measured at that level.
    pub n_proofs: usize,
}

#[derive(Clone, Debug)]
struct RewriteAggTargets {
    proofs: Vec<ProofWithPublicInputsTarget<D>>,
    verifier_data: VerifierCircuitTarget,
    enabled: Vec<BoolTarget>,
    expected_msg_commitment: HashOutTarget,
    expected_frag_commitment: HashOutTarget,
    expected_request_randomness_binding: HashOutTarget,
    expected_start_idx: Target,
    expected_end_idx: Target,
    expected_rewritten_count_in: Target,
    expected_rewritten_count_out: Target,
    expected_prev_present: Target,
    expected_last_present: Target,
    expected_user_acc_in: HashOutTarget,
    expected_user_root: HashOutTarget,
    expected_mod_acc_in: HashOutTarget,
    expected_mod_root: HashOutTarget,
    /// Present only for a root-statement aggregator. This remains a private
    /// witness and opens the internal request-randomness binding.
    request_randomness: Option<HashOutTarget>,
}

#[derive(Clone, Debug)]
struct RewriteFinalTargets {
    proof: ProofWithPublicInputsTarget<D>,
    verifier_data: VerifierCircuitTarget,
    request_randomness: HashOutTarget,
}

pub const FINAL_PI_MESSAGE_BANK_COMMITMENT: usize = 0;
pub const FINAL_PI_FRAGMENT_BANK_COMMITMENT: usize = FINAL_PI_MESSAGE_BANK_COMMITMENT + 4;
pub const FINAL_PI_CLIENT_MESSAGE_COUNT: usize = FINAL_PI_FRAGMENT_BANK_COMMITMENT + 4;
pub const FINAL_PI_CLIENT_REQUEST_HASH: usize = FINAL_PI_CLIENT_MESSAGE_COUNT + 1;
pub const FINAL_PI_REWRITTEN_MESSAGE_COUNT: usize = FINAL_PI_CLIENT_REQUEST_HASH + 4;
pub const FINAL_PI_REQUEST_COMMITMENT: usize = FINAL_PI_REWRITTEN_MESSAGE_COUNT + 1;
pub const FINAL_PUBLIC_INPUTS: usize = FINAL_PI_REQUEST_COMMITMENT + 4;

fn pi_hash_target(public_inputs: &[Target], start: usize) -> HashOutTarget {
    HashOutTarget {
        elements: std::array::from_fn(|offset| public_inputs[start + offset]),
    }
}

fn pi_hash(public_inputs: &[F], start: usize) -> HashOut<F> {
    HashOut {
        elements: std::array::from_fn(|offset| public_inputs[start + offset]),
    }
}

fn connect_hash_if_enabled(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    left: HashOutTarget,
    right: HashOutTarget,
) {
    let zero = builder.zero();
    for offset in 0..4 {
        let diff = builder.sub(left.elements[offset], right.elements[offset]);
        let gated = builder.mul(enabled.target, diff);
        builder.connect(gated, zero);
    }
}

fn select_hash(
    builder: &mut CircuitBuilder<F, D>,
    enabled: BoolTarget,
    when_true: HashOutTarget,
    when_false: HashOutTarget,
) -> HashOutTarget {
    HashOutTarget {
        elements: std::array::from_fn(|offset| {
            builder.select(
                enabled,
                when_true.elements[offset],
                when_false.elements[offset],
            )
        }),
    }
}

/// Aggregation circuit for Rewrite proofs over ordered batch transitions:
/// - verifies `fan_in` child proofs of `inner_common`
/// - enforces all enabled children share `msg_root` and `frag_root`
/// - enforces continuity across enabled children:
///   `start_idx[i] == end_idx(prev)`,
///   `prev_present[i] == last_present(prev)`,
///   `user_acc_in[i] == user_root(prev)`,
///   `mod_acc_in[i] == mod_root(prev)`
/// - outputs public inputs:
///   `[msg_root, frag_root, start_idx_group, end_idx_group, prev_present_group,
///     last_present_group, user_acc_in_group, user_root_group, mod_acc_in_group,
///     mod_root_group]`
///
/// The full transition layout is deliberately preserved at every recursion level.
/// Higher levels need the boundaries and accumulator inputs to enforce continuity
/// between already-aggregated groups.
fn build_rewrite_agg_circuit(
    fan_in: usize,
    inner_common: &CommonCircuitData<F, D>,
    inner_verifier_only: &VerifierOnlyCircuitData<C, D>,
    zero_knowledge: bool,
    final_statement: bool,
) -> (CircuitData<F, C, D>, RewriteAggTargets)
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = zero_knowledge;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let proofs = (0..fan_in)
        .map(|_| builder.add_virtual_proof_with_pis(inner_common))
        .collect::<Vec<_>>();
    // Bind every recursive proof to the approved child circuit, including its
    // circuit digest and constants/sigmas cap. CommonCircuitData alone does not
    // identify the child relation.
    let verifier_data = builder.constant_verifier_data::<C>(inner_verifier_only);
    let enabled = (0..fan_in)
        .map(|_| builder.add_virtual_bool_target_safe())
        .collect::<Vec<_>>();

    let expected_msg_commitment = builder.add_virtual_hash();
    let expected_frag_commitment = builder.add_virtual_hash();
    let expected_request_randomness_binding = builder.add_virtual_hash();
    let expected_start_idx = builder.add_virtual_target();
    let expected_end_idx = builder.add_virtual_target();
    let expected_rewritten_count_in = builder.add_virtual_target();
    let expected_rewritten_count_out = builder.add_virtual_target();
    let expected_prev_present = builder.add_virtual_target();
    let expected_last_present = builder.add_virtual_target();
    let expected_user_acc_in = builder.add_virtual_hash();
    let expected_user_root = builder.add_virtual_hash();
    let expected_mod_acc_in = builder.add_virtual_hash();
    let expected_mod_root = builder.add_virtual_hash();

    builder.assert_one(enabled[0].target);

    // child0 defines the group's left boundary.
    builder.connect(proofs[0].public_inputs[PI_START_IDX], expected_start_idx);
    builder.connect(
        proofs[0].public_inputs[PI_PREV_PRESENT],
        expected_prev_present,
    );
    builder.connect_hashes(
        pi_hash_target(&proofs[0].public_inputs, PI_USER_ACC_IN),
        expected_user_acc_in,
    );
    builder.connect_hashes(
        pi_hash_target(&proofs[0].public_inputs, PI_MOD_ACC_IN),
        expected_mod_acc_in,
    );
    builder.connect(
        proofs[0].public_inputs[PI_REWRITTEN_COUNT_IN],
        expected_rewritten_count_in,
    );

    let mut prev_end = proofs[0].public_inputs[PI_END_IDX];
    let mut prev_rewritten_count_out = proofs[0].public_inputs[PI_REWRITTEN_COUNT_OUT];
    let mut prev_last = proofs[0].public_inputs[PI_LAST_PRESENT];
    let mut prev_user_root = pi_hash_target(&proofs[0].public_inputs, PI_USER_ROOT);
    let mut prev_mod_root = pi_hash_target(&proofs[0].public_inputs, PI_MOD_ROOT);
    let mut end_idx_group = prev_end;
    let mut rewritten_count_out_group = prev_rewritten_count_out;
    let mut last_present_group = prev_last;
    let mut user_root_group = prev_user_root;
    let mut mod_root_group = prev_mod_root;

    for i in 0..fan_in {
        builder.verify_proof::<C>(&proofs[i], &verifier_data, inner_common);

        // If enabled: enforce all bank commitments and the hidden request-salt
        // handle match the group's expected values.
        let msg_commitment_i = pi_hash_target(&proofs[i].public_inputs, PI_MSG_COMMITMENT);
        let frag_commitment_i = pi_hash_target(&proofs[i].public_inputs, PI_FRAG_COMMITMENT);
        let request_randomness_binding_i =
            pi_hash_target(&proofs[i].public_inputs, PI_REQUEST_RANDOMNESS_BINDING);
        connect_hash_if_enabled(
            &mut builder,
            enabled[i],
            msg_commitment_i,
            expected_msg_commitment,
        );
        connect_hash_if_enabled(
            &mut builder,
            enabled[i],
            frag_commitment_i,
            expected_frag_commitment,
        );
        connect_hash_if_enabled(
            &mut builder,
            enabled[i],
            request_randomness_binding_i,
            expected_request_randomness_binding,
        );
        let z = builder.zero();

        if i > 0 {
            // Enabled children form a prefix. This keeps disabled padding from
            // creating an unconstrained hole before a later enabled child.
            let one = builder.one();
            let one_minus_prev = builder.sub(one, enabled[i - 1].target);
            let invalid_enable = builder.mul(enabled[i].target, one_minus_prev);
            builder.connect(invalid_enable, z);
        }

        // Ordered transition continuity for enabled children (i > 0).
        if i > 0 {
            let start_idx_i = proofs[i].public_inputs[PI_START_IDX];
            let prev_present_i = proofs[i].public_inputs[PI_PREV_PRESENT];
            let user_acc_in_i = pi_hash_target(&proofs[i].public_inputs, PI_USER_ACC_IN);
            let mod_acc_in_i = pi_hash_target(&proofs[i].public_inputs, PI_MOD_ACC_IN);
            let rewritten_count_in_i = proofs[i].public_inputs[PI_REWRITTEN_COUNT_IN];

            let diff_start = builder.sub(start_idx_i, prev_end);
            let gated_start = builder.mul(enabled[i].target, diff_start);
            builder.connect(gated_start, z);

            let diff_prev = builder.sub(prev_present_i, prev_last);
            let gated_prev = builder.mul(enabled[i].target, diff_prev);
            builder.connect(gated_prev, z);

            connect_hash_if_enabled(&mut builder, enabled[i], user_acc_in_i, prev_user_root);
            connect_hash_if_enabled(&mut builder, enabled[i], mod_acc_in_i, prev_mod_root);

            let diff_rewritten = builder.sub(rewritten_count_in_i, prev_rewritten_count_out);
            let gated_rewritten = builder.mul(enabled[i].target, diff_rewritten);
            builder.connect(gated_rewritten, z);
        }

        // Update group right boundary to the last enabled child's values.
        let end_idx_i = proofs[i].public_inputs[PI_END_IDX];
        let rewritten_count_out_i = proofs[i].public_inputs[PI_REWRITTEN_COUNT_OUT];
        let last_present_i = proofs[i].public_inputs[PI_LAST_PRESENT];
        let user_root_i = pi_hash_target(&proofs[i].public_inputs, PI_USER_ROOT);
        let mod_root_i = pi_hash_target(&proofs[i].public_inputs, PI_MOD_ROOT);
        end_idx_group = builder.select(enabled[i], end_idx_i, end_idx_group);
        rewritten_count_out_group =
            builder.select(enabled[i], rewritten_count_out_i, rewritten_count_out_group);
        last_present_group = builder.select(enabled[i], last_present_i, last_present_group);
        user_root_group = select_hash(&mut builder, enabled[i], user_root_i, user_root_group);
        mod_root_group = select_hash(&mut builder, enabled[i], mod_root_i, mod_root_group);
        prev_end = builder.select(enabled[i], end_idx_i, prev_end);
        prev_rewritten_count_out =
            builder.select(enabled[i], rewritten_count_out_i, prev_rewritten_count_out);
        prev_last = builder.select(enabled[i], last_present_i, prev_last);
        prev_user_root = select_hash(&mut builder, enabled[i], user_root_i, prev_user_root);
        prev_mod_root = select_hash(&mut builder, enabled[i], mod_root_i, prev_mod_root);
    }

    builder.connect(end_idx_group, expected_end_idx);
    builder.connect(rewritten_count_out_group, expected_rewritten_count_out);
    builder.connect(last_present_group, expected_last_present);
    builder.connect_hashes(user_root_group, expected_user_root);
    builder.connect_hashes(mod_root_group, expected_mod_root);

    let request_randomness = if final_statement {
        let zero = builder.zero();
        let one = builder.one();
        builder.connect(expected_start_idx, zero);
        builder.connect(expected_rewritten_count_in, zero);
        builder.connect(expected_prev_present, one);
        builder.connect(expected_last_present, zero);
        let iv = transcript_init();
        for (offset, value) in iv.into_iter().enumerate() {
            let expected = builder.constant(F::from_canonical_u64(value));
            builder.connect(expected_user_acc_in.elements[offset], expected);
            builder.connect(expected_mod_acc_in.elements[offset], expected);
        }

        // The root opens the internal salt handle with a private witness, then
        // commits to the fully finalized rewritten request digest.
        let request_randomness = builder.add_virtual_hash();
        let request_randomness_binding =
            request_randomness_binding_target(&mut builder, request_randomness);
        builder.connect_hashes(
            request_randomness_binding,
            expected_request_randomness_binding,
        );

        builder.range_check(expected_end_idx, 32);
        builder.range_check(expected_rewritten_count_out, 32);
        let client_request_hash =
            request_hash_target(&mut builder, expected_end_idx, expected_user_root);
        let rewritten_request_hash = request_hash_target(
            &mut builder,
            expected_rewritten_count_out,
            expected_mod_root,
        );
        let request_commitment =
            commit_digest_target(&mut builder, rewritten_request_hash, request_randomness);

        builder.register_public_inputs(&expected_msg_commitment.elements);
        builder.register_public_inputs(&expected_frag_commitment.elements);
        builder.register_public_input(expected_end_idx);
        builder.register_public_inputs(&client_request_hash.elements);
        builder.register_public_input(expected_rewritten_count_out);
        builder.register_public_inputs(&request_commitment.elements);
        Some(request_randomness)
    } else {
        builder.register_public_inputs(&expected_msg_commitment.elements);
        builder.register_public_inputs(&expected_frag_commitment.elements);
        builder.register_public_inputs(&expected_request_randomness_binding.elements);
        builder.register_public_input(expected_start_idx);
        builder.register_public_input(expected_end_idx);
        builder.register_public_input(expected_rewritten_count_in);
        builder.register_public_input(expected_rewritten_count_out);
        builder.register_public_input(expected_prev_present);
        builder.register_public_input(expected_last_present);
        builder.register_public_inputs(&expected_user_acc_in.elements);
        builder.register_public_inputs(&expected_user_root.elements);
        builder.register_public_inputs(&expected_mod_acc_in.elements);
        builder.register_public_inputs(&expected_mod_root.elements);
        None
    };

    let data = builder.build::<C>();
    (
        data,
        RewriteAggTargets {
            proofs,
            verifier_data,
            enabled,
            expected_msg_commitment,
            expected_frag_commitment,
            expected_request_randomness_binding,
            expected_start_idx,
            expected_end_idx,
            expected_rewritten_count_in,
            expected_rewritten_count_out,
            expected_prev_present,
            expected_last_present,
            expected_user_acc_in,
            expected_user_root,
            expected_mod_acc_in,
            expected_mod_root,
            request_randomness,
        },
    )
}

struct NativeGroupState {
    start_idx: F,
    end_idx: F,
    rewritten_count_in: F,
    rewritten_count_out: F,
    prev_present: F,
    last_present: F,
    user_acc_in: HashOut<F>,
    user_root: HashOut<F>,
    mod_acc_in: HashOut<F>,
    mod_root: HashOut<F>,
}

fn native_group_state(
    child_proofs: &[ProofWithPublicInputs<F, C, D>],
    enabled: &[bool],
) -> Result<NativeGroupState>
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    anyhow::ensure!(!child_proofs.is_empty(), "child_proofs must be non-empty");
    anyhow::ensure!(child_proofs.len() == enabled.len(), "enabled len mismatch");
    anyhow::ensure!(enabled[0], "first child must be enabled");

    let mut last = 0usize;
    for (i, en) in enabled.iter().enumerate() {
        if *en {
            last = i;
        }
    }

    for i in 1..child_proofs.len() {
        if enabled[i] {
            anyhow::ensure!(
                child_proofs[i].public_inputs[PI_START_IDX]
                    == child_proofs[i - 1].public_inputs[PI_END_IDX],
                "enabled child start_idx does not match previous end_idx"
            );
            anyhow::ensure!(
                child_proofs[i].public_inputs[PI_PREV_PRESENT]
                    == child_proofs[i - 1].public_inputs[PI_LAST_PRESENT],
                "enabled child prev_present does not match previous last_present"
            );
            anyhow::ensure!(
                pi_hash(&child_proofs[i].public_inputs, PI_USER_ACC_IN)
                    == pi_hash(&child_proofs[i - 1].public_inputs, PI_USER_ROOT),
                "enabled child user_acc_in does not match previous user_root"
            );
            anyhow::ensure!(
                pi_hash(&child_proofs[i].public_inputs, PI_MOD_ACC_IN)
                    == pi_hash(&child_proofs[i - 1].public_inputs, PI_MOD_ROOT),
                "enabled child mod_acc_in does not match previous mod_root"
            );
            anyhow::ensure!(
                child_proofs[i].public_inputs[PI_REWRITTEN_COUNT_IN]
                    == child_proofs[i - 1].public_inputs[PI_REWRITTEN_COUNT_OUT],
                "enabled child rewritten_count_in does not match previous rewritten_count_out"
            );
            anyhow::ensure!(
                pi_hash(
                    &child_proofs[i].public_inputs,
                    PI_REQUEST_RANDOMNESS_BINDING
                ) == pi_hash(
                    &child_proofs[i - 1].public_inputs,
                    PI_REQUEST_RANDOMNESS_BINDING
                ),
                "enabled children use different request randomness bindings"
            );
        }
    }

    Ok(NativeGroupState {
        start_idx: child_proofs[0].public_inputs[PI_START_IDX],
        end_idx: child_proofs[last].public_inputs[PI_END_IDX],
        rewritten_count_in: child_proofs[0].public_inputs[PI_REWRITTEN_COUNT_IN],
        rewritten_count_out: child_proofs[last].public_inputs[PI_REWRITTEN_COUNT_OUT],
        prev_present: child_proofs[0].public_inputs[PI_PREV_PRESENT],
        last_present: child_proofs[last].public_inputs[PI_LAST_PRESENT],
        user_acc_in: pi_hash(&child_proofs[0].public_inputs, PI_USER_ACC_IN),
        user_root: pi_hash(&child_proofs[last].public_inputs, PI_USER_ROOT),
        mod_acc_in: pi_hash(&child_proofs[0].public_inputs, PI_MOD_ACC_IN),
        mod_root: pi_hash(&child_proofs[last].public_inputs, PI_MOD_ROOT),
    })
}

fn prove_rewrite_agg(
    data: &CircuitData<F, C, D>,
    t: &RewriteAggTargets,
    inner_verifier_only: &plonky2::plonk::circuit_data::VerifierOnlyCircuitData<C, D>,
    child_proofs: &[ProofWithPublicInputs<F, C, D>],
    enabled: &[bool],
    request_randomness: Option<Digest>,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    anyhow::ensure!(
        child_proofs.len() == t.proofs.len(),
        "child_proofs len mismatch"
    );
    anyhow::ensure!(enabled.len() == t.enabled.len(), "enabled len mismatch");
    anyhow::ensure!(enabled[0], "first child must be enabled");

    let mut pw = PartialWitness::<F>::new();
    pw.set_verifier_data_target(&t.verifier_data, inner_verifier_only)?;
    for i in 0..t.proofs.len() {
        pw.set_proof_with_pis_target(&t.proofs[i], &child_proofs[i])?;
        pw.set_bool_target(t.enabled[i], enabled[i])?;
    }

    let state = native_group_state(child_proofs, enabled)?;
    pw.set_hash_target(
        t.expected_msg_commitment,
        pi_hash(&child_proofs[0].public_inputs, PI_MSG_COMMITMENT),
    )?;
    pw.set_hash_target(
        t.expected_frag_commitment,
        pi_hash(&child_proofs[0].public_inputs, PI_FRAG_COMMITMENT),
    )?;
    pw.set_hash_target(
        t.expected_request_randomness_binding,
        pi_hash(
            &child_proofs[0].public_inputs,
            PI_REQUEST_RANDOMNESS_BINDING,
        ),
    )?;
    pw.set_target(t.expected_start_idx, state.start_idx)?;
    pw.set_target(t.expected_end_idx, state.end_idx)?;
    pw.set_target(t.expected_rewritten_count_in, state.rewritten_count_in)?;
    pw.set_target(t.expected_rewritten_count_out, state.rewritten_count_out)?;
    pw.set_target(t.expected_prev_present, state.prev_present)?;
    pw.set_target(t.expected_last_present, state.last_present)?;
    pw.set_hash_target(t.expected_user_acc_in, state.user_acc_in)?;
    pw.set_hash_target(t.expected_user_root, state.user_root)?;
    pw.set_hash_target(t.expected_mod_acc_in, state.mod_acc_in)?;
    pw.set_hash_target(t.expected_mod_root, state.mod_root)?;
    match (t.request_randomness, request_randomness) {
        (Some(target), Some(randomness)) => {
            pw.set_hash_target(target, digest_to_hashout(randomness))?;
        }
        (None, None) => {}
        (Some(_), None) => anyhow::bail!("root aggregation requires request randomness"),
        (None, Some(_)) => anyhow::bail!("request randomness supplied to non-root aggregation"),
    }

    Ok(data.prove(pw)?)
}

/// Verify the recursively aggregated transition and export the complete rewrite
/// statement. This final recursive layer fixes the global left boundary and the
/// canonical terminal row; only finalized request/bank commitments and their
/// public 32-bit message counts leave this layer.
fn build_rewrite_final_circuit(
    inner_common: &CommonCircuitData<F, D>,
    inner_verifier_only: &VerifierOnlyCircuitData<C, D>,
    zero_knowledge: bool,
) -> (CircuitData<F, C, D>, RewriteFinalTargets)
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = zero_knowledge;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let proof = builder.add_virtual_proof_with_pis(inner_common);
    // Bind every recursive proof to the approved child circuit, including its
    // circuit digest and constants/sigmas cap. CommonCircuitData alone does not
    // identify the child relation.
    let verifier_data = builder.constant_verifier_data::<C>(inner_verifier_only);
    builder.verify_proof::<C>(&proof, &verifier_data, inner_common);

    let zero = builder.zero();
    let one = builder.one();
    builder.connect(proof.public_inputs[PI_START_IDX], zero);
    builder.connect(proof.public_inputs[PI_REWRITTEN_COUNT_IN], zero);
    builder.connect(proof.public_inputs[PI_PREV_PRESENT], one);
    builder.connect(proof.public_inputs[PI_LAST_PRESENT], zero);

    let iv = transcript_init();
    for (offset, value) in iv.into_iter().enumerate() {
        let expected = builder.constant(F::from_canonical_u64(value));
        builder.connect(proof.public_inputs[PI_USER_ACC_IN + offset], expected);
        builder.connect(proof.public_inputs[PI_MOD_ACC_IN + offset], expected);
    }

    let client_message_count = proof.public_inputs[PI_END_IDX];
    let rewritten_message_count = proof.public_inputs[PI_REWRITTEN_COUNT_OUT];
    builder.range_check(client_message_count, 32);
    builder.range_check(rewritten_message_count, 32);

    let client_request_hash = request_hash_target(
        &mut builder,
        client_message_count,
        pi_hash_target(&proof.public_inputs, PI_USER_ROOT),
    );
    let rewritten_request_hash = request_hash_target(
        &mut builder,
        rewritten_message_count,
        pi_hash_target(&proof.public_inputs, PI_MOD_ROOT),
    );
    let request_randomness = builder.add_virtual_hash();
    let request_randomness_binding =
        request_randomness_binding_target(&mut builder, request_randomness);
    builder.connect_hashes(
        request_randomness_binding,
        pi_hash_target(&proof.public_inputs, PI_REQUEST_RANDOMNESS_BINDING),
    );
    let request_commitment =
        commit_digest_target(&mut builder, rewritten_request_hash, request_randomness);

    builder.register_public_inputs(&proof.public_inputs[PI_MSG_COMMITMENT..PI_MSG_COMMITMENT + 4]);
    builder
        .register_public_inputs(&proof.public_inputs[PI_FRAG_COMMITMENT..PI_FRAG_COMMITMENT + 4]);
    builder.register_public_input(client_message_count);
    builder.register_public_inputs(&client_request_hash.elements);
    builder.register_public_input(rewritten_message_count);
    builder.register_public_inputs(&request_commitment.elements);

    (
        builder.build::<C>(),
        RewriteFinalTargets {
            proof,
            verifier_data,
            request_randomness,
        },
    )
}

fn prove_rewrite_final(
    data: &CircuitData<F, C, D>,
    targets: &RewriteFinalTargets,
    inner_verifier_only: &plonky2::plonk::circuit_data::VerifierOnlyCircuitData<C, D>,
    inner_proof: &ProofWithPublicInputs<F, C, D>,
    request_randomness: Digest,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
{
    let mut pw = PartialWitness::<F>::new();
    pw.set_verifier_data_target(&targets.verifier_data, inner_verifier_only)?;
    pw.set_proof_with_pis_target(&targets.proof, inner_proof)?;
    pw.set_hash_target(
        targets.request_randomness,
        digest_to_hashout(request_randomness),
    )?;
    Ok(data.prove(pw)?)
}

fn rounds_needed(mut n: usize, fan_in: usize) -> usize {
    let mut rounds = 0usize;
    while n > 1 {
        n = (n + fan_in - 1) / fan_in;
        rounds += 1;
    }
    rounds
}

impl RewriteRecursionCache {
    /// Build the recursion cache once. Aggregation levels are built lazily as needed.
    pub fn new(
        fan_in: usize,
        leaf_data: Arc<CircuitData<F, C, D>>,
    ) -> Result<(Self, RewriteAggStats)>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        anyhow::ensure!(fan_in >= 2, "fan_in must be >= 2");

        let zero_knowledge = leaf_data.common.config.zero_knowledge;
        let cache = Self {
            fan_in,
            zero_knowledge,
            leaf_data,
            agg_levels: Vec::new(),
            finalizers: Vec::new(),
            root_aggs: Vec::new(),
        };
        Ok((
            cache,
            RewriteAggStats {
                agg_setup_s: 0.0,
                agg_prove_s: 0.0,
            },
        ))
    }

    fn ensure_agg_levels(&mut self, needed: usize) -> Result<f64>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        let mut setup_s = 0f64;
        while self.agg_levels.len() < needed {
            let inner_data = if self.agg_levels.is_empty() {
                self.leaf_data.clone()
            } else {
                self.agg_levels.last().unwrap().0.clone()
            };
            let t = std::time::Instant::now();
            let (agg_data, agg_t) = build_rewrite_agg_circuit(
                self.fan_in,
                &inner_data.common,
                &inner_data.verifier_only,
                self.zero_knowledge,
                false,
            );
            setup_s += t.elapsed().as_secs_f64();
            self.agg_levels.push((Arc::new(agg_data), agg_t));
        }
        Ok(setup_s)
    }

    fn inner_data_for_rounds(&self, rounds: usize) -> Arc<CircuitData<F, C, D>> {
        if rounds == 0 {
            self.leaf_data.clone()
        } else {
            self.agg_levels[rounds - 1].0.clone()
        }
    }

    fn ensure_finalizer(&mut self, rounds: usize) -> Result<f64>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        while self.finalizers.len() <= rounds {
            self.finalizers.push(None);
        }
        if self.finalizers[rounds].is_some() {
            return Ok(0.0);
        }
        let inner_data = self.inner_data_for_rounds(rounds);
        let started = std::time::Instant::now();
        let (data, targets) = build_rewrite_final_circuit(
            &inner_data.common,
            &inner_data.verifier_only,
            self.zero_knowledge,
        );
        let elapsed = started.elapsed().as_secs_f64();
        self.finalizers[rounds] = Some((Arc::new(data), targets));
        Ok(elapsed)
    }

    fn ensure_root_statement(&mut self, rounds: usize) -> Result<f64>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        if rounds == 0 {
            return self.ensure_finalizer(0);
        }
        self.ensure_agg_levels(rounds - 1)?;
        while self.root_aggs.len() <= rounds {
            self.root_aggs.push(None);
        }
        if self.root_aggs[rounds].is_some() {
            return Ok(0.0);
        }
        let inner_data = self.inner_data_for_rounds(rounds - 1);
        let started = std::time::Instant::now();
        let (data, targets) = build_rewrite_agg_circuit(
            self.fan_in,
            &inner_data.common,
            &inner_data.verifier_only,
            self.zero_knowledge,
            true,
        );
        let elapsed = started.elapsed().as_secs_f64();
        self.root_aggs[rounds] = Some((Arc::new(data), targets));
        Ok(elapsed)
    }

    pub fn root_data_for_leaf_count(
        &mut self,
        leaf_count: usize,
    ) -> Result<Arc<CircuitData<F, C, D>>>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        anyhow::ensure!(leaf_count > 0, "leaf_count must be positive");
        let levels_needed = rounds_needed(leaf_count, self.fan_in);
        self.ensure_agg_levels(levels_needed.saturating_sub(1))?;
        self.ensure_root_statement(levels_needed)?;
        if levels_needed == 0 {
            Ok(self.finalizers[0].as_ref().unwrap().0.clone())
        } else {
            Ok(self.root_aggs[levels_needed].as_ref().unwrap().0.clone())
        }
    }

    /// Aggregate leaves, optionally recording proof sizes for each recursion level.
    pub fn aggregate_with_data(
        &mut self,
        leaf_proofs: &[ProofWithPublicInputs<F, C, D>],
        request_randomness: Digest,
        collect_level_sizes: bool,
    ) -> Result<(
        ProofWithPublicInputs<F, C, D>,
        Arc<CircuitData<F, C, D>>,
        RewriteAggStats,
        Vec<LevelSizeInfo>,
    )>
    where
        <C as GenericConfig<D>>::Hasher: AlgebraicHasher<F>,
    {
        anyhow::ensure!(!leaf_proofs.is_empty(), "leaf_proofs must be non-empty");

        let mut infos: Vec<LevelSizeInfo> = Vec::new();

        if collect_level_sizes {
            infos.push(LevelSizeInfo {
                level: "leaf".to_string(),
                degree_bits: self.leaf_data.common.degree_bits(),
                proof_bytes_avg: leaf_proofs
                    .iter()
                    .map(|p| p.to_bytes().len())
                    .sum::<usize>()
                    / leaf_proofs.len(),
                n_proofs: leaf_proofs.len(),
            });
        }

        let mut proofs = leaf_proofs.to_vec();

        let levels_needed = rounds_needed(proofs.len(), self.fan_in);
        let mut agg_setup_s = self.ensure_agg_levels(levels_needed.saturating_sub(1))?;
        agg_setup_s += self.ensure_root_statement(levels_needed)?;

        let t_agg_prove = std::time::Instant::now();
        let mut level = 0usize;
        while proofs.len() > self.fan_in {
            let inner_verifier_only = if level == 0 {
                self.leaf_data.verifier_only.clone()
            } else {
                self.agg_levels[level - 1].0.verifier_only.clone()
            };
            let (agg_data, agg_t) = &self.agg_levels[level];

            let next: Vec<ProofWithPublicInputs<F, C, D>> = proofs
                .par_chunks(self.fan_in)
                .map(|chunk| {
                    let mut child = chunk.to_vec();
                    let mut enabled = vec![true; chunk.len()];
                    if child.len() < self.fan_in {
                        let last = child.last().cloned().unwrap();
                        child.resize(self.fan_in, last);
                        enabled.resize(self.fan_in, false);
                    }
                    prove_rewrite_agg(
                        agg_data.as_ref(),
                        agg_t,
                        &inner_verifier_only,
                        &child,
                        &enabled,
                        None,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            if collect_level_sizes {
                infos.push(LevelSizeInfo {
                    level: format!("agg_{level}"),
                    degree_bits: agg_data.common.degree_bits(),
                    proof_bytes_avg: next.iter().map(|p| p.to_bytes().len()).sum::<usize>()
                        / next.len(),
                    n_proofs: next.len(),
                });
            }

            proofs = next;
            level += 1;
        }
        let transition_data = self.inner_data_for_rounds(level);
        let (root_proof, root_data) = if levels_needed == 0 {
            let transition_proof = proofs.pop().expect("non-empty");
            let (root_data, final_targets) = self.finalizers[0].as_ref().unwrap();
            (
                prove_rewrite_final(
                    root_data,
                    final_targets,
                    &transition_data.verifier_only,
                    &transition_proof,
                    request_randomness,
                )?,
                root_data.clone(),
            )
        } else {
            let mut enabled = vec![true; proofs.len()];
            let last = proofs.last().cloned().expect("non-empty");
            proofs.resize(self.fan_in, last);
            enabled.resize(self.fan_in, false);
            let (root_data, root_targets) = self.root_aggs[levels_needed].as_ref().unwrap();
            (
                prove_rewrite_agg(
                    root_data,
                    root_targets,
                    &transition_data.verifier_only,
                    &proofs,
                    &enabled,
                    Some(request_randomness),
                )?,
                root_data.clone(),
            )
        };
        let agg_prove_s = t_agg_prove.elapsed().as_secs_f64();
        if collect_level_sizes {
            infos.push(LevelSizeInfo {
                level: "root_statement".to_string(),
                degree_bits: root_data.common.degree_bits(),
                proof_bytes_avg: root_proof.to_bytes().len(),
                n_proofs: 1,
            });
        }

        Ok((
            root_proof,
            root_data,
            RewriteAggStats {
                agg_setup_s,
                agg_prove_s,
            },
            infos,
        ))
    }
}
