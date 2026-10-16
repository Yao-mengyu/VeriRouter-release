use plonky2::hash::hash_types::HashOutTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitData, VerifierCircuitTarget};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};

use crate::commitment::{commit_digest_target, digest_to_hashout};
use crate::util::{recursion_config, C, D, F};
use crate::witness::CommitmentRandomness;

#[derive(Clone)]
pub struct FinalAggTargets {
    pub reference_commitment_pi: HashOutTarget,
    pub q_hash_pi: HashOutTarget,
    pub graph_commitment_pi: HashOutTarget,
    pub output_id_pi: Target,
    pub entry_pi: Target,
    pub entry_layer_pi: Target,
    pub ef_search_pi: Target,
    pub top_k_pi: Target,
    pub reference_root: HashOutTarget,
    pub graph_root: HashOutTarget,
    pub reference_randomness: HashOutTarget,
    pub graph_randomness: HashOutTarget,
    pub score_table_hash: HashOutTarget,
    pub nbr_commit: HashOutTarget,
    pub upper_len_pi: Target,
    pub rounds_len_pi: Target,
    pub dist_proof_t: ProofWithPublicInputsTarget<D>,
    pub dist_vd_t: VerifierCircuitTarget,
    pub upper_proof_t: ProofWithPublicInputsTarget<D>,
    pub upper_vd_t: VerifierCircuitTarget,
    pub ef_proof_t: ProofWithPublicInputsTarget<D>,
    pub ef_vd_t: VerifierCircuitTarget,
}

fn hash_from_slice4(slice: &[Target]) -> HashOutTarget {
    HashOutTarget {
        elements: [slice[0], slice[1], slice[2], slice[3]],
    }
}

fn connect_hashes(builder: &mut CircuitBuilder<F, D>, a: &HashOutTarget, b: &HashOutTarget) {
    for i in 0..4 {
        builder.connect(a.elements[i], b.elements[i]);
    }
}

fn connect_verifier_data(
    builder: &mut CircuitBuilder<F, D>,
    a: &VerifierCircuitTarget,
    b: &VerifierCircuitTarget,
) {
    connect_hashes(builder, &a.circuit_digest, &b.circuit_digest);
    assert_eq!(
        a.constants_sigmas_cap.0.len(),
        b.constants_sigmas_cap.0.len()
    );
    for (ha, hb) in a
        .constants_sigmas_cap
        .0
        .iter()
        .zip(b.constants_sigmas_cap.0.iter())
    {
        connect_hashes(builder, ha, hb);
    }
}

pub fn build_final_agg_circuit(
    dist_data: &CircuitData<F, C, D>,
    upper_data: &CircuitData<F, C, D>,
    ef_data: &CircuitData<F, C, D>,
) -> (CircuitData<F, C, D>, FinalAggTargets) {
    let config = recursion_config(dist_data.common.config.zero_knowledge);
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let reference_commitment_pi = builder.add_virtual_hash_public_input();
    let q_hash_pi = builder.add_virtual_hash_public_input();
    let graph_commitment_pi = builder.add_virtual_hash_public_input();
    let output_id_pi = builder.add_virtual_public_input();
    let entry_pi = builder.add_virtual_public_input();
    let entry_layer_pi = builder.add_virtual_public_input();
    let ef_search_pi = builder.add_virtual_public_input();
    let top_k_pi = builder.add_virtual_public_input();
    let reference_root = builder.add_virtual_hash();
    let graph_root = builder.add_virtual_hash();
    let reference_randomness = builder.add_virtual_hash();
    let graph_randomness = builder.add_virtual_hash();
    let score_table_hash = builder.add_virtual_hash();
    let nbr_commit = builder.add_virtual_hash();
    let upper_len_pi = builder.add_virtual_target();
    let rounds_len_pi = builder.add_virtual_target();

    let reference_commitment_calc =
        commit_digest_target(&mut builder, reference_root, reference_randomness);
    connect_hashes(
        &mut builder,
        &reference_commitment_pi,
        &reference_commitment_calc,
    );
    let graph_commitment_calc = commit_digest_target(&mut builder, graph_root, graph_randomness);
    connect_hashes(&mut builder, &graph_commitment_pi, &graph_commitment_calc);

    let dist_vd_const = builder.constant_verifier_data(&dist_data.verifier_only);
    let dist_proof_t = builder.add_virtual_proof_with_pis(&dist_data.common);
    let dist_vd_t =
        builder.add_virtual_verifier_data(dist_data.common.config.fri_config.cap_height);
    connect_verifier_data(&mut builder, &dist_vd_t, &dist_vd_const);
    builder.verify_proof::<C>(&dist_proof_t, &dist_vd_t, &dist_data.common);

    let upper_vd_const = builder.constant_verifier_data(&upper_data.verifier_only);
    let upper_proof_t = builder.add_virtual_proof_with_pis(&upper_data.common);
    let upper_vd_t =
        builder.add_virtual_verifier_data(upper_data.common.config.fri_config.cap_height);
    connect_verifier_data(&mut builder, &upper_vd_t, &upper_vd_const);
    builder.verify_proof::<C>(&upper_proof_t, &upper_vd_t, &upper_data.common);

    let ef_vd_const = builder.constant_verifier_data(&ef_data.verifier_only);
    let ef_proof_t = builder.add_virtual_proof_with_pis(&ef_data.common);
    let ef_vd_t = builder.add_virtual_verifier_data(ef_data.common.config.fri_config.cap_height);
    connect_verifier_data(&mut builder, &ef_vd_t, &ef_vd_const);
    builder.verify_proof::<C>(&ef_proof_t, &ef_vd_t, &ef_data.common);

    let dist_pi = &dist_proof_t.public_inputs;
    let upper_pi = &upper_proof_t.public_inputs;
    let ef_pi = &ef_proof_t.public_inputs;

    connect_hashes(
        &mut builder,
        &reference_root,
        &hash_from_slice4(&dist_pi[0..4]),
    );
    connect_hashes(&mut builder, &q_hash_pi, &hash_from_slice4(&dist_pi[4..8]));
    connect_hashes(
        &mut builder,
        &score_table_hash,
        &hash_from_slice4(&dist_pi[8..12]),
    );

    connect_hashes(
        &mut builder,
        &score_table_hash,
        &hash_from_slice4(&upper_pi[0..4]),
    );
    connect_hashes(
        &mut builder,
        &graph_root,
        &hash_from_slice4(&upper_pi[4..8]),
    );
    builder.connect(upper_len_pi, upper_pi[10]);

    connect_hashes(
        &mut builder,
        &score_table_hash,
        &hash_from_slice4(&ef_pi[0..4]),
    );
    connect_hashes(&mut builder, &graph_root, &hash_from_slice4(&ef_pi[4..8]));
    connect_hashes(&mut builder, &nbr_commit, &hash_from_slice4(&ef_pi[8..12]));
    builder.connect(output_id_pi, ef_pi[12]);
    builder.connect(rounds_len_pi, ef_pi[15]);
    builder.connect(upper_len_pi, ef_pi[14]);

    builder.connect(entry_pi, upper_pi[8]);
    builder.connect(entry_layer_pi, upper_pi[11]);
    builder.connect(upper_pi[9], ef_pi[13]);
    builder.connect(ef_search_pi, ef_pi[16]);
    builder.connect(top_k_pi, ef_pi[17]);

    let data = builder.build::<C>();
    let targets = FinalAggTargets {
        reference_commitment_pi,
        q_hash_pi,
        graph_commitment_pi,
        output_id_pi,
        entry_pi,
        entry_layer_pi,
        ef_search_pi,
        top_k_pi,
        reference_root,
        graph_root,
        reference_randomness,
        graph_randomness,
        score_table_hash,
        nbr_commit,
        upper_len_pi,
        rounds_len_pi,
        dist_proof_t,
        dist_vd_t,
        upper_proof_t,
        upper_vd_t,
        ef_proof_t,
        ef_vd_t,
    };
    (data, targets)
}

#[allow(clippy::too_many_arguments)]
pub fn fill_final_agg_witness(
    pw: &mut PartialWitness<F>,
    targets: &FinalAggTargets,
    dist_proof: &ProofWithPublicInputs<F, C, D>,
    upper_proof: &ProofWithPublicInputs<F, C, D>,
    ef_proof: &ProofWithPublicInputs<F, C, D>,
    dist_vd: &plonky2::plonk::circuit_data::VerifierOnlyCircuitData<C, D>,
    upper_vd: &plonky2::plonk::circuit_data::VerifierOnlyCircuitData<C, D>,
    ef_vd: &plonky2::plonk::circuit_data::VerifierOnlyCircuitData<C, D>,
    commitment_randomness: &CommitmentRandomness,
) {
    let _ = pw.set_proof_with_pis_target(&targets.dist_proof_t, dist_proof);
    let _ = pw.set_verifier_data_target(&targets.dist_vd_t, dist_vd);
    let _ = pw.set_proof_with_pis_target(&targets.upper_proof_t, upper_proof);
    let _ = pw.set_verifier_data_target(&targets.upper_vd_t, upper_vd);
    let _ = pw.set_proof_with_pis_target(&targets.ef_proof_t, ef_proof);
    let _ = pw.set_verifier_data_target(&targets.ef_vd_t, ef_vd);
    pw.set_hash_target(
        targets.reference_randomness,
        digest_to_hashout(commitment_randomness.reference),
    )
    .expect("set private reference commitment randomness");
    pw.set_hash_target(
        targets.graph_randomness,
        digest_to_hashout(commitment_randomness.graph),
    )
    .expect("set private graph commitment randomness");
}
