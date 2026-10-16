//! Generate rewrite proofs, serialize reports, and verify pinned public statements.
use crate::commitment;
use anyhow::Result;

use plonky2::field::types::PrimeField64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::circuits::aggregate::{
    FINAL_PI_CLIENT_MESSAGE_COUNT, FINAL_PI_CLIENT_REQUEST_HASH, FINAL_PI_FRAGMENT_BANK_COMMITMENT,
    FINAL_PI_MESSAGE_BANK_COMMITMENT, FINAL_PI_REQUEST_COMMITMENT,
    FINAL_PI_REWRITTEN_MESSAGE_COUNT, FINAL_PUBLIC_INPUTS,
};
use crate::circuits::batch::{
    build_rewrite_tree_batch_circuit_with_zk, build_rewrite_tree_batch_public_orig_circuit_with_zk,
    prove_rewrite_tree_batch, RewriteParams, BATCH_PUBLIC_INPUTS, PI_END_IDX, PI_FRAG_COMMITMENT,
    PI_LAST_PRESENT, PI_MOD_ACC_IN, PI_MOD_ROOT, PI_MSG_COMMITMENT, PI_PREV_PRESENT,
    PI_REQUEST_RANDOMNESS_BINDING, PI_REWRITTEN_COUNT_IN, PI_REWRITTEN_COUNT_OUT, PI_START_IDX,
    PI_USER_ACC_IN, PI_USER_ROOT,
};
use crate::util::{Digest, C, D, F};
use crate::witness;

use crate::witness::InputParams;
const REWRITE_REPORT_VERSION: &str =
    "promptmod_report_v4_request_commitment_count_bound_salted_banks";
const REWRITE_VERIFY_POLICY_VERSION: &str = "rewrite_verification_policy_v1";

fn default_fan_in() -> usize {
    2
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct RewritePublicInputsReport {
    message_bank_commitment: Digest,
    fragment_bank_commitment: Digest,
    request_randomness_binding: Digest,
    #[serde(default)]
    start_idx: u64,
    #[serde(default)]
    end_idx: u64,
    rewritten_count_in: u64,
    rewritten_count_out: u64,
    #[serde(default)]
    prev_present: u64,
    #[serde(default)]
    last_present: u64,
    #[serde(default)]
    user_acc_in: Digest,
    #[serde(alias = "client_request_hash", alias = "user_prompt_hash")]
    user_root: Digest,
    #[serde(default)]
    mod_acc_in: Digest,
    #[serde(alias = "rewritten_request_hash", alias = "executed_prompt_hash")]
    mod_root: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RewriteBatchReport {
    batch_index: usize,
    total_public_inputs: usize,
    leaf_proof_size_bytes: usize,
    public_inputs: RewritePublicInputsReport,
    #[serde(default)]
    native_last_present: bool,
    #[serde(default)]
    native_user_acc_in: Digest,
    #[serde(default)]
    native_user_root: Digest,
    #[serde(default)]
    native_mod_acc_in: Digest,
    #[serde(default)]
    native_mod_root: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RewriteAggStatsReport {
    leaf_setup_time_sec: f64,
    agg_setup_time_sec: f64,
    agg_prove_time_sec: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RewriteRootReport {
    total_public_inputs: usize,
    proof_size_bytes: usize,
    proof_path: Option<String>,
    verify_time_sec: f64,
    public_inputs: RewriteRootPublicInputsReport,
    agg_stats: RewriteAggStatsReport,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct RewriteRootPublicInputsReport {
    message_bank_commitment: Digest,
    fragment_bank_commitment: Digest,
    client_message_count: u64,
    client_request_hash: Digest,
    rewritten_message_count: u64,
    request_commitment: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RewriteLevelSizeReport {
    level: String,
    degree_bits: usize,
    proof_bytes_avg: usize,
    n_proofs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RewriteTotalsReport {
    batches: usize,
    #[serde(default)]
    leaf_merkle_prep_time_sec: f64,
    #[serde(default)]
    leaf_setup_time_sec: f64,
    #[serde(default)]
    leaf_witness_build_time_sec: f64,
    #[serde(default)]
    leaf_prove_wall_time_sec: f64,
    #[serde(default)]
    agg_prove_wall_time_sec: Option<f64>,
    #[serde(default)]
    total_prove_wall_time_sec: f64,
    #[serde(default)]
    leaf_verify_wall_time_sec: f64,
    #[serde(default)]
    leaf_proof_bytes_avg: usize,
    #[serde(default)]
    leaf_proof_bytes_total: usize,
    #[serde(default)]
    root_only_proof_size_bytes: Option<usize>,
    message_bank_commitment: Digest,
    fragment_bank_commitment: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewriteRunReport {
    version: String,
    input_path: Option<String>,
    aggregated: bool,
    public_orig: bool,
    zero_knowledge: bool,
    #[serde(default)]
    dedup_merkle: bool,
    strict_commit: bool,
    #[serde(default = "default_fan_in")]
    fan_in: usize,
    params: RewriteParams,
    batch_count: usize,
    message_bank_commitment: Digest,
    fragment_bank_commitment: Digest,
    batch_reports: Vec<RewriteBatchReport>,
    root_report: Option<RewriteRootReport>,
    recursion_levels: Option<Vec<RewriteLevelSizeReport>>,
    totals: RewriteTotalsReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct RewriteClientVerifyReport {
    report_path: String,
    proof_path: String,
    verify_policy_path: String,
    verified: bool,
    proof_bytes: usize,
    public_inputs: RewriteRootPublicInputsReport,
    verifier_data_hex: String,
    zero_knowledge: bool,
    recursion_depth: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RewriteVerifyPolicyFile {
    version: String,
    public_inputs: RewriteRootPublicInputsReport,
}

fn digest_from_public_inputs(pi: &[F], start: usize) -> Digest {
    std::array::from_fn(|offset| pi[start + offset].to_canonical_u64())
}

fn public_inputs_report(pi: &[F]) -> RewritePublicInputsReport {
    assert!(pi.len() >= BATCH_PUBLIC_INPUTS);
    RewritePublicInputsReport {
        message_bank_commitment: digest_from_public_inputs(pi, PI_MSG_COMMITMENT),
        fragment_bank_commitment: digest_from_public_inputs(pi, PI_FRAG_COMMITMENT),
        request_randomness_binding: digest_from_public_inputs(pi, PI_REQUEST_RANDOMNESS_BINDING),
        start_idx: pi[PI_START_IDX].to_canonical_u64(),
        end_idx: pi[PI_END_IDX].to_canonical_u64(),
        rewritten_count_in: pi[PI_REWRITTEN_COUNT_IN].to_canonical_u64(),
        rewritten_count_out: pi[PI_REWRITTEN_COUNT_OUT].to_canonical_u64(),
        prev_present: pi[PI_PREV_PRESENT].to_canonical_u64(),
        last_present: pi[PI_LAST_PRESENT].to_canonical_u64(),
        user_acc_in: digest_from_public_inputs(pi, PI_USER_ACC_IN),
        user_root: digest_from_public_inputs(pi, PI_USER_ROOT),
        mod_acc_in: digest_from_public_inputs(pi, PI_MOD_ACC_IN),
        mod_root: digest_from_public_inputs(pi, PI_MOD_ROOT),
    }
}

fn root_public_inputs_report(pi: &[F]) -> RewriteRootPublicInputsReport {
    assert_eq!(pi.len(), FINAL_PUBLIC_INPUTS);
    RewriteRootPublicInputsReport {
        message_bank_commitment: digest_from_public_inputs(pi, FINAL_PI_MESSAGE_BANK_COMMITMENT),
        fragment_bank_commitment: digest_from_public_inputs(pi, FINAL_PI_FRAGMENT_BANK_COMMITMENT),
        client_message_count: pi[FINAL_PI_CLIENT_MESSAGE_COUNT].to_canonical_u64(),
        client_request_hash: digest_from_public_inputs(pi, FINAL_PI_CLIENT_REQUEST_HASH),
        rewritten_message_count: pi[FINAL_PI_REWRITTEN_MESSAGE_COUNT].to_canonical_u64(),
        request_commitment: digest_from_public_inputs(pi, FINAL_PI_REQUEST_COMMITMENT),
    }
}

fn ensure_expected_statement(
    actual: &RewriteRootPublicInputsReport,
    expected: &RewriteRootPublicInputsReport,
) -> Result<()> {
    anyhow::ensure!(
        actual == expected,
        "root proof public inputs do not match caller-pinned verify policy"
    );
    Ok(())
}

fn default_root_proof_path(report_json_path: &str) -> std::path::PathBuf {
    let report_path = std::path::Path::new(report_json_path);
    let parent = report_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = report_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rewrite_report");
    parent.join(format!("{stem}.root.proof"))
}

fn resolve_report_relative(report_path: &std::path::Path, value: &str) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        report_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(path)
    }
}

/// Verify a minimal public root artifact; diagnostic leaf reports stay private.
fn verify_public_rewrite_report(
    report_json_path: &str,
    proof_override: Option<&str>,
    verify_policy_path: &str,
    verifier_path: &str,
) -> Result<RewriteClientVerifyReport> {
    let report_path = std::path::Path::new(report_json_path);
    let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(report_path)?)?;
    anyhow::ensure!(
        report["version"].as_str() == Some(REWRITE_REPORT_VERSION),
        "unsupported public report version"
    );
    anyhow::ensure!(
        report["aggregated"].as_bool() == Some(true),
        "root proof required"
    );
    let root = report
        .get("root_report")
        .ok_or_else(|| anyhow::anyhow!("missing root_report"))?;
    let reported_inputs: RewriteRootPublicInputsReport = serde_json::from_value(
        root.get("public_inputs")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing public_inputs"))?,
    )?;
    let proof_path = if let Some(path) = proof_override {
        std::path::PathBuf::from(path)
    } else {
        resolve_report_relative(
            report_path,
            root["proof_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing proof_path"))?,
        )
    };
    let policy: RewriteVerifyPolicyFile =
        serde_json::from_str(&std::fs::read_to_string(verify_policy_path)?)?;
    anyhow::ensure!(
        policy.version == REWRITE_VERIFY_POLICY_VERSION,
        "unsupported policy version"
    );
    let expected_statement = policy.public_inputs;
    let vk_bytes = std::fs::read(verifier_path)?;
    let data = plonky2::plonk::circuit_data::VerifierCircuitData::<F, C, D>::from_bytes(
        vk_bytes.clone(),
        &plonky2::util::serialization::DefaultGateSerializer,
    )
    .map_err(|e| anyhow::anyhow!("invalid verifier data: {e:?}"))?;
    anyhow::ensure!(
        data.common.config.zero_knowledge,
        "approved key must enable ZK"
    );
    let bytes = std::fs::read(&proof_path)?;
    let proof = plonky2::plonk::proof::ProofWithPublicInputs::<F, C, D>::from_bytes(
        bytes.clone(),
        &data.common,
    )?;
    data.verify(proof.clone())?;
    anyhow::ensure!(
        proof.public_inputs.len() == FINAL_PUBLIC_INPUTS,
        "wrong root statement length"
    );
    let public_inputs = root_public_inputs_report(&proof.public_inputs);
    ensure_expected_statement(&public_inputs, &expected_statement)?;
    anyhow::ensure!(
        public_inputs == reported_inputs,
        "public inputs do not match report"
    );
    Ok(RewriteClientVerifyReport {
        report_path: report_path.display().to_string(),
        proof_path: proof_path.display().to_string(),
        verify_policy_path: verify_policy_path.to_string(),
        verified: true,
        proof_bytes: bytes.len(),
        public_inputs,
        verifier_data_hex: vk_bytes.iter().map(|b| format!("{b:02x}")).collect(),
        zero_knowledge: true,
        recursion_depth: 0,
    })
}

pub fn verify_rewrite_report(
    report_json_path: &str,
    proof_override: Option<&str>,
    verify_policy_path: &str,
    verifier_path: Option<&str>,
) -> Result<RewriteClientVerifyReport> {
    if let Some(path) = verifier_path {
        return verify_public_rewrite_report(
            report_json_path,
            proof_override,
            verify_policy_path,
            path,
        );
    }
    let report_path = std::path::Path::new(report_json_path);
    let payload = std::fs::read_to_string(report_path)?;
    let report: RewriteRunReport = serde_json::from_str(&payload)?;
    anyhow::ensure!(
        report.version == REWRITE_REPORT_VERSION,
        "unsupported rewrite report version: {}",
        report.version
    );
    anyhow::ensure!(
        report.aggregated,
        "client verification requires an aggregated rewrite report; rerun without --no-agg"
    );
    let root_report = report
        .root_report
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("aggregated report is missing root_report"))?;
    let proof_path = if let Some(value) = proof_override {
        std::path::PathBuf::from(value)
    } else {
        let value = root_report
            .proof_path
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("root_report is missing proof_path"))?;
        resolve_report_relative(report_path, value)
    };
    let policy_payload = std::fs::read_to_string(verify_policy_path)?;
    let policy = serde_json::from_str::<RewriteVerifyPolicyFile>(&policy_payload)?;
    anyhow::ensure!(
        policy.version == REWRITE_VERIFY_POLICY_VERSION,
        "unsupported rewrite verification policy version: {}",
        policy.version
    );
    let expected_statement = policy.public_inputs;
    let params = &report.params;
    let (leaf_data, _, _) = if report.public_orig {
        build_rewrite_tree_batch_public_orig_circuit_with_zk(params, report.zero_knowledge)
    } else {
        build_rewrite_tree_batch_circuit_with_zk(params, report.zero_knowledge)
    };
    let leaf_data = Arc::new(leaf_data);
    let mut cache =
        crate::circuits::aggregate::RewriteRecursionCache::new(report.fan_in, leaf_data)?.0;
    let root_data = cache.root_data_for_leaf_count(report.batch_count)?;
    let proof_bytes = std::fs::read(&proof_path)?;
    let proof = plonky2::plonk::proof::ProofWithPublicInputs::<F, C, D>::from_bytes(
        proof_bytes.clone(),
        &root_data.common,
    )?;
    root_data.verify(proof.clone())?;
    let public_inputs = root_public_inputs_report(&proof.public_inputs);
    anyhow::ensure!(
        public_inputs == root_report.public_inputs,
        "root proof public inputs do not match report"
    );
    ensure_expected_statement(&public_inputs, &expected_statement)?;
    Ok(RewriteClientVerifyReport {
        report_path: report_path.display().to_string(),
        proof_path: proof_path.display().to_string(),
        verify_policy_path: verify_policy_path.to_string(),
        verified: true,
        proof_bytes: proof_bytes.len(),
        public_inputs,
        verifier_data_hex: root_data
            .verifier_data()
            .to_bytes(&plonky2::util::serialization::DefaultGateSerializer)
            .map_err(|e| anyhow::anyhow!("verifier serialization failed: {e:?}"))?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        zero_knowledge: report.zero_knowledge,
        recursion_depth: {
            let mut n = report.batch_count;
            let mut depth = 1;
            while n > report.fan_in {
                n = n.div_ceil(report.fan_in);
                depth += 1;
            }
            depth
        },
    })
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub public_orig: bool,
    pub print_public_orig: bool,
    pub strict_commit: bool,
    pub do_agg: bool,
    pub fan_in: usize,
    pub verify_leaves: bool,
    pub dedup_merkle: bool,
    pub zero_knowledge: bool,
    pub save_proof: bool,
    pub print_level_sizes: bool,
}

pub fn run_input(
    input_path: Option<&str>,
    defaults: InputParams,
    options: RunOptions,
    report_json_path: Option<&str>,
) -> Result<RewriteRunReport> {
    let RunOptions {
        public_orig,
        print_public_orig,
        strict_commit,
        do_agg,
        fan_in,
        verify_leaves,
        dedup_merkle,
        zero_knowledge,
        save_proof,
        print_level_sizes,
    } = options;
    let input_path_for_report = input_path.map(str::to_string);
    let (mut params, commit, messages) = witness::load_input(input_path, defaults)?;
    let t_leaf_merkle_prep = Instant::now();
    let banks = crate::commit_check::check_commit_data(&mut params, commit.as_ref(), &messages)?;
    let request_randomness = banks.request_randomness;
    let message_bank_commitment =
        commitment::commit_digest(banks.message.root, banks.msg_randomness);
    let fragment_bank_commitment =
        commitment::commit_digest(banks.fragment.root, banks.frag_randomness);
    let leaf_merkle_prep_s = t_leaf_merkle_prep.elapsed().as_secs_f64();
    eprintln!("[VeriRouteRewrite] setup rewrite circuits");
    let t_leaf_setup = Instant::now();
    let (data, _, wit_t) = if public_orig {
        build_rewrite_tree_batch_public_orig_circuit_with_zk(&params, zero_knowledge)
    } else {
        build_rewrite_tree_batch_circuit_with_zk(&params, zero_knowledge)
    };
    let leaf_setup_s = t_leaf_setup.elapsed().as_secs_f64();
    let data = Arc::new(data);
    let t_leaf_witness_build = Instant::now();
    let witnesses = witness::prepare_batches(
        &params,
        commit.as_ref(),
        &messages,
        &banks,
        strict_commit,
        dedup_merkle,
    )?;
    let n_batches = witnesses.len();
    let expected_message_count = messages.iter().filter(|message| message.present).count() as u64;
    let expected_rewritten_message_count = messages
        .iter()
        .map(|message| message.gap_before.len() as u64 + u64::from(message.present))
        .sum::<u64>();
    let last = &witnesses
        .last()
        .expect("input includes a terminal row")
        .witness;
    anyhow::ensure!(
        last.end_idx == expected_message_count,
        "message interval does not cover the input"
    );
    anyhow::ensure!(
        last.rewritten_count_out == expected_rewritten_message_count,
        "rewritten count does not cover the input"
    );
    let leaf_witness_build_s = t_leaf_witness_build.elapsed().as_secs_f64();
    // Prove all leaf proofs in parallel.
    eprintln!("[VeriRouteRewrite] prove rewrite leaves in parallel");
    let t_leaf_prove = Instant::now();
    let leaf_proofs: Vec<plonky2::plonk::proof::ProofWithPublicInputs<F, C, D>> = witnesses
        .par_iter()
        .map(|w| {
            let p = prove_rewrite_tree_batch(data.as_ref(), &wit_t, &w.witness)?;
            Ok(p)
        })
        .collect::<Result<Vec<_>>>()?;
    let leaf_prove_wall_s = t_leaf_prove.elapsed().as_secs_f64();
    eprintln!("[VeriRouteRewrite] rewrite leaf proofs done");
    let leaf_sizes: Vec<usize> = leaf_proofs.iter().map(|p| p.to_bytes().len()).collect();
    let leaf_bytes_total: usize = leaf_sizes.iter().sum();
    let leaf_bytes_avg: usize = if leaf_sizes.is_empty() {
        0
    } else {
        leaf_bytes_total / leaf_sizes.len()
    };

    let should_verify_leaves = verify_leaves || !do_agg;
    let leaf_verify_wall_s: f64 = if should_verify_leaves {
        let t = Instant::now();
        leaf_proofs
            .par_iter()
            .map(|p| {
                data.verify(p.clone())?;
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
        t.elapsed().as_secs_f64()
    } else {
        0.0
    };

    let mut batch_reports = Vec::with_capacity(leaf_proofs.len());

    // Print per-batch info (in order). In `--agg` mode we intentionally do NOT
    // print per-leaf timings/sizes (root proof is what matters).
    for (b, proof) in leaf_proofs.iter().enumerate() {
        // Public inputs:
        // Transition PIs expose only salted bank commitments and an internal
        // request-randomness handle; raw bank roots remain private witnesses.
        let pi = &proof.public_inputs;
        let pi_report = public_inputs_report(pi);
        println!("batch {}/{}: {} public inputs", b + 1, n_batches, pi.len());
        if !do_agg {
            println!("  leaf_proof_size={} bytes", leaf_sizes[b]);
        }

        if public_orig {
            println!("  (public-orig enabled) total_public_inputs={}", pi.len());
            if print_public_orig {
                let base = BATCH_PUBLIC_INPUTS;
                let per_msg = 3usize + 3usize * params.max_orig_chunks;
                anyhow::ensure!(
                    pi.len() >= base + params.batch * per_msg,
                    "public_inputs too short for public-orig decoding: got {}, need at least {}",
                    pi.len(),
                    base + params.batch * per_msg
                );
                println!("  public-orig fields (decoded):");
                for mi in 0..params.batch {
                    let off = base + mi * per_msg;
                    let role_id = pi[off + 0].to_canonical_u64();
                    let present = pi[off + 1].to_canonical_u64() != 0;
                    let total_len = pi[off + 2].to_canonical_u64();
                    println!(
                        "    msg[{mi}]: role_id={role_id} present={present} total_len={total_len}"
                    );
                    for k in 0..params.max_orig_chunks {
                        let c_off = off + 3 + 3 * k;
                        let ch_en = pi[c_off + 0].to_canonical_u64() != 0;
                        let ch_len = pi[c_off + 1].to_canonical_u64();
                        let ch_packed = pi[c_off + 2].to_canonical_u64();
                        println!("      chunk[{k}]: enabled={ch_en} len={ch_len} packed_le_u64={ch_packed}");
                    }
                }
            }
        } else {
            println!("  (commit-orig) total_public_inputs={}", pi.len());
        }

        batch_reports.push(RewriteBatchReport {
            batch_index: b,
            total_public_inputs: pi.len(),
            leaf_proof_size_bytes: leaf_sizes[b],
            public_inputs: pi_report.clone(),
            native_last_present: witnesses[b].last_present,
            native_user_acc_in: witnesses[b].witness.user_acc_in,
            native_user_root: witnesses[b].user_root,
            native_mod_acc_in: witnesses[b].witness.mod_acc_in,
            native_mod_root: witnesses[b].mod_root,
        });

        // If we are not aggregating: do the external consistency check only once on the final batch proof.
        if !do_agg && (b + 1 == n_batches.max(1)) {
            let expected = RewritePublicInputsReport {
                message_bank_commitment,
                fragment_bank_commitment,
                request_randomness_binding: commitment::request_randomness_binding_digest(
                    request_randomness,
                ),
                start_idx: witnesses[b].witness.start_idx,
                end_idx: witnesses[b].witness.end_idx,
                rewritten_count_in: witnesses[b].witness.rewritten_count_in,
                rewritten_count_out: witnesses[b].witness.rewritten_count_out,
                prev_present: u64::from(witnesses[b].witness.prev_present),
                last_present: u64::from(witnesses[b].last_present),
                user_acc_in: witnesses[b].witness.user_acc_in,
                user_root: witnesses[b].user_root,
                mod_acc_in: witnesses[b].witness.mod_acc_in,
                mod_root: witnesses[b].mod_root,
            };
            anyhow::ensure!(
                pi_report == expected,
                "leaf public inputs do not match the native transition"
            );
        }
    }

    let mut root_proof_bytes: Option<usize> = None;
    let mut agg_prove_wall_s: Option<f64> = None;
    let mut root_report: Option<RewriteRootReport> = None;
    let mut recursion_levels_report: Option<Vec<RewriteLevelSizeReport>> = None;
    if do_agg {
        anyhow::ensure!(fan_in >= 2, "--fan-in must be >= 2");
        let (mut cache, _) =
            crate::circuits::aggregate::RewriteRecursionCache::new(fan_in, data.clone())?;
        let (root_proof, root_data, stats, infos) =
            cache.aggregate_with_data(&leaf_proofs, request_randomness, print_level_sizes)?;
        let infos_opt = print_level_sizes.then_some(infos);

        let t_verify = Instant::now();
        root_data.verify(root_proof.clone())?;
        let root_verify_s = t_verify.elapsed().as_secs_f64();
        let root_proof_raw = root_proof.to_bytes();
        root_proof_bytes = Some(root_proof_raw.len());
        let root_proof_path = if save_proof {
            if let Some(report_path) = report_json_path {
                let proof_path = default_root_proof_path(report_path);
                if let Some(parent) = proof_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&proof_path, &root_proof_raw)?;
                Some(std::fs::canonicalize(&proof_path)?.display().to_string())
            } else {
                None
            }
        } else {
            None
        };
        agg_prove_wall_s = Some(stats.agg_prove_s);

        if let Some(infos) = infos_opt {
            recursion_levels_report = Some(
                infos
                    .iter()
                    .map(|info| RewriteLevelSizeReport {
                        level: info.level.clone(),
                        degree_bits: info.degree_bits,
                        proof_bytes_avg: info.proof_bytes_avg,
                        n_proofs: info.n_proofs,
                    })
                    .collect(),
            );
            println!("recursion_levels (zero_knowledge={}):", zero_knowledge);
            for info in infos {
                println!(
                    "  - level={} degree_bits={} proof_bytes_avg={} n_proofs={}",
                    info.level, info.degree_bits, info.proof_bytes_avg, info.n_proofs
                );
            }
        }

        let rpi = &root_proof.public_inputs;
        let root_pi_report = root_public_inputs_report(rpi);
        println!(
            "root (aggregated):\n  public_inputs:\n    message_bank_commitment={:?} fragment_bank_commitment={:?} client_message_count={} client_request_hash={:?} rewritten_message_count={} request_commitment={:?}\n  stats:\n    root_verify_time={:.3}s root_proof_size={} bytes",
            root_pi_report.message_bank_commitment,
            root_pi_report.fragment_bank_commitment,
            root_pi_report.client_message_count,
            root_pi_report.client_request_hash,
            root_pi_report.rewritten_message_count,
            root_pi_report.request_commitment,
            root_verify_s,
            root_proof_bytes.unwrap()
        );
        println!(
            "  agg_stats:\n    leaf_setup_time={:.3}s\n    agg_setup_time={:.3}s agg_prove_time={:.3}s",
            leaf_setup_s,
            stats.agg_setup_s,
            stats.agg_prove_s
        );
        root_report = Some(RewriteRootReport {
            total_public_inputs: rpi.len(),
            proof_size_bytes: root_proof_bytes.unwrap_or(0),
            proof_path: root_proof_path,
            verify_time_sec: root_verify_s,
            public_inputs: root_pi_report.clone(),
            agg_stats: RewriteAggStatsReport {
                leaf_setup_time_sec: leaf_setup_s,
                agg_setup_time_sec: stats.agg_setup_s,
                agg_prove_time_sec: stats.agg_prove_s,
            },
        });

        // External consistency checks against independently computed native
        // statements. These checks mirror what a caller pins in verify-policy.
        let final_user_fold = witnesses
            .last()
            .map(|batch| batch.user_root)
            .unwrap_or_else(commitment::transcript_init);
        let final_rewritten_fold = witnesses
            .last()
            .map(|batch| batch.mod_root)
            .unwrap_or_else(commitment::transcript_init);
        let expected_client_request_hash =
            commitment::request_hash_digest(expected_message_count, final_user_fold);
        let expected_rewritten_request_hash =
            commitment::request_hash_digest(expected_rewritten_message_count, final_rewritten_fold);
        let expected_request_commitment =
            commitment::commit_digest(expected_rewritten_request_hash, request_randomness);
        ensure_expected_statement(
            &root_pi_report,
            &RewriteRootPublicInputsReport {
                message_bank_commitment,
                fragment_bank_commitment,
                client_message_count: expected_message_count,
                client_request_hash: expected_client_request_hash,
                rewritten_message_count: expected_rewritten_message_count,
                request_commitment: expected_request_commitment,
            },
        )?;
    }

    let total_prove_wall_time = if do_agg {
        leaf_prove_wall_s + agg_prove_wall_s.unwrap_or(0.0)
    } else {
        leaf_prove_wall_s
    };

    if do_agg {
        let agg_prove_wall_time = agg_prove_wall_s.unwrap_or(0.0);
        println!(
            "total:\n  batches={}\n  leaf_merkle_prep_time={:.3}s\n  leaf_setup_time={:.3}s\n  leaf_witness_build_time={:.3}s\n  leaf_prove_wall_time={:.3}s\n  agg_prove_wall_time={:.3}s\n  total_prove_wall_time={:.3}s\n  leaf_verify_wall_time={:.3}s\n  root_only_proof_size={} bytes\n  message_bank_commitment={:?} fragment_bank_commitment={:?}",
            n_batches.max(1),
            leaf_merkle_prep_s,
            leaf_setup_s,
            leaf_witness_build_s,
            leaf_prove_wall_s,
            agg_prove_wall_time,
            total_prove_wall_time,
            leaf_verify_wall_s,
            root_proof_bytes.unwrap_or(0),
            message_bank_commitment,
            fragment_bank_commitment
        );
    } else {
        println!(
            "total:\n  batches={}\n  leaf_merkle_prep_time={:.3}s\n  leaf_setup_time={:.3}s\n  leaf_witness_build_time={:.3}s\n  leaf_prove_wall_time={:.3}s\n  leaf_verify_wall_time={:.3}s\n  leaf_proof_bytes_avg={} bytes\n  leaf_proof_bytes_total={} bytes\n  message_bank_commitment={:?} fragment_bank_commitment={:?}",
            n_batches.max(1),
            leaf_merkle_prep_s,
            leaf_setup_s,
            leaf_witness_build_s,
            leaf_prove_wall_s,
            leaf_verify_wall_s,
            leaf_bytes_avg,
            leaf_bytes_total,
            message_bank_commitment,
            fragment_bank_commitment
        );
    }

    let report = RewriteRunReport {
        version: REWRITE_REPORT_VERSION.to_string(),
        input_path: input_path_for_report,
        aggregated: do_agg,
        public_orig,
        zero_knowledge,
        dedup_merkle,
        strict_commit,
        fan_in,
        params: params.clone(),
        batch_count: n_batches.max(1),
        message_bank_commitment,
        fragment_bank_commitment,
        batch_reports,
        root_report,
        recursion_levels: recursion_levels_report,
        totals: RewriteTotalsReport {
            batches: n_batches.max(1),
            leaf_merkle_prep_time_sec: leaf_merkle_prep_s,
            leaf_setup_time_sec: leaf_setup_s,
            leaf_witness_build_time_sec: leaf_witness_build_s,
            leaf_prove_wall_time_sec: leaf_prove_wall_s,
            agg_prove_wall_time_sec: agg_prove_wall_s,
            total_prove_wall_time_sec: total_prove_wall_time,
            leaf_verify_wall_time_sec: leaf_verify_wall_s,
            leaf_proof_bytes_avg: leaf_bytes_avg,
            leaf_proof_bytes_total: leaf_bytes_total,
            root_only_proof_size_bytes: root_proof_bytes,
            message_bank_commitment,
            fragment_bank_commitment,
        },
    };
    eprintln!("[VeriRouteRewrite] rewrite proof done");
    Ok(report)
}
