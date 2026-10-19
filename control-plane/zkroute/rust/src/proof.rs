//! Build route witnesses, generate proofs, and verify proof artifacts.

use anyhow::{bail, Context, Result};
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use rayon::join;
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::circuits::common::NodeIdRangeMode;
use crate::circuits::ef_search::{
    build_ef_search_circuit, fill_ef_search_witness, EfSearchTargets,
};
use crate::circuits::final_agg::{build_final_agg_circuit, fill_final_agg_witness};
use crate::circuits::score_open::{build_score_open_circuit, fill_score_open_witness};
use crate::circuits::upper::{build_upper_circuit, fill_upper_witness, UpperTargets};
use crate::commit_check::check_commit_data;
use crate::commitment::{commit_digest, compute_nbr_commit, hash_query, score_table_hash};
use crate::util::{canonical_node_id, u32_to_f, u64_to_f, NodeId, C, D, F};
use crate::witness::{
    ef_search_unused_ref_len, final_output_model_id, load_input, upper_end, ProofInput, RouteShape,
    ScoreTableCompaction,
};

pub const ROUTE_VERIFICATION_POLICY_VERSION: &str = "route_verification_policy_v2_commitments";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofReport {
    pub trace_path: String,
    pub use_zk: bool,
    pub query_index: u32,
    pub node_count: usize,
    pub score_table_compaction: ScoreTableCompaction,
    pub expand_row_count: usize,
    pub upper_steps: usize,
    pub ef_rounds: usize,
    pub final_output_model_id: u32,
    pub final_statement: FinalRouteStatement,
    #[serde(default)]
    pub final_proof_path: Option<String>,
    #[serde(default)]
    pub final_proof_bytes: Option<usize>,
    pub modules: std::collections::BTreeMap<String, ModuleBench>,
    pub final_public_inputs_len: usize,
    pub setup_sec: f64,
    pub prove_sec: f64,
    pub verify_sec: f64,
    pub proof_bytes_total: usize,
    pub total_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FinalRouteStatement {
    pub reference_commitment: [u64; 4],
    pub q_hash: [u64; 4],
    pub graph_commitment: [u64; 4],
    pub output_model_id: u32,
    #[serde(default)]
    pub entry_point: NodeId,
    #[serde(default)]
    pub entry_layer: u32,
    #[serde(default)]
    pub ef_search: u32,
    #[serde(default)]
    pub top_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RouteVerificationPolicy {
    pub version: String,
    pub reference_commitment: [u64; 4],
    pub q_hash: [u64; 4],
    pub graph_commitment: [u64; 4],
    pub entry_point: NodeId,
    pub entry_layer: u32,
    pub ef_search: u32,
    pub top_k: u32,
}

impl RouteVerificationPolicy {
    pub fn matches(&self, statement: &FinalRouteStatement) -> bool {
        self.version == ROUTE_VERIFICATION_POLICY_VERSION
            && self.reference_commitment == statement.reference_commitment
            && self.q_hash == statement.q_hash
            && self.graph_commitment == statement.graph_commitment
            && self.entry_point == statement.entry_point
            && self.entry_layer == statement.entry_layer
            && self.ef_search == statement.ef_search
            && self.top_k == statement.top_k
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleBench {
    pub setup_sec: f64,
    pub prove_sec: f64,
    pub verify_sec: f64,
    pub proof_bytes: usize,
    pub public_inputs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientVerifyReport {
    pub report_path: String,
    pub proof_path: String,
    pub verified: bool,
    pub proof_bytes: usize,
    pub public_inputs_len: usize,
    pub final_statement: FinalRouteStatement,
    pub verifier_data_hex: String,
    pub zero_knowledge: bool,
    pub recursion_depth: usize,
}

impl FinalRouteStatement {
    pub fn public_inputs(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(17);
        out.extend(self.reference_commitment.map(u64_to_f));
        out.extend(self.q_hash.map(u64_to_f));
        out.extend(self.graph_commitment.map(u64_to_f));
        out.push(u32_to_f(self.output_model_id));
        out.push(u64_to_f(canonical_node_id(self.entry_point)));
        out.push(u32_to_f(self.entry_layer));
        out.push(u32_to_f(self.ef_search));
        out.push(u32_to_f(self.top_k));
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub modules_only: bool,
    pub verify_subproofs: bool,
    pub parallel_subproofs: bool,
    pub save_final_proof: bool,
}

fn log_info(message: impl AsRef<str>) {
    eprintln!("[VeriRouteRoute] {}", message.as_ref());
}

fn log_stage_start(label: impl AsRef<str>) -> Instant {
    let label = label.as_ref().to_string();
    log_info(format!("start {}", label));
    Instant::now()
}

fn log_stage_end(label: impl AsRef<str>, started: Instant) {
    log_info(format!(
        "end {} ({:.3}s)",
        label.as_ref(),
        started.elapsed().as_secs_f64()
    ));
}

pub fn final_statement(
    reference_commitment: &[u64; 4],
    q_hash: &[u64; 4],
    graph_commitment: &[u64; 4],
    output_id: u32,
    entry_point: NodeId,
    entry_layer: u32,
    ef_search: u32,
    top_k: u32,
) -> FinalRouteStatement {
    FinalRouteStatement {
        reference_commitment: *reference_commitment,
        q_hash: *q_hash,
        graph_commitment: *graph_commitment,
        output_model_id: output_id,
        entry_point: canonical_node_id(entry_point),
        entry_layer,
        ef_search,
        top_k,
    }
}

fn resolve_input_relative(trace_path: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        trace_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn default_report_path(trace: &Path) -> PathBuf {
    let parent = trace.parent().unwrap_or_else(|| Path::new("."));
    let stem = trace
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("proof_input");
    parent.join(format!("{}.prove_report.json", stem))
}

fn default_final_proof_path(report_path: &Path) -> PathBuf {
    let parent = report_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = report_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("route_report");
    parent.join(format!("{stem}.final.proof"))
}

fn resolve_report_relative(report_path: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        report_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

/// Verifies a proof against a caller-supplied trace for development and regression testing.
///
/// This does not establish a trust boundary on its own because the caller supplies the trace
/// shape and artifact locations. Production callers should use
/// [`verify_final_proof_artifact_with_policy`] and pin the policy out of band.
pub fn verify_final_proof_artifact(
    trace_path: &Path,
    report_path: &Path,
    proof_path_override: Option<&Path>,
) -> Result<ClientVerifyReport> {
    verify_final_proof_artifact_inner(trace_path, report_path, proof_path_override, None)
}

pub fn verify_final_proof_artifact_with_policy(
    trace_path: &Path,
    report_path: &Path,
    proof_path_override: Option<&Path>,
    policy: &RouteVerificationPolicy,
) -> Result<ClientVerifyReport> {
    verify_final_proof_artifact_inner(trace_path, report_path, proof_path_override, Some(policy))
}

fn verify_final_proof_artifact_inner(
    trace_path: &Path,
    report_path: &Path,
    proof_path_override: Option<&Path>,
    policy: Option<&RouteVerificationPolicy>,
) -> Result<ClientVerifyReport> {
    log_info(format!(
        "client verify: trace={} report={}",
        trace_path.display(),
        report_path.display()
    ));
    let report_payload = fs::read_to_string(report_path)
        .with_context(|| format!("failed to read {}", report_path.display()))?;
    let report_json: serde_json::Value = serde_json::from_str(&report_payload)
        .with_context(|| format!("failed to parse {}", report_path.display()))?;
    let (input, _, shape) = load_input(trace_path)?;
    let use_zk = report_json
        .get("use_zk")
        .and_then(|v| v.as_bool())
        .context("route report is missing use_zk")?;
    let report_statement: FinalRouteStatement = serde_json::from_value(
        report_json
            .get("final_statement")
            .cloned()
            .context("route report is missing final_statement")?,
    )?;
    let proof_path = if let Some(path) = proof_path_override {
        path.to_path_buf()
    } else {
        let value = report_json
            .get("final_proof_path")
            .and_then(|v| v.as_str())
            .context("route report is missing final_proof_path")?;
        resolve_report_relative(report_path, value)
    };

    let global_node_path =
        resolve_input_relative(trace_path, &input.global_artifacts.node_table_path);
    let global_graph_path =
        resolve_input_relative(trace_path, &input.global_artifacts.graph_rows_path);
    let (node_merkle, graph_merkle) =
        check_commit_data(&global_node_path, &global_graph_path, input.query.dimension)?;
    let node_root = node_merkle.root_u64s();
    let reference_commitment = commit_digest(node_root, input.commitment_randomness.reference);
    let q_hash = hash_query(&input);
    let graph_root = graph_merkle.root_u64s();
    let graph_commitment = commit_digest(graph_root, input.commitment_randomness.graph);
    let output_model_id = final_output_model_id(&input)?;
    let entry_layer = graph_merkle.max_layer();
    let statement = final_statement(
        &reference_commitment,
        &q_hash,
        &graph_commitment,
        output_model_id,
        input.entry_point,
        entry_layer,
        input.ef_search.ef_search,
        input.ef_search.k,
    );
    if let Some(policy) = policy {
        if !policy.matches(&statement) {
            bail!("route proof does not match the pinned verification policy");
        }
    }
    if statement != report_statement {
        bail!("route report final_statement does not match the supplied trace/artifacts");
    }

    let (dist_data, _) = build_score_open_circuit(
        use_zk,
        shape.node_count,
        input.query.dimension,
        node_merkle.tree_height(),
    );
    let (upper_data, _) = build_upper(use_zk, &shape, graph_merkle.tree_height());
    let (ef_data, _) = build_ef(use_zk, &shape, &input, graph_merkle.tree_height());
    let (final_data, _) = build_final_agg_circuit(&dist_data, &upper_data, &ef_data);
    let proof_bytes = fs::read(&proof_path)
        .with_context(|| format!("failed to read {}", proof_path.display()))?;
    let proof =
        ProofWithPublicInputs::<F, C, D>::from_bytes(proof_bytes.clone(), &final_data.common)
            .with_context(|| format!("failed to decode {}", proof_path.display()))?;
    let expected_pi = final_statement(
        &reference_commitment,
        &q_hash,
        &graph_commitment,
        output_model_id,
        input.entry_point,
        entry_layer,
        input.ef_search.ef_search,
        input.ef_search.k,
    )
    .public_inputs();
    if proof.public_inputs != expected_pi {
        bail!("final proof public inputs do not match derived statement");
    }
    final_data.verify(proof.clone())?;
    Ok(ClientVerifyReport {
        report_path: report_path.display().to_string(),
        proof_path: proof_path.display().to_string(),
        verified: true,
        proof_bytes: proof_bytes.len(),
        public_inputs_len: proof.public_inputs.len(),
        final_statement: statement,
        verifier_data_hex: final_data
            .verifier_data()
            .to_bytes(&plonky2::util::serialization::DefaultGateSerializer)
            .map_err(|e| anyhow::anyhow!("verifier serialization failed: {e:?}"))?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        zero_knowledge: use_zk,
        recursion_depth: 1,
    })
}

fn build_upper(
    use_zk: bool,
    shape: &RouteShape,
    graph_tree_height: usize,
) -> (CircuitData<F, C, D>, UpperTargets) {
    let (data, targets, _) = build_upper_circuit(
        use_zk,
        shape.node_count,
        shape.max_upper_steps,
        shape.upper_neighbor_cap,
        graph_tree_height,
        shape.node_count,
        NodeIdRangeMode::Wide56,
    );
    (data, targets)
}

fn build_ef(
    use_zk: bool,
    shape: &RouteShape,
    input: &ProofInput,
    graph_tree_height: usize,
) -> (CircuitData<F, C, D>, EfSearchTargets) {
    build_ef_search_circuit(
        use_zk,
        shape.node_count,
        &shape.row_neighbor_lens(),
        shape.max_upper_steps,
        shape.max_rounds,
        input.ef_search.ef_search as usize,
        input.ef_search.k as usize,
        shape.visit_len,
        shape.c_final_len,
        shape.w_final_len,
        graph_tree_height,
        0,
    )
}

struct ModuleProofRun {
    proof: ProofWithPublicInputs<F, C, D>,
    prove_sec: f64,
    verify_sec: f64,
    proof_bytes: usize,
    public_inputs: usize,
}

fn prove_module(
    name: &str,
    data: &CircuitData<F, C, D>,
    pw: PartialWitness<F>,
    verify_subproofs: bool,
) -> Result<ModuleProofRun> {
    log_info(format!("module {}: prove start", name));
    let prove_t0 = Instant::now();
    let proof = data.prove(pw).with_context(|| format!("{} proof", name))?;
    let prove_sec = prove_t0.elapsed().as_secs_f64();
    let proof_bytes = proof.to_bytes().len();
    let public_inputs = proof.public_inputs.len();
    log_info(format!(
        "module {}: prove end ({:.3}s, {} bytes, {} public inputs)",
        name, prove_sec, proof_bytes, public_inputs
    ));
    let verify_sec = if verify_subproofs {
        log_info(format!("module {}: verify start", name));
        let verify_t0 = Instant::now();
        data.verify(proof.clone())?;
        let sec = verify_t0.elapsed().as_secs_f64();
        log_info(format!("module {}: verify end ({:.3}s)", name, sec));
        sec
    } else {
        0.0
    };
    Ok(ModuleProofRun {
        proof,
        prove_sec,
        verify_sec,
        proof_bytes,
        public_inputs,
    })
}

fn record_module_bench(
    modules: &mut BTreeMap<String, ModuleBench>,
    name: &str,
    setup_sec: f64,
    run: &ModuleProofRun,
    setup_sec_total: &mut f64,
    prove_sec_total: &mut f64,
    verify_sec_total: &mut f64,
    proof_bytes_total: &mut usize,
) {
    *setup_sec_total += setup_sec;
    *prove_sec_total += run.prove_sec;
    *verify_sec_total += run.verify_sec;
    *proof_bytes_total += run.proof_bytes;
    modules.insert(
        name.to_string(),
        ModuleBench {
            setup_sec,
            prove_sec: run.prove_sec,
            verify_sec: run.verify_sec,
            proof_bytes: run.proof_bytes,
            public_inputs: run.public_inputs,
        },
    );
}

pub fn run_trace(
    trace_path: &Path,
    use_zk: bool,
    opts: RunOptions,
    report_path: Option<&Path>,
) -> Result<ProofReport> {
    log_info(format!("loading trace {}", trace_path.display()));
    let (input, score_table_compaction, shape) = load_input(trace_path)?;
    log_info(format!(
        "trace loaded: query_index={} query_dim={} nodes={} upper_steps={} ef_rounds={} neighbor_visits={} zk={} parallel_subproofs={}",
        input.query_index,
        input.query.dimension,
        input.nodes.len(),
        input.upper_trace.len(),
        input.ef_search.expansions.len(),
        input.ef_search.neighbor_visits.len(),
        use_zk,
        opts.parallel_subproofs
    ));
    let global_node_path =
        resolve_input_relative(trace_path, &input.global_artifacts.node_table_path);
    let global_graph_path =
        resolve_input_relative(trace_path, &input.global_artifacts.graph_rows_path);
    let root_t0 = log_stage_start("derive_roots");
    let (node_merkle, graph_merkle) =
        check_commit_data(&global_node_path, &global_graph_path, input.query.dimension)?;
    let node_root = node_merkle.root_u64s();
    let reference_commitment = commit_digest(node_root, input.commitment_randomness.reference);
    let q_hash = hash_query(&input);
    let score_table_hash = score_table_hash(&input);
    let graph_root = graph_merkle.root_u64s();
    let graph_commitment = commit_digest(graph_root, input.commitment_randomness.graph);
    let entry_layer = graph_merkle.max_layer();
    let nbr_commit = compute_nbr_commit(&input);
    let upper_end = upper_end(&input);
    let ef_entry = upper_end;
    if ef_search_unused_ref_len(&input) != 0 {
        bail!("score table has nodes that are not referenced by the search trace");
    }
    let output_model_id = final_output_model_id(&input)?;
    log_stage_end("derive_roots", root_t0);

    let mut modules = BTreeMap::<String, ModuleBench>::new();
    let mut setup_sec_total = 0.0;
    let mut prove_sec_total = 0.0;
    let mut verify_sec_total = 0.0;
    let mut proof_bytes_total = 0usize;
    let t0 = Instant::now();

    log_info("prebuilding subproof circuits");

    log_info("module score_open: circuit build start");
    let dist_build_t0 = Instant::now();
    let (dist_data, dist_targets) = build_score_open_circuit(
        use_zk,
        shape.node_count,
        input.query.dimension,
        node_merkle.tree_height(),
    );
    let dist_build_sec = dist_build_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module score_open: circuit build end ({:.3}s)",
        dist_build_sec
    ));
    log_info("module score_open: witness fill start");
    let dist_fill_t0 = Instant::now();
    let mut dist_pw = PartialWitness::<F>::new();
    fill_score_open_witness(
        &mut dist_pw,
        &dist_targets,
        &input,
        &node_merkle,
        &node_root,
        &q_hash,
        &score_table_hash,
    );
    let dist_fill_sec = dist_fill_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module score_open: witness fill end ({:.3}s)",
        dist_fill_sec
    ));
    let dist_setup_sec = dist_build_sec + dist_fill_sec;

    let upper_entry = input.entry_point;
    log_info("module upper: circuit build start");
    let upper_build_t0 = Instant::now();
    let (upper_data, upper_targets) = build_upper(use_zk, &shape, graph_merkle.tree_height());
    let upper_build_sec = upper_build_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module upper: circuit build end ({:.3}s)",
        upper_build_sec
    ));
    log_info("module upper: witness fill start");
    let upper_fill_t0 = Instant::now();
    let mut upper_pw = PartialWitness::<F>::new();
    fill_upper_witness(
        &mut upper_pw,
        &upper_targets,
        &input,
        &score_table_hash,
        &graph_root,
        upper_entry,
        upper_end,
        entry_layer,
        &graph_merkle,
    );
    let upper_fill_sec = upper_fill_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module upper: witness fill end ({:.3}s)",
        upper_fill_sec
    ));
    let upper_setup_sec = upper_build_sec + upper_fill_sec;

    let ef_module_name = "ef_search";
    log_info(format!("module {ef_module_name}: circuit build start"));
    let ef_build_t0 = Instant::now();
    let (ef_data, ef_targets) = build_ef(use_zk, &shape, &input, graph_merkle.tree_height());
    let ef_build_sec = ef_build_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module {ef_module_name}: circuit build end ({:.3}s)",
        ef_build_sec
    ));
    log_info(format!("module {ef_module_name}: witness fill start"));
    let ef_fill_t0 = Instant::now();
    let mut ef_pw = PartialWitness::<F>::new();
    fill_ef_search_witness(
        &mut ef_pw,
        &ef_targets,
        &input,
        ef_entry,
        output_model_id,
        &score_table_hash,
        &graph_root,
        &nbr_commit,
        &graph_merkle,
    );
    let ef_fill_sec = ef_fill_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module {ef_module_name}: witness fill end ({:.3}s)",
        ef_fill_sec
    ));
    let ef_setup_sec = ef_build_sec + ef_fill_sec;

    let final_prebuilt = if opts.modules_only {
        None
    } else {
        log_info("module final_agg: circuit build start");
        let final_build_t0 = Instant::now();
        let (final_data, final_targets) =
            build_final_agg_circuit(&dist_data, &upper_data, &ef_data);
        let final_build_sec = final_build_t0.elapsed().as_secs_f64();
        log_info(format!(
            "module final_agg: circuit build end ({:.3}s)",
            final_build_sec
        ));
        Some((final_data, final_targets, final_build_sec))
    };

    let verify_subproofs = opts.verify_subproofs;
    let (dist_run, upper_run, ef_run) = if opts.parallel_subproofs {
        log_info("parallel_subproofs enabled: proving subproofs in parallel");
        let ((dist, upper), ef) = join(
            || {
                join(
                    || prove_module("score_open", &dist_data, dist_pw, verify_subproofs),
                    || prove_module("upper", &upper_data, upper_pw, verify_subproofs),
                )
            },
            || prove_module(ef_module_name, &ef_data, ef_pw, verify_subproofs),
        );
        (dist?, upper?, ef?)
    } else {
        (
            prove_module("score_open", &dist_data, dist_pw, verify_subproofs)?,
            prove_module("upper", &upper_data, upper_pw, verify_subproofs)?,
            prove_module(ef_module_name, &ef_data, ef_pw, verify_subproofs)?,
        )
    };

    record_module_bench(
        &mut modules,
        "score_open",
        dist_setup_sec,
        &dist_run,
        &mut setup_sec_total,
        &mut prove_sec_total,
        &mut verify_sec_total,
        &mut proof_bytes_total,
    );
    record_module_bench(
        &mut modules,
        "upper",
        upper_setup_sec,
        &upper_run,
        &mut setup_sec_total,
        &mut prove_sec_total,
        &mut verify_sec_total,
        &mut proof_bytes_total,
    );
    record_module_bench(
        &mut modules,
        ef_module_name,
        ef_setup_sec,
        &ef_run,
        &mut setup_sec_total,
        &mut prove_sec_total,
        &mut verify_sec_total,
        &mut proof_bytes_total,
    );
    if opts.modules_only {
        log_info("modules_only enabled, skipping final_agg");
        let report = ProofReport {
            trace_path: trace_path.display().to_string(),
            use_zk,
            query_index: input.query_index,
            node_count: input.nodes.len(),
            score_table_compaction,
            expand_row_count: input.expand_rows.len(),
            upper_steps: input.upper_trace.len(),
            ef_rounds: input.ef_search.expansions.len(),
            final_output_model_id: output_model_id,
            final_statement: final_statement(
                &reference_commitment,
                &q_hash,
                &graph_commitment,
                output_model_id,
                upper_entry,
                entry_layer,
                input.ef_search.ef_search,
                input.ef_search.k,
            ),
            final_proof_path: None,
            final_proof_bytes: None,
            modules,
            final_public_inputs_len: 0,
            setup_sec: setup_sec_total,
            prove_sec: prove_sec_total,
            verify_sec: verify_sec_total,
            proof_bytes_total,
            total_sec: t0.elapsed().as_secs_f64(),
        };
        let out_path = report_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| default_report_path(trace_path));
        fs::write(&out_path, to_string_pretty(&report)?)
            .with_context(|| format!("failed to write {}", out_path.display()))?;
        log_info(format!("report written to {}", out_path.display()));
        return Ok(report);
    }

    let (final_data, final_targets, final_build_sec) =
        final_prebuilt.expect("final_agg prebuild missing");
    log_info("module final_agg: witness fill start");
    let final_fill_t0 = Instant::now();
    let mut final_pw = PartialWitness::<F>::new();
    fill_final_agg_witness(
        &mut final_pw,
        &final_targets,
        &dist_run.proof,
        &upper_run.proof,
        &ef_run.proof,
        &dist_data.verifier_only,
        &upper_data.verifier_only,
        &ef_data.verifier_only,
        &input.commitment_randomness,
    );
    let final_fill_sec = final_fill_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module final_agg: witness fill end ({:.3}s)",
        final_fill_sec
    ));
    let final_setup_sec = final_build_sec + final_fill_sec;
    log_info("module final_agg: prove start");
    let final_prove_t0 = Instant::now();
    let final_proof = final_data
        .prove(final_pw)
        .with_context(|| "final agg proof")?;
    let final_prove_sec = final_prove_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module final_agg: prove end ({:.3}s, {} bytes, {} public inputs)",
        final_prove_sec,
        final_proof.to_bytes().len(),
        final_proof.public_inputs.len()
    ));
    log_info("module final_agg: verify start");
    let final_verify_t0 = Instant::now();
    final_data.verify(final_proof.clone())?;
    let final_verify_sec = final_verify_t0.elapsed().as_secs_f64();
    log_info(format!(
        "module final_agg: verify end ({:.3}s)",
        final_verify_sec
    ));
    setup_sec_total += final_setup_sec;
    prove_sec_total += final_prove_sec;
    verify_sec_total += final_verify_sec;
    proof_bytes_total += final_proof.to_bytes().len();
    modules.insert(
        "final_agg".to_string(),
        ModuleBench {
            setup_sec: final_setup_sec,
            prove_sec: final_prove_sec,
            verify_sec: final_verify_sec,
            proof_bytes: final_proof.to_bytes().len(),
            public_inputs: final_proof.public_inputs.len(),
        },
    );

    let expected_pi = final_statement(
        &reference_commitment,
        &q_hash,
        &graph_commitment,
        output_model_id,
        upper_entry,
        entry_layer,
        input.ef_search.ef_search,
        input.ef_search.k,
    )
    .public_inputs();
    if final_proof.public_inputs != expected_pi {
        bail!("final proof public inputs mismatch expected values");
    }

    let out_path = report_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| default_report_path(trace_path));
    let final_proof_bytes = final_proof.to_bytes();
    let final_proof_path = if opts.save_final_proof {
        let path = default_final_proof_path(&out_path);
        fs::write(&path, &final_proof_bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Some(fs::canonicalize(&path)?)
    } else {
        None
    };

    let report = ProofReport {
        trace_path: trace_path.display().to_string(),
        use_zk,
        query_index: input.query_index,
        node_count: input.nodes.len(),
        score_table_compaction,
        expand_row_count: input.expand_rows.len(),
        upper_steps: input.upper_trace.len(),
        ef_rounds: input.ef_search.expansions.len(),
        final_output_model_id: output_model_id,
        final_statement: final_statement(
            &reference_commitment,
            &q_hash,
            &graph_commitment,
            output_model_id,
            upper_entry,
            entry_layer,
            input.ef_search.ef_search,
            input.ef_search.k,
        ),
        final_proof_path: final_proof_path
            .as_ref()
            .map(|path| path.display().to_string()),
        final_proof_bytes: Some(final_proof_bytes.len()),
        final_public_inputs_len: final_proof.public_inputs.len(),
        modules,
        setup_sec: setup_sec_total,
        prove_sec: prove_sec_total,
        verify_sec: verify_sec_total,
        proof_bytes_total,
        total_sec: t0.elapsed().as_secs_f64(),
    };
    fs::write(&out_path, to_string_pretty(&report)?)
        .with_context(|| format!("failed to write {}", out_path.display()))?;
    log_info(format!(
        "all proving stages finished in {:.3}s",
        t0.elapsed().as_secs_f64()
    ));
    log_info(format!("report written to {}", out_path.display()));
    Ok(report)
}

/// Client verifier: only an approved public key, public policy and proof.
/// The caller authenticates the key; this function never reads index/trace/salts.
pub fn verify_public_route_artifact(
    report_path: &Path,
    proof_override: Option<&Path>,
    verifier_path: &Path,
    policy: &RouteVerificationPolicy,
) -> Result<ClientVerifyReport> {
    let report: serde_json::Value = serde_json::from_slice(&fs::read(report_path)?)?;
    let statement: FinalRouteStatement = serde_json::from_value(
        report
            .get("final_statement")
            .cloned()
            .context("missing final_statement")?,
    )?;
    if !policy.matches(&statement) {
        bail!("route proof does not match pinned policy");
    }
    let vk_bytes = fs::read(verifier_path)?;
    let data = plonky2::plonk::circuit_data::VerifierCircuitData::<F, C, D>::from_bytes(
        vk_bytes.clone(),
        &plonky2::util::serialization::DefaultGateSerializer,
    )
    .map_err(|e| anyhow::anyhow!("invalid verifier data: {e:?}"))?;
    if !data.common.config.zero_knowledge {
        bail!("approved key must enable ZK");
    }
    let proof_path = match proof_override {
        Some(path) => path.to_path_buf(),
        None => resolve_report_relative(
            report_path,
            report
                .get("final_proof_path")
                .and_then(|v| v.as_str())
                .context("missing final_proof_path")?,
        ),
    };
    let bytes = fs::read(&proof_path)?;
    let proof = ProofWithPublicInputs::<F, C, D>::from_bytes(bytes.clone(), &data.common)?;
    let expected = statement.public_inputs();
    if proof.public_inputs != expected {
        bail!("public inputs do not match route statement");
    }
    data.verify(proof.clone())?;
    Ok(ClientVerifyReport {
        report_path: report_path.display().to_string(),
        proof_path: proof_path.display().to_string(),
        verified: true,
        proof_bytes: bytes.len(),
        public_inputs_len: proof.public_inputs.len(),
        final_statement: statement,
        verifier_data_hex: vk_bytes.iter().map(|b| format!("{b:02x}")).collect(),
        zero_knowledge: true,
        recursion_depth: 1,
    })
}
