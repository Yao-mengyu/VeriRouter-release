use anyhow::{Context, Result};
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::proof::ProofWithPublicInputs;
use rayon::join;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use veriroute_route::circuits::common::NodeIdRangeMode;
use veriroute_route::circuits::ef_search::{
    build_ef_search_circuit_with_options, fill_ef_search_witness,
};
use veriroute_route::circuits::final_agg::{build_final_agg_circuit, fill_final_agg_witness};
use veriroute_route::circuits::score_open::{
    build_score_open_circuit_with_options, fill_score_open_witness, ScoreDotFormula,
};
use veriroute_route::circuits::upper::{build_upper_circuit, fill_upper_witness};
use veriroute_route::commit_check::check_commit_data;
use veriroute_route::commitment::{
    commit_digest, compute_nbr_commit, hash_query, score_table_hash,
};
use veriroute_route::proof::{final_statement, FinalRouteStatement};
use veriroute_route::util::{C, D, F, MULTISET_CHALLENGE_COUNT};
use veriroute_route::witness::{
    compact_score_table, ef_search_unused_ref_len, final_output_model_id, upper_end, ProofInput,
    RouteShape,
};

const ROUTE_CIRCUIT_FINGERPRINT: &str = "route_circuit_v21";

#[derive(Clone, Debug, Deserialize)]
struct RouteBenchTraceItem {
    padded_input_path: String,
    query_index: u32,
    actual: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
struct RouteBenchManifest {
    version: String,
    shape: RouteShape,
    traces: Vec<RouteBenchTraceItem>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TimingSummary {
    count: usize,
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    p95: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModuleBenchReport {
    witness_fill_sec: f64,
    prove_sec: f64,
    verify_sec: f64,
    proof_bytes: usize,
    public_inputs: usize,
    #[serde(default)]
    gate_rows: usize,
    #[serde(default)]
    degree_bits: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TraceBenchReport {
    query_index: u32,
    actual: serde_json::Value,
    padded_trace_path: String,
    final_output_model_id: u32,
    final_statement: FinalRouteStatement,
    modules: BTreeMap<String, ModuleBenchReport>,
    witness_fill_total_sec: f64,
    subproof_parallel_prove_wall_sec: f64,
    final_agg_fill_sec: f64,
    final_agg_prove_sec: f64,
    final_verify_sec: f64,
    total_prove_wall_sec: f64,
    total_verify_sec: f64,
    proof_bytes_total: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SetupReport {
    shape: RouteShape,
    trace_count: usize,
    use_zk: bool,
    #[serde(default = "default_true")]
    parallel_subproofs: bool,
    #[serde(default)]
    circuit_fingerprint: String,
    #[serde(default)]
    score_open_dummy_vector_zero_constraints: bool,
    #[serde(default)]
    score_open_dot_formula: String,
    #[serde(default)]
    score_dot_product_gate_mode: String,
    #[serde(default)]
    custom_dot_product_gate: bool,
    #[serde(default)]
    uses_standard_plonky2_specialized_hash_and_range_gates: bool,
    #[serde(default)]
    score_open_pre_padding_gate_count: usize,
    #[serde(default)]
    score_open_node_merkle_gate_count: usize,
    #[serde(default)]
    score_open_node_merkle_leaf_hash_gate_count: usize,
    #[serde(default)]
    score_open_node_merkle_path_gate_count: usize,
    #[serde(default)]
    score_open_formula_gate_count: usize,
    #[serde(default)]
    score_open_score_table_hash_gate_count: usize,
    #[serde(default)]
    upper_pre_padding_gate_count: usize,
    #[serde(default)]
    upper_score_table_hash_gate_count: usize,
    #[serde(default)]
    upper_graph_merkle_gate_count: usize,
    #[serde(default)]
    upper_graph_merkle_leaf_hash_gate_count: usize,
    #[serde(default)]
    upper_graph_merkle_path_gate_count: usize,
    #[serde(default)]
    upper_dist_ref_gate_count: usize,
    #[serde(default)]
    ef_search_pre_padding_gate_count: usize,
    #[serde(default)]
    ef_search_score_table_hash_gate_count: usize,
    #[serde(default)]
    ef_search_graph_merkle_gate_count: usize,
    #[serde(default)]
    ef_search_graph_merkle_leaf_hash_gate_count: usize,
    #[serde(default)]
    ef_search_graph_merkle_path_gate_count: usize,
    #[serde(default)]
    ef_search_dist_ref_gate_count: usize,
    #[serde(default)]
    node_id_range_mode: String,
    #[serde(default)]
    ef_graph_static_open_rows: usize,
    #[serde(default)]
    ef_graph_static_padding_rows: usize,
    #[serde(default)]
    score_table_hash_mode: String,
    #[serde(default)]
    upper_ef_score_table_path_openings: bool,
    #[serde(default)]
    upper_ref_unused_cap: usize,
    #[serde(default)]
    ef_ref_unused_cap: usize,
    #[serde(default)]
    nbr_commit_mode: String,
    #[serde(default)]
    dist_ref_membership_mode: String,
    #[serde(default)]
    multiset_challenge_count: usize,
    setup_total_sec: f64,
    score_open_setup_sec: f64,
    upper_setup_sec: f64,
    ef_search_setup_sec: f64,
    final_agg_setup_sec: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RouteBenchReport {
    benchmark_type: String,
    version: String,
    manifest_version: String,
    setup: SetupReport,
    summary_sec: serde_json::Value,
    summary_bytes: serde_json::Value,
    traces: Vec<TraceBenchReport>,
}

fn default_true() -> bool {
    true
}

fn build_report(
    manifest_version: &str,
    setup: &SetupReport,
    traces: &[TraceBenchReport],
    total_prove_secs: &[f64],
    total_verify_secs: &[f64],
    proof_bytes: &[usize],
) -> RouteBenchReport {
    RouteBenchReport {
        benchmark_type: "route_overhead".to_string(),
        version: "route_overhead_v2".to_string(),
        manifest_version: manifest_version.to_string(),
        setup: setup.clone(),
        summary_sec: serde_json::json!({
            "total_prove_wall_sec": summarize(total_prove_secs),
            "total_verify_sec": summarize(total_verify_secs),
        }),
        summary_bytes: serde_json::json!({
            "proof_bytes_total": summarize_usize(proof_bytes),
        }),
        traces: traces.to_vec(),
    }
}

fn write_report(path: &Path, report: &RouteBenchReport) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, serde_json::to_string_pretty(report)?)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn load_existing_report(path: &Path) -> Option<RouteBenchReport> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str::<RouteBenchReport>(&raw).ok()
}

struct ModuleRun {
    proof: ProofWithPublicInputs<F, C, D>,
    prove_sec: f64,
    verify_sec: f64,
    proof_bytes: usize,
    public_inputs: usize,
}

fn summarize(values: &[f64]) -> TimingSummary {
    if values.is_empty() {
        return TimingSummary {
            count: 0,
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            p95: 0.0,
        };
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.total_cmp(b));
    let p95_index = (((ordered.len() - 1) as f64) * 0.95).round() as usize;
    TimingSummary {
        count: values.len(),
        mean: values.iter().sum::<f64>() / (values.len() as f64),
        median: if ordered.len() % 2 == 1 {
            ordered[ordered.len() / 2]
        } else {
            let hi = ordered.len() / 2;
            (ordered[hi - 1] + ordered[hi]) / 2.0
        },
        min: *ordered.first().unwrap(),
        max: *ordered.last().unwrap(),
        p95: ordered[p95_index],
    }
}

fn summarize_usize(values: &[usize]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::json!({
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "p95": 0.0,
        });
    }
    let as_f64 = values.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let summary = summarize(&as_f64);
    serde_json::json!({
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "min": *values.iter().min().unwrap(),
        "max": *values.iter().max().unwrap(),
        "p95": summary.p95,
    })
}

fn parse_flag_string(argv: &[String], key: &str) -> Option<String> {
    let mut i = 0usize;
    while i < argv.len() {
        if argv[i] == key {
            return argv.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

fn parse_flag_bool(argv: &[String], key: &str, default: bool) -> bool {
    let no_key = if let Some(stripped) = key.strip_prefix("--") {
        format!("--no-{}", stripped)
    } else {
        format!("no-{}", key)
    };
    let mut i = 0usize;
    while i < argv.len() {
        let arg = &argv[i];
        if arg == &no_key {
            return false;
        }
        if arg == key {
            if let Some(next) = argv.get(i + 1) {
                match next.as_str() {
                    "true" | "1" | "yes" => return true,
                    "false" | "0" | "no" => return false,
                    _ if next.starts_with("--") => return true,
                    _ => return true,
                }
            }
            return true;
        }
        if let Some(value) = arg.strip_prefix(&(key.to_string() + "=")) {
            match value {
                "true" | "1" | "yes" => return true,
                "false" | "0" | "no" => return false,
                _ => return default,
            }
        }
        i += 1;
    }
    default
}

fn parse_score_node_id_mode(argv: &[String]) -> Result<NodeIdRangeMode> {
    let raw = parse_flag_string(argv, "--score-node-id-mode").unwrap_or_else(|| "wide".to_string());
    match raw.as_str() {
        "u32" | "u32-or-sentinel" | "fast" => Ok(NodeIdRangeMode::U32OrSentinel),
        "wide" | "wide56" | "u56" => Ok(NodeIdRangeMode::Wide56),
        other => anyhow::bail!(
            "unsupported --score-node-id-mode {}; expected u32 or wide",
            other
        ),
    }
}

fn parse_score_dot_formula(argv: &[String]) -> Result<ScoreDotFormula> {
    let raw =
        parse_flag_string(argv, "--score-dot-formula").unwrap_or_else(|| "centered".to_string());
    match raw.as_str() {
        "centered" | "centered-query" | "precomputed" => Ok(ScoreDotFormula::CenteredQuery),
        "original" => Ok(ScoreDotFormula::Original),
        other => anyhow::bail!(
            "unsupported --score-dot-formula {}; expected centered or original",
            other
        ),
    }
}

fn node_id_mode_name(mode: NodeIdRangeMode) -> &'static str {
    match mode {
        NodeIdRangeMode::U32OrSentinel => "u32",
        NodeIdRangeMode::Wide56 => "wide",
    }
}

fn score_formula_name(formula: ScoreDotFormula) -> &'static str {
    match formula {
        ScoreDotFormula::CenteredQuery => "precomputed_centered_query_dot_gate_v3",
        ScoreDotFormula::Original => "original_sum_qv_sumv_v1",
    }
}

fn load_input(path: &Path) -> Result<ProofInput> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str::<ProofInput>(&raw)?)
}

fn prove_module(
    data: &plonky2::plonk::circuit_data::CircuitData<F, C, D>,
    pw: PartialWitness<F>,
) -> Result<ModuleRun> {
    let prove_t0 = Instant::now();
    let proof = data.prove(pw)?;
    let prove_sec = prove_t0.elapsed().as_secs_f64();
    let verify_t0 = Instant::now();
    data.verify(proof.clone())?;
    let verify_sec = verify_t0.elapsed().as_secs_f64();
    Ok(ModuleRun {
        proof_bytes: proof.to_bytes().len(),
        public_inputs: proof.public_inputs.len(),
        proof,
        prove_sec,
        verify_sec,
    })
}

fn gate_rows(data: &plonky2::plonk::circuit_data::CircuitData<F, C, D>) -> usize {
    1usize << data.common.degree_bits()
}

fn module_report(
    witness_fill_sec: f64,
    run: &ModuleRun,
    data: &plonky2::plonk::circuit_data::CircuitData<F, C, D>,
) -> ModuleBenchReport {
    ModuleBenchReport {
        witness_fill_sec,
        prove_sec: run.prove_sec,
        verify_sec: run.verify_sec,
        proof_bytes: run.proof_bytes,
        public_inputs: run.public_inputs,
        gate_rows: gate_rows(data),
        degree_bits: data.common.degree_bits(),
    }
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let manifest_path = PathBuf::from(
        parse_flag_string(&argv, "--manifest")
            .ok_or_else(|| anyhow::anyhow!("missing --manifest"))?,
    );
    let out_path = PathBuf::from(
        parse_flag_string(&argv, "--out")
            .unwrap_or_else(|| "route_experiment_bench.json".to_string()),
    );
    let use_zk = parse_flag_bool(&argv, "--zk", true);
    let resume = parse_flag_bool(&argv, "--resume", false);
    let parallel_subproofs = parse_flag_bool(&argv, "--parallel-subproofs", true);
    let score_node_id_mode = parse_score_node_id_mode(&argv)?;
    let score_dot_formula = parse_score_dot_formula(&argv)?;
    eprintln!(
        "[route_experiment_bench] parsed args: use_zk={} resume={} parallel_subproofs={} score_node_id_mode={} score_dot_formula={} manifest={} out={}",
        use_zk,
        resume,
        parallel_subproofs,
        node_id_mode_name(score_node_id_mode),
        score_formula_name(score_dot_formula),
        manifest_path.display(),
        out_path.display(),
    );

    let manifest =
        serde_json::from_str::<RouteBenchManifest>(&fs::read_to_string(&manifest_path)?)?;
    anyhow::ensure!(
        !manifest.traces.is_empty(),
        "manifest.traces must be non-empty"
    );
    let first_trace_path = PathBuf::from(&manifest.traces[0].padded_input_path);
    let mut first_input = load_input(&first_trace_path)?;
    compact_score_table(&mut first_input, Some(manifest.shape.node_count))?;
    let global_node_path = PathBuf::from(&first_input.global_artifacts.node_table_path);
    let global_graph_path = PathBuf::from(&first_input.global_artifacts.graph_rows_path);
    let (node_merkle, graph_merkle) = check_commit_data(
        &global_node_path,
        &global_graph_path,
        first_input.query.dimension,
    )?;

    let row_neighbor_lens = manifest.shape.row_neighbor_lens();
    let graph_static_enabled = vec![true; row_neighbor_lens.len()];
    let upper_ref_unused_cap = manifest.shape.node_count;
    let ef_ref_unused_cap = 0;
    let visit_len = manifest.shape.visit_len;
    anyhow::ensure!(
        visit_len == manifest.shape.max_rounds * manifest.shape.round_neighbor_cap,
        "manifest shape mismatch: visit_len={} but max_rounds*round_neighbor_cap={}",
        visit_len,
        manifest.shape.max_rounds * manifest.shape.round_neighbor_cap
    );

    let setup_t0 = Instant::now();

    let dist_t0 = Instant::now();
    let (dist_data, dist_targets, dist_stats) = build_score_open_circuit_with_options(
        use_zk,
        manifest.shape.node_count,
        first_input.query.dimension,
        node_merkle.tree_height(),
        score_node_id_mode,
        score_dot_formula,
    );
    let dist_setup_sec = dist_t0.elapsed().as_secs_f64();

    let upper_t0 = Instant::now();
    let (upper_data, upper_targets, upper_stats) = build_upper_circuit(
        use_zk,
        manifest.shape.node_count,
        manifest.shape.max_upper_steps,
        manifest.shape.upper_neighbor_cap,
        graph_merkle.tree_height(),
        upper_ref_unused_cap,
        score_node_id_mode,
    );
    let upper_setup_sec = upper_t0.elapsed().as_secs_f64();

    let ef_t0 = Instant::now();
    let (ef_data, ef_targets, ef_stats) = build_ef_search_circuit_with_options(
        use_zk,
        manifest.shape.node_count,
        &row_neighbor_lens,
        manifest.shape.max_upper_steps,
        manifest.shape.max_rounds,
        first_input.ef_search.ef_search as usize,
        first_input.ef_search.k as usize,
        visit_len,
        manifest.shape.c_final_len,
        manifest.shape.w_final_len,
        graph_merkle.tree_height(),
        ef_ref_unused_cap,
        &graph_static_enabled,
        score_node_id_mode,
    );
    let ef_setup_sec = ef_t0.elapsed().as_secs_f64();

    let final_t0 = Instant::now();
    let (final_data, final_targets) = build_final_agg_circuit(&dist_data, &upper_data, &ef_data);
    let final_agg_setup_sec = final_t0.elapsed().as_secs_f64();

    let setup = SetupReport {
        shape: manifest.shape.clone(),
        trace_count: manifest.traces.len(),
        use_zk,
        parallel_subproofs,
        circuit_fingerprint: ROUTE_CIRCUIT_FINGERPRINT.to_string(),
        score_open_dummy_vector_zero_constraints: false,
        score_open_dot_formula: score_formula_name(score_dot_formula).to_string(),
        score_dot_product_gate_mode: "custom_degree2_shared_query_four_lane_v1".to_string(),
        custom_dot_product_gate: true,
        uses_standard_plonky2_specialized_hash_and_range_gates: true,
        score_open_pre_padding_gate_count: dist_stats.pre_padding_gate_count,
        score_open_node_merkle_gate_count: dist_stats.node_merkle_opening_gate_count,
        score_open_node_merkle_leaf_hash_gate_count: dist_stats.node_merkle_leaf_hash_gate_count,
        score_open_node_merkle_path_gate_count: dist_stats.node_merkle_path_gate_count,
        score_open_formula_gate_count: dist_stats.score_formula_gate_count,
        score_open_score_table_hash_gate_count: dist_stats.score_table_hash_gate_count,
        upper_pre_padding_gate_count: upper_stats.pre_padding_gate_count,
        upper_score_table_hash_gate_count: upper_stats.score_table_hash_gate_count,
        upper_graph_merkle_gate_count: upper_stats.graph_merkle_opening_gate_count,
        upper_graph_merkle_leaf_hash_gate_count: upper_stats.graph_merkle_leaf_hash_gate_count,
        upper_graph_merkle_path_gate_count: upper_stats.graph_merkle_path_gate_count,
        upper_dist_ref_gate_count: upper_stats.dist_ref_membership_gate_count,
        ef_search_pre_padding_gate_count: ef_stats.pre_padding_gate_count,
        ef_search_score_table_hash_gate_count: ef_stats.score_table_hash_gate_count,
        ef_search_graph_merkle_gate_count: ef_stats.graph_merkle_opening_gate_count,
        ef_search_graph_merkle_leaf_hash_gate_count: ef_stats.graph_merkle_leaf_hash_gate_count,
        ef_search_graph_merkle_path_gate_count: ef_stats.graph_merkle_path_gate_count,
        ef_search_dist_ref_gate_count: ef_stats.dist_ref_membership_gate_count,
        node_id_range_mode: node_id_mode_name(score_node_id_mode).to_string(),
        ef_graph_static_open_rows: graph_static_enabled
            .iter()
            .filter(|&&enabled| enabled)
            .count(),
        ef_graph_static_padding_rows: graph_static_enabled
            .iter()
            .filter(|&&enabled| !enabled)
            .count(),
        score_table_hash_mode: "poseidon_flat_domain_len_rows_v2".to_string(),
        upper_ef_score_table_path_openings: false,
        upper_ref_unused_cap,
        ef_ref_unused_cap,
        nbr_commit_mode: "flat_visit_transcript_poseidon_v1".to_string(),
        dist_ref_membership_mode:
            "poseidon_fiat_shamir_triple_permutation_table_hash_dual_challenge_v5".to_string(),
        multiset_challenge_count: MULTISET_CHALLENGE_COUNT,
        setup_total_sec: setup_t0.elapsed().as_secs_f64(),
        score_open_setup_sec: dist_setup_sec,
        upper_setup_sec,
        ef_search_setup_sec: ef_setup_sec,
        final_agg_setup_sec,
    };
    eprintln!(
        "[route_experiment_bench] setup complete: traces={} node_count={} rounds={} c_final_len={} w_final_len={}",
        manifest.traces.len(),
        manifest.shape.node_count,
        manifest.shape.max_rounds,
        manifest.shape.c_final_len,
        manifest.shape.w_final_len,
    );

    let node_root = node_merkle.root_u64s();
    let graph_root = graph_merkle.root_u64s();
    let entry_layer = graph_merkle.max_layer();

    let mut traces = Vec::with_capacity(manifest.traces.len());
    let mut total_prove_secs = Vec::with_capacity(manifest.traces.len());
    let mut total_verify_secs = Vec::with_capacity(manifest.traces.len());
    let mut proof_bytes = Vec::with_capacity(manifest.traces.len());
    let mut completed_trace_paths = HashSet::new();
    let requested_trace_paths = manifest
        .traces
        .iter()
        .map(|item| item.padded_input_path.as_str())
        .collect::<HashSet<_>>();

    if resume {
        if let Some(existing) = load_existing_report(&out_path) {
            let compatible = existing.setup.circuit_fingerprint == setup.circuit_fingerprint
                && existing.setup.shape == setup.shape
                && existing.setup.use_zk == setup.use_zk
                && existing.setup.parallel_subproofs == setup.parallel_subproofs
                && existing.setup.node_id_range_mode == setup.node_id_range_mode
                && existing.setup.score_open_dot_formula == setup.score_open_dot_formula;
            if compatible {
                for trace in existing.traces {
                    if !requested_trace_paths.contains(trace.padded_trace_path.as_str()) {
                        continue;
                    }
                    if completed_trace_paths.insert(trace.padded_trace_path.clone()) {
                        total_prove_secs.push(trace.total_prove_wall_sec);
                        total_verify_secs.push(trace.total_verify_sec);
                        proof_bytes.push(trace.proof_bytes_total);
                        traces.push(trace);
                    }
                }
            } else {
                eprintln!(
                    "[route_experiment_bench] ignoring incompatible resume report: existing_fingerprint={:?} current_fingerprint={:?}",
                    existing.setup.circuit_fingerprint, setup.circuit_fingerprint
                );
            }
        }
    }

    write_report(
        &out_path,
        &build_report(
            &manifest.version,
            &setup,
            &traces,
            &total_prove_secs,
            &total_verify_secs,
            &proof_bytes,
        ),
    )?;

    for item in &manifest.traces {
        if completed_trace_paths.contains(&item.padded_input_path) {
            continue;
        }
        eprintln!(
            "[route_experiment_bench] start trace query_index={} path={}",
            item.query_index, item.padded_input_path
        );
        let trace_path = PathBuf::from(&item.padded_input_path);
        let mut input = load_input(&trace_path)?;
        compact_score_table(&mut input, Some(manifest.shape.node_count))?;
        let ef_unused_refs = ef_search_unused_ref_len(&input);
        anyhow::ensure!(
            ef_unused_refs == 0,
            "ef_search score-table invariant violated for query {}: {} real score nodes are not referenced by upper/ef rows or EF queue refs",
            item.query_index,
            ef_unused_refs
        );

        let q_hash = hash_query(&input);
        let score_table_hash = score_table_hash(&input);
        let reference_commitment = commit_digest(node_root, input.commitment_randomness.reference);
        let graph_commitment = commit_digest(graph_root, input.commitment_randomness.graph);
        let nbr_commit = compute_nbr_commit(&input);
        let ef_entry = upper_end(&input);
        let output_model_id = final_output_model_id(&input)?;
        let final_statement = final_statement(
            &reference_commitment,
            &q_hash,
            &graph_commitment,
            output_model_id,
            input.entry_point,
            entry_layer,
            input.ef_search.ef_search,
            input.ef_search.k,
        );

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

        let upper_fill_t0 = Instant::now();
        let mut upper_pw = PartialWitness::<F>::new();
        fill_upper_witness(
            &mut upper_pw,
            &upper_targets,
            &input,
            &score_table_hash,
            &graph_root,
            input.entry_point,
            ef_entry,
            entry_layer,
            &graph_merkle,
        );
        let upper_fill_sec = upper_fill_t0.elapsed().as_secs_f64();

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

        let subproof_t0 = Instant::now();
        let (dist_run, upper_run, ef_run) = if parallel_subproofs {
            let ((dist_run, upper_run), ef_run) = join(
                || {
                    join(
                        || {
                            prove_module(&dist_data, dist_pw).with_context(|| {
                                format!("prove score_open for query {}", item.query_index)
                            })
                        },
                        || {
                            prove_module(&upper_data, upper_pw).with_context(|| {
                                format!("prove upper for query {}", item.query_index)
                            })
                        },
                    )
                },
                || {
                    prove_module(&ef_data, ef_pw)
                        .with_context(|| format!("prove ef_search for query {}", item.query_index))
                },
            );
            (dist_run?, upper_run?, ef_run?)
        } else {
            let dist_run = prove_module(&dist_data, dist_pw)
                .with_context(|| format!("prove score_open for query {}", item.query_index))?;
            let upper_run = prove_module(&upper_data, upper_pw)
                .with_context(|| format!("prove upper for query {}", item.query_index))?;
            let ef_run = prove_module(&ef_data, ef_pw)
                .with_context(|| format!("prove ef_search for query {}", item.query_index))?;
            (dist_run, upper_run, ef_run)
        };
        let subproof_prove_wall_sec = subproof_t0.elapsed().as_secs_f64();

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

        let final_prove_t0 = Instant::now();
        let final_proof = final_data
            .prove(final_pw)
            .with_context(|| format!("prove final_agg for query {}", item.query_index))?;
        let final_prove_sec = final_prove_t0.elapsed().as_secs_f64();

        let final_verify_t0 = Instant::now();
        final_data
            .verify(final_proof.clone())
            .with_context(|| format!("verify final_agg for query {}", item.query_index))?;
        let final_verify_sec = final_verify_t0.elapsed().as_secs_f64();
        anyhow::ensure!(
            final_proof.public_inputs == final_statement.public_inputs(),
            "final_agg public inputs do not match the reported statement for query {}",
            item.query_index
        );

        let mut modules = BTreeMap::new();
        modules.insert(
            "score_open".to_string(),
            module_report(dist_fill_sec, &dist_run, &dist_data),
        );
        modules.insert(
            "upper".to_string(),
            module_report(upper_fill_sec, &upper_run, &upper_data),
        );
        modules.insert(
            "ef_search".to_string(),
            module_report(ef_fill_sec, &ef_run, &ef_data),
        );
        modules.insert(
            "final_agg".to_string(),
            ModuleBenchReport {
                witness_fill_sec: final_fill_sec,
                prove_sec: final_prove_sec,
                verify_sec: final_verify_sec,
                proof_bytes: final_proof.to_bytes().len(),
                public_inputs: final_proof.public_inputs.len(),
                gate_rows: gate_rows(&final_data),
                degree_bits: final_data.common.degree_bits(),
            },
        );

        let witness_fill_total_sec = dist_fill_sec + upper_fill_sec + ef_fill_sec;
        let total_prove_wall_sec =
            witness_fill_total_sec + subproof_prove_wall_sec + final_fill_sec + final_prove_sec;
        let total_verify_sec =
            dist_run.verify_sec + upper_run.verify_sec + ef_run.verify_sec + final_verify_sec;
        let proof_bytes_total = dist_run.proof_bytes
            + upper_run.proof_bytes
            + ef_run.proof_bytes
            + final_proof.to_bytes().len();
        total_prove_secs.push(total_prove_wall_sec);
        total_verify_secs.push(total_verify_sec);
        proof_bytes.push(proof_bytes_total);

        completed_trace_paths.insert(item.padded_input_path.clone());
        traces.push(TraceBenchReport {
            query_index: item.query_index,
            actual: item.actual.clone(),
            padded_trace_path: trace_path.display().to_string(),
            final_output_model_id: output_model_id,
            final_statement,
            modules,
            witness_fill_total_sec,
            subproof_parallel_prove_wall_sec: subproof_prove_wall_sec,
            final_agg_fill_sec: final_fill_sec,
            final_agg_prove_sec: final_prove_sec,
            final_verify_sec,
            total_prove_wall_sec,
            total_verify_sec,
            proof_bytes_total,
        });

        write_report(
            &out_path,
            &build_report(
                &manifest.version,
                &setup,
                &traces,
                &total_prove_secs,
                &total_verify_secs,
                &proof_bytes,
            ),
        )?;
        eprintln!(
            "[route_experiment_bench] finished trace query_index={} proof_bytes_total={} total_prove_wall_sec={:.3}",
            item.query_index,
            traces.last().map(|t| t.proof_bytes_total).unwrap_or(0),
            traces.last().map(|t| t.total_prove_wall_sec).unwrap_or(0.0),
        );
    }

    let report = build_report(
        &manifest.version,
        &setup,
        &traces,
        &total_prove_secs,
        &total_verify_secs,
        &proof_bytes,
    );
    write_report(&out_path, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
