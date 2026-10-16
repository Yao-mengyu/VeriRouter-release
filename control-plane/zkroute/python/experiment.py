from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

ZKROUTE_ROOT = Path(__file__).resolve().parents[1]
ROOT = ZKROUTE_ROOT.parents[1]

import core  # noqa: E402
import export_trace  # noqa: E402


DEFAULT_RAW_DATA_DIR = ZKROUTE_ROOT / "raw_data"
DEFAULT_CACHE_DIR = ZKROUTE_ROOT / "cache" / "zkHNSW_new"
DEFAULT_EXPORT_OUT = ZKROUTE_ROOT / "out" / "route_experiment_export"
DEFAULT_PAD_DIR = ZKROUTE_ROOT / "out" / "route_experiment_padded"
DEFAULT_OUT = ROOT / "out" / "route_experiment_overhead.json"
DEFAULT_ROUTE_BIN_DIR = ROOT / "out" / "route_inputs"
ROUTE_MANIFEST = "route_experiment_manifest.json"
ROUTE_MANIFEST_VERSION = "route_experiment_manifest_v4_input_provenance"
EXPORT_TRACE_DIRNAME = "traces"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary = Path(handle.name)
    temporary.replace(path)


# These values define both the exporter command and its resume identity.
EXPORT_DEFAULTS = {
    "raw_data_dir": str(DEFAULT_RAW_DATA_DIR),
    "train_parquet": "train-00000-of-00001-embedding.parquet",
    "test_parquet": "train-00000-of-00001-test.parquet",
    "embedding_id_csv": "train-00000-of-00001-embedding.id.csv",
    "cache_dir": str(DEFAULT_CACHE_DIR),
    "n_train": 0, "synthetic_train_seed": 12345,
    "train_subset_mode": "prefix", "dense_coverage_prefix_size": 5000,
    "dense_coverage_anchor_count": 36, "dense_coverage_candidate_count": 1024,
    "n_test": 0, "query_seed": 0,
    "coverage_probe_count": 0, "coverage_probe_seed": 20260722, "coverage_probe_noise": 0.005,
    "m": 16, "ef_construction": 200, "ef_search": 64, "k": 20,
    "faiss_seed": 12345, "faiss_threads": 1,
    "scheme": "affine", "proj_dim": 384, "proj_seed": 0, "proj_percentile": 99.9,
    "synthetic_chunk_rows": 8192, "faiss_add_batch_rows": 50000, "trace_source": "replay",
}
EXPORT_CHOICES = {
    "train_subset_mode": ("prefix", "dense-coverage"),
    "scheme": ("affine", "proj_affine"), "trace_source": ("auto", "direct", "replay"),
}
EXPORT_FLAGS = ("allow_synthetic_train_extension", "skip_real_search_comparison")
SHAPE_CAPS = {
    "node_count_cap": "node_count_cap", "score_node_count_cap": "node_count",
    "upper_steps_cap": "max_upper_steps", "visit_len_cap": "visit_len",
    "c_final_len_cap": "c_final_len", "w_final_len_cap": "w_final_len",
}
QUERY_METADATA = ("query_position", "query_index", "query_source", "query_family", "query_vector_sha256")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verirouter.py experiment", description="Prepare fixed-shape route input bundles and optionally benchmark route proofs.")
    for name, default in EXPORT_DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default,
                            choices=EXPORT_CHOICES.get(name), help={
                                "n_train": "Number of train vectors; 0 uses all real vectors.",
                                "n_test": "Sampled held-out query pool size; 0 uses all queries.",
                            }.get(name))
    for name in EXPORT_FLAGS:
        parser.add_argument("--" + name.replace("_", "-"), action="store_true")
    for name, default in {
        "export_out_dir": DEFAULT_EXPORT_OUT, "pad_dir": DEFAULT_PAD_DIR,
        "out": DEFAULT_OUT, "route_bin_dir": DEFAULT_ROUTE_BIN_DIR,
    }.items():
        parser.add_argument("--" + name.replace("_", "-"), default=str(default))
    for name in SHAPE_CAPS:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=0,
                            help=("Legacy bucket label; --score-node-count-cap sets the score circuit capacity."
                                  if name == "node_count_cap" else
                                  "Fixed bucket capacity; 0 derives it from the selected traces."))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--query-indices-file", default="", help="File of sampled query positions.")
    parser.add_argument("--resume", action="store_true", help="Reuse inputs with matching data and settings.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs without proving.")
    parser.add_argument("--scan-only", action="store_true", help="Record actual trace sizes without padding or proving.")
    parser.add_argument("--score-node-id-mode", choices=("u32", "wide"), default="wide")
    parser.add_argument("--binary", default="", help="Prebuilt route_experiment_bench binary.")
    return parser.parse_args(argv)


def export_parameters(args: argparse.Namespace) -> dict[str, Any]:
    values = {name: getattr(args, name) for name in (*EXPORT_DEFAULTS, *EXPORT_FLAGS)}
    for name in ("raw_data_dir", "cache_dir"):
        values[name] = str(Path(values[name]).resolve())
    return values


def ensure_route_binary(binary_override: str) -> Path:
    if binary_override:
        return Path(binary_override).resolve()
    manifest = ZKROUTE_ROOT / "rust" / "Cargo.toml"
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(manifest.parent / "target")))
    if not target.is_absolute():
        target = ROOT / target
    binary = target / "release" / "route_experiment_bench"
    subprocess.run(
        [
            "cargo",
            "+nightly",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--bin",
            "route_experiment_bench",
        ],
        check=True,
        cwd=ROOT,
    )
    return binary


def parse_peak_rss_bytes(stderr_text: str) -> int | None:
    linux = re.search(
        r"(?im)^\s*maximum resident set size \(kbytes\):\s*(\d+)\s*$",
        stderr_text,
    )
    if linux:
        return int(linux.group(1)) * 1024
    for pattern in (
        r"(?im)^\s*(\d+)\s+maximum resident set size\s*$",
        r"(?im)^\s*maximum resident set size:\s*(\d+)\s*$",
    ):
        match = re.search(pattern, stderr_text)
        if match:
            return int(match.group(1))
    return None


def load_query_indices(path: Path) -> list[int]:
    values: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.replace(",", " ").split():
            values.append(int(token))
    if not values:
        raise ValueError(f"no query indices found in {path}")
    return values


def export_trace_record_path(export_out_dir: Path, trace_query_pos: int) -> Path:
    return export_out_dir / EXPORT_TRACE_DIRNAME / f"export_trace_query_{int(trace_query_pos)}.json"


def padded_trace_path(pad_dir: Path, query_index: int) -> Path:
    return pad_dir / f"proof_input_padded_query_{int(query_index)}.json"


def run_export(args: argparse.Namespace, trace_query_pos: int) -> dict[str, Any]:
    export_run_dir = Path(args.export_out_dir).resolve() / "runs" / Path(args.out).stem
    export_run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
           "--out-dir", str(Path(args.export_out_dir).resolve()),
           "--results-dir", str(export_run_dir), "--trace-query-pos", str(trace_query_pos)]
    for name, value in export_parameters(args).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    if args.refresh_cache:
        cmd.append("--refresh-cache")
    export_trace.main(cmd)
    return load_json(export_run_dir / "export_report.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_resume_key(args: argparse.Namespace, trace_query_pos: int | None = None) -> dict[str, Any]:
    parameters = export_parameters(args)
    raw = Path(parameters["raw_data_dir"])
    files = [args.train_parquet, args.test_parquet, args.embedding_id_csv, "train.csv"]
    return {
        "version": "route_export_resume_v4_input_provenance",
        "input_semantics": core.INPUT_SEMANTICS_VERSION,
        "export_cache_version": export_trace.CACHE_VERSION,
        "parameters": parameters,
        "inputs": [core.file_fingerprint(raw / name) for name in files],
        "trace_query_pos": trace_query_pos,
    }


def load_resumable_export_trace(
    args: argparse.Namespace,
    export_out_dir: Path,
    trace_query_pos: int,
) -> dict[str, Any] | None:
    record_path = export_trace_record_path(export_out_dir, trace_query_pos)
    if not record_path.exists():
        return None
    payload = load_json(record_path)
    if payload.get("route_experiment_resume_key") != export_resume_key(args, trace_query_pos):
        return None
    files = payload.get("files", {})
    if not isinstance(files, dict):
        return None
    proof_input_path = files.get("proof_input")
    search_trace_path = files.get("search_trace")
    if not proof_input_path or not search_trace_path:
        return None
    proof_input_file = Path(str(proof_input_path))
    search_trace_file = Path(str(search_trace_path))
    if not proof_input_file.exists() or not search_trace_file.exists():
        return None
    return payload


def save_export_trace(
    args: argparse.Namespace,
    export_out_dir: Path,
    trace_query_pos: int,
    payload: dict[str, Any],
) -> None:
    write_json(
        export_trace_record_path(export_out_dir, trace_query_pos),
        {
            **payload,
            "route_experiment_resume_key": export_resume_key(args, trace_query_pos),
        },
    )


def load_resumable_pad_state(
    pad_dir: Path, export_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], Path] | None:
    manifest_path = pad_dir / ROUTE_MANIFEST
    padded_graph_rows_path = pad_dir / "global_graph_rows_padded.jsonl"
    if not manifest_path.exists() or not padded_graph_rows_path.exists():
        return None

    manifest = load_json(manifest_path)
    if (manifest.get("version") != ROUTE_MANIFEST_VERSION
            or manifest.get("export_context") != export_context):
        return None
    traces = manifest.get("traces", [])
    shape = manifest.get("shape", {})
    if not isinstance(traces, list) or not isinstance(shape, dict):
        return None

    trace_map: dict[int, dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        try:
            query_position = int(trace["query_position"])
        except (KeyError, TypeError, ValueError):
            continue
        padded_input_path = Path(str(trace.get("padded_input_path", "")))
        if not padded_input_path.exists():
            continue
        trace_map[query_position] = trace

    if not trace_map:
        return None
    return shape, trace_map, padded_graph_rows_path


def shape_matches_requested_caps(shape: dict[str, Any], args: argparse.Namespace) -> bool:
    for name, field in SHAPE_CAPS.items():
        cap = getattr(args, name)
        if cap > 0 and int(shape.get(field, 0)) != cap:
            return False
    return True


def trace_dimensions(proof_input: dict[str, Any]) -> dict[str, int]:
    upper = len(proof_input["upper_trace"])
    ef = proof_input["ef_search"]
    rows = proof_input["expand_rows"]
    return {
        "node_count": len(proof_input["nodes"]), "upper_steps": upper,
        "ef_rounds": len(ef["expansions"]), "expand_row_count": len(rows),
        "neighbor_visits": len(ef.get("neighbor_visits", [])),
        "c_final_len": len(ef["c_pq"]["final_refs"]), "w_final_len": len(ef["w_pq"]["final_refs"]),
        "upper_neighbor_cap": max((len(row["neighbors"]) for row in rows[:upper]), default=0),
        "round_neighbor_cap": max((len(row["neighbors"]) for row in rows[upper:]), default=0),
    }


def apply_shape_caps(shape: dict[str, int], args: argparse.Namespace) -> None:
    shape["node_count_cap"] = shape["node_count"]
    for name, field in SHAPE_CAPS.items():
        cap = getattr(args, name)
        if name == "visit_len_cap" or cap <= 0:
            continue
        if cap < shape[field]:
            raise ValueError(f"{name} {cap} is smaller than required {shape[field]}")
        shape[field] = cap
    shape["score_node_count_cap"] = shape["node_count"]
    width = shape["round_neighbor_cap"]
    if args.visit_len_cap > 0:
        if width == 0 or args.visit_len_cap % width:
            raise ValueError("visit_len_cap requires a positive round width and must be divisible by it")
        rounds = args.visit_len_cap // width
        if rounds < shape["max_rounds"]:
            raise ValueError("visit_len_cap is too small for the exported rounds")
        shape["max_rounds"] = rounds
    shape["visit_len"] = shape["max_rounds"] * width


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{lineno}")
            yield payload


def write_zlib_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(text.encode("utf-8"), level=9))


def write_bin_json(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    write_zlib_text(path, raw)


def selected_model_id(proof_input: dict[str, Any]) -> int:
    declared_selected_model_id = proof_input.get("selected_model_id")
    if declared_selected_model_id is not None:
        return int(declared_selected_model_id)
    topk = proof_input.get("ef_search", {}).get("final_topk", [])
    if not topk:
        raise ValueError("proof input has no ef_search.final_topk entries")
    return int(topk[0]["model_id"])


def export_demo_route_bins(
    *,
    route_bin_dir: Path,
    manifest_traces: list[dict[str, Any]],
    padded_graph_rows_path: Path,
    shape: dict[str, int],
    n_train: int,
    node_count_cap: int,
    visit_len_cap: int,
) -> list[dict[str, Any]]:
    route_bin_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    graph_rows_text = padded_graph_rows_path.read_text(encoding="utf-8")
    for trace in manifest_traces:
        query_index = int(trace["query_index"])
        proof_source = Path(str(trace["padded_input_path"])).resolve()
        proof_input = load_json(proof_source)
        node_table_source = Path(str(proof_input["global_artifacts"]["node_table_path"])).resolve()
        query_dir = route_bin_dir / f"query_{query_index}"
        proof_bin = query_dir / "proof_input.bin"
        node_bin = query_dir / "global_quantized_nodes.bin"
        graph_bin = query_dir / "global_graph_rows.bin"
        route_input_bin = route_bin_dir / f"route_input_query_{query_index}.bin"

        proof_payload = deepcopy(proof_input)
        proof_payload["global_artifacts"]["node_table_path"] = node_bin.name
        proof_payload["global_artifacts"]["graph_rows_path"] = graph_bin.name
        write_bin_json(proof_bin, proof_payload)
        write_zlib_text(node_bin, node_table_source.read_text(encoding="utf-8"))
        write_zlib_text(graph_bin, graph_rows_text)

        route_input = {
            "expected_output_model_id": selected_model_id(proof_payload),
            "proof_input_path": str(proof_bin),
            "source": {
                "bucket": (
                    f"scorecap_{int(shape['node_count'])}"
                    f"__nodecap_{int(node_count_cap or shape['node_count'])}"
                    f"__visitcap_{int(shape['visit_len'])}"
                ),
                "experiment": "control-plane route overhead",
                "kind": "experiment_route_input",
                "n_train": int(n_train),
                "score_node_count": int(shape["node_count"]),
                "score_node_count_cap": int(shape.get("score_node_count_cap", shape["node_count"])),
                "node_count": int(node_count_cap or shape["node_count"]),
                "node_count_cap": int(node_count_cap or shape["node_count"]),
                "query_index": query_index,
                "ef_rounds": int(shape["max_rounds"]),
                "neighbor_visits": int(shape["visit_len"]),
                "visit_len_cap": int(visit_len_cap or shape["visit_len"]),
            },
        }
        write_bin_json(route_input_bin, route_input)
        exported.append(
            {
                "query_index": query_index,
                "route_input_bin": str(route_input_bin),
                "proof_input_bin": str(proof_bin),
            }
        )
    write_json(route_bin_dir / "route_inputs_manifest.json", {"version": "route_inputs_v1", "inputs": exported})
    return exported


def dummy_ref() -> dict[str, int]:
    return {
        "node_id": core.NODE_ID_SENTINEL,
        "score_enc": 0,
        "model_id": 0,
    }


def build_padded_graph_rows_sidecar(
    source_path: Path,
    out_path: Path,
    upper_neighbor_cap: int,
    round_neighbor_cap: int,
) -> Path:
    sent_id = core.NODE_ID_SENTINEL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source_path):
            padded = deepcopy(row)
            layer = int(padded.get("layer", 0))
            target_cap = upper_neighbor_cap if layer > 0 else round_neighbor_cap
            padded["src"] = core.canonical_node_id(int(padded.get("src", sent_id)))
            neighbors = [core.canonical_node_id(int(x)) for x in padded.get("neighbors", [])]
            if len(neighbors) > target_cap:
                raise ValueError(
                    f"row src={padded.get('src')} layer={layer} has {len(neighbors)} neighbors > cap {target_cap}"
                )
            while len(neighbors) < target_cap:
                neighbors.append(sent_id)
            padded["neighbors"] = neighbors
            handle.write(json.dumps(padded, ensure_ascii=False))
            handle.write("\n")
    return out_path


def shared_padded_graph_rows_sidecar(
    *,
    cache_dir: Path,
    source_path: Path,
    upper_neighbor_cap: int,
    round_neighbor_cap: int,
) -> Path:
    source = source_path.resolve()
    stat = source.stat()
    key_payload = {
        "version": "padded_graph_rows_shared_v1",
        "source_path": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "upper_neighbor_cap": int(upper_neighbor_cap),
        "round_neighbor_cap": int(round_neighbor_cap),
    }
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    shared_path = (
        cache_dir
        / "padded_graph_rows"
        / digest
        / "global_graph_rows_padded.jsonl"
    )
    if shared_path.exists():
        return shared_path

    shared_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = shared_path.with_name(
        f"{shared_path.name}.tmp.{os.getpid()}"
    )
    build_padded_graph_rows_sidecar(
        source,
        temporary_path,
        int(upper_neighbor_cap),
        int(round_neighbor_cap),
    )
    temporary_path.replace(shared_path)
    return shared_path


def link_shared_graph_rows(shared_path: Path, local_path: Path) -> Path:
    shared = shared_path.resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() or local_path.is_symlink():
        if local_path.is_symlink() and local_path.resolve() == shared:
            return local_path
        local_path.unlink()
    relative_target = os.path.relpath(shared, local_path.parent)
    local_path.symlink_to(relative_target)
    return local_path


def graph_row_caps(source_path: Path) -> tuple[int, int]:
    upper_neighbor_cap = 0
    round_neighbor_cap = 0
    for row in iter_jsonl(source_path):
        layer = int(row.get("layer", 0))
        count = len(row.get("neighbors", []))
        if layer > 0:
            upper_neighbor_cap = max(upper_neighbor_cap, count)
        else:
            round_neighbor_cap = max(round_neighbor_cap, count)
    return upper_neighbor_cap, round_neighbor_cap


def node_vector_map(proof_input: dict[str, Any]) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for node in proof_input["nodes"]:
        node_id = core.canonical_node_id(int(node["node_id"]))
        if core.is_node_id_sentinel(node_id):
            continue
        out[int(node_id)] = core.np.ascontiguousarray(
            core.np.array(node["vector_u16"], dtype=core.np.uint16)
        )
    return out


def pad_trace_bundle(
    proof_input: dict[str, Any],
    search_trace: dict[str, Any],
    shape: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sent_id = core.NODE_ID_SENTINEL
    dref = dummy_ref()

    original_node_count = len(proof_input["nodes"])
    original_upper_steps = len(proof_input["upper_trace"])
    original_rounds = len(proof_input["ef_search"]["expansions"])
    original_expand_rows = len(proof_input["expand_rows"])
    original_neighbor_visits = len(proof_input["ef_search"]["neighbor_visits"])

    query_u16 = core.np.ascontiguousarray(core.np.array(proof_input["query"]["vector_u16"], dtype=core.np.uint16))
    vectors_u16 = node_vector_map(proof_input)
    score_map = core.compute_quantized_scores(
        query_u16=query_u16,
        node_vectors_u16=vectors_u16,
        zero_point=int(proof_input["quantizer"]["zero_point"]),
    )
    real_nodes_by_id: dict[int, dict[str, Any]] = {}
    for node in proof_input["nodes"]:
        node_id = core.canonical_node_id(int(node["node_id"]))
        if core.is_node_id_sentinel(node_id):
            continue
        clone = deepcopy(node)
        clone["node_id"] = int(node_id)
        real_nodes_by_id.setdefault(int(node_id), clone)
    compact_nodes = [real_nodes_by_id[node_id] for node_id in sorted(real_nodes_by_id)]
    ref_map = {
        int(node["node_id"]): {
            "node_id": int(node["node_id"]),
            "score_enc": int(node["score_enc"]),
            "model_id": int(node["model_id"]),
        }
        for node in compact_nodes
    }
    if not compact_nodes:
        raise ValueError("proof_input.nodes is empty")
    if len(compact_nodes) > int(shape["node_count"]):
        raise ValueError(
            f"unique score node count {len(compact_nodes)} exceeds node_count cap {shape['node_count']}"
        )
    dummy_node = {
        "node_id": int(sent_id),
        "model_id": 0,
        "vector_u16": [0 for _ in range(int(proof_input["query"]["dimension"]))],
        "score_enc": 0,
    }
    padded_nodes = deepcopy(compact_nodes)
    while len(padded_nodes) < shape["node_count"]:
        padded_nodes.append(deepcopy(dummy_node))

    def clean_ref(value: dict[str, Any]) -> dict[str, int]:
        node_id = core.canonical_node_id(int(value.get("node_id", sent_id)))
        if core.is_node_id_sentinel(node_id):
            return deepcopy(dref)
        return deepcopy(ref_map[int(node_id)])

    padded_upper_rows = []
    for row in proof_input["expand_rows"][:original_upper_steps]:
        padded_row = deepcopy(row)
        padded_row["src"] = clean_ref(padded_row["src"])
        padded_row["neighbors"] = [clean_ref(value) for value in padded_row["neighbors"]]
        while len(padded_row["neighbors"]) < shape["upper_neighbor_cap"]:
            padded_row["neighbors"].append(deepcopy(dref))
        padded_upper_rows.append(padded_row)
    while len(padded_upper_rows) < shape["max_upper_steps"]:
        padded_upper_rows.append(
            {
                "layer": 1,
                "src": deepcopy(dref),
                "neighbors": [deepcopy(dref) for _ in range(shape["upper_neighbor_cap"])],
            }
        )

    padded_upper_trace = [
        {"selected": clean_ref(step["selected"])} for step in proof_input["upper_trace"]
    ]
    while len(padded_upper_trace) < shape["max_upper_steps"]:
        padded_upper_trace.append({"selected": deepcopy(dref)})

    level0_trace = deepcopy(search_trace["level0"])
    padded_expansions = []
    for expansion in level0_trace["expansions"]:
        padded_expansion = deepcopy(expansion)
        padded_expansion["pop_id"] = core.canonical_node_id(int(padded_expansion.get("pop_id", sent_id)))
        padded_expansion["neighbors"] = [
            core.canonical_node_id(int(x)) for x in padded_expansion.get("neighbors", [])
        ]
        padded_expansion["neighbor_visits"] = [
            {
                "node_id": core.canonical_node_id(int(visit.get("node_id", nb))),
                "is_new": bool(visit.get("is_new", False)),
            }
            for visit, nb in zip(
                padded_expansion.get("neighbor_visits", []),
                padded_expansion["neighbors"],
            )
        ]
        while len(padded_expansion["neighbors"]) < shape["round_neighbor_cap"]:
            padded_expansion["neighbors"].append(sent_id)
            padded_expansion["neighbor_visits"].append({"node_id": sent_id, "is_new": False})
        padded_expansions.append(padded_expansion)
    while len(padded_expansions) < shape["max_rounds"]:
        padded_expansions.append(
            {
                "pop_id": sent_id,
                "neighbors": [sent_id for _ in range(shape["round_neighbor_cap"])],
                "neighbor_visits": [
                    {"node_id": sent_id, "is_new": False}
                    for _ in range(shape["round_neighbor_cap"])
                ],
            }
        )
    level0_trace["expansions"] = padded_expansions
    level0_trace["round_width"] = int(shape["round_neighbor_cap"])

    padded_ef_search = core.level0_trace_from_quantized_scores(
        level0_trace,
        score_map,
        ref_map,
        score_node_capacity=len(padded_nodes),
    )
    target_c_final_len = int(shape.get("c_final_len", len(padded_ef_search["c_pq"]["final_refs"])))
    target_w_final_len = int(shape.get("w_final_len", len(padded_ef_search["w_pq"]["final_refs"])))
    if len(padded_ef_search["c_pq"]["final_refs"]) > target_c_final_len:
        raise ValueError(
            f"c_final_len overflow after padding: got {len(padded_ef_search['c_pq']['final_refs'])}, cap {target_c_final_len}"
        )
    if len(padded_ef_search["w_pq"]["final_refs"]) > target_w_final_len:
        raise ValueError(
            f"w_final_len overflow after padding: got {len(padded_ef_search['w_pq']['final_refs'])}, cap {target_w_final_len}"
        )
    while len(padded_ef_search["c_pq"]["final_refs"]) < target_c_final_len:
        padded_ef_search["c_pq"]["final_refs"].append(deepcopy(dref))
        padded_ef_search["c_pq"]["final_src_times"].append(0)
        padded_ef_search["c_pq"]["final_dummy_ordinals"].append(0)
        padded_ef_search["c_pq"]["final_tie_positions"].append(0)
    if len(padded_ef_search["w_pq"]["final_refs"]) != target_w_final_len:
        raise ValueError(
            "w_final_len must equal ef_search: WorkQ has fixed cardinality and "
            "does not admit unauthenticated final padding items"
        )

    position_max = core.c_position_limit(
        len(padded_expansions),
        int(shape["round_neighbor_cap"]),
    )
    core.assert_packed_priority_domain(
        position_max,
        int(padded_ef_search["ef_search"]),
    )
    c_pq = padded_ef_search["c_pq"]
    refs = c_pq["del_refs"] + c_pq["final_refs"]
    ties = c_pq["del_tie_positions"] + c_pq["final_tie_positions"]
    enabled = [not core.is_node_id_sentinel(int(ref["node_id"])) for ref in refs]
    keys = [core.c_packed_priority_key(int(ref["score_enc"]), int(tie), position_max) if active else 0
            for ref, tie, active in zip(refs, ties, enabled)]
    times = [core.c_pop_position(r, shape["round_neighbor_cap"]) if active else 0
             for r, active in enumerate(enabled[:len(padded_expansions)])]
    padded_ef_search["c_prev_lower"] = core.build_prev_lower_trace(
        keys, times + c_pq["final_src_times"], enabled)

    padded_round_rows = []
    for expansion in padded_expansions:
        pop_id = int(expansion["pop_id"])
        src_ref = deepcopy(dref) if core.is_node_id_sentinel(pop_id) else deepcopy(ref_map[pop_id])
        neighbor_refs = [
            deepcopy(dref) if core.is_node_id_sentinel(int(node_id)) else deepcopy(ref_map[int(node_id)])
            for node_id in expansion["neighbors"]
        ]
        padded_round_rows.append({"layer": 0, "src": src_ref, "neighbors": neighbor_refs})

    padded_proof_input = deepcopy(proof_input)
    padded_proof_input["nodes"] = padded_nodes
    padded_proof_input["expand_rows"] = padded_upper_rows + padded_round_rows
    padded_proof_input["upper_trace"] = padded_upper_trace
    padded_proof_input["ef_search"] = padded_ef_search
    source = deepcopy(padded_proof_input.get("source", {}))
    if not isinstance(source, dict):
        source = {}
    source["score_node_count"] = int(len(padded_nodes))
    source["score_node_count_cap"] = int(shape["score_node_count_cap"])
    source["node_count_cap"] = int(shape["node_count_cap"])
    source["bucket"] = (
        f"scorecap_{int(shape['score_node_count_cap'])}"
        f"__nodecap_{int(shape['node_count_cap'])}"
        f"__visitcap_{int(shape['visit_len'])}"
    )
    padded_proof_input["source"] = source
    padded_proof_input["benchmark_actual"] = {
        "node_count": original_node_count,
        "score_node_count": len(padded_nodes),
        "upper_steps": original_upper_steps,
        "ef_rounds": original_rounds,
        "expand_row_count": original_expand_rows,
        "neighbor_visits": original_neighbor_visits,
    }
    return padded_proof_input, padded_proof_input["benchmark_actual"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    resume = args.resume and not args.refresh_cache
    export_context = export_resume_key(args)
    raw_data_dir = Path(args.raw_data_dir).resolve()
    test_parquet = raw_data_dir / args.test_parquet
    cache_dir = Path(args.cache_dir).resolve()
    export_out_dir = Path(args.export_out_dir).resolve()
    pad_dir = Path(args.pad_dir).resolve()
    out_path = Path(args.out).resolve()
    route_bin_dir = Path(args.route_bin_dir).resolve() if str(args.route_bin_dir).strip() else None
    export_out_dir.mkdir(parents=True, exist_ok=True)
    pad_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_total_queries = export_trace.parquet_row_count(test_parquet)
    heldout_query_pool_count = (
        min(int(args.n_test), heldout_total_queries)
        if int(args.n_test) > 0
        else heldout_total_queries
    )
    coverage_probe_count = max(0, int(args.coverage_probe_count))
    total_queries = heldout_query_pool_count + coverage_probe_count
    if str(args.query_indices_file).strip():
        query_indices_path = Path(str(args.query_indices_file)).resolve()
        query_positions = load_query_indices(query_indices_path)
        if any(idx < 0 or idx >= total_queries for idx in query_positions):
            raise ValueError(f"query index out of range for total_queries={total_queries}: {query_positions}")
    else:
        heldout_query_count = heldout_query_pool_count
        if int(args.limit) > 0:
            heldout_query_count = min(heldout_query_count, int(args.limit))
        query_positions = list(range(heldout_query_count))
        query_positions.extend(
            range(
                heldout_query_pool_count,
                heldout_query_pool_count + coverage_probe_count,
            )
        )
    query_count = len(query_positions)

    exports: list[dict[str, Any]] = []
    manifest_traces: list[dict[str, Any]] = []
    manifest_path = pad_dir / ROUTE_MANIFEST
    padded_graph_rows_path: Path | None = None

    def write_stage(stage: str, extra: dict[str, Any]) -> None:
        payload = {
            "experiment_type": "route_overhead",
            "version": "route_overhead_progress_v1",
            "stage": stage,
            "query_count_total": int(query_count),
            "query_positions": [int(x) for x in query_positions],
            **extra,
        }
        write_json(out_path, payload)

    if not (resume and out_path.exists()):
        write_stage(
            "exporting",
            {
                "exports_completed": 0,
                "export_out_dir": str(export_out_dir),
                "pad_dir": str(pad_dir),
            },
        )
    existing_pad = load_resumable_pad_state(pad_dir, export_context) if resume else None
    resumed_pad = None
    if existing_pad is not None and not args.scan_only:
        existing_shape, existing_trace_map, existing_graph_path = existing_pad
        if all(int(query_position) in existing_trace_map for query_position in query_positions):
            resumed_pad = (
                existing_shape,
                [existing_trace_map[int(query_position)] for query_position in query_positions],
                existing_graph_path,
            )
    if resumed_pad is not None and not shape_matches_requested_caps(resumed_pad[0], args):
        resumed_pad = None
    if resumed_pad is not None:
        max_shape, manifest_traces, padded_graph_rows_path = resumed_pad
        if not manifest_traces:
            raise ValueError("no padded traces available for resume")
        manifest = {
            "version": ROUTE_MANIFEST_VERSION,
            "export_context": export_context,
            "shape": max_shape,
            "traces": manifest_traces,
        }
        write_json(manifest_path, manifest)
    else:
        for trace_query_pos in query_positions:
            payload = (
                load_resumable_export_trace(args, export_out_dir, trace_query_pos)
                if resume
                else None
            )
            if payload is None:
                payload = run_export(args, trace_query_pos)
                save_export_trace(args, export_out_dir, trace_query_pos, payload)
            exports.append(payload)
            if not (resume and out_path.exists()):
                write_stage(
                    "exporting",
                    {
                        "exports_completed": len(exports),
                        "export_out_dir": str(export_out_dir),
                        "pad_dir": str(pad_dir),
                    },
                )

        trace_bundles = []
        max_shape = {
            "node_count": 0,
            "node_count_cap": 0,
            "score_node_count_cap": 0,
            "max_upper_steps": 0,
            "max_rounds": 0,
            "upper_neighbor_cap": 0,
            "round_neighbor_cap": 0,
            "visit_len": 0,
            "c_final_len": 0,
            "w_final_len": 0,
        }
        original_graph_rows_path: Path | None = None
        for export_position, payload in enumerate(exports):
            files = payload["files"]
            proof_input = load_json(Path(files["proof_input"]))
            search_trace = load_json(Path(files["search_trace"]))
            if original_graph_rows_path is None:
                original_graph_rows_path = Path(proof_input["global_artifacts"]["graph_rows_path"]).resolve()
            actual = trace_dimensions(proof_input)
            for field, source in {
                "node_count": "node_count", "max_upper_steps": "upper_steps",
                "max_rounds": "ef_rounds", "visit_len": "neighbor_visits",
                "c_final_len": "c_final_len", "w_final_len": "w_final_len",
                "upper_neighbor_cap": "upper_neighbor_cap", "round_neighbor_cap": "round_neighbor_cap",
            }.items():
                max_shape[field] = max(max_shape[field], actual[source])
            trace_bundles.append(
                {
                    "query_position": int(payload.get("trace_query_pos", query_positions[export_position])),
                    "query_index": int(proof_input["query_index"]),
                    "query_source": str(
                        payload.get("trace_query_source", "heldout")
                    ),
                    "query_family": str(
                        payload.get("trace_query_family", "heldout")
                    ),
                    "query_vector_sha256": str(
                        payload.get("trace_query_vector_sha256", "")
                    ),
                    "proof_input": proof_input,
                    "search_trace": search_trace,
                }
            )

        if original_graph_rows_path is None:
            raise ValueError("no graph rows path discovered from exported traces")

        graph_upper_cap, graph_round_cap = graph_row_caps(original_graph_rows_path)
        max_shape["upper_neighbor_cap"] = max(max_shape["upper_neighbor_cap"], graph_upper_cap)
        max_shape["round_neighbor_cap"] = max(max_shape["round_neighbor_cap"], graph_round_cap)
        if bool(args.scan_only):
            scan_traces = []
            embedding_relations: set[tuple[str, int]] = set()
            for bundle in trace_bundles:
                proof_input = bundle["proof_input"]
                quantizer = (
                    proof_input.get("quantizer")
                    if isinstance(proof_input.get("quantizer"), dict)
                    else {}
                )
                embedding_relations.add(
                    (
                        str(quantizer.get("scheme") or ""),
                        int(quantizer.get("projected_dim") or 0),
                    )
                )
                actual = trace_dimensions(proof_input)
                actual.update(
                    upper_neighbor_cap=max_shape["upper_neighbor_cap"],
                    round_neighbor_cap=max_shape["round_neighbor_cap"],
                    required_visit_cap=actual["ef_rounds"] * max_shape["round_neighbor_cap"],
                )
                scan_traces.append({**{key: bundle[key] for key in QUERY_METADATA}, "actual": actual})
            if len(embedding_relations) != 1:
                raise ValueError(
                    f"scan inputs do not share one embedding relation: {sorted(embedding_relations)}"
                )
            embedding_scheme, embedding_dimension = next(iter(embedding_relations))
            scan_manifest = {
                "version": "route_actual_trace_scan_v1",
                "embedding_scheme": embedding_scheme,
                "embedding_dimension": embedding_dimension,
                "shape": max_shape,
                "traces": scan_traces,
            }
            write_json(manifest_path, scan_manifest)
            write_json(
                out_path,
                {
                    "experiment_type": "route_overhead",
                    "version": "route_actual_trace_scan_v1",
                    "stage": "scanned",
                    "embedding_scheme": embedding_scheme,
                    "embedding_dimension": embedding_dimension,
                    "query_count_total": int(query_count),
                    "query_positions": [int(x) for x in query_positions],
                    "artifacts": {
                        "manifest_path": str(manifest_path),
                        "export_out_dir": str(export_out_dir),
                    },
                    "shape": max_shape,
                    "traces": scan_traces,
                },
            )
            return
        apply_shape_caps(max_shape, args)

        existing_shape: dict[str, Any] = {}
        existing_trace_map: dict[int, dict[str, Any]] = {}
        if existing_pad is not None:
            existing_shape, existing_trace_map, existing_graph_path = existing_pad
        else:
            existing_graph_path = pad_dir / "global_graph_rows_padded.jsonl"
        if (
            existing_graph_path.exists()
            and int(existing_shape.get("upper_neighbor_cap") or -1) == int(max_shape["upper_neighbor_cap"])
            and int(existing_shape.get("round_neighbor_cap") or -1) == int(max_shape["round_neighbor_cap"])
        ):
            padded_graph_rows_path = existing_graph_path
        else:
            shared_graph_rows_path = shared_padded_graph_rows_sidecar(
                cache_dir=cache_dir,
                source_path=original_graph_rows_path,
                upper_neighbor_cap=int(max_shape["upper_neighbor_cap"]),
                round_neighbor_cap=int(max_shape["round_neighbor_cap"]),
            )
            padded_graph_rows_path = link_shared_graph_rows(
                shared_graph_rows_path,
                pad_dir / "global_graph_rows_padded.jsonl",
            )
        reuse_existing_padded = existing_shape == max_shape

        for bundle in trace_bundles:
            query_index = int(bundle["query_index"])
            query_position = int(bundle["query_position"])
            if reuse_existing_padded and query_position in existing_trace_map:
                manifest_traces.append(existing_trace_map[query_position])
                continue
            padded_input, actual = pad_trace_bundle(bundle["proof_input"], bundle["search_trace"], max_shape)
            padded_input["global_artifacts"]["graph_rows_path"] = str(padded_graph_rows_path)
            expected_visit_len = int(max_shape["max_rounds"]) * int(max_shape["round_neighbor_cap"])
            actual_visit_len = len(padded_input["ef_search"]["neighbor_visits"])
            if actual_visit_len != expected_visit_len:
                raise ValueError(
                    f"inconsistent padded neighbor_visits length: got {actual_visit_len}, expected {expected_visit_len}"
                )
            padded_path = padded_trace_path(pad_dir, query_index)
            write_json(padded_path, padded_input)
            if actual_visit_len != max_shape["visit_len"]:
                raise ValueError("inconsistent padded visit_len across traces")
            if len(padded_input["ef_search"]["c_pq"]["final_refs"]) != max_shape["c_final_len"]:
                raise ValueError("inconsistent padded c_final_len across traces")
            if len(padded_input["ef_search"]["w_pq"]["final_refs"]) != max_shape["w_final_len"]:
                raise ValueError("inconsistent padded w_final_len across traces")
            manifest_traces.append(
                {
                    "padded_input_path": str(padded_path),
                    **{key: bundle[key] for key in QUERY_METADATA},
                    "actual": actual,
                }
            )

        if not manifest_traces:
            raise ValueError("no padded traces generated")

        manifest = {
            "version": ROUTE_MANIFEST_VERSION,
            "export_context": export_context,
            "shape": max_shape,
            "traces": manifest_traces,
        }
        write_json(manifest_path, manifest)

    if padded_graph_rows_path is None:
        raise ValueError("no padded graph rows path available")
    route_bin_inputs: list[dict[str, Any]] = []
    if route_bin_dir is not None:
        route_bin_inputs = export_demo_route_bins(
            route_bin_dir=route_bin_dir,
            manifest_traces=manifest_traces,
            padded_graph_rows_path=padded_graph_rows_path,
            shape=max_shape,
            n_train=int(args.n_train),
            node_count_cap=int(args.node_count_cap),
            visit_len_cap=int(args.visit_len_cap),
        )
    if bool(args.prepare_only):
        report = {
            "experiment_type": "route_overhead",
            "version": "route_preparation_v3_active_prev_lower",
            "stage": "prepared",
            "query_count_total": int(query_count),
            "query_positions": [int(x) for x in query_positions],
            "artifacts": {
                "manifest_path": str(manifest_path),
                "pad_dir": str(pad_dir),
                "padded_graph_rows_path": str(padded_graph_rows_path),
                "route_bin_dir": str(route_bin_dir) if route_bin_dir is not None else None,
                "route_bin_inputs": route_bin_inputs,
            },
        }
        write_json(out_path, report)
        print(
            json.dumps(
                {
                    "out": str(out_path),
                    "route_bin_dir": str(route_bin_dir) if route_bin_dir is not None else None,
                    "route_bin_inputs": route_bin_inputs,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not (resume and out_path.exists()):
        write_stage(
            "proving",
            {
                "exports_completed": len(exports),
                "manifest_path": str(manifest_path),
                "trace_count": len(manifest_traces),
            },
        )

    binary = ensure_route_binary(str(args.binary))
    cmd = [
        str(binary),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_path),
        "--zk",
        "--score-node-id-mode",
        str(args.score_node_id_mode),
    ]
    if (resumed_pad is not None and out_path.exists()
            and load_json(out_path).get("export_context") == export_context):
        cmd.append("--resume")
    wrapped_cmd = cmd
    if bool(args.measure_memory):
        wrapped_cmd = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v", *cmd]
    result = subprocess.run(wrapped_cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    report["export_context"] = export_context
    report["memory"] = {
        "enabled": bool(args.measure_memory),
        "peak_rss_bytes_process": parse_peak_rss_bytes(result.stderr) if bool(args.measure_memory) else None,
        "time_l_stderr": result.stderr,
        "time_l_returncode": result.returncode,
    }
    report["artifacts"] = {
        "manifest_path": str(manifest_path),
        "pad_dir": str(pad_dir),
        "padded_graph_rows_path": str(padded_graph_rows_path),
        "route_bin_dir": str(route_bin_dir) if route_bin_dir is not None else None,
        "route_bin_inputs": route_bin_inputs,
        "export_run_dir": str((export_out_dir / "runs" / out_path.stem).resolve()),
    }
    write_json(out_path, report)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "manifest_path": str(manifest_path),
                "trace_count": len(manifest_traces),
                "peak_rss_bytes_process": report["memory"]["peak_rss_bytes_process"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
