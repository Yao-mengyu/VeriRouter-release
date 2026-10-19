#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


EXACT_HNSW_LAYER_RANGE_NODE_LIMIT = 250_000
INPUT_SEMANTICS_VERSION = "canonical_labels_zero_inclusive_quantizer_v1"

SCORE_BITS = 48
SCORE_BIAS = 1 << (SCORE_BITS - 1)
NODE_ID_BITS = 56
POSITION_BITS = 32
NODE_ID_SENTINEL = (1 << NODE_ID_BITS) - 1
LEGACY_NODE_ID_SENTINEL = (1 << 32) - 1
MAX_REAL_NODE_ID = NODE_ID_SENTINEL - 1
POSITION_SCHEMA_PAPER_WORKUPD_V1 = "paper_workupd_v1"
PROOF_INPUT_VERSION = (
    "zkhnsw_new_global_graph_quant_search_v17_physical_workupd_"
    "packed_priority_direct_score_hash_active_prev_lower"
)
GOLDILOCKS_MODULUS = (1 << 64) - (1 << 32) + 1
PACKED_PRIORITY_BITS = 62


def is_node_id_sentinel(node_id: int) -> bool:
    return int(node_id) in {int(NODE_ID_SENTINEL), int(LEGACY_NODE_ID_SENTINEL)}


def canonical_node_id(node_id: int) -> int:
    node_id = int(node_id)
    if is_node_id_sentinel(node_id):
        return int(NODE_ID_SENTINEL)
    if node_id > int(MAX_REAL_NODE_ID):
        raise ValueError(
            f"node_id {node_id} exceeds the 56-bit real-id domain; max real node_id is {MAX_REAL_NODE_ID}"
        )
    return node_id


def validate_commitment_randomness(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise ValueError("commitment_randomness must be an object")
    if "score" in value:
        raise ValueError(
            "commitment_randomness.score is forbidden in v17; "
            "the canonical score table uses a direct unrandomized hash"
        )
    expected = {"reference", "graph"}
    if set(value) != expected:
        raise ValueError(
            "commitment_randomness must contain exactly reference and graph"
        )
    out: dict[str, list[int]] = {}
    for label in ("reference", "graph"):
        digest = value.get(label)
        if not isinstance(digest, list) or len(digest) != 4:
            raise ValueError(f"commitment_randomness.{label} must contain four limbs")
        if any(not isinstance(limb, int) or isinstance(limb, bool) for limb in digest):
            raise ValueError(
                f"commitment_randomness.{label} limbs must be integers"
            )
        limbs = [int(limb) for limb in digest]
        if any(limb < 0 or limb >= GOLDILOCKS_MODULUS for limb in limbs):
            raise ValueError(
                f"commitment_randomness.{label} limbs must be canonical Goldilocks values"
            )
        out[label] = limbs
    return out


def require_faiss(*, prefer_trace: bool = False):
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "faiss is required. Install faiss-cpu (or faiss-gpu) in this Python environment."
        ) from exc
    if bool(prefer_trace) and not (
        hasattr(faiss, "hnsw_search_prove_trace_json")
        or hasattr(faiss, "hnsw_search_trace_json")
    ):
        raise RuntimeError(
            "the installed faiss package does not expose trace hooks; use the replay input path"
        )
    return faiss


def configure_faiss_repro(faiss, seed: int, threads: int) -> dict[str, int | None]:
    used_seed: int | None = None
    used_threads: int | None = None
    if int(threads) > 0:
        try:
            faiss.omp_set_num_threads(int(threads))
            used_threads = int(threads)
        except Exception:
            pass
    if int(seed) >= 0:
        try:
            faiss.rand_seed(int(seed))
            used_seed = int(seed)
        except Exception:
            pass
    return {"seed": used_seed, "threads": used_threads}


def resolve_run_dir(out_dir: Path, results_dir_arg: str) -> Path:
    if results_dir_arg.strip():
        run_dir = Path(results_dir_arg).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (out_dir / "runs" / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_embeddings(parquet_path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "embeddings" not in df.columns:
        raise ValueError(f"missing embeddings column in {parquet_path}")
    arr = np.stack(df["embeddings"].to_numpy(), axis=0).astype(np.float32)
    return np.ascontiguousarray(arr)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, eps)
    return np.ascontiguousarray((x / norms).astype(np.float32))


def winner_model_name_from_train_row(row: Any) -> str:
    flags = [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]]
    if any(flag not in (0, 1) for flag in flags) or sum(flags) != 1:
        raise ValueError(f"invalid winner flags for example {row['id']}")
    # A tie uses model_a deterministically.
    return str(row["model_b"] if flags[1] == 1 else row["model_a"])


def build_embedding_winner_model_ids(example_ids: np.ndarray, train_csv_path: Path) -> np.ndarray:
    import pandas as pd

    train = pd.read_csv(train_csv_path, low_memory=False)
    required = {"id", "model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"}
    if not required.issubset(train.columns):
        raise ValueError(f"missing label columns in {train_csv_path}: {sorted(required - set(train.columns))}")
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise ValueError("training example IDs must be present and unique")
    for column in ("model_a", "model_b"):
        if train[column].isna().any() or train[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing model names in {column}")
    models = sorted(pd.concat([train["model_a"], train["model_b"]]).astype(str).unique())
    model_to_id = {name: i + 1 for i, name in enumerate(models)}
    train_by_id = train.set_index("id", drop=False)
    missing = pd.Index(example_ids).difference(train_by_id.index)
    if not missing.empty:
        raise ValueError(f"embedding IDs have no training labels: {missing[:5].tolist()}")
    labels = [model_to_id[winner_model_name_from_train_row(train_by_id.loc[rid])]
              for rid in example_ids.tolist()]
    return np.ascontiguousarray(labels, dtype=np.uint32)


def load_embedding_model_ids(id_csv_path: Path, expected_n: int, train_csv_path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(id_csv_path)
    if "id" not in df.columns or len(df) != int(expected_n):
        raise ValueError(f"embedding ID table must have an id column and exactly {expected_n} rows")
    ids = df["id"]
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids.dtype) or (ids < 0).any():
        raise ValueError("embedding IDs must be nonnegative integers")
    return build_embedding_winner_model_ids(ids.to_numpy(), train_csv_path)


def create_faiss_hnsw_index(faiss, dimension: int, m: int, ef_construction: int, metric: str):
    if metric == "dot":
        index = faiss.IndexHNSWFlat(int(dimension), int(m), faiss.METRIC_INNER_PRODUCT)
    elif metric == "l2":
        index = faiss.IndexHNSWFlat(int(dimension), int(m), faiss.METRIC_L2)
    else:
        raise ValueError(f"unsupported metric: {metric}")
    index.hnsw.efConstruction = int(ef_construction)
    return index


def build_faiss_hnsw_index(faiss, vectors: np.ndarray, m: int, ef_construction: int, metric: str):
    index = create_faiss_hnsw_index(
        faiss,
        int(vectors.shape[1]),
        int(m),
        int(ef_construction),
        str(metric),
    )
    index.add(np.ascontiguousarray(vectors.astype(np.float32, copy=False)))
    return index


def configure_hnsw_search_mode(
    index,
    ef_search: int,
    *,
    search_bounded_queue: bool | None = None,
    check_relative_distance: bool | None = None,
) -> dict[str, int | bool | None]:
    index.hnsw.efSearch = int(ef_search)
    used_bounded: bool | None = None
    used_relative: bool | None = None
    if search_bounded_queue is not None and hasattr(index.hnsw, "search_bounded_queue"):
        index.hnsw.search_bounded_queue = bool(search_bounded_queue)
        used_bounded = bool(index.hnsw.search_bounded_queue)
    elif hasattr(index.hnsw, "search_bounded_queue"):
        used_bounded = bool(index.hnsw.search_bounded_queue)
    if check_relative_distance is not None and hasattr(index.hnsw, "check_relative_distance"):
        index.hnsw.check_relative_distance = bool(check_relative_distance)
        used_relative = bool(index.hnsw.check_relative_distance)
    elif hasattr(index.hnsw, "check_relative_distance"):
        used_relative = bool(index.hnsw.check_relative_distance)
    return {
        "ef_search": int(ef_search),
        "search_bounded_queue": used_bounded,
        "check_relative_distance": used_relative,
    }


def search_faiss_hnsw(
    index,
    queries: np.ndarray,
    k: int,
    ef_search: int,
    *,
    search_bounded_queue: bool | None = None,
    check_relative_distance: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    configure_hnsw_search_mode(
        index,
        int(ef_search),
        search_bounded_queue=search_bounded_queue,
        check_relative_distance=check_relative_distance,
    )
    distances, ids = index.search(np.ascontiguousarray(queries.astype(np.float32, copy=False)), int(k))
    return np.ascontiguousarray(ids.astype(np.int32)), np.ascontiguousarray(distances.astype(np.float32))


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, bytes):
        try:
            return json.loads(x.decode("utf-8"))
        except Exception:
            return x.decode("utf-8", errors="replace")
    return x


def parse_trace_blob(blob: Any) -> dict[str, Any]:
    if isinstance(blob, dict):
        return to_jsonable(blob)
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    if isinstance(blob, str):
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            raise RuntimeError("trace JSON is not an object")
        return to_jsonable(parsed)
    raise RuntimeError(f"unsupported trace payload type: {type(blob).__name__}")


def get_faiss_hnsw_trace(
    faiss,
    index,
    query: np.ndarray,
    k: int,
    ef_search: int,
    *,
    prefer_prove: bool = True,
) -> tuple[dict[str, Any], str]:
    q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
    tried: list[str] = []

    if bool(prefer_prove) and hasattr(faiss, "hnsw_search_prove_trace_json"):
        tried.append("faiss.hnsw_search_prove_trace_json")
        fn = getattr(faiss, "hnsw_search_prove_trace_json")
        try:
            payload = fn(index, q, int(k), int(ef_search))
        except TypeError:
            if not hasattr(faiss, "swig_ptr"):
                raise
            payload = fn(index, int(q.shape[1]), faiss.swig_ptr(q), int(k), int(ef_search))
        return parse_trace_blob(payload), "faiss.hnsw_search_prove_trace_json"

    if hasattr(faiss, "hnsw_search_trace_json"):
        tried.append("faiss.hnsw_search_trace_json")
        fn = getattr(faiss, "hnsw_search_trace_json")
        try:
            payload = fn(index, q, int(k), int(ef_search))
        except TypeError:
            if not hasattr(faiss, "swig_ptr"):
                raise
            payload = fn(index, int(q.shape[1]), faiss.swig_ptr(q), int(k), int(ef_search))
        return parse_trace_blob(payload), "faiss.hnsw_search_trace_json"

    tried_msg = ", ".join(tried) if tried else "none"
    raise RuntimeError(
        "no patched FAISS HNSW trace API found; "
        f"tried: {tried_msg}"
    )


@dataclass
class HnswRawGraph:
    levels: np.ndarray
    offsets: np.ndarray
    neighbors: np.ndarray
    nb_neighbors_by_layer: list[int]
    layer_ranges: list[list[tuple[int, int]]]
    max_level: int
    entry_point: int


class HnswGraphAccessor:
    def __init__(self, raw: HnswRawGraph):
        self.raw = raw

    def node_max_level(self, node: int) -> int:
        return int(self.raw.levels[node]) - 1

    def neighbors_by_layer(self, node: int) -> dict[int, list[int]]:
        lv = self.node_max_level(node)
        start = int(self.raw.offsets[node])
        end = int(self.raw.offsets[node + 1])
        flat = self.raw.neighbors[start:end]
        if not self.raw.nb_neighbors_by_layer or all(int(v) <= 0 for v in self.raw.nb_neighbors_by_layer):
            return {0: [int(v) for v in flat.tolist() if int(v) >= 0]}
        if lv <= 0:
            return {0: [int(v) for v in flat.tolist() if int(v) >= 0]}

        # Prefer the exact neighbor_range() slices captured from FAISS. These
        # are the most authoritative representation of how the flattened
        # neighbor array is partitioned per layer on this build.
        if self.raw.layer_ranges and node < len(self.raw.layer_ranges):
            node_ranges = self.raw.layer_ranges[node]
            if len(node_ranges) >= int(lv) + 1:
                out: dict[int, list[int]] = {}
                for layer in range(int(lv) + 1):
                    begin, finish = node_ranges[layer]
                    chunk = self.raw.neighbors[int(begin) : int(finish)]
                    out[int(layer)] = [int(v) for v in chunk.tolist() if int(v) >= 0]
                return out

        out: dict[int, list[int]] = {}
        pos = 0
        flat_len = int(end - start)

        # Fallback when neighbor_range() is unavailable: interpret the flattened
        # span in the natural layer order [0, 1, ..., lv], with the documented
        # per-layer caps returned by nb_neighbors(layer).
        for layer in range(0, int(lv) + 1):
            width = int(self.raw.nb_neighbors_by_layer[layer]) if layer < len(self.raw.nb_neighbors_by_layer) else 0
            width = max(0, min(width, flat_len - pos))
            chunk = flat[pos : pos + width]
            out[int(layer)] = [int(v) for v in chunk.tolist() if int(v) >= 0]
            pos += width
        return out

    def neighbors_on_layer(self, node: int, layer: int) -> list[int]:
        return self.neighbors_by_layer(node).get(int(layer), [])


def extract_hnsw_raw_graph(faiss, index) -> HnswRawGraph:
    h = index.hnsw
    try:
        levels = faiss.vector_to_array(h.levels).astype(np.int32, copy=False)
        offsets = faiss.vector_to_array(h.offsets).astype(np.int64, copy=False)
        neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32, copy=False)
    except Exception as exc:
        raise RuntimeError(
            "Cannot read HNSW internals from this faiss build. Need hnsw.levels/offsets/neighbors."
        ) from exc

    ntotal = int(index.ntotal)
    if offsets.shape[0] != ntotal + 1:
        raise RuntimeError(
            f"Unexpected offsets length: got {offsets.shape[0]}, expected {ntotal + 1}"
        )

    try:
        max_level = int(h.max_level)
    except Exception:
        max_level = int(np.max(levels) - 1) if levels.size else -1

    try:
        entry_point = int(h.entry_point)
    except Exception:
        entry_point = -1

    nb_neighbors_by_layer: list[int] = []
    for layer in range(max(0, max_level) + 1):
        try:
            nb_neighbors_by_layer.append(int(h.nb_neighbors(layer)))
        except Exception:
            nb_neighbors_by_layer.append(0)

    layer_ranges: list[list[tuple[int, int]]] = []
    # Large graphs use the equivalent fixed-width offset decoder in
    # HnswGraphAccessor and avoid millions of Python range objects.
    if ntotal <= EXACT_HNSW_LAYER_RANGE_NODE_LIMIT:
        try:
            for node in range(ntotal):
                node_ranges: list[tuple[int, int]] = []
                node_lv = int(levels[node]) - 1
                for layer in range(max(0, node_lv) + 1):
                    begin = np.zeros(1, dtype=np.uint64)
                    end = np.zeros(1, dtype=np.uint64)
                    h.neighbor_range(
                        int(node),
                        int(layer),
                        faiss.swig_ptr(begin),
                        faiss.swig_ptr(end),
                    )
                    node_ranges.append((int(begin[0]), int(end[0])))
                layer_ranges.append(node_ranges)
        except Exception:
            layer_ranges = []

    return HnswRawGraph(
        levels=levels,
        offsets=offsets,
        neighbors=neighbors,
        nb_neighbors_by_layer=nb_neighbors_by_layer,
        layer_ranges=layer_ranges,
        max_level=max_level,
        entry_point=entry_point,
    )


def summarize_degree(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def dump_graph_json(
    scheme_dir: Path,
    raw: HnswRawGraph,
    accessor: HnswGraphAccessor,
    dim: int,
    metric: str,
    m: int,
    ef_construction: int,
    graph_nodes_limit: int,
) -> tuple[Path, Path]:
    ntotal = int(raw.levels.shape[0])
    limit = ntotal if graph_nodes_limit <= 0 else min(ntotal, int(graph_nodes_limit))
    levels_hist = np.bincount(
        np.maximum(raw.levels - 1, 0),
        minlength=max(1, raw.max_level + 1),
    ).tolist()

    degree_per_layer: dict[int, list[int]] = {layer: [] for layer in range(max(0, raw.max_level) + 1)}
    nodes_path = scheme_dir / "graph_nodes.jsonl"
    with nodes_path.open("w", encoding="utf-8") as f:
        for node in range(limit):
            lv = accessor.node_max_level(node)
            layer_neighbors = accessor.neighbors_by_layer(node)
            for layer, nbrs in layer_neighbors.items():
                if layer in degree_per_layer:
                    degree_per_layer[layer].append(len(nbrs))
            start = int(raw.offsets[node])
            end = int(raw.offsets[node + 1])
            rec = {
                "node": int(node),
                "max_level": int(lv),
                "neighbors": {str(k): v for k, v in layer_neighbors.items()},
                "neighbor_count": {str(k): len(v) for k, v in layer_neighbors.items()},
                "raw_neighbor_span": [int(v) for v in raw.neighbors[start:end].tolist()],
            }
            f.write(json.dumps(rec) + "\n")

    meta = {
        "node_count": ntotal,
        "dumped_node_count": int(limit),
        "dim": int(dim),
        "metric": str(metric),
        "m": int(m),
        "ef_construction": int(ef_construction),
        "entry_point": int(raw.entry_point),
        "max_level": int(raw.max_level),
        "levels_histogram": levels_hist,
        "nb_neighbors_by_layer": [int(v) for v in raw.nb_neighbors_by_layer],
        "degree_stats_per_layer": {
            str(layer): summarize_degree(vals) for layer, vals in degree_per_layer.items()
        },
    }
    meta_path = scheme_dir / "graph_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path, nodes_path


def validate_model_ids(
    model_ids: np.ndarray, node_count: int, required_ids: list[int] | None = None,
) -> None:
    if (model_ids is None or model_ids.ndim != 1 or len(model_ids) != node_count
            or not np.issubdtype(model_ids.dtype, np.integer)):
        raise ValueError("each node must have an explicit nonzero u32 model ID")
    values = model_ids if required_ids is None else model_ids[required_ids]
    if np.any(values < 1) or np.any(values > np.iinfo(np.uint32).max):
        raise ValueError("each node must have an explicit nonzero u32 model ID")


def write_global_quantized_nodes_jsonl(
    path: Path,
    vectors_u16: np.ndarray,
    model_ids: np.ndarray,
) -> Path:
    validate_model_ids(model_ids, int(vectors_u16.shape[0]))
    with path.open("w", encoding="utf-8") as f:
        for node_id in range(int(vectors_u16.shape[0])):
            model_id = int(model_ids[node_id])
            rec = {
                "node_id": int(node_id),
                "model_id": int(model_id),
                "vector_u16": [int(v) for v in vectors_u16[node_id].tolist()],
            }
            f.write(json.dumps(rec) + "\n")
    return path


def write_global_graph_rows_jsonl(
    path: Path,
    raw: HnswRawGraph,
    accessor: HnswGraphAccessor,
) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for node in range(int(raw.levels.shape[0])):
            layer_map = accessor.neighbors_by_layer(node)
            for layer in sorted(layer_map):
                rec = {
                    "layer": int(layer),
                    "src": int(node),
                    "neighbors": [int(v) for v in layer_map[int(layer)]],
                }
                f.write(json.dumps(rec) + "\n")
    return path


def write_search_results_jsonl(
    path: Path,
    query_indices: np.ndarray,
    topk_ids: np.ndarray,
    distances: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(topk_ids.shape[0]):
            rec = {
                "query_index": int(query_indices[i]),
                "topk_ids": [int(v) for v in topk_ids[i].tolist()],
                "distances": [float(v) for v in distances[i].tolist()],
            }
            f.write(json.dumps(rec) + "\n")


def runtime_score(query: np.ndarray, vector: np.ndarray, metric: str) -> float:
    if metric == "dot":
        return float(np.dot(query, vector))
    if metric == "l2":
        diff = query - vector
        return -float(np.dot(diff, diff))
    raise ValueError(f"unsupported metric: {metric}")


def better_pair(a_score: float, a_seen: int, b_score: float, b_seen: int) -> bool:
    """Return whether (a_score, -a_seen) is the larger total key."""

    if a_score > b_score:
        return True
    if a_score < b_score:
        return False
    return int(a_seen) < int(b_seen)


def better_int_pair(a_score: int, a_seen: int, b_score: int, b_seen: int) -> bool:
    """Integer-score form of ``better_pair``."""

    if a_score > b_score:
        return True
    if a_score < b_score:
        return False
    return int(a_seen) < int(b_seen)


def sort_node_ids_by_real_score(
    node_ids: list[int],
    scores: dict[int, float],
    seen_orders: dict[int, int] | None = None,
) -> list[int]:
    if seen_orders is None:
        # Python's sort is stable: this preserves current-then-neighbor scan
        # order in an upper-layer step.
        return sorted(node_ids, key=lambda node_id: -float(scores[int(node_id)]))
    return sorted(
        node_ids,
        key=lambda node_id: (-float(scores[int(node_id)]), int(seen_orders[int(node_id)])),
    )


def sort_node_ids_by_int_score(
    node_ids: list[int],
    scores: dict[int, int],
    seen_orders: dict[int, int] | None = None,
) -> list[int]:
    if seen_orders is None:
        return sorted(node_ids, key=lambda node_id: -int(scores[int(node_id)]))
    return sorted(
        node_ids,
        key=lambda node_id: (-int(scores[int(node_id)]), int(seen_orders[int(node_id)])),
    )


def sort_result_ids_by_real_score(
    node_ids: list[int],
    scores: dict[int, float],
    births: dict[int, int],
) -> list[int]:
    return sorted(
        node_ids,
        key=lambda node_id: (
            -float(scores[int(node_id)]),
            int(births[int(node_id)]),
        ),
    )


def sort_result_ids_by_int_score(
    node_ids: list[int],
    scores: dict[int, int],
    births: dict[int, int],
) -> list[int]:
    return sorted(
        node_ids,
        key=lambda node_id: (
            -int(scores[int(node_id)]),
            int(births[int(node_id)]),
        ),
    )


def topk_lists_match_up_to_ties(
    topk_a: list[int],
    topk_b: list[int],
    score_map: dict[int, int],
) -> bool:
    if len(topk_a) != len(topk_b):
        return False
    if not topk_a and not topk_b:
        return True

    # This helper compares an external FAISS result, which does not expose the
    # protocol's first-seen order.  It therefore deliberately remains an
    # up-to-score-ties equivalence check rather than a protocol comparator.
    ordered_a = sorted(
        [int(x) for x in topk_a],
        key=lambda node_id: (-int(score_map[int(node_id)]), int(node_id)),
    )
    ordered_b = sorted(
        [int(x) for x in topk_b],
        key=lambda node_id: (-int(score_map[int(node_id)]), int(node_id)),
    )

    boundary_a = int(score_map[int(ordered_a[-1])])
    boundary_b = int(score_map[int(ordered_b[-1])])
    if boundary_a != boundary_b:
        return False
    boundary = boundary_a
    prefix_a = [int(x) for x in ordered_a if int(score_map[int(x)]) > boundary]
    prefix_b = [int(x) for x in ordered_b if int(score_map[int(x)]) > boundary]
    if prefix_a != prefix_b:
        return False
    suffix_a = [int(x) for x in ordered_a if int(score_map[int(x)]) == boundary]
    suffix_b = [int(x) for x in ordered_b if int(score_map[int(x)]) == boundary]
    return len(suffix_a) == len(suffix_b)


def compute_real_scores_for_nodes(
    query: np.ndarray,
    vectors: np.ndarray,
    node_ids: list[int],
    metric: str,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for node_id in node_ids:
        out[int(node_id)] = runtime_score(query, vectors[int(node_id)], metric)
    return out


def upper_greedy_trace_real(
    query: np.ndarray,
    vectors: np.ndarray,
    accessor: HnswGraphAccessor,
    raw: HnswRawGraph,
    metric: str,
) -> tuple[list[dict[str, Any]], int]:
    if int(raw.entry_point) < 0:
        raise RuntimeError("HNSW entry point unavailable")
    current = int(raw.entry_point)
    trace: list[dict[str, Any]] = []
    for layer in range(int(raw.max_level), 0, -1):
        while True:
            neighbors = [int(x) for x in accessor.neighbors_on_layer(current, layer)]
            candidate_ids = [int(current)] + neighbors
            score_map = compute_real_scores_for_nodes(query, vectors, candidate_ids, metric)
            ordered = sort_node_ids_by_real_score(candidate_ids, score_map)
            selected = int(ordered[0])
            trace.append(
                {
                    "layer": int(layer),
                    "current": int(current),
                    "neighbors": neighbors,
                    "selected": int(selected),
                }
            )
            if selected == current:
                break
            current = int(selected)
            if len(trace) > 200000:
                raise RuntimeError("upper-layer replay exceeded safety steps")
    return trace, int(current)


def result_worst_real(
    result_ids: set[int],
    score_map: dict[int, float],
    births: dict[int, int],
    ef_search: int,
) -> tuple[int, float] | None:
    if len(result_ids) < int(ef_search):
        return None
    ordered = sort_result_ids_by_real_score(list(result_ids), score_map, births)
    worst_id = int(ordered[int(ef_search) - 1])
    return int(worst_id), float(score_map[int(worst_id)])


def result_worst_int(
    result_ids: set[int],
    score_map: dict[int, int],
    births: dict[int, int],
    ef_search: int,
) -> tuple[int, int] | None:
    if len(result_ids) < int(ef_search):
        return None
    ordered = sort_result_ids_by_int_score(list(result_ids), score_map, births)
    worst_id = int(ordered[int(ef_search) - 1])
    return int(worst_id), int(score_map[int(worst_id)])


def best_first_level0_trace_real(
    query: np.ndarray,
    vectors: np.ndarray,
    accessor: HnswGraphAccessor,
    entry: int,
    ef_search: int,
    k: int,
    metric: str,
) -> dict[str, Any]:
    seen: set[int] = {int(entry)}
    candidates: set[int] = {int(entry)}
    results: set[int] = {int(entry)}
    seen_orders: dict[int, int] = {int(entry): 1}
    score_map: dict[int, float] = {int(entry): runtime_score(query, vectors[int(entry)], metric)}
    next_seen_order = 2
    expansions: list[dict[str, Any]] = []
    stop_info: dict[str, Any] | None = None

    while candidates:
        candidate_ordered = sort_node_ids_by_real_score(
            list(candidates), score_map, seen_orders
        )
        pop_id = int(candidate_ordered[0])
        pop_score = float(score_map[int(pop_id)])
        worst_before = result_worst_real(results, score_map, seen_orders, ef_search)
        if worst_before is not None:
            worst_id, lower_bound_before = worst_before
            if better_pair(
                float(lower_bound_before),
                int(seen_orders[int(worst_id)]),
                pop_score,
                int(seen_orders[int(pop_id)]),
            ):
                stop_info = {
                    "candidate": int(pop_id),
                    "candidate_score": float(pop_score),
                    "lower_bound_node": int(worst_id),
                    "lower_bound_score": float(lower_bound_before),
                    "reason": "candidate_not_better_than_lower_bound",
                }
                break

        candidates.remove(int(pop_id))

        neighbors = [int(x) for x in accessor.neighbors_on_layer(pop_id, 0)]
        neighbor_visits: list[dict[str, Any]] = []
        for nb in neighbors:
            is_new = int(nb) not in seen
            accepted = False
            if is_new:
                seen.add(int(nb))
                seen_orders[int(nb)] = int(next_seen_order)
                next_seen_order += 1
                score_map[int(nb)] = runtime_score(query, vectors[int(nb)], metric)
                worst_now = result_worst_real(results, score_map, seen_orders, ef_search)
                if worst_now is None or better_pair(
                    float(score_map[int(nb)]),
                    int(seen_orders[int(nb)]),
                    float(worst_now[1]),
                    int(seen_orders[int(worst_now[0])]),
                ):
                    accepted = True
                    candidates.add(int(nb))
                    results.add(int(nb))
                    if len(results) > int(ef_search):
                        result_ordered = sort_result_ids_by_real_score(
                            list(results), score_map, seen_orders
                        )
                        removed = int(result_ordered[-1])
                        results.remove(removed)
            neighbor_visits.append(
                {
                    "node_id": int(nb),
                    "is_new": bool(is_new),
                    "accepted": bool(accepted),
                }
            )

        expansions.append(
            {
                "pop_id": int(pop_id),
                "neighbors": neighbors,
                "neighbor_visits": neighbor_visits,
            }
        )

        if len(expansions) > 500000:
            raise RuntimeError("level-0 best-first replay exceeded safety steps")

    final_order = sort_result_ids_by_real_score(list(results), score_map, seen_orders)
    final_topk = [int(x) for x in final_order[: int(k)]]
    return {
        "entry": int(entry),
        "ef_search": int(ef_search),
        "k": int(k),
        "expansions": expansions,
        "final_topk": final_topk,
        "stop": stop_info,
    }


def replay_query_real(
    query: np.ndarray,
    vectors: np.ndarray,
    accessor: HnswGraphAccessor,
    raw: HnswRawGraph,
    ef_search: int,
    k: int,
    metric: str,
) -> dict[str, Any]:
    upper_trace, entry = upper_greedy_trace_real(query, vectors, accessor, raw, metric)
    level0 = best_first_level0_trace_real(query, vectors, accessor, entry, ef_search, k, metric)
    # This is an unpadded trace over authenticated variable-length graph rows.
    # Its physical WorkUpd width must match the rows actually exported, rather
    # than FAISS's larger allocation capacity. Fixed-shape exporters set their
    # cap only after padding every row with sentinels.
    level0["round_width"] = max(
        (len(exp["neighbors"]) for exp in level0["expansions"]),
        default=0,
    )
    return {
        "metric": str(metric),
        "entry_point": int(raw.entry_point),
        "upper_trace": upper_trace,
        "level0": level0,
        "final_topk": level0["final_topk"],
    }


def compute_quantized_scores_for_nodes(
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    node_ids: list[int],
    zero_point: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for node_id in node_ids:
        out[int(node_id)] = quantized_dot_score(query_u16, vectors_u16[int(node_id)], int(zero_point))
    return out


def upper_greedy_trace_quantized(
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    accessor: HnswGraphAccessor,
    raw: HnswRawGraph,
    zero_point: int,
) -> tuple[list[dict[str, Any]], int]:
    if int(raw.entry_point) < 0:
        raise RuntimeError("HNSW entry point unavailable")
    current = int(raw.entry_point)
    trace: list[dict[str, Any]] = []
    for layer in range(int(raw.max_level), 0, -1):
        while True:
            neighbors = [int(x) for x in accessor.neighbors_on_layer(current, layer)]
            current_score = int(
                quantized_dot_score(query_u16, vectors_u16[int(current)], int(zero_point))
            )
            selected = int(current)
            selected_score = int(current_score)
            for nb in neighbors:
                nb_score = int(
                    quantized_dot_score(query_u16, vectors_u16[int(nb)], int(zero_point))
                )
                # Stable scan order breaks ties: keep current/the first
                # neighbor encountered at the maximum score.
                if nb_score > selected_score:
                    selected = int(nb)
                    selected_score = int(nb_score)
            trace.append(
                {
                    "layer": int(layer),
                    "current": int(current),
                    "neighbors": neighbors,
                    "selected": int(selected),
                }
            )
            if selected == current:
                break
            current = int(selected)
            if len(trace) > 200000:
                raise RuntimeError("upper-layer quantized replay exceeded safety steps")
    return trace, int(current)


def best_first_level0_trace_quantized(
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    accessor: HnswGraphAccessor,
    entry: int,
    ef_search: int,
    k: int,
    zero_point: int,
) -> dict[str, Any]:
    import heapq

    seen: set[int] = {int(entry)}
    score_map: dict[int, int] = {
        int(entry): int(quantized_dot_score(query_u16, vectors_u16[int(entry)], int(zero_point)))
    }
    seen_orders: dict[int, int] = {int(entry): 1}
    next_seen_order = 2
    # Python's heap is a min-heap.  CandQ uses (-score, seen_order), while
    # WorkQ uses (score, -seen_order).  Both implement the same total key
    # (score, -seen_order): higher score wins and an equal-score earlier first
    # visit wins.
    candidates: list[tuple[int, int, int]] = [
        (-int(score_map[int(entry)]), 1, int(entry))
    ]
    top_candidates: list[tuple[int, int, int]] = [
        (int(score_map[int(entry)]), -1, int(entry))
    ]
    expansions: list[dict[str, Any]] = []
    stop_info: dict[str, Any] | None = None

    def worst_top_key() -> tuple[int, int]:
        return int(top_candidates[0][0]), int(top_candidates[0][1])

    def push_top(node_id: int) -> bool:
        score = int(score_map[int(node_id)])
        neg_seen = -int(seen_orders[int(node_id)])
        key = (score, neg_seen)
        if len(top_candidates) < int(ef_search):
            heapq.heappush(top_candidates, (score, neg_seen, int(node_id)))
            return True
        if key > worst_top_key():
            _removed_score, _removed_seen_key, _removed_id = heapq.heapreplace(
                top_candidates,
                (score, neg_seen, int(node_id)),
            )
            return True
        return False

    while candidates:
        pop_neg_score, pop_seen_order, pop_id = heapq.heappop(candidates)
        pop_score = int(-pop_neg_score)
        if (pop_score, -int(pop_seen_order)) < worst_top_key():
            stop_info = {
                "candidate": int(pop_id),
                "candidate_score_int": int(pop_score),
                "lower_bound_node": int(top_candidates[0][2]),
                "lower_bound_score_int": int(worst_top_key()[0]),
                "reason": "candidate_worse_than_worst_top_unbounded",
            }
            break

        neighbors = [int(x) for x in accessor.neighbors_on_layer(pop_id, 0)]
        neighbor_visits: list[dict[str, Any]] = []
        for nb in neighbors:
            is_new = int(nb) not in seen
            accepted = False
            if is_new:
                seen.add(int(nb))
                seen_orders[int(nb)] = int(next_seen_order)
                next_seen_order += 1
                score_map[int(nb)] = int(
                    quantized_dot_score(
                        query_u16, vectors_u16[int(nb)], int(zero_point)
                    )
                )
                accepted = push_top(int(nb))
                if accepted:
                    heapq.heappush(
                        candidates,
                        (
                            -int(score_map[int(nb)]),
                            int(seen_orders[int(nb)]),
                            int(nb),
                        ),
                    )
            neighbor_visits.append(
                {
                    "node_id": int(nb),
                    "is_new": bool(is_new),
                    "accepted": bool(accepted),
                }
            )

        expansions.append(
            {
                "pop_id": int(pop_id),
                "neighbors": neighbors,
                "neighbor_visits": neighbor_visits,
            }
        )

        if len(expansions) > 500000:
            raise RuntimeError("level-0 quantized replay exceeded safety steps")

    final_order = sort_result_ids_by_int_score(
        [int(node_id) for _, _, node_id in top_candidates],
        score_map,
        seen_orders,
    )
    final_topk = [int(x) for x in final_order[: int(k)]]
    return {
        "entry": int(entry),
        "ef_search": int(ef_search),
        "k": int(k),
        "expansions": expansions,
        "final_topk": final_topk,
        "stop": stop_info,
    }


def replay_query_quantized(
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    accessor: HnswGraphAccessor,
    raw: HnswRawGraph,
    ef_search: int,
    k: int,
    zero_point: int,
) -> dict[str, Any]:
    upper_trace, entry = upper_greedy_trace_quantized(query_u16, vectors_u16, accessor, raw, zero_point)
    level0 = best_first_level0_trace_quantized(
        query_u16,
        vectors_u16,
        accessor,
        entry,
        ef_search,
        k,
        zero_point,
    )
    # Keep base proof shape equal to the authenticated rows actually present.
    # The route-experiment padding path writes a larger fixed cap only after
    # materializing sentinel slots in both trace and graph rows.
    level0["round_width"] = max(
        (len(exp["neighbors"]) for exp in level0["expansions"]),
        default=0,
    )
    return {
        "metric": "dot",
        "entry_point": int(raw.entry_point),
        "upper_trace": upper_trace,
        "level0": level0,
        "final_topk": level0["final_topk"],
    }


def _clean_topk_ids(values: list[Any], k: int) -> list[int]:
    out: list[int] = []
    for raw in values:
        node_id = int(raw)
        if node_id < 0:
            continue
        out.append(node_id)
        if len(out) >= int(k):
            break
    return out


def minimal_search_trace_from_quantized_replay(replay_trace: dict[str, Any]) -> dict[str, Any]:
    upper_trace = [
        {
            "layer": int(step["layer"]),
            "current": int(step["current"]),
            "neighbors": [int(x) for x in step["neighbors"]],
            "selected": int(step["selected"]),
        }
        for step in replay_trace["upper_trace"]
    ]
    expansions: list[dict[str, Any]] = []
    for exp in replay_trace["level0"]["expansions"]:
        neighbors = [int(x) for x in exp["neighbors"]]
        visits = exp.get("neighbor_visits", [])
        if not isinstance(visits, list) or len(visits) != len(neighbors):
            raise ValueError(
                f"replay neighbor_visits length mismatch at pop {exp.get('pop_id')}: "
                f"neighbors={len(neighbors)} visits={len(visits) if isinstance(visits, list) else 'invalid'}"
            )
        expansions.append(
            {
                "pop_id": int(exp["pop_id"]),
                "neighbors": neighbors,
                "neighbor_visits": [
                    {
                        "node_id": int(visit.get("node_id", nb)),
                        "is_new": bool(visit.get("is_new", False)),
                    }
                    for nb, visit in zip(neighbors, visits)
                ],
            }
        )
    level0 = replay_trace["level0"]
    return {
        "trace_source": "replay_quantized_fallback",
        "entry_point": int(replay_trace["entry_point"]),
        "upper_trace": upper_trace,
        "level0": {
            "entry": int(level0["entry"]),
            "ef_search": int(level0["ef_search"]),
            "k": int(level0["k"]),
            # The minimal/base trace has not had authenticated graph rows
            # padded. Use the observed row shape; a fixed cap is introduced
            # later only by the padding exporter.
            "round_width": max(
                (len(exp["neighbors"]) for exp in expansions),
                default=0,
            ),
            "expansions": expansions,
            "final_topk": [int(x) for x in level0["final_topk"]],
            "stop": level0.get("stop"),
        },
    }


def minimal_search_trace_from_faiss_trace(trace_payload: dict[str, Any]) -> dict[str, Any]:
    upper_events = trace_payload.get("upper_trace", [])
    if not isinstance(upper_events, list):
        raise ValueError("trace.upper_trace must be a list")
    upper_trace = []
    for ev_any in upper_events:
        ev = dict(ev_any) if isinstance(ev_any, dict) else {}
        upper_trace.append(
            {
                "layer": int(ev.get("layer", 0)),
                "current": int(ev.get("current", trace_payload.get("entry_point", 0))),
                "neighbors": [int(x) for x in ev.get("neighbors", [])],
                "selected": int(ev.get("selected", ev.get("current", trace_payload.get("entry_point", 0)))),
            }
        )

    entry_point = int(trace_payload.get("entry_point", 0))
    level0_entry = upper_trace[-1]["selected"] if upper_trace else entry_point
    level0_events = trace_payload.get("level0_trace", [])
    if not isinstance(level0_events, list):
        raise ValueError("trace.level0_trace must be a list")
    seen: set[int] = {int(level0_entry)}
    expansions: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    for ev_any in level0_events:
        ev = dict(ev_any) if isinstance(ev_any, dict) else {}
        event_name = str(ev.get("event", "")).lower()
        if event_name == "expand":
            neighbors = [int(x) for x in ev.get("neighbors", [])]
            steps_raw = ev.get("neighbor_steps", [])
            use_steps = isinstance(steps_raw, list) and len(steps_raw) == len(neighbors)
            neighbor_visits: list[dict[str, Any]] = []
            for idx, nb in enumerate(neighbors):
                if use_steps:
                    step = steps_raw[idx] if isinstance(steps_raw[idx], dict) else {}
                    step_node = int(step.get("node", nb))
                    if step_node != nb:
                        raise ValueError(
                            f"neighbor_steps[{idx}] node={step_node} does not match neighbors[{idx}]={nb}"
                        )
                    is_new = bool(step.get("was_new", False))
                else:
                    is_new = int(nb) not in seen
                neighbor_visits.append({"node_id": int(nb), "is_new": bool(is_new)})
                if is_new:
                    seen.add(int(nb))
            expansions.append(
                {
                    "pop_id": int(ev.get("node", -1)),
                    "neighbors": neighbors,
                    "neighbor_visits": neighbor_visits,
                }
            )
        elif event_name == "stop":
            stop = {
                "candidate": int(ev.get("node", -1)),
                "reason": str(ev.get("reason", "")),
            }

    k = int(trace_payload.get("k", 0))
    return {
        "trace_source": "faiss_direct_quantized",
        "trace_api": str(trace_payload.get("trace_api", "")),
        "entry_point": int(entry_point),
        "upper_trace": upper_trace,
        "level0": {
            "entry": int(level0_entry),
            "ef_search": int(trace_payload.get("ef_search", 0)),
            "k": int(k),
            # Direct FAISS traces also contain variable-length authenticated
            # rows. Ignore an allocation-capacity hint until rows are actually
            # padded by the fixed-shape exporter.
            "round_width": max(
                (len(exp["neighbors"]) for exp in expansions),
                default=0,
            ),
            "expansions": expansions,
            "final_topk": _clean_topk_ids(list(trace_payload.get("topk_ids", [])), k),
            "stop": stop,
        },
    }


@dataclass(frozen=True)
class AffineQuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    min_value: float
    max_value: float


@dataclass(frozen=True)
class SearchSchemeParams:
    name: str
    quantizer: AffineQuantizationParams
    projected_dim: int
    projection_seed: int
    projection_percentile: float
    projection_matrix: np.ndarray | None


def fit_affine_quantizer(values: np.ndarray, qmin: int = 0, qmax: int = 65535) -> AffineQuantizationParams:
    arr = np.ascontiguousarray(values.astype(np.float64, copy=False))
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("quantizer input must be nonempty and finite")
    if not 0 <= qmin < qmax <= 65535:
        raise ValueError("quantizer bounds must satisfy 0 <= qmin < qmax <= 65535")
    # Include zero so that clamping the zero point cannot saturate one-sided data.
    min_value = min(0.0, float(np.min(arr)))
    max_value = max(0.0, float(np.max(arr)))
    if max_value == min_value:
        return AffineQuantizationParams(
            scale=1.0,
            zero_point=int(qmin),
            qmin=int(qmin),
            qmax=int(qmax),
            min_value=min_value,
            max_value=max_value,
        )
    scale = float((max_value - min_value) / float(qmax - qmin))
    zero_point = int(round(float(qmin) - min_value / scale))
    zero_point = max(int(qmin), min(int(qmax), zero_point))
    return AffineQuantizationParams(
        scale=scale,
        zero_point=zero_point,
        qmin=int(qmin),
        qmax=int(qmax),
        min_value=min_value,
        max_value=max_value,
    )


def quantize_affine(values: np.ndarray, params: AffineQuantizationParams) -> np.ndarray:
    arr = np.ascontiguousarray(values.astype(np.float64, copy=False))
    q = np.rint(arr / float(params.scale) + float(params.zero_point))
    q = np.clip(q, int(params.qmin), int(params.qmax)).astype(np.uint16)
    return np.ascontiguousarray(q)


def dequantize_affine(values_u16: np.ndarray, params: AffineQuantizationParams) -> np.ndarray:
    q = np.ascontiguousarray(values_u16.astype(np.float32, copy=False))
    return np.ascontiguousarray((q - float(params.zero_point)) * float(params.scale), dtype=np.float32)


def proj_sign_matrix(seed: int, in_dim: int, out_dim: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=(in_dim, out_dim), dtype=np.int8)
    return np.ascontiguousarray((signs * 2 - 1).astype(np.float32))


def project_vectors(values: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    in_dim = int(projection_matrix.shape[0])
    projected = (np.ascontiguousarray(values.astype(np.float32, copy=False)) @ projection_matrix) / math.sqrt(in_dim)
    return np.ascontiguousarray(projected.astype(np.float32, copy=False))


def fit_search_scheme(
    x_train: np.ndarray,
    scheme: str,
    proj_dim: int,
    proj_seed: int,
    proj_percentile: float,
) -> SearchSchemeParams:
    scheme = str(scheme)
    if scheme == "affine":
        transformed = np.ascontiguousarray(x_train.astype(np.float32, copy=False))
        projection_matrix = None
        projected_dim = int(x_train.shape[1])
        used_seed = 0
        used_percentile = 0.0
    elif scheme == "proj_affine":
        projected_dim = int(proj_dim)
        projection_matrix = proj_sign_matrix(int(proj_seed), int(x_train.shape[1]), projected_dim)
        transformed = project_vectors(x_train, projection_matrix)
        used_seed = int(proj_seed)
        used_percentile = float(proj_percentile)
    else:
        raise ValueError(f"unsupported search scheme: {scheme}")
    quantizer = fit_affine_quantizer(transformed, qmin=0, qmax=65535)
    return SearchSchemeParams(
        name=scheme,
        quantizer=quantizer,
        projected_dim=int(projected_dim),
        projection_seed=int(used_seed),
        projection_percentile=float(used_percentile),
        projection_matrix=projection_matrix,
    )


def transform_search_values(values: np.ndarray, scheme: SearchSchemeParams) -> np.ndarray:
    arr = np.ascontiguousarray(values.astype(np.float32, copy=False))
    if scheme.projection_matrix is not None:
        arr = project_vectors(arr, scheme.projection_matrix)
    return np.ascontiguousarray(arr)


def quantize_search_values(values: np.ndarray, scheme: SearchSchemeParams) -> np.ndarray:
    transformed = transform_search_values(values, scheme)
    return quantize_affine(transformed, scheme.quantizer)


def decode_search_values(values_u16: np.ndarray, scheme: SearchSchemeParams) -> np.ndarray:
    return dequantize_affine(values_u16, scheme.quantizer)


def centered_u16(values: np.ndarray, zero_point: int) -> np.ndarray:
    q = np.ascontiguousarray(values.astype(np.uint16, copy=False))
    return np.ascontiguousarray(q.astype(np.int64) - int(zero_point))


def quantized_dot_score(query_u16: np.ndarray, node_u16: np.ndarray, zero_point: int) -> int:
    q_center = centered_u16(query_u16, zero_point)
    node_center = centered_u16(node_u16, zero_point)
    return int(np.dot(q_center.astype(np.int64), node_center.astype(np.int64)))


def compute_quantized_scores(
    query_u16: np.ndarray,
    node_vectors_u16: dict[int, np.ndarray],
    zero_point: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for node_id, vector_u16 in node_vectors_u16.items():
        out[int(node_id)] = quantized_dot_score(query_u16, vector_u16, int(zero_point))
    return out


def upper_trace_from_quantized_scores(
    trace: list[dict[str, Any]],
    score_map: dict[int, int],
    ref_map: dict[int, dict[str, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in trace:
        current = int(step["current"])
        neighbors = [int(x) for x in step["neighbors"]]
        # Stable scan order breaks ties: current is examined before neighbors.
        selected = int(current)
        selected_score = int(score_map[int(current)])
        for nb in neighbors:
            nb_score = int(score_map[int(nb)])
            if nb_score > selected_score:
                selected = int(nb)
                selected_score = int(nb_score)
        out.append({"selected": dict(ref_map[int(selected)])})
    return out


def _checked_position(value: int, label: str) -> int:
    value = int(value)
    if value < 1 or value >= (1 << POSITION_BITS):
        raise ValueError(f"{label} {value} is outside the {POSITION_BITS}-bit position domain")
    return value


def c_entry_position() -> int:
    return 1


def c_pop_position(round_idx: int, round_width: int) -> int:
    return _checked_position(
        2 + int(round_idx) * (int(round_width) + 1),
        "candidate pop position",
    )


def c_neighbor_position(round_idx: int, slot_idx: int, round_width: int) -> int:
    if not 0 <= int(slot_idx) < int(round_width):
        raise ValueError("candidate neighbor slot is outside the fixed round width")
    return _checked_position(
        c_pop_position(round_idx, round_width) + int(slot_idx) + 1,
        "candidate neighbor position",
    )


def c_position_limit(rounds: int, round_width: int) -> int:
    return _checked_position(
        1 + int(rounds) * (int(round_width) + 1),
        "candidate position limit",
    )


def w_initial_position(slot_idx: int) -> int:
    return _checked_position(int(slot_idx) + 1, "working initial position")


def w_push_position(step_idx: int, ef_search: int) -> int:
    return _checked_position(
        int(ef_search) + 2 * int(step_idx) + 1,
        "working push position",
    )


def w_pop_position(step_idx: int, ef_search: int) -> int:
    return _checked_position(
        int(ef_search) + 2 * int(step_idx) + 2,
        "working pop position",
    )


def w_position_limit(work_update_count: int, ef_search: int) -> int:
    if int(work_update_count) == 0:
        return _checked_position(int(ef_search), "working position limit")
    return w_pop_position(int(work_update_count) - 1, ef_search)


def _checked_score_enc(score_enc: int) -> int:
    score_enc = int(score_enc)
    if not 0 <= score_enc < (1 << SCORE_BITS):
        raise ValueError(f"score_enc is outside the {SCORE_BITS}-bit score domain")
    return score_enc


def assert_packed_priority_domain(position_max: int, ef_search: int) -> None:
    """Reject shapes whose packed comparison key would exceed 62 bits."""

    position_max = _checked_position(position_max, "candidate position limit")
    ef_search = int(ef_search)
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if position_max > (1 << (PACKED_PRIORITY_BITS - SCORE_BITS)) - 1:
        raise ValueError(
            "packed priority exceeds the 62-bit domain because position_max is above 16383: "
            f"position_max={position_max}"
        )
    score_max = (1 << SCORE_BITS) - 1
    c_max = score_max * position_max + (position_max - 1)
    w_max = ef_search + score_max * position_max + (position_max - 1)
    packed_limit = 1 << PACKED_PRIORITY_BITS
    if c_max >= packed_limit or w_max >= packed_limit:
        raise ValueError(
            "packed priority key exceeds the 62-bit sound comparison domain: "
            f"position_max={position_max} ef_search={ef_search} "
            f"c_max={c_max} w_max={w_max}"
        )


def c_packed_priority_key(score_enc: int, tie_position: int, position_max: int) -> int:
    """Encode CandQ's best-first order as a min-PQ key."""

    score_enc = _checked_score_enc(score_enc)
    tie_position = _checked_position(tie_position, "candidate tie position")
    position_max = _checked_position(position_max, "candidate position limit")
    if tie_position > position_max:
        raise ValueError("candidate tie position exceeds its public maximum")
    score_max = (1 << SCORE_BITS) - 1
    return (score_max - score_enc) * position_max + (tie_position - 1)


def w_packed_priority_key(
    score_enc: int,
    tie_position: int,
    position_max: int,
    ef_search: int,
) -> int:
    """Encode WorkQ's worst-first order as a min-PQ key."""

    score_enc = _checked_score_enc(score_enc)
    tie_position = _checked_position(tie_position, "working tie position")
    position_max = _checked_position(position_max, "candidate position limit")
    if tie_position > position_max:
        raise ValueError("working tie position exceeds its public maximum")
    return (
        int(ef_search)
        + score_enc * position_max
        + (position_max - tie_position)
    )


def build_prev_lower_trace(
    keys: list[int],
    event_times: list[int],
    enabled: list[bool] | None = None,
    active_capacity: int | None = None,
) -> dict[str, Any]:
    """Build the latest-preceding-lower-priority monotone-stack witness.

    ``key`` is a packed lower-priority order encoding: larger values have
    lower queue priority. Consequently, the nearest preceding lower-priority
    item is the nearest preceding item with a greater encoded key.
    Disabled physical slots remain in the arrays for a fixed circuit shape but
    are skipped by the logical stack.
    """

    count = len(keys)
    if enabled is None:
        enabled = [True for _ in range(count)]
    if len(event_times) != count or len(enabled) != count:
        raise ValueError(
            "prev_lower length mismatch: "
            f"keys={count} event_times={len(event_times)} enabled={len(enabled)}"
        )
    if active_capacity is None:
        active_capacity = count
    active_capacity = int(active_capacity)
    if active_capacity < 0 or active_capacity > count:
        raise ValueError(
            "prev_lower active capacity must be between zero and the physical "
            f"key length: active_capacity={active_capacity} keys={count}"
        )
    active_count = sum(bool(flag) for flag in enabled)
    if active_count > active_capacity:
        raise ValueError(
            "prev_lower active rows exceed witness capacity: "
            f"active={active_count} capacity={active_capacity}"
        )
    sentinel_idx = int(count)
    sentinel_key = 0
    sentinel_time = 0
    parent = [sentinel_idx for _ in range(count)]
    parent_time = [sentinel_time for _ in range(count)]
    stack: list[int] = []
    events: list[dict[str, Any]] = []

    for cur_idx, cur_key in enumerate(keys):
        if not bool(enabled[cur_idx]):
            # Disabled physical slots do not participate in the OutSeq stack.
            # Their canonical parent remains the full-layout sentinel.
            continue
        cur_time = int(event_times[cur_idx])
        while stack and int(keys[stack[-1]]) <= int(cur_key):
            top_before = int(stack.pop())
            top_after = int(stack[-1]) if stack else sentinel_idx
            events.append(
                {
                    "active": True,
                    "is_push": False,
                    "cur_index": int(cur_idx),
                    "cur_key": int(cur_key),
                    "cur_time": int(cur_time),
                    "top_before_index": int(top_before),
                    "top_before_key": int(keys[top_before]),
                    "top_before_time": int(event_times[top_before]),
                    "top_after_index": int(top_after),
                    "top_after_key": int(keys[top_after]) if top_after != sentinel_idx else int(sentinel_key),
                    "top_after_time": int(event_times[top_after]) if top_after != sentinel_idx else int(sentinel_time),
                }
            )
        top_before = int(stack[-1]) if stack else sentinel_idx
        parent[cur_idx] = top_before
        parent_time[cur_idx] = (
            int(event_times[top_before]) if top_before != sentinel_idx else int(sentinel_time)
        )
        events.append(
            {
                "active": True,
                "is_push": True,
                "cur_index": int(cur_idx),
                "cur_key": int(cur_key),
                "cur_time": int(cur_time),
                "top_before_index": int(top_before),
                "top_before_key": int(keys[top_before]) if top_before != sentinel_idx else int(sentinel_key),
                "top_before_time": int(parent_time[cur_idx]),
                "top_after_index": int(cur_idx),
                "top_after_key": int(cur_key),
                "top_after_time": int(cur_time),
            }
        )
        stack.append(int(cur_idx))

    final_nodes = []
    stack_top_down = list(reversed(stack))
    for offset, node_idx in enumerate(stack_top_down):
        parent_index = int(parent[node_idx])
        if offset + 1 < len(stack_top_down):
            parent_key = int(keys[stack_top_down[offset + 1]])
        else:
            parent_key = int(sentinel_key)
        final_nodes.append(
            {
                "node_index": int(node_idx),
                "node_key": int(keys[node_idx]),
                "node_time": int(event_times[node_idx]),
                "parent_index": int(parent_index),
                "parent_key": int(parent_key),
                "parent_time": int(parent_time[node_idx]),
            }
        )
    event_capacity = 2 * max(1, active_capacity)
    if len(events) > event_capacity:
        raise ValueError(
            "prev_lower events exceed witness capacity: "
            f"events={len(events)} capacity={event_capacity}"
        )
    while len(events) < event_capacity:
        events.append(
            {
                "active": False,
                "is_push": False,
                "cur_index": 0,
                "cur_key": 0,
                "cur_time": 0,
                "top_before_index": int(sentinel_idx),
                "top_before_key": int(sentinel_key),
                "top_before_time": int(sentinel_time),
                "top_after_index": int(sentinel_idx),
                "top_after_key": int(sentinel_key),
                "top_after_time": int(sentinel_time),
            }
        )
    final_capacity = max(1, active_capacity)
    if len(final_nodes) > final_capacity:
        raise ValueError(
            "prev_lower final stack exceeds witness capacity: "
            f"nodes={len(final_nodes)} capacity={final_capacity}"
        )
    while len(final_nodes) < final_capacity:
        final_nodes.append(
            {
                "node_index": int(sentinel_idx),
                "node_key": 0,
                "node_time": 0,
                "parent_index": int(sentinel_idx),
                "parent_key": 0,
                "parent_time": 0,
            }
        )
    return {
        "prev_lower": [int(x) for x in parent],
        "prev_lower_times": [int(x) for x in parent_time],
        "events": events,
        "final_nodes": final_nodes,
    }


def encode_score_with_bias(value: int, bits: int = SCORE_BITS) -> int:
    bias = 1 << (bits - 1)
    lo = -bias
    hi = bias
    if int(value) < lo or int(value) >= hi:
        raise ValueError(f"signed value {value} does not fit into {bits} bits")
    return int(value) + bias


def decode_score_with_bias(value: int, bits: int = SCORE_BITS) -> int:
    bias = 1 << (bits - 1)
    value = int(value)
    if value < 0 or value >= (1 << bits):
        raise ValueError(f"encoded value {value} does not fit into {bits} bits")
    return value - bias


def collect_referenced_node_ids(search_trace: dict[str, Any]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []

    def push(node_id: int) -> None:
        node_id = canonical_node_id(int(node_id))
        if is_node_id_sentinel(node_id):
            return
        if node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)

    push(int(search_trace["entry_point"]))
    for step in search_trace["upper_trace"]:
        push(int(step["current"]))
        push(int(step["selected"]))
        for nb in step["neighbors"]:
            push(int(nb))

    level0 = search_trace["level0"]
    push(int(level0["entry"]))
    for node_id in level0["final_topk"]:
        push(int(node_id))
    for exp in level0["expansions"]:
        push(int(exp["pop_id"]))
        for nb in exp["neighbors"]:
            push(int(nb))
        for visit in exp["neighbor_visits"]:
            push(int(visit["node_id"]))
    stop_info = level0.get("stop")
    if isinstance(stop_info, dict) and "candidate" in stop_info:
        push(int(stop_info["candidate"]))

    return ordered


def level0_trace_from_quantized_scores(
    level0_trace: dict[str, Any],
    score_map: dict[int, int],
    ref_map: dict[int, dict[str, int]],
    *,
    score_node_capacity: int | None = None,
) -> dict[str, Any]:
    ef_search = int(level0_trace["ef_search"])
    k = int(level0_trace["k"])
    entry = int(level0_trace["entry"])
    expansions = level0_trace["expansions"]
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if k <= 0 or k > ef_search:
        raise ValueError("k must be positive and no larger than ef_search")
    if not isinstance(expansions, list):
        raise ValueError("level-0 expansions must be a list")
    observed_round_width = max(
        (len(exp.get("neighbors", [])) for exp in expansions), default=0
    )
    round_width = int(level0_trace.get("round_width", observed_round_width))
    if "round_width" in level0_trace and round_width != observed_round_width:
        raise ValueError(
            "explicit round_width must equal the observed neighbor-row width; "
            f"got {round_width}, observed {observed_round_width}. "
            "Pad every authenticated row with sentinel neighbors before using a larger fixed width."
        )
    position_max = c_position_limit(len(expansions), round_width)
    assert_packed_priority_domain(position_max, ef_search)
    sent_id = NODE_ID_SENTINEL

    score_enc_map = {
        int(node_id): int(encode_score_with_bias(int(score)))
        for node_id, score in score_map.items()
    }
    if not score_map:
        raise ValueError("node_count for quantized level-0 trace must be positive")
    if score_node_capacity is None:
        score_node_capacity = len(score_map)
    score_node_capacity = int(score_node_capacity)
    if score_node_capacity < len(score_map):
        raise ValueError(
            "score_node_capacity cannot be smaller than the real score table: "
            f"capacity={score_node_capacity} real_rows={len(score_map)}"
        )
    dummy_ref = {
        "node_id": int(sent_id),
        "score_enc": 0,
        "model_id": 0,
    }

    def dist_ref(node_id: int) -> dict[str, int]:
        node_id = canonical_node_id(int(node_id))
        if int(node_id) == int(sent_id):
            return dict(dummy_ref)
        ref = dict(ref_map[int(node_id)])
        expected_score = int(score_enc_map[int(node_id)])
        if int(ref["score_enc"]) != expected_score:
            raise ValueError(
                f"score_enc mismatch for node {node_id}: ref={ref['score_enc']} expected={expected_score}"
            )
        return ref

    def c_real_token(node_id: int, tie_position: int) -> dict[str, int]:
        score_enc = int(score_enc_map[int(node_id)])
        tie_position = _checked_position(tie_position, "candidate tie position")
        return {
            "node_id": int(node_id),
            "score_enc": int(score_enc),
            "src_time": int(tie_position),
            "dummy_ordinal": 0,
            "tie_position": int(tie_position),
            "ord": int(c_packed_priority_key(score_enc, tie_position, position_max)),
        }

    def c_disabled_token(src_time: int) -> dict[str, int]:
        return {
            "node_id": int(sent_id),
            "score_enc": 0,
            "src_time": int(src_time),
            "dummy_ordinal": 0,
            "tie_position": 0,
            "ord": 0,
        }

    def w_real_token(node_id: int, src_time: int, tie_position: int) -> dict[str, int]:
        score_enc = int(score_enc_map[int(node_id)])
        tie_position = _checked_position(tie_position, "working tie position")
        return {
            "node_id": int(node_id),
            "score_enc": int(score_enc),
            "src_time": int(src_time),
            "dummy_ordinal": 0,
            "tie_position": int(tie_position),
            "ord": int(
                w_packed_priority_key(
                    score_enc,
                    tie_position,
                    position_max,
                    ef_search,
                )
            ),
        }

    def w_dummy_token(dummy_ordinal: int) -> dict[str, int]:
        src_time = w_initial_position(int(dummy_ordinal) - 1)
        return {
            "node_id": int(sent_id),
            "score_enc": 0,
            "src_time": int(src_time),
            "dummy_ordinal": int(dummy_ordinal),
            "tie_position": 0,
            # Dummies occupy keys 0..EfCap-1. Every real WorkQ key starts at
            # EfCap, so all distinct initial dummies are removed first.
            "ord": int(dummy_ordinal) - 1,
        }

    def sort_c_live(tokens: list[dict[str, int]]) -> list[dict[str, int]]:
        return sorted(
            tokens,
            key=lambda tok: (
                int(tok["ord"]),
                int(tok["src_time"]),
            ),
        )

    def sort_w_live_for_delete(tokens: list[dict[str, int]]) -> list[dict[str, int]]:
        return sorted(
            tokens,
            key=lambda tok: (
                int(tok["ord"]),
                int(tok["src_time"]),
            ),
        )

    def sort_c_final(tokens: list[dict[str, int]]) -> list[dict[str, int]]:
        return sort_c_live(tokens)

    def sort_w_final(tokens: list[dict[str, int]]) -> list[dict[str, int]]:
        return sort_w_live_for_delete(tokens)

    seen: set[int] = set()
    neighbor_visits: list[dict[str, Any]] = []
    expansions_out: list[dict[str, Any]] = []
    work_updates: list[dict[str, Any]] = []

    c_live: list[dict[str, int]] = []
    c_delete_keys: list[int] = []
    c_delete_enabled: list[bool] = []
    c_del_times: list[int] = []
    c_del_src_times: list[int] = []
    c_del_node_ids: list[int] = []
    c_del_dummy_ordinals: list[int] = []
    c_del_tie_positions: list[int] = []

    # Paper Eq. (WorkQ fixed positions): initialize all Ef slots with distinct
    # dummy tokens before any real update.  A dummy's (src_time, sentinel-id)
    # pair is its unique PQ identity.
    w_insertions: list[dict[str, int]] = []
    w_live: list[dict[str, int]] = []
    for dummy_ordinal in range(1, int(ef_search) + 1):
        dummy = w_dummy_token(dummy_ordinal)
        w_insertions.append(dummy)
        w_live.append(dummy)
    w_delete_keys: list[int] = []
    w_delete_enabled: list[bool] = []
    w_del_times: list[int] = []
    w_del_src_times: list[int] = []
    w_del_node_ids: list[int] = []
    w_del_dummy_ordinals: list[int] = []
    w_del_tie_positions: list[int] = []

    def apply_work_update(
        node_id: int,
        assigned_c_position: int,
        *,
        active: bool,
    ) -> tuple[bool, dict[str, int] | None]:
        """Append one physical WorkUpd row and apply it when enabled."""

        update_idx = len(work_updates)
        push_time = w_push_position(update_idx, ef_search)
        pop_time = w_pop_position(update_idx, ef_search)
        if not active:
            work_updates.append(
                {
                    "ref": dict(dummy_ref),
                    "active": False,
                    "assigned_c_position": 0,
                    "w_push_position": 0,
                    "w_pop_position": 0,
                    "admitted": False,
                }
            )
            w_delete_keys.append(0)
            w_delete_enabled.append(False)
            w_del_times.append(0)
            w_del_src_times.append(0)
            w_del_node_ids.append(int(sent_id))
            w_del_dummy_ordinals.append(0)
            w_del_tie_positions.append(0)
            return False, None

        node_id = canonical_node_id(node_id)
        if is_node_id_sentinel(node_id):
            raise ValueError("padding cannot enter an active WorkUpd row")
        assigned_c_position = _checked_position(
            assigned_c_position,
            "assigned candidate position",
        )
        inserted = w_real_token(node_id, push_time, assigned_c_position)
        w_insertions.append(inserted)
        w_live.append(inserted)
        w_live.sort(
            key=lambda tok: (
                int(tok["ord"]),
                int(tok["src_time"]),
            )
        )
        evicted = w_live.pop(0)
        admitted = int(evicted["src_time"]) != int(push_time)
        w_delete_keys.append(int(evicted["ord"]))
        w_delete_enabled.append(True)
        w_del_times.append(int(pop_time))
        w_del_src_times.append(int(evicted["src_time"]))
        w_del_node_ids.append(int(evicted["node_id"]))
        w_del_dummy_ordinals.append(int(evicted["dummy_ordinal"]))
        w_del_tie_positions.append(int(evicted["tie_position"]))
        work_updates.append(
            {
                "ref": dist_ref(node_id),
                "active": True,
                "assigned_c_position": int(assigned_c_position),
                "w_push_position": int(push_time),
                "w_pop_position": int(pop_time),
                "admitted": bool(admitted),
            }
        )
        if admitted:
            c_live.append(c_real_token(node_id, assigned_c_position))
        return bool(admitted), evicted

    # X starts with entry, and WorkUpd is the exact stable first-visit filter of
    # X.  Entry is therefore update 1, rather than a special initial W item.
    entry = canonical_node_id(entry)
    if is_node_id_sentinel(entry):
        raise ValueError("entry must be a real node")
    seen.add(int(entry))
    entry_admitted, _ = apply_work_update(
        int(entry), c_entry_position(), active=True
    )
    if not entry_admitted:
        raise ValueError("entry update must displace a dummy")

    inactive_suffix_started = False
    for round_idx, exp in enumerate(expansions):
        pop_id = canonical_node_id(int(exp["pop_id"]))
        round_active = int(pop_id) != int(sent_id)
        if round_active:
            if inactive_suffix_started:
                raise ValueError(
                    f"active round {round_idx} appears after the inactive suffix began"
                )
            c_live = sort_c_live(c_live)
            if not c_live:
                raise ValueError(f"C is empty before round {round_idx}")
            pop_token = c_live.pop(0)
            if int(pop_token["node_id"]) != int(pop_id):
                raise ValueError(
                    f"quantized best-first pop mismatch: expected {pop_token['node_id']}, got {pop_id}"
                )
            c_delete_keys.append(int(pop_token["ord"]))
            c_delete_enabled.append(True)
            c_del_src_times.append(int(pop_token["src_time"]))
            c_del_node_ids.append(int(pop_token["node_id"]))
            c_del_dummy_ordinals.append(0)
            c_del_tie_positions.append(int(pop_token["tie_position"]))
        else:
            inactive_suffix_started = True
            pop_token = c_disabled_token(0)
            c_delete_keys.append(0)
            c_delete_enabled.append(False)
            c_del_src_times.append(0)
            c_del_node_ids.append(int(sent_id))
            c_del_dummy_ordinals.append(0)
            c_del_tie_positions.append(0)
        c_del_times.append(c_pop_position(round_idx, round_width))

        neighbors = [canonical_node_id(int(x)) for x in exp["neighbors"]]
        visits = exp.get("neighbor_visits", [])
        if not isinstance(visits, list) or len(visits) != len(neighbors):
            raise ValueError(
                f"neighbor_visits length mismatch at pop {pop_id}: "
                f"neighbors={len(neighbors)} visits={len(visits) if isinstance(visits, list) else 'invalid'}"
            )
        if not round_active:
            for pos, (nb, visit_any) in enumerate(zip(neighbors, visits)):
                if int(nb) != int(sent_id):
                    raise ValueError(
                        f"inactive round {round_idx} has real neighbor {nb} at slot {pos}"
                    )
                if not isinstance(visit_any, dict):
                    raise ValueError(
                        f"inactive round {round_idx} visit {pos} must be an object"
                    )
                visit_id = canonical_node_id(
                    int(visit_any.get("node_id", sent_id))
                )
                if int(visit_id) != int(sent_id):
                    raise ValueError(
                        f"inactive round {round_idx} visit {pos} must reference the sentinel"
                    )
                if bool(visit_any.get("is_new", False)):
                    raise ValueError(
                        f"inactive round {round_idx} visit {pos} must have is_new=false"
                    )
                if bool(visit_any.get("accepted", False)):
                    raise ValueError(
                        f"inactive round {round_idx} visit {pos} must have accepted=false"
                    )

        first_admission_seen = False
        for pos in range(int(round_width)):
            step_c_ins_time = c_neighbor_position(round_idx, pos, round_width)
            nb = int(sent_id)
            is_new = False
            admitted = False

            if pos < len(neighbors):
                nb = int(neighbors[pos])
                visit = visits[pos] if isinstance(visits[pos], dict) else {}
                visit_id = canonical_node_id(int(visit.get("node_id", nb)))
                if visit_id != int(nb):
                    raise ValueError(
                        f"neighbor_visits[{pos}] node_id={visit_id} does not match neighbors[{pos}]={nb}"
                    )
                is_new = int(nb) != int(sent_id) and int(nb) not in seen
                trace_is_new = bool(visit.get("is_new", is_new))
                if bool(trace_is_new) != bool(is_new):
                    raise ValueError(
                        f"is_new mismatch at pop {pop_id} for neighbor {nb}: "
                        f"trace={trace_is_new} expected={is_new}"
                    )
                if is_new:
                    seen.add(int(nb))
                admitted, evicted = apply_work_update(
                    int(nb),
                    step_c_ins_time,
                    active=bool(is_new),
                )
                if admitted and not first_admission_seen:
                    first_admission_seen = True
                    if (
                        evicted is not None
                        and int(evicted["node_id"]) != int(sent_id)
                        and better_int_pair(
                            int(evicted["score_enc"]),
                            int(evicted["tie_position"]),
                            int(pop_token["score_enc"]),
                            int(pop_token["tie_position"]),
                        )
                    ):
                        raise ValueError(
                            "round threshold violation: current C pop "
                            f"{pop_id} is below the evicted WorkQ item "
                            f"{evicted['node_id']} under the stable (score, -position) key"
                        )
                trace_admitted = visit.get("accepted")
                if trace_admitted is not None and bool(trace_admitted) != bool(admitted):
                    raise ValueError(
                        f"accepted mismatch at pop {pop_id} for neighbor {nb}: "
                        f"trace={bool(trace_admitted)} expected={bool(admitted)}"
                    )
                neighbor_visits.append(
                    {
                        "node_id": int(nb),
                        "is_new": bool(is_new),
                    }
                )
            else:
                apply_work_update(
                    int(sent_id),
                    step_c_ins_time,
                    active=False,
                )
        expansions_out.append({"pop_id": int(pop_id)})

    work_update_capacity = 1 + len(expansions) * int(round_width)
    if len(work_updates) != work_update_capacity:
        raise ValueError(
            "physical WorkUpd length mismatch: "
            f"got {len(work_updates)}, expected {work_update_capacity}"
        )

    c_final_tokens = sort_c_final(c_live)
    w_final_tokens = sort_w_final(w_live)

    c_best_real = next((tok for tok in c_final_tokens if int(tok["node_id"]) != int(sent_id)), None)
    w_real_final = [tok for tok in w_final_tokens if int(tok["node_id"]) != int(sent_id)]
    if c_best_real is not None:
        if len(w_real_final) < int(ef_search):
            raise ValueError("best-first would continue because W is not full-real at final state")
        worst_real = w_real_final[0]
        # Search terminates only when the best remaining candidate is
        # strictly worse than the worst retained result under the protocol's
        # stable first-seen total key.  Node identifiers are deliberately not
        # part of this ordering.
        if not better_int_pair(
            int(worst_real["score_enc"]),
            int(worst_real["tie_position"]),
            int(c_best_real["score_enc"]),
            int(c_best_real["tie_position"]),
        ):
            raise ValueError(
                f"quantized best-first would continue beyond exported trace at node {c_best_real['node_id']}"
            )

    final_topk = [int(tok["node_id"]) for tok in reversed(w_real_final[-int(k) :])]
    if len(final_topk) != int(k):
        raise ValueError(
            f"final_topk length mismatch: expected {k}, got {len(final_topk)}"
        )
    c_pq = {
        "del_src_times": [int(x) for x in c_del_src_times],
        "del_refs": [dist_ref(int(node_id)) for node_id in c_del_node_ids],
        "del_dummy_ordinals": [int(x) for x in c_del_dummy_ordinals],
        "del_tie_positions": [int(x) for x in c_del_tie_positions],
        "final_src_times": [int(tok["src_time"]) for tok in c_final_tokens],
        "final_refs": [dist_ref(int(tok["node_id"])) for tok in c_final_tokens],
        "final_dummy_ordinals": [0 for _ in c_final_tokens],
        "final_tie_positions": [int(tok["tie_position"]) for tok in c_final_tokens],
    }
    w_pq = {
        "del_src_times": [int(x) for x in w_del_src_times],
        "del_refs": [dist_ref(int(node_id)) for node_id in w_del_node_ids],
        "del_dummy_ordinals": [int(x) for x in w_del_dummy_ordinals],
        "del_tie_positions": [int(x) for x in w_del_tie_positions],
        "final_src_times": [int(tok["src_time"]) for tok in w_final_tokens],
        "final_refs": [dist_ref(int(tok["node_id"])) for tok in w_final_tokens],
        "final_dummy_ordinals": [int(tok["dummy_ordinal"]) for tok in w_final_tokens],
        "final_tie_positions": [int(tok["tie_position"]) for tok in w_final_tokens],
    }

    c_prev_lower = build_prev_lower_trace(
        [int(x) for x in c_delete_keys] + [int(tok["ord"]) for tok in c_final_tokens],
        [int(x) for x in c_del_times] + [int(tok["src_time"]) for tok in c_final_tokens],
        [bool(x) for x in c_delete_enabled] + [True for _ in c_final_tokens],
    )
    w_prev_lower = build_prev_lower_trace(
        [int(x) for x in w_delete_keys] + [int(tok["ord"]) for tok in w_final_tokens],
        [int(x) for x in w_del_times] + [int(tok["src_time"]) for tok in w_final_tokens],
        [bool(x) for x in w_delete_enabled] + [True for _ in w_final_tokens],
        active_capacity=min(len(work_updates), score_node_capacity) + int(ef_search),
    )

    ef_search_trace = {
        "ef_search": int(ef_search),
        "k": int(k),
        "expansions": expansions_out,
        "neighbor_visits": neighbor_visits,
        "work_updates": work_updates,
        "final_topk": [dist_ref(int(node_id)) for node_id in final_topk],
        "c_pq": c_pq,
        "w_pq": w_pq,
        "c_prev_lower": c_prev_lower,
        "w_prev_lower": w_prev_lower,
    }
    return ef_search_trace


def build_proof_input(
    query_index: int,
    search_trace: dict[str, Any],
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    metric: str,
    model_ids: np.ndarray,
    quantizer: AffineQuantizationParams,
    scheme_name: str,
    projected_dim: int,
    global_node_table_path: Path,
    global_graph_rows_path: Path,
    commitment_randomness: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if metric != "dot":
        raise ValueError("proof export currently supports metric=dot only")

    ref_node_ids = collect_referenced_node_ids(search_trace)
    validate_model_ids(model_ids, int(vectors_u16.shape[0]), ref_node_ids)
    node_vectors_u16: dict[int, np.ndarray] = {}
    proof_nodes: list[dict[str, Any]] = []
    for node_id in ref_node_ids:
        node_vec_u16 = np.ascontiguousarray(vectors_u16[int(node_id)].astype(np.uint16, copy=False))
        node_vectors_u16[int(node_id)] = node_vec_u16
        score_int = int(quantized_dot_score(query_u16, node_vec_u16, quantizer.zero_point))
        model_id = int(model_ids[int(node_id)])
        proof_nodes.append(
            {
                "node_id": int(node_id),
                "model_id": int(model_id),
                "vector_u16": [int(v) for v in node_vec_u16.tolist()],
                "score_enc": int(encode_score_with_bias(score_int)),
            }
        )
    proof_nodes.sort(key=lambda node: int(node["node_id"]))
    ref_map = {
        int(node["node_id"]): {
            "node_id": int(node["node_id"]),
            "score_enc": int(node["score_enc"]),
            "model_id": int(node["model_id"]),
        }
        for node in proof_nodes
    }

    score_map = compute_quantized_scores(query_u16, node_vectors_u16, quantizer.zero_point)
    proof_upper = upper_trace_from_quantized_scores(search_trace["upper_trace"], score_map, ref_map)
    proof_ef_search = level0_trace_from_quantized_scores(
        search_trace["level0"],
        score_map,
        ref_map,
    )

    trace_upper_selected = [int(step["selected"]) for step in search_trace["upper_trace"]]
    proof_upper_selected = [int(step["selected"]["node_id"]) for step in proof_upper]
    trace_pop_order = [int(exp["pop_id"]) for exp in search_trace["level0"]["expansions"]]
    proof_pop_order = [int(x["pop_id"]) for x in proof_ef_search["expansions"]]
    trace_topk = [int(x) for x in search_trace["level0"]["final_topk"]]
    proof_topk = [int(x["node_id"]) for x in proof_ef_search["final_topk"]]
    topk_match_up_to_ties = topk_lists_match_up_to_ties(trace_topk, proof_topk, score_map)

    consistency = {
        "trace_source": str(search_trace.get("trace_source", "")),
        "upper_selected_match": bool(trace_upper_selected == proof_upper_selected),
        "level0_pop_order_match": bool(trace_pop_order == proof_pop_order),
        "final_topk_match": bool(trace_topk == proof_topk),
        "final_topk_match_up_to_ties": bool(topk_match_up_to_ties),
    }
    if not (
        bool(consistency["upper_selected_match"])
        and bool(consistency["level0_pop_order_match"])
        and bool(consistency["final_topk_match_up_to_ties"])
    ):
        raise ValueError(
            "proof trace is not self-consistent with the quantized replay decisions: "
            f"{consistency}"
        )

    expand_rows: list[dict[str, Any]] = []
    for step in search_trace["upper_trace"]:
        expand_rows.append(
            {
                "layer": int(step["layer"]),
                "src": dict(ref_map[int(step["current"])]),
                "neighbors": [dict(ref_map[int(x)]) for x in step["neighbors"]],
            }
        )
    for exp in search_trace["level0"]["expansions"]:
        expand_rows.append(
            {
                "layer": 0,
                "src": dict(ref_map[int(exp["pop_id"])]),
                "neighbors": [dict(ref_map[int(x)]) for x in exp["neighbors"]],
            }
        )

    proof_input = {
        "version": str(PROOF_INPUT_VERSION),
        "position_schema": str(POSITION_SCHEMA_PAPER_WORKUPD_V1),
        "commitment_randomness": validate_commitment_randomness(commitment_randomness),
        "metric": str(metric),
        "query_index": int(query_index),
        "score_bits": int(SCORE_BITS),
        "entry_point": int(search_trace["entry_point"]),
        # The circuit proves this selected model is a member of final_topk.
        # Keep rank one as the default routing policy while allowing callers to
        # select another authenticated top-k backend before proving.
        "selected_model_id": int(proof_ef_search["final_topk"][0]["model_id"]),
        "global_artifacts": {
            "node_table_path": str(global_node_table_path),
            "graph_rows_path": str(global_graph_rows_path),
        },
        "query": {
            "dimension": int(query_u16.shape[0]),
            "vector_u16": [int(v) for v in query_u16.tolist()],
        },
        "quantizer": {
            "scheme": str(scheme_name),
            "scale": float(quantizer.scale),
            "zero_point": int(quantizer.zero_point),
            "projected_dim": int(projected_dim),
        },
        "nodes": proof_nodes,
        "expand_rows": expand_rows,
        "upper_trace": proof_upper,
        "ef_search": proof_ef_search,
    }
    return proof_input, consistency


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
