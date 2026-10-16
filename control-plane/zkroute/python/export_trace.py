from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import core

CACHE_VERSION = "20260823_cache_v8_direct_score_hash_no_score_randomness"
PROOF_INPUT_FILENAME = "proof_input_linear.json"
LEGACY_PROOF_INPUT_FILENAMES = ("proof_input.json",)
SYNTHETIC_NOISE_SCALE_NORMALIZED = 0.02
SYNTHETIC_NOISE_SCALE_UNNORMALIZED = 0.15


def fresh_randomness_digest() -> list[int]:
    """Sample four independent uniform canonical Goldilocks elements."""
    limbs: list[int] = []
    while len(limbs) < 4:
        candidate = secrets.randbits(64)
        if candidate < core.GOLDILOCKS_MODULUS:
            limbs.append(candidate)
    return limbs


def load_or_create_index_randomness(
    path: Path,
    *,
    refresh: bool,
) -> dict[str, list[int]]:
    if not refresh and path.exists():
        payload = read_json(path)
        # Validate the complete object so a legacy `score` member is rejected
        # explicitly instead of being projected away during cache loading.
        validated = core.validate_commitment_randomness(payload)
        return {
            "reference": validated["reference"],
            "graph": validated["graph"],
        }
    payload = {
        "reference": fresh_randomness_digest(),
        "graph": fresh_randomness_digest(),
    }
    atomic_dump_json(path, payload)
    return payload


def proof_input_cache_path(cache_dir: Path) -> Path:
    preferred = cache_dir / PROOF_INPUT_FILENAME
    if preferred.exists():
        return preferred
    for legacy_name in LEGACY_PROOF_INPUT_FILENAMES:
        legacy_path = cache_dir / legacy_name
        if legacy_path.exists():
            return legacy_path
    return preferred


def trace_cache_complete(cache_dir: Path) -> bool:
    return (
        (cache_dir / "search_trace.json").exists()
        and proof_input_cache_path(cache_dir).exists()
        and (cache_dir / "proof_equivalence.json").exists()
        and (cache_dir / "trace_manifest.json").exists()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    raw_default = root / "raw_data"
    out_default = root / "VeriRouteRoute_out"
    cache_default = root / "cache" / "VeriRouteRoute"

    ap = argparse.ArgumentParser(
        description="Build or reuse cached quantized-embedding FAISS HNSW artifacts, export a proof-aligned trace, and emit global-root proof input."
    )
    ap.add_argument("--raw-data-dir", type=str, default=str(raw_default))
    ap.add_argument("--train-parquet", type=str, default="train-00000-of-00001-embedding.parquet")
    ap.add_argument("--test-parquet", type=str, default="train-00000-of-00001-test.parquet")
    ap.add_argument(
        "--embedding-id-csv",
        type=str,
        default="train-00000-of-00001-embedding.id.csv",
    )
    ap.add_argument("--metric", type=str, default="dot", choices=["dot"])
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--n-train", type=int, default=0, help="0 means all train vectors")
    ap.add_argument(
        "--allow-synthetic-train-extension",
        action="store_true",
        default=False,
        help=(
            "When n-train exceeds real data size, append deterministic clustered "
            "bootstrap samples around real vectors to reach the target graph size."
        ),
    )
    ap.add_argument(
        "--synthetic-train-seed",
        type=int,
        default=12345,
        help="Seed used for deterministic synthetic train-vector generation.",
    )
    ap.add_argument(
        "--train-subset-mode",
        choices=["prefix", "dense-coverage"],
        default="prefix",
        help=(
            "Select real train rows by source prefix, or use one deterministic nested "
            "ordering that inserts a unique dense real-neighbor community into the "
            "smallest coverage graph."
        ),
    )
    ap.add_argument("--dense-coverage-prefix-size", type=int, default=5000)
    ap.add_argument("--dense-coverage-anchor-count", type=int, default=36)
    ap.add_argument("--dense-coverage-candidate-count", type=int, default=1024)
    ap.add_argument(
        "--n-test",
        type=int,
        default=64,
        help=(
            "size of the deterministic sampled test-query pool; 0 means all test "
            "queries, and this is not a per-bucket sample count"
        ),
    )
    ap.add_argument("--query-seed", type=int, default=0)
    ap.add_argument("--coverage-probe-count", type=int, default=0)
    ap.add_argument("--coverage-probe-seed", type=int, default=20260722)
    ap.add_argument("--coverage-probe-noise", type=float, default=0.005)
    ap.add_argument(
        "--coverage-probe-anchor-query-position",
        action="append",
        type=int,
        default=[],
    )
    ap.add_argument("--coverage-probe-anchor-count", type=int, default=0)
    ap.add_argument(
        "--coverage-probe-anchor-mode",
        choices=[
            "local",
            "interpolate",
            "generated-local",
            "refined-local",
            "score-boundary",
        ],
        default="local",
    )
    ap.add_argument(
        "--coverage-probe-generated-anchor-offset",
        action="append",
        type=int,
        default=[],
    )
    ap.add_argument(
        "--coverage-probe-direction-node",
        action="append",
        type=int,
        default=[],
    )
    ap.add_argument("--trace-query-pos", type=int, default=0)
    ap.add_argument("--scan-query-positions-file", type=str, default="")
    ap.add_argument("--scan-report", type=str, default="")
    ap.add_argument("--scan-resume", action="store_true", default=False)
    ap.add_argument("--scan-checkpoint-every", type=int, default=8)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--faiss-seed", type=int, default=12345)
    ap.add_argument("--faiss-threads", type=int, default=1)
    ap.add_argument("--scheme", type=str, default="affine", choices=["affine", "proj_affine"])
    ap.add_argument("--proj-dim", type=int, default=384)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--proj-percentile", type=float, default=99.9)
    ap.add_argument("--synthetic-chunk-rows", type=int, default=8192)
    ap.add_argument("--faiss-add-batch-rows", type=int, default=50000)
    ap.add_argument("--graph-nodes-limit", type=int, default=1000)
    ap.add_argument("--trace-source", type=str, default="auto", choices=["auto", "direct", "replay"])
    ap.add_argument("--out-dir", type=str, default=str(out_default))
    ap.add_argument("--results-dir", type=str, default="")
    ap.add_argument("--cache-dir", type=str, default=str(cache_default))
    ap.add_argument("--refresh-cache", action="store_true", default=False)
    ap.add_argument(
        "--skip-real-search-comparison",
        action="store_true",
        default=False,
        help="Skip the float-search comparison pass. Useful for very large synthetic graphs.",
    )
    return ap.parse_args(argv)


def sample_queries(q_all: np.ndarray, n_test: int, query_seed: int) -> tuple[np.ndarray, np.ndarray]:
    q_indices_all = np.arange(q_all.shape[0], dtype=np.int64)
    if int(n_test) > 0 and int(n_test) < q_all.shape[0]:
        rng = np.random.default_rng(int(query_seed))
        sample_idx = np.sort(rng.choice(q_all.shape[0], size=int(n_test), replace=False))
        sample_idx = np.asarray(sample_idx, dtype=np.int64)
        return (
            np.ascontiguousarray(q_all[sample_idx]),
            np.ascontiguousarray(q_indices_all[sample_idx]),
        )
    if int(n_test) > 0:
        return (
            np.ascontiguousarray(q_all[: int(n_test)]),
            np.ascontiguousarray(q_indices_all[: int(n_test)]),
        )
    return np.ascontiguousarray(q_all), np.ascontiguousarray(q_indices_all)


def synthesize_random_train_vectors(
    base_vectors: np.ndarray,
    synthetic_count: int,
    *,
    normalize: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    synthetic_count = int(synthetic_count)
    if synthetic_count <= 0:
        dim = int(base_vectors.shape[1])
        return (
            np.empty((0, dim), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )

    rng = np.random.default_rng(int(seed))
    if base_vectors.size == 0:
        synthetic = rng.standard_normal((synthetic_count, 0)).astype(np.float32, copy=False)
        return np.ascontiguousarray(synthetic), np.empty(0, dtype=np.uint32)

    sampled_idx = rng.integers(
        0,
        int(base_vectors.shape[0]),
        size=synthetic_count,
        endpoint=False,
        dtype=np.uint32,
    )
    anchors = np.ascontiguousarray(base_vectors[sampled_idx].astype(np.float32, copy=False))
    noise = rng.standard_normal((synthetic_count, int(base_vectors.shape[1]))).astype(np.float32, copy=False)

    if bool(normalize):
        noise_scale = np.float32(SYNTHETIC_NOISE_SCALE_NORMALIZED)
        synthetic = anchors + noise * noise_scale
        synthetic = core.l2_normalize(np.ascontiguousarray(synthetic))
    else:
        std = np.std(base_vectors, axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.where(std > 1e-6, std, np.float32(1.0))
        noise_scale = np.float32(SYNTHETIC_NOISE_SCALE_UNNORMALIZED)
        synthetic = anchors + noise * (std * noise_scale)
        synthetic = np.ascontiguousarray(synthetic.astype(np.float32, copy=False))
    return (
        np.ascontiguousarray(synthetic.astype(np.float32, copy=False)),
        np.ascontiguousarray(sampled_idx),
    )


def extend_model_ids(
    model_ids: np.ndarray,
    target_count: int,
    synthetic_anchor_positions: np.ndarray | None = None,
) -> np.ndarray:
    target_count = int(target_count)
    base = np.ascontiguousarray(model_ids.astype(np.uint32, copy=False))
    if target_count <= int(base.shape[0]):
        return np.ascontiguousarray(base[:target_count])
    if int(base.shape[0]) == 0:
        raise ValueError("cannot extend an empty model-ID table")
    synthetic_count = target_count - int(base.shape[0])
    if synthetic_anchor_positions is None:
        raise ValueError("synthetic model IDs require their sampled anchor positions")
    anchors = np.ascontiguousarray(
        synthetic_anchor_positions.astype(np.int64, copy=False)
    )
    if int(anchors.shape[0]) != synthetic_count:
        raise ValueError(
            "synthetic anchor count does not match requested model-ID extension: "
            f"{int(anchors.shape[0])} vs {synthetic_count}"
        )
    if np.any(anchors < 0) or np.any(anchors >= int(base.shape[0])):
        raise ValueError("synthetic anchor position is outside the real model-ID table")
    extended = np.concatenate([base, base[anchors]], axis=0)
    return np.ascontiguousarray(extended.astype(np.uint32, copy=False))


def dense_coverage_real_order(
    faiss,
    x_all: np.ndarray,
    *,
    prefix_size: int,
    anchor_count: int,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    row_count = int(x_all.shape[0])
    prefix_size = int(prefix_size)
    anchor_count = int(anchor_count)
    candidate_count = int(candidate_count)
    if anchor_count <= 0 or prefix_size < anchor_count or prefix_size > row_count:
        raise ValueError(
            "dense coverage requires 0 < anchor_count <= prefix_size <= real row count"
        )
    if candidate_count <= 0:
        raise ValueError("dense coverage candidate count must be positive")

    density_vectors = core.l2_normalize(
        np.ascontiguousarray(x_all.astype(np.float32, copy=False))
    )
    candidate_ids = np.linspace(
        0,
        row_count - 1,
        num=min(candidate_count, row_count),
        dtype=np.int64,
    )
    index = faiss.IndexFlatIP(int(density_vectors.shape[1]))
    index.add(density_vectors)
    search_k = min(row_count, max(256, anchor_count * 4))
    scores, ids = index.search(
        np.ascontiguousarray(density_vectors[candidate_ids]), search_k
    )
    row_hashes = [
        hashlib.blake2b(
            memoryview(np.ascontiguousarray(row)), digest_size=8
        ).digest()
        for row in density_vectors
    ]

    selected_anchors: list[int] = []
    selected_center = -1
    selected_density: float | None = None
    for candidate_position, candidate_id in enumerate(candidate_ids):
        unique_hashes: set[bytes] = set()
        members: list[int] = []
        last_score = float("-inf")
        for rank, raw_id in enumerate(ids[candidate_position]):
            row_id = int(raw_id)
            if row_id < 0 or row_hashes[row_id] in unique_hashes:
                continue
            unique_hashes.add(row_hashes[row_id])
            members.append(row_id)
            last_score = float(scores[candidate_position, rank])
            if len(members) == anchor_count:
                break
        if len(members) != anchor_count:
            continue
        if selected_density is None or last_score > selected_density:
            selected_anchors = members
            selected_center = int(candidate_id)
            selected_density = last_score
    if len(selected_anchors) != anchor_count or selected_density is None:
        raise RuntimeError("could not identify a complete unique dense coverage community")

    anchor_set = set(selected_anchors)
    remaining = [row_id for row_id in range(row_count) if row_id not in anchor_set]
    insertion_position = prefix_size - anchor_count
    order = np.asarray(
        remaining[:insertion_position]
        + selected_anchors
        + remaining[insertion_position:],
        dtype=np.int64,
    )
    if int(order.shape[0]) != row_count or len(set(order.tolist())) != row_count:
        raise RuntimeError("dense coverage row ordering is not a permutation")
    center = np.mean(x_all[selected_anchors], axis=0, keepdims=True, dtype=np.float64)
    center = core.l2_normalize(center.astype(np.float32, copy=False))[0]
    return (
        np.ascontiguousarray(order),
        np.ascontiguousarray(center),
        {
            "mode": "dense-coverage",
            "selection": "exact_ip_unique_neighbor_density",
            "prefix_size": prefix_size,
            "anchor_count": anchor_count,
            "candidate_count": min(candidate_count, row_count),
            "candidate_schedule": "linspace_over_all_real_rows",
            "anchor_source_indices": selected_anchors,
            "center_candidate_source_index": selected_center,
            "anchor_boundary_cosine": float(selected_density),
            "insertion_position": insertion_position,
            "nested_prefix_order": True,
        },
    )


def select_real_train_vectors(
    faiss,
    args: argparse.Namespace,
    x_all: np.ndarray,
    real_count_used: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if str(args.train_subset_mode) == "dense-coverage":
        order, center, metadata = dense_coverage_real_order(
            faiss,
            x_all,
            prefix_size=int(args.dense_coverage_prefix_size),
            anchor_count=int(args.dense_coverage_anchor_count),
            candidate_count=int(args.dense_coverage_candidate_count),
        )
        if int(real_count_used) < int(args.dense_coverage_prefix_size):
            raise ValueError(
                "dense coverage graph is smaller than dense-coverage-prefix-size: "
                f"{real_count_used} < {int(args.dense_coverage_prefix_size)}"
            )
    else:
        order = np.arange(int(x_all.shape[0]), dtype=np.int64)
        center = None
        metadata = {
            "mode": "prefix",
            "nested_prefix_order": True,
        }
    source_indices = np.ascontiguousarray(order[: int(real_count_used)])
    return (
        np.ascontiguousarray(x_all[source_indices]),
        source_indices,
        center,
        metadata,
    )


def build_train_vectors(
    faiss,
    args: argparse.Namespace,
    train_parquet: Path,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    x_all = core.load_embeddings(train_parquet)
    if bool(args.normalize):
        x_all = core.l2_normalize(x_all)

    real_count_available = int(x_all.shape[0])
    requested_count = real_count_available if int(args.n_train) <= 0 else int(args.n_train)
    real_count_used = min(int(requested_count), int(real_count_available))
    x_real, real_source_indices, coverage_center, subset_metadata = (
        select_real_train_vectors(faiss, args, x_all, real_count_used)
    )

    synthetic_count = 0
    synthetic_anchor_positions = np.empty(0, dtype=np.uint32)
    if int(requested_count) > int(real_count_available) and bool(args.allow_synthetic_train_extension):
        synthetic_count = int(requested_count) - int(real_count_available)
        synthetic, synthetic_anchor_positions = synthesize_random_train_vectors(
            x_real,
            synthetic_count,
            normalize=bool(args.normalize),
            seed=int(args.synthetic_train_seed),
        )
        x_train = np.ascontiguousarray(np.concatenate([x_real, synthetic], axis=0))
    else:
        x_train = x_real

    metadata = {
        "real_count_available": int(real_count_available),
        "real_count_used": int(real_count_used),
        "requested_count": int(requested_count),
        "used_count": int(x_train.shape[0]),
        "synthetic_count": int(synthetic_count),
        "synthetic_extension_requested": bool(args.allow_synthetic_train_extension),
        "synthetic_extension_used": bool(synthetic_count > 0),
        "synthetic_train_seed": int(args.synthetic_train_seed),
        "real_subset": subset_metadata,
        "synthetic_generation": {
            "strategy": "empirical_anchor_bootstrap_with_isotropic_gaussian_perturbation",
            "normalize": bool(args.normalize),
            "noise_scale_if_normalized": SYNTHETIC_NOISE_SCALE_NORMALIZED,
            "noise_scale_if_not_normalized": SYNTHETIC_NOISE_SCALE_UNNORMALIZED,
            "noise_scale_semantics": "per_coordinate_before_optional_l2_normalization",
            "anchor_sampling": "uniform_with_replacement_over_real_rows",
            "model_id_policy": "inherit_sampled_real_anchor",
            "intended_use": "scalability_stress_not_distribution_fidelity",
        },
    }
    return (
        x_train,
        metadata,
        real_source_indices,
        synthetic_anchor_positions,
        coverage_center,
    )


def build_streamed_train_vectors(
    faiss,
    args,
    train_parquet: Path,
    vectors_path: Path,
    synthetic_anchor_positions_path: Path,
) -> tuple[
    np.ndarray,
    core.SearchSchemeParams,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    x_all = core.load_embeddings(train_parquet)
    if bool(args.normalize):
        x_all = core.l2_normalize(x_all)

    real_count_available = int(x_all.shape[0])
    requested_count = real_count_available if int(args.n_train) <= 0 else int(args.n_train)
    real_count_used = min(requested_count, real_count_available)
    if requested_count <= real_count_available or not bool(args.allow_synthetic_train_extension):
        raise ValueError("streamed train extension requires a synthetic train extension")

    x_real, real_source_indices, coverage_center, subset_metadata = (
        select_real_train_vectors(faiss, args, x_all, real_count_used)
    )
    scheme = core.fit_search_scheme(
        x_train=x_real,
        scheme=str(args.scheme),
        proj_dim=int(args.proj_dim),
        proj_seed=int(args.proj_seed),
        proj_percentile=float(args.proj_percentile),
    )
    transformed_real = core.transform_search_values(x_real, scheme)
    vectors_u16 = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.uint16,
        shape=(requested_count, int(scheme.projected_dim)),
    )
    vectors_u16[:real_count_used] = core.quantize_affine(transformed_real, scheme.quantizer)

    synthetic_count = requested_count - real_count_used
    rng = np.random.default_rng(int(args.synthetic_train_seed))
    chunk_rows = max(1, int(args.synthetic_chunk_rows))
    sampled_idx = np.lib.format.open_memmap(
        synthetic_anchor_positions_path,
        mode="w+",
        dtype=np.uint32,
        shape=(synthetic_count,),
    )
    for start in range(0, synthetic_count, chunk_rows):
        stop = min(synthetic_count, start + chunk_rows)
        sampled_idx[start:stop] = rng.integers(
            0,
            real_count_used,
            size=stop - start,
            endpoint=False,
            dtype=np.uint32,
        )
    sampled_idx.flush()
    unnormalized_std: np.ndarray | None = None
    if not bool(args.normalize):
        unnormalized_std = np.std(
            x_real, axis=0, dtype=np.float64
        ).astype(np.float32, copy=False)
        unnormalized_std = np.where(
            unnormalized_std > 1e-6, unnormalized_std, np.float32(1.0)
        )
    for start in range(0, synthetic_count, chunk_rows):
        stop = min(synthetic_count, start + chunk_rows)
        anchors = np.ascontiguousarray(x_real[sampled_idx[start:stop]])
        noise = rng.standard_normal((stop - start, int(x_real.shape[1]))).astype(
            np.float32,
            copy=False,
        )
        if bool(args.normalize):
            synthetic = anchors + noise * np.float32(
                SYNTHETIC_NOISE_SCALE_NORMALIZED
            )
            synthetic = core.l2_normalize(synthetic)
        else:
            assert unnormalized_std is not None
            synthetic = anchors + noise * (
                unnormalized_std * np.float32(SYNTHETIC_NOISE_SCALE_UNNORMALIZED)
            )
        transformed = core.transform_search_values(synthetic, scheme)
        vectors_u16[real_count_used + start : real_count_used + stop] = (
            core.quantize_affine(transformed, scheme.quantizer)
        )
    vectors_u16.flush()

    metadata = {
        "real_count_available": real_count_available,
        "real_count_used": real_count_used,
        "requested_count": requested_count,
        "used_count": requested_count,
        "synthetic_count": synthetic_count,
        "synthetic_extension_requested": True,
        "synthetic_extension_used": True,
        "synthetic_train_seed": int(args.synthetic_train_seed),
        "real_subset": subset_metadata,
        "synthetic_generation": {
            "strategy": (
                "chunked_bootstrap_real_vectors_then_project"
                if scheme.projection_matrix is not None
                else "chunked_bootstrap_real_vectors_affine1536"
            ),
            "normalize": bool(args.normalize),
            "noise_scale_if_normalized": SYNTHETIC_NOISE_SCALE_NORMALIZED,
            "noise_scale_if_not_normalized": SYNTHETIC_NOISE_SCALE_UNNORMALIZED,
            "noise_scale_semantics": "per_coordinate_before_optional_l2_normalization",
            "chunk_rows": chunk_rows,
            "anchor_sampling": "uniform_with_replacement_over_real_rows",
            "model_id_policy": "inherit_sampled_real_anchor",
            "intended_use": "scalability_stress_not_distribution_fidelity",
            "quantizer_fit_scope": "real_rows",
            "projection_fit_scope": (
                "real_rows" if scheme.projection_matrix is not None else None
            ),
        },
    }
    return (
        vectors_u16,
        scheme,
        metadata,
        real_source_indices,
        sampled_idx,
        coverage_center,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def maybe_fingerprint(path: Path) -> dict[str, int | str] | None:
    return core.file_fingerprint(path) if path.exists() else None


def parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as parquet

    return int(parquet.ParquetFile(path).metadata.num_rows)


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(core.to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def faiss_build_id(faiss) -> dict[str, Any]:
    return {
        "module_file": str(getattr(faiss, "__file__", "")),
        "has_hnsw_search_prove_trace_json": bool(hasattr(faiss, "hnsw_search_prove_trace_json")),
        "has_hnsw_search_trace_json": bool(hasattr(faiss, "hnsw_search_trace_json")),
    }


def save_scheme(cache_dir: Path, scheme: core.SearchSchemeParams) -> None:
    meta = {
        "name": str(scheme.name),
        "projected_dim": int(scheme.projected_dim),
        "projection_seed": int(scheme.projection_seed),
        "projection_percentile": float(scheme.projection_percentile),
        "quantizer": {
            "scale": float(scheme.quantizer.scale),
            "zero_point": int(scheme.quantizer.zero_point),
            "qmin": int(scheme.quantizer.qmin),
            "qmax": int(scheme.quantizer.qmax),
            "min_value": float(scheme.quantizer.min_value),
            "max_value": float(scheme.quantizer.max_value),
        },
        "has_projection_matrix": bool(scheme.projection_matrix is not None),
    }
    core.dump_json(cache_dir / "scheme.json", meta)
    if scheme.projection_matrix is not None:
        np.save(cache_dir / "projection_matrix.npy", np.ascontiguousarray(scheme.projection_matrix))


def load_scheme(cache_dir: Path) -> core.SearchSchemeParams:
    meta = read_json(cache_dir / "scheme.json")
    projection_matrix = None
    proj_path = cache_dir / "projection_matrix.npy"
    if bool(meta.get("has_projection_matrix")):
        projection_matrix = np.ascontiguousarray(np.load(proj_path).astype(np.float32, copy=False))
    quant_meta = dict(meta["quantizer"])
    quantizer = core.AffineQuantizationParams(
        scale=float(quant_meta["scale"]),
        zero_point=int(quant_meta["zero_point"]),
        qmin=int(quant_meta["qmin"]),
        qmax=int(quant_meta["qmax"]),
        min_value=float(quant_meta["min_value"]),
        max_value=float(quant_meta["max_value"]),
    )
    return core.SearchSchemeParams(
        name=str(meta["name"]),
        quantizer=quantizer,
        projected_dim=int(meta["projected_dim"]),
        projection_seed=int(meta["projection_seed"]),
        projection_percentile=float(meta["projection_percentile"]),
        projection_matrix=projection_matrix,
    )


def materialize_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)


def clear_optional_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def prepare_train_cache(
    faiss,
    args: argparse.Namespace,
    train_parquet: Path,
    embedding_id_csv: Path,
    train_csv: Path,
    cache_root: Path,
) -> dict[str, Any]:
    key_payload = {
        "version": CACHE_VERSION,
        "kind": "train_quant",
        "input_semantics": core.INPUT_SEMANTICS_VERSION,
        "train_parquet": core.file_fingerprint(train_parquet),
        "embedding_id_csv": maybe_fingerprint(embedding_id_csv),
        "train_csv": maybe_fingerprint(train_csv),
        "normalize": bool(args.normalize),
        "n_train": int(args.n_train),
        "allow_synthetic_train_extension": bool(args.allow_synthetic_train_extension),
        "synthetic_train_seed": int(args.synthetic_train_seed),
        "scheme": str(args.scheme),
        "proj_dim": int(args.proj_dim),
        "proj_seed": int(args.proj_seed),
        "proj_percentile": float(args.proj_percentile),
    }
    if str(args.train_subset_mode) != "prefix":
        key_payload["train_subset"] = {
            "version": "dense_coverage_exact_ip_v1",
            "mode": str(args.train_subset_mode),
            "prefix_size": int(args.dense_coverage_prefix_size),
            "anchor_count": int(args.dense_coverage_anchor_count),
            "candidate_count": int(args.dense_coverage_candidate_count),
        }
    if bool(args.allow_synthetic_train_extension) and int(args.n_train) > 0:
        key_payload["synthetic_streaming_extension"] = {
            "version": "chunked_v4_anchor_labels_distribution_provenance",
            "noise_scale_normalized": SYNTHETIC_NOISE_SCALE_NORMALIZED,
            "noise_scale_unnormalized": SYNTHETIC_NOISE_SCALE_UNNORMALIZED,
        }
    train_key = cache_key(key_payload)
    train_dir = cache_root / "train_quant" / train_key
    train_dir.mkdir(parents=True, exist_ok=True)
    meta_path = train_dir / "meta.json"
    vectors_path = train_dir / "x_train_u16.npy"
    model_ids_path = train_dir / "model_ids.npy"
    real_source_indices_path = train_dir / "real_source_indices.npy"
    synthetic_anchor_positions_path = train_dir / "synthetic_anchor_positions.npy"
    coverage_probe_center_path = train_dir / "coverage_probe_center_f32.npy"
    global_nodes_path = train_dir / "global_quantized_nodes.jsonl"
    real_count_available = parquet_row_count(train_parquet)

    cache_hit = (
        (not bool(args.refresh_cache))
        and meta_path.exists()
        and vectors_path.exists()
        and model_ids_path.exists()
        and (train_dir / "scheme.json").exists()
        and (
            str(args.train_subset_mode) == "prefix"
            or (
                real_source_indices_path.exists()
                and coverage_probe_center_path.exists()
            )
        )
        and (
            not bool(args.allow_synthetic_train_extension)
            or int(args.n_train) <= real_count_available
            or synthetic_anchor_positions_path.exists()
        )
    )
    if not cache_hit:
        use_streamed_extension = (
            bool(args.allow_synthetic_train_extension)
            and int(args.n_train) > 0
            and int(args.n_train) > real_count_available
        )
        if use_streamed_extension:
            (
                x_train_u16,
                scheme,
                train_build,
                real_source_indices,
                synthetic_anchor_positions,
                coverage_probe_center,
            ) = build_streamed_train_vectors(
                faiss,
                args,
                train_parquet,
                vectors_path,
                synthetic_anchor_positions_path,
            )
        else:
            (
                x_train,
                train_build,
                real_source_indices,
                synthetic_anchor_positions,
                coverage_probe_center,
            ) = build_train_vectors(
                faiss,
                args,
                train_parquet,
            )
            scheme = core.fit_search_scheme(
                x_train=x_train,
                scheme=str(args.scheme),
                proj_dim=int(args.proj_dim),
                proj_seed=int(args.proj_seed),
                proj_percentile=float(args.proj_percentile),
            )
            x_train_u16 = core.quantize_search_values(x_train, scheme)
            np.save(vectors_path, np.ascontiguousarray(x_train_u16))
            if int(synthetic_anchor_positions.shape[0]) > 0:
                np.save(
                    synthetic_anchor_positions_path,
                    np.ascontiguousarray(synthetic_anchor_positions),
                )
        np.save(real_source_indices_path, np.ascontiguousarray(real_source_indices))
        if coverage_probe_center is not None:
            np.save(
                coverage_probe_center_path,
                np.ascontiguousarray(coverage_probe_center, dtype=np.float32),
            )
        elif coverage_probe_center_path.exists():
            coverage_probe_center_path.unlink()

        model_ids_all = core.load_embedding_model_ids(
            embedding_id_csv,
            int(train_build["real_count_available"]),
            train_csv_path=train_csv,
        )
        model_ids_real = np.ascontiguousarray(model_ids_all[real_source_indices])
        model_ids = extend_model_ids(
            model_ids_real,
            int(train_build["used_count"]),
            synthetic_anchor_positions,
        )
        save_scheme(train_dir, scheme)
        np.save(model_ids_path, np.ascontiguousarray(model_ids))
        core.write_global_quantized_nodes_jsonl(global_nodes_path, x_train_u16, model_ids)
        core.dump_json(
            meta_path,
            {
                "train_key": train_key,
                "n_train_used": int(train_build["used_count"]),
                "train_dim": int(scheme.projected_dim),
                "normalize": bool(args.normalize),
                "embedding_id_csv_exists": bool(embedding_id_csv.exists()),
                "train_csv_exists": bool(train_csv.exists()),
                "train_parquet": core.file_fingerprint(train_parquet),
                "synthetic_train": train_build,
                "real_source_indices_path": str(real_source_indices_path.resolve()),
                "coverage_probe_center_path": (
                    str(coverage_probe_center_path.resolve())
                    if coverage_probe_center_path.exists()
                    else None
                ),
                "synthetic_anchor_positions_path": (
                    str(synthetic_anchor_positions_path.resolve())
                    if synthetic_anchor_positions_path.exists()
                    else None
                ),
            },
        )

    meta = read_json(meta_path)
    scheme = load_scheme(train_dir)
    x_train_u16 = np.load(vectors_path, mmap_mode="r")
    model_ids = np.load(model_ids_path)
    core.validate_model_ids(model_ids, int(x_train_u16.shape[0]))
    real_source_indices = (
        np.load(real_source_indices_path, mmap_mode="r")
        if real_source_indices_path.exists()
        else np.arange(
            int((meta.get("synthetic_train") or {}).get("real_count_used", 0)),
            dtype=np.int64,
        )
    )
    coverage_probe_center = (
        np.ascontiguousarray(
            np.load(coverage_probe_center_path).astype(np.float32, copy=False)
        )
        if coverage_probe_center_path.exists()
        else None
    )
    if not global_nodes_path.exists():
        core.write_global_quantized_nodes_jsonl(global_nodes_path, x_train_u16, model_ids)
    return {
        "key": train_key,
        "dir": train_dir,
        "meta": meta,
        "scheme": scheme,
        "x_train_u16": x_train_u16,
        "model_ids": model_ids,
        "real_source_indices": real_source_indices,
        "coverage_probe_center": coverage_probe_center,
        "global_nodes_path": global_nodes_path,
        "cache_hit": cache_hit,
    }


def build_coverage_probe_bank(
    args: argparse.Namespace,
    train_cache: dict[str, Any],
    heldout_queries: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    requested = max(0, int(args.coverage_probe_count))
    if requested == 0:
        dim = int(train_cache["scheme"].projected_dim)
        return np.empty((0, dim), dtype=np.float32), []
    scheme = train_cache["scheme"]
    if scheme.projection_matrix is not None:
        raise ValueError("coverage probes currently require the affine non-projected scheme")
    if not bool(args.normalize):
        raise ValueError("coverage probes require normalized train and query vectors")
    center = train_cache.get("coverage_probe_center")
    if not isinstance(center, np.ndarray):
        raise ValueError(
            "coverage probes require --train-subset-mode dense-coverage"
        )

    rng = np.random.default_rng(int(args.coverage_probe_seed))
    dim = int(center.shape[0])
    arrays: list[np.ndarray] = []
    families: list[str] = []

    def append_normalized(family: str, values: np.ndarray) -> None:
        normalized = core.l2_normalize(
            np.ascontiguousarray(values, dtype=np.float32)
        )
        arrays.append(normalized)
        families.extend([family] * int(normalized.shape[0]))

    local_count = min(8, requested)
    local = np.repeat(center[None, :], local_count, axis=0)
    local += rng.standard_normal(local.shape).astype(np.float32) * np.float32(
        args.coverage_probe_noise
    )
    append_normalized("dense_local", local)

    remaining_budget = max(0, requested - len(families))
    if remaining_budget > 0:
        boundary_count = min(8, remaining_budget)
        boundary = np.repeat(center[None, :], boundary_count, axis=0)
        boundary += rng.standard_normal(boundary.shape).astype(np.float32) * np.float32(
            max(float(args.coverage_probe_noise) * 16.0, 0.08)
        )
        append_normalized("dense_boundary", boundary)

    empirical_target = min(16, max(0, requested - len(families)))
    empirical = np.empty((0, dim), dtype=np.float32)
    if empirical_target > 0:
        real_count = int(
            (train_cache["meta"].get("synthetic_train") or {}).get(
                "real_count_used", 0
            )
        )
        dense_prefix = max(
            1,
            min(
                real_count,
                int(args.dense_coverage_prefix_size)
                - int(args.dense_coverage_anchor_count),
            ),
        )
        empirical_target = min(empirical_target, dense_prefix)
        positions = np.sort(
            rng.choice(dense_prefix, size=empirical_target, replace=False)
        )
        empirical = core.decode_search_values(
            np.ascontiguousarray(train_cache["x_train_u16"][positions]), scheme
        )
        append_normalized("empirical_anchor", empirical)

    if len(families) < requested and len(empirical) > 0:
        count = min(8, requested - len(families), len(empirical))
        noisy = empirical[:count] + rng.standard_normal(
            empirical[:count].shape
        ).astype(np.float32) * np.float32(SYNTHETIC_NOISE_SCALE_NORMALIZED)
        append_normalized("empirical_noise", noisy)

    if len(families) < requested and len(empirical) > 1:
        count = min(8, requested - len(families), len(empirical) // 2)
        mixed = (
            empirical[:count] * np.float32(0.5)
            + empirical[-count:] * np.float32(0.5)
        )
        append_normalized("empirical_mix", mixed)

    if len(families) < requested:
        count = min(8, requested - len(families))
        append_normalized(
            "random_unit",
            rng.standard_normal((count, dim)).astype(np.float32),
        )

    if len(families) < requested and len(empirical) > 0:
        count = min(requested - len(families), len(empirical))
        append_normalized("negative_empirical", -empirical[:count])

    anchor_positions = [
        int(position) for position in args.coverage_probe_anchor_query_position
    ]
    generated_anchor_offsets = [
        int(offset) for offset in args.coverage_probe_generated_anchor_offset
    ]
    direction_nodes = [
        int(node_id) for node_id in args.coverage_probe_direction_node
    ]
    anchor_budget = min(
        max(0, int(args.coverage_probe_anchor_count)),
        requested - len(families),
    )
    if anchor_budget > 0:
        anchor_mode = str(args.coverage_probe_anchor_mode)
        if anchor_mode in {"generated-local", "refined-local"}:
            if not generated_anchor_offsets:
                raise ValueError(
                    f"{anchor_mode} coverage probes require at least one "
                    "--coverage-probe-generated-anchor-offset"
                )
            if min(generated_anchor_offsets) < 0:
                raise ValueError("generated coverage-probe offsets must be non-negative")
            source_args = argparse.Namespace(**vars(args))
            source_args.coverage_probe_count = max(
                requested, max(generated_anchor_offsets) + 1
            )
            if anchor_mode == "generated-local":
                source_args.coverage_probe_anchor_count = 0
                source_args.coverage_probe_anchor_query_position = []
            else:
                if not anchor_positions:
                    raise ValueError(
                        "refined-local coverage probes require at least one "
                        "--coverage-probe-anchor-query-position"
                    )
                source_args.coverage_probe_anchor_mode = "local"
                source_args.coverage_probe_anchor_count = int(
                    source_args.coverage_probe_count
                )
            source_args.coverage_probe_generated_anchor_offset = []
            source_args.coverage_probe_direction_node = []
            source_probes, _ = build_coverage_probe_bank(
                source_args, train_cache, heldout_queries
            )
            anchor_vectors = np.ascontiguousarray(
                source_probes[np.asarray(generated_anchor_offsets, dtype=np.int64)],
                dtype=np.float32,
            )
        else:
            if not anchor_positions:
                raise ValueError(
                    "--coverage-probe-anchor-count requires at least one "
                    "--coverage-probe-anchor-query-position"
                )
            invalid = [
                position
                for position in anchor_positions
                if position < 0 or position >= int(heldout_queries.shape[0])
            ]
            if invalid:
                raise ValueError(
                    f"coverage-probe anchor query positions are out of range: {invalid}"
                )
            anchor_vectors = np.ascontiguousarray(
                heldout_queries[np.asarray(anchor_positions, dtype=np.int64)],
                dtype=np.float32,
            )
        row_indices = np.arange(anchor_budget, dtype=np.int64)
        if anchor_mode == "score-boundary":
            if len(anchor_vectors) != 1 or len(direction_nodes) != 2:
                raise ValueError(
                    "score-boundary probes require exactly one anchor query and "
                    "two direction nodes: candidate then retained-bound node"
                )
            invalid_nodes = [
                node_id
                for node_id in direction_nodes
                if node_id < 0 or node_id >= int(train_cache["x_train_u16"].shape[0])
            ]
            if invalid_nodes:
                raise ValueError(
                    f"coverage-probe direction nodes are out of range: {invalid_nodes}"
                )
            direction_vectors = core.decode_search_values(
                np.ascontiguousarray(
                    train_cache["x_train_u16"][
                        np.asarray(direction_nodes, dtype=np.int64)
                    ]
                ),
                scheme,
            )
            direction = np.ascontiguousarray(
                direction_vectors[0] - direction_vectors[1], dtype=np.float32
            )
            alphas = np.geomspace(1.0e-3, 256.0, num=128).astype(np.float32)
            alpha_values = alphas[row_indices % len(alphas)]
            refinements = (
                anchor_vectors[0][None, :]
                + alpha_values[:, None] * direction[None, :]
            )
            jitter_scales = np.asarray(
                [0.0, 0.02, 0.05, 0.1], dtype=np.float32
            ) * np.float32(args.coverage_probe_noise)
            jitter_indices = (row_indices // len(alphas)) % len(jitter_scales)
            perturbations = rng.standard_normal(refinements.shape).astype(np.float32)
            refinements += perturbations * jitter_scales[jitter_indices, None]
            append_normalized("score_boundary_refinement", refinements)
        elif anchor_mode == "interpolate":
            if len(anchor_vectors) < 2:
                raise ValueError(
                    "interpolated coverage probes require at least two anchor queries"
                )
            alphas = np.asarray(
                [
                    0.02,
                    0.05,
                    0.08,
                    0.10,
                    0.125,
                    0.15,
                    0.175,
                    0.20,
                    0.25,
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                    0.70,
                    0.75,
                    0.80,
                    0.825,
                    0.85,
                    0.875,
                    0.90,
                    0.92,
                    0.95,
                    0.98,
                ],
                dtype=np.float32,
            )
            noise_scales = np.asarray(
                [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
                dtype=np.float32,
            ) * np.float32(args.coverage_probe_noise)
            alpha_values = alphas[row_indices % len(alphas)]
            scale_indices = (row_indices // len(alphas)) % len(noise_scales)
            left = anchor_vectors[0]
            right = anchor_vectors[1]
            refinements = (
                alpha_values[:, None] * left[None, :]
                + (np.float32(1.0) - alpha_values[:, None]) * right[None, :]
            )
            perturbations = rng.standard_normal(refinements.shape).astype(np.float32)
            refinements += perturbations * noise_scales[scale_indices, None]
            append_normalized("heldout_anchor_interpolation", refinements)
        else:
            anchor_indices = row_indices % len(anchor_vectors)
            noise_scales = np.asarray(
                [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
                dtype=np.float32,
            ) * np.float32(args.coverage_probe_noise)
            scale_indices = (row_indices // len(anchor_vectors)) % len(noise_scales)
            anchors = np.ascontiguousarray(anchor_vectors[anchor_indices])
            perturbations = rng.standard_normal(anchors.shape).astype(np.float32)
            refinements = anchors + perturbations * noise_scales[scale_indices, None]
            append_normalized(
                (
                    "generated_probe_local_refinement"
                    if anchor_mode == "generated-local"
                    else "refined_probe_local_refinement"
                    if anchor_mode == "refined-local"
                    else "heldout_local_refinement"
                ),
                refinements,
            )

    if len(families) < requested:
        append_normalized(
            "random_unit_extra",
            rng.standard_normal((requested - len(families), dim)).astype(np.float32),
        )
    probes = np.ascontiguousarray(np.concatenate(arrays, axis=0)[:requested])
    return probes, families[:requested]


def prepare_query_cache(
    args: argparse.Namespace,
    test_parquet: Path,
    train_cache: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    key_payload = {
        "version": CACHE_VERSION,
        "kind": "query_sample",
        "train_key": str(train_cache["key"]),
        "test_parquet": core.file_fingerprint(test_parquet),
        "normalize": bool(args.normalize),
        "n_test": int(args.n_test),
        "query_seed": int(args.query_seed),
    }
    if int(args.coverage_probe_count) > 0:
        anchor_positions = [
            int(position)
            for position in args.coverage_probe_anchor_query_position
        ]
        key_payload["coverage_probes"] = {
            "version": (
                "empirical_coverage_bank_v7_targeted_refinement"
                if int(args.coverage_probe_anchor_count) > 0
                else "empirical_coverage_bank_v2_vector_hashes"
            ),
            "count": int(args.coverage_probe_count),
            "seed": int(args.coverage_probe_seed),
            "local_noise": float(args.coverage_probe_noise),
            "anchor_query_positions": anchor_positions,
            "anchor_count": int(args.coverage_probe_anchor_count),
            "anchor_mode": str(args.coverage_probe_anchor_mode),
            "generated_anchor_offsets": [
                int(offset)
                for offset in args.coverage_probe_generated_anchor_offset
            ],
            "direction_nodes": [
                int(node_id) for node_id in args.coverage_probe_direction_node
            ],
        }
    query_key = cache_key(key_payload)
    query_dir = cache_root / "query_sample" / query_key
    query_dir.mkdir(parents=True, exist_ok=True)
    meta_path = query_dir / "meta.json"
    q_test_real_path = query_dir / "q_test_f32.npy"
    q_test_u16_path = query_dir / "q_test_u16.npy"
    q_indices_path = query_dir / "q_indices.npy"
    query_records_path = query_dir / "query_records.json"

    cache_hit = (
        (not bool(args.refresh_cache))
        and meta_path.exists()
        and q_test_real_path.exists()
        and q_test_u16_path.exists()
        and q_indices_path.exists()
        and query_records_path.exists()
    )
    if not cache_hit:
        q_all = core.load_embeddings(test_parquet)
        if bool(args.normalize):
            q_all = core.l2_normalize(q_all)
        q_test, q_indices = sample_queries(q_all, int(args.n_test), int(args.query_seed))
        heldout_count = int(q_test.shape[0])
        query_records: list[dict[str, Any]] = [
            {
                "query_position": position,
                "query_index": int(q_indices[position]),
                "query_source": "heldout",
                "query_family": "heldout",
                "query_vector_sha256": hashlib.sha256(
                    np.ascontiguousarray(q_test[position], dtype=np.float32).tobytes()
                ).hexdigest(),
            }
            for position in range(heldout_count)
        ]
        probes, probe_families = build_coverage_probe_bank(
            args, train_cache, q_test
        )
        if int(probes.shape[0]) > 0:
            probe_indices = np.arange(
                int(q_all.shape[0]),
                int(q_all.shape[0]) + int(probes.shape[0]),
                dtype=np.int64,
            )
            q_test = np.ascontiguousarray(
                np.concatenate([q_test, probes], axis=0)
            )
            q_indices = np.ascontiguousarray(
                np.concatenate([q_indices, probe_indices], axis=0)
            )
            query_records.extend(
                {
                    "query_position": heldout_count + offset,
                    "query_index": int(probe_indices[offset]),
                    "query_source": "coverage_probe",
                    "query_family": str(probe_families[offset]),
                    "query_vector_sha256": hashlib.sha256(
                        np.ascontiguousarray(probes[offset], dtype=np.float32).tobytes()
                    ).hexdigest(),
                }
                for offset in range(len(probe_families))
            )
        q_test_u16 = core.quantize_search_values(q_test, train_cache["scheme"])
        np.save(q_test_real_path, np.ascontiguousarray(q_test.astype(np.float32, copy=False)))
        np.save(q_test_u16_path, np.ascontiguousarray(q_test_u16))
        np.save(q_indices_path, np.ascontiguousarray(q_indices))
        query_records_path.write_text(
            json.dumps(query_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        core.dump_json(
            meta_path,
            {
                "query_key": query_key,
                "n_test_used": int(q_test.shape[0]),
                "heldout_query_count": heldout_count,
                "coverage_probe_count": int(q_test.shape[0]) - heldout_count,
                "coverage_probe_generator": (
                    (
                        "empirical_coverage_bank_v7_targeted_refinement"
                        if int(args.coverage_probe_anchor_count) > 0
                        else "empirical_coverage_bank_v2_vector_hashes"
                    )
                    if int(q_test.shape[0]) > heldout_count
                    else None
                ),
                "coverage_probe_anchor_query_positions": [
                    int(position)
                    for position in args.coverage_probe_anchor_query_position
                ],
                "coverage_probe_anchor_count": int(
                    args.coverage_probe_anchor_count
                ),
                "coverage_probe_anchor_mode": str(
                    args.coverage_probe_anchor_mode
                ),
                "coverage_probe_generated_anchor_offsets": [
                    int(offset)
                    for offset in args.coverage_probe_generated_anchor_offset
                ],
                "coverage_probe_direction_nodes": [
                    int(node_id) for node_id in args.coverage_probe_direction_node
                ],
                "query_dim": int(q_test.shape[1]),
                "test_parquet": core.file_fingerprint(test_parquet),
            },
        )

    meta = read_json(meta_path)
    q_test_real = np.ascontiguousarray(np.load(q_test_real_path).astype(np.float32, copy=False))
    q_test_u16 = np.load(q_test_u16_path, mmap_mode="r")
    q_indices = np.ascontiguousarray(np.load(q_indices_path).astype(np.int64, copy=False))
    query_records_raw = json.loads(query_records_path.read_text(encoding="utf-8"))
    if not isinstance(query_records_raw, list) or len(query_records_raw) != len(q_indices):
        raise ValueError("query cache records do not match the cached query array")
    return {
        "key": query_key,
        "dir": query_dir,
        "meta": meta,
        "q_test_real": q_test_real,
        "q_test_u16": q_test_u16,
        "q_indices": q_indices,
        "query_records": query_records_raw,
        "cache_hit": cache_hit,
    }


def prepare_graph_cache(
    faiss,
    args: argparse.Namespace,
    train_cache: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    synthetic_meta = train_cache["meta"].get("synthetic_train", {})
    explicit_hnsw_rng = bool(str(args.train_subset_mode) != "prefix") or bool(
        isinstance(synthetic_meta, dict)
        and synthetic_meta.get("synthetic_extension_used")
    )
    key_payload = {
        "version": CACHE_VERSION,
        "kind": "quant_graph",
        "train_key": str(train_cache["key"]),
        "metric": str(args.metric),
        "m": int(args.m),
        "ef_construction": int(args.ef_construction),
        "faiss_seed": int(args.faiss_seed),
        "faiss_build": faiss_build_id(faiss),
    }
    if explicit_hnsw_rng:
        key_payload["hnsw_rng_binding"] = "explicit_index_hnsw_rng_v1"
    if isinstance(synthetic_meta, dict) and bool(synthetic_meta.get("synthetic_extension_used")):
        key_payload["faiss_add_batch_rows"] = int(args.faiss_add_batch_rows)
    graph_key = cache_key(key_payload)
    graph_dir = cache_root / "quant_graph" / graph_key
    graph_dir.mkdir(parents=True, exist_ok=True)
    index_path = graph_dir / "hnsw.index"
    graph_meta_path = graph_dir / "graph_meta.json"
    graph_nodes_path = graph_dir / "graph_nodes.jsonl"
    graph_rows_path = graph_dir / "global_graph_rows.jsonl"
    index_randomness_path = graph_dir / "commitment_randomness.json"

    cache_hit = (not bool(args.refresh_cache)) and index_path.exists()
    if cache_hit:
        try:
            index = faiss.read_index(str(index_path))
        except Exception:
            cache_hit = False
    if not cache_hit:
        vectors_u16 = train_cache["x_train_u16"]
        index = core.create_faiss_hnsw_index(
            faiss,
            int(train_cache["scheme"].projected_dim),
            int(args.m),
            int(args.ef_construction),
            str(args.metric),
        )
        if (
            explicit_hnsw_rng
            and int(args.faiss_seed) >= 0
            and hasattr(index.hnsw, "rng")
        ):
            index.hnsw.rng = faiss.RandomGenerator(int(args.faiss_seed))
        add_batch_rows = max(1, int(args.faiss_add_batch_rows))
        for start in range(0, int(vectors_u16.shape[0]), add_batch_rows):
            stop = min(int(vectors_u16.shape[0]), start + add_batch_rows)
            decoded = core.decode_search_values(vectors_u16[start:stop], train_cache["scheme"])
            index.add(np.ascontiguousarray(decoded))
        faiss.write_index(index, str(index_path))

    raw_graph = core.extract_hnsw_raw_graph(faiss, index)
    accessor = core.HnswGraphAccessor(raw_graph)
    expected_dumped = (
        int(raw_graph.levels.shape[0])
        if int(args.graph_nodes_limit) <= 0
        else min(int(raw_graph.levels.shape[0]), int(args.graph_nodes_limit))
    )
    regenerate_graph_dump = bool(args.refresh_cache) or not graph_meta_path.exists() or not graph_nodes_path.exists()
    if not regenerate_graph_dump and graph_meta_path.exists():
        try:
            graph_meta = read_json(graph_meta_path)
            regenerate_graph_dump = int(graph_meta.get("dumped_node_count", -1)) != int(expected_dumped)
        except Exception:
            regenerate_graph_dump = True
    if regenerate_graph_dump:
        core.dump_graph_json(
            scheme_dir=graph_dir,
            raw=raw_graph,
            accessor=accessor,
            dim=int(train_cache["scheme"].projected_dim),
            metric=str(args.metric),
            m=int(args.m),
            ef_construction=int(args.ef_construction),
            graph_nodes_limit=int(args.graph_nodes_limit),
        )
    if bool(args.refresh_cache) or not graph_rows_path.exists():
        core.write_global_graph_rows_jsonl(graph_rows_path, raw_graph, accessor)
    index_commitment_randomness = load_or_create_index_randomness(
        index_randomness_path,
        refresh=bool(args.refresh_cache) or not cache_hit,
    )

    return {
        "key": graph_key,
        "dir": graph_dir,
        "index": index,
        "raw_graph": raw_graph,
        "accessor": accessor,
        "graph_meta_path": graph_meta_path,
        "graph_nodes_path": graph_nodes_path,
        "graph_rows_path": graph_rows_path,
        "index_commitment_randomness": index_commitment_randomness,
        "hnsw_rng_binding": (
            "explicit_index_hnsw_rng_v1"
            if explicit_hnsw_rng
            else "legacy_faiss_default"
        ),
        "cache_hit": cache_hit,
    }


def load_real_train_vectors(
    faiss,
    args: argparse.Namespace,
    train_parquet: Path,
) -> np.ndarray:
    x_train, _, _, _, _ = build_train_vectors(
        faiss,
        args,
        train_parquet,
    )
    return x_train


def build_seeded_real_hnsw_index(
    faiss,
    args: argparse.Namespace,
    vectors: np.ndarray,
):
    index = core.create_faiss_hnsw_index(
        faiss,
        int(vectors.shape[1]),
        int(args.m),
        int(args.ef_construction),
        str(args.metric),
    )
    if int(args.faiss_seed) >= 0 and hasattr(index.hnsw, "rng"):
        index.hnsw.rng = faiss.RandomGenerator(int(args.faiss_seed))
    index.add(np.ascontiguousarray(vectors.astype(np.float32, copy=False)))
    return index


def prepare_search_cache(
    cache_kind: str,
    key_payload: dict[str, Any],
    compute_fn,
    cache_root: Path,
    *,
    refresh_cache: bool,
) -> tuple[str, Path, np.ndarray, np.ndarray, bool]:
    cache_key_value = cache_key(key_payload)
    cache_dir = cache_root / cache_kind / cache_key_value
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / "ids.npy"
    distances_path = cache_dir / "distances.npy"
    cache_hit = (not bool(refresh_cache)) and ids_path.exists() and distances_path.exists()
    if not cache_hit:
        ids, distances = compute_fn()
        np.save(ids_path, np.ascontiguousarray(ids.astype(np.int32, copy=False)))
        np.save(distances_path, np.ascontiguousarray(distances.astype(np.float32, copy=False)))
    ids = np.ascontiguousarray(np.load(ids_path).astype(np.int32, copy=False))
    distances = np.ascontiguousarray(np.load(distances_path).astype(np.float32, copy=False))
    return cache_key_value, cache_dir, ids, distances, cache_hit


def build_equivalence_payload(
    query_index: int,
    equivalence: dict[str, Any],
    faiss_ids: np.ndarray,
    search_trace: dict[str, Any],
    proof_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query_index": int(query_index),
        "equivalence": equivalence,
        "faiss_topk": [int(x) for x in faiss_ids.tolist()],
        "search_trace_topk": [int(x) for x in search_trace["level0"]["final_topk"]],
        "proof_topk": [int(x["node_id"]) for x in proof_input["ef_search"]["final_topk"]],
    }


def topk_ids_match_up_to_ties(
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    zero_point: int,
    lhs: list[int],
    rhs: list[int],
) -> bool:
    if len(lhs) != len(rhs):
        return False
    union_ids = sorted({int(x) for x in lhs} | {int(x) for x in rhs})
    score_map = {
        int(node_id): int(
            core.quantized_dot_score(
                query_u16,
                np.ascontiguousarray(vectors_u16[int(node_id)].astype(np.uint16, copy=False)),
                int(zero_point),
            )
        )
        for node_id in union_ids
    }
    return core.topk_lists_match_up_to_ties(lhs, rhs, score_map)


def route_output_id_matches(lhs: list[int], rhs: list[int]) -> bool:
    if not lhs or not rhs:
        return False
    return int(lhs[0]) == int(rhs[0])


def save_trace_cache_payload(
    cache_dir: Path,
    search_trace: dict[str, Any],
    proof_input: dict[str, Any],
    proof_equivalence: dict[str, Any],
    manifest: dict[str, Any],
    raw_trace_payload: dict[str, Any] | None,
) -> None:
    core.dump_json(cache_dir / "search_trace.json", search_trace)
    core.dump_json(cache_dir / PROOF_INPUT_FILENAME, proof_input)
    core.dump_json(cache_dir / "proof_equivalence.json", proof_equivalence)
    core.dump_json(cache_dir / "trace_manifest.json", manifest)
    if raw_trace_payload is not None:
        core.dump_json(cache_dir / "faiss_prove_trace.json", raw_trace_payload)


def load_trace_cache_payload(cache_dir: Path) -> dict[str, Any]:
    payload = {
        "search_trace": read_json(cache_dir / "search_trace.json"),
        "proof_input": read_json(proof_input_cache_path(cache_dir)),
        "proof_equivalence": read_json(cache_dir / "proof_equivalence.json"),
        "manifest": read_json(cache_dir / "trace_manifest.json"),
        "raw_trace_path": cache_dir / "faiss_prove_trace.json",
    }
    return payload


def build_trace_cache_key(
    mode: str,
    graph_key: str,
    query_key: str,
    query_index: int,
    trace_query_pos: int,
    ef_search: int,
    k: int,
) -> str:
    return cache_key(
        {
            "version": CACHE_VERSION,
            "kind": "trace_export",
            "mode": str(mode),
            "graph_key": str(graph_key),
            "query_key": str(query_key),
            "query_index": int(query_index),
            "trace_query_pos": int(trace_query_pos),
            "ef_search": int(ef_search),
            "k": int(k),
            "proof_version": str(core.PROOF_INPUT_VERSION),
        }
    )


def load_scan_query_positions(path: Path, query_count: int) -> list[int]:
    positions: list[int] = []
    seen: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        position = int(value)
        if position < 0 or position >= int(query_count):
            raise ValueError(
                f"scan query position {position} is outside [0, {int(query_count)})"
            )
        if position in seen:
            continue
        seen.add(position)
        positions.append(position)
    if not positions:
        raise ValueError("scan query position file did not contain any positions")
    return positions


def scan_shape_from_traces(
    traces: list[dict[str, Any]],
    *,
    upper_neighbor_cap: int,
    round_neighbor_cap: int,
) -> dict[str, int]:
    actuals = [
        trace.get("actual")
        for trace in traces
        if isinstance(trace, dict) and isinstance(trace.get("actual"), dict)
    ]

    def maximum(name: str) -> int:
        return max((int(actual.get(name) or 0) for actual in actuals), default=0)

    return {
        "node_count": maximum("node_count"),
        "node_count_cap": 0,
        "score_node_count_cap": 0,
        "max_upper_steps": maximum("upper_steps"),
        "max_rounds": maximum("ef_rounds"),
        "upper_neighbor_cap": int(upper_neighbor_cap),
        "round_neighbor_cap": int(round_neighbor_cap),
        "visit_len": maximum("neighbor_visits"),
        "c_final_len": maximum("c_final_len"),
        "w_final_len": maximum("w_final_len"),
    }


def atomic_dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    core.dump_json(temporary, payload)
    temporary.replace(path)


def proof_shape_from_quantized_trace(
    *,
    search_trace: dict[str, Any],
    query_u16: np.ndarray,
    vectors_u16: np.ndarray,
    model_ids: np.ndarray,
    zero_point: int,
    upper_neighbor_cap: int,
    round_neighbor_cap: int,
) -> dict[str, int]:
    ref_node_ids = core.collect_referenced_node_ids(search_trace)
    core.validate_model_ids(model_ids, int(vectors_u16.shape[0]), ref_node_ids)
    score_map: dict[int, int] = {}
    ref_map: dict[int, dict[str, int]] = {}
    for node_id in ref_node_ids:
        node_vector = np.ascontiguousarray(
            vectors_u16[int(node_id)].astype(np.uint16, copy=False)
        )
        score_int = int(
            core.quantized_dot_score(query_u16, node_vector, int(zero_point))
        )
        score_map[int(node_id)] = score_int
        model_id = int(model_ids[int(node_id)])
        ref_map[int(node_id)] = {
            "node_id": int(node_id),
            "score_enc": int(core.encode_score_with_bias(score_int)),
            "model_id": model_id,
        }

    proof_ef_search = core.level0_trace_from_quantized_scores(
        search_trace["level0"],
        score_map,
        ref_map,
    )
    actual_upper_steps = len(search_trace["upper_trace"])
    actual_rounds = len(proof_ef_search["expansions"])
    return {
        "node_count": len(ref_node_ids),
        "upper_steps": actual_upper_steps,
        "ef_rounds": actual_rounds,
        "expand_row_count": actual_upper_steps + actual_rounds,
        "neighbor_visits": len(proof_ef_search.get("neighbor_visits", [])),
        "c_final_len": len(proof_ef_search["c_pq"]["final_refs"]),
        "w_final_len": len(proof_ef_search["w_pq"]["final_refs"]),
        "upper_neighbor_cap": int(upper_neighbor_cap),
        "round_neighbor_cap": int(round_neighbor_cap),
        "required_visit_cap": actual_rounds * int(round_neighbor_cap),
    }


def scan_replay_trace_shapes(
    *,
    args: argparse.Namespace,
    report_path: Path,
    positions: list[int],
    train_cache: dict[str, Any],
    query_cache: dict[str, Any],
    graph_cache: dict[str, Any],
    q_test_u16: np.ndarray,
    q_indices: np.ndarray,
    faiss_repro: dict[str, Any],
    faiss,
) -> dict[str, Any]:
    scheme = train_cache["scheme"]
    raw_graph = graph_cache["raw_graph"]
    neighbor_caps = [int(value) for value in raw_graph.nb_neighbors_by_layer]
    round_neighbor_cap = neighbor_caps[0] if neighbor_caps else int(args.m) * 2
    upper_neighbor_cap = max(neighbor_caps[1:], default=int(args.m))
    scan_key = cache_key(
        {
            "version": "route_actual_trace_scan_v2_inprocess",
            "graph_key": str(graph_cache["key"]),
            "query_key": str(query_cache["key"]),
            "positions": positions,
            "ef_search": int(args.ef_search),
            "k": int(args.k),
            "proof_version": str(core.PROOF_INPUT_VERSION),
        }
    )

    trace_map: dict[int, dict[str, Any]] = {}
    if bool(args.scan_resume) and report_path.exists():
        try:
            prior = read_json(report_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            prior = {}
        if (
            prior.get("version") == "route_actual_trace_scan_v2_inprocess"
            and prior.get("scan_key") == scan_key
            and isinstance(prior.get("traces"), list)
        ):
            trace_map = {
                int(trace["query_position"]): trace
                for trace in prior["traces"]
                if isinstance(trace, dict)
                and isinstance(trace.get("query_position"), int)
                and int(trace["query_position"]) in positions
            }

    def payload(stage: str) -> dict[str, Any]:
        traces = [trace_map[position] for position in positions if position in trace_map]
        return {
            "version": "route_actual_trace_scan_v2_inprocess",
            "stage": stage,
            "scan_key": scan_key,
            "embedding_scheme": str(scheme.name),
            "embedding_dimension": int(scheme.projected_dim),
            "shape": scan_shape_from_traces(
                traces,
                upper_neighbor_cap=upper_neighbor_cap,
                round_neighbor_cap=round_neighbor_cap,
            ),
            "query_count_total": len(positions),
            "query_count_scanned": len(traces),
            "query_positions": positions,
            "traces": traces,
            "trace_source": "quantized_replay",
            "faiss_repro": faiss_repro,
            "hnsw_rng_seed": int(args.faiss_seed),
            "hnsw_rng_binding": str(graph_cache["hnsw_rng_binding"]),
            "faiss_build": faiss_build_id(faiss),
            "cache": {
                "root": str(Path(args.cache_dir).expanduser().resolve()),
                "train_key": str(train_cache["key"]),
                "query_key": str(query_cache["key"]),
                "graph_key": str(graph_cache["key"]),
            },
        }

    checkpoint_every = max(1, int(args.scan_checkpoint_every))
    completed_since_checkpoint = 0
    for trace_query_pos in positions:
        if trace_query_pos in trace_map:
            continue
        query_index = int(q_indices[trace_query_pos])
        query_record = query_cache["query_records"][trace_query_pos]
        replay_trace = core.replay_query_quantized(
            query_u16=np.ascontiguousarray(
                q_test_u16[trace_query_pos].astype(np.uint16, copy=False)
            ),
            vectors_u16=train_cache["x_train_u16"],
            accessor=graph_cache["accessor"],
            raw=raw_graph,
            ef_search=int(args.ef_search),
            k=int(args.k),
            zero_point=int(scheme.quantizer.zero_point),
        )
        search_trace = core.minimal_search_trace_from_quantized_replay(replay_trace)
        search_trace["query_index"] = query_index
        query_u16 = np.ascontiguousarray(
            q_test_u16[trace_query_pos].astype(np.uint16, copy=False)
        )
        actual = proof_shape_from_quantized_trace(
            search_trace=search_trace,
            query_u16=query_u16,
            vectors_u16=train_cache["x_train_u16"],
            model_ids=train_cache["model_ids"],
            zero_point=int(scheme.quantizer.zero_point),
            upper_neighbor_cap=upper_neighbor_cap,
            round_neighbor_cap=round_neighbor_cap,
        )
        trace_map[trace_query_pos] = {
            "query_position": int(trace_query_pos),
            "query_index": query_index,
            "query_source": str(query_record.get("query_source", "heldout")),
            "query_family": str(query_record.get("query_family", "heldout")),
            "query_vector_sha256": str(
                query_record.get("query_vector_sha256", "")
            ),
            "actual": actual,
        }
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            atomic_dump_json(report_path, payload("scanning"))
            completed_since_checkpoint = 0

    report = payload("scanned")
    atomic_dump_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    train_parquet = raw_data_dir / args.train_parquet
    test_parquet = raw_data_dir / args.test_parquet
    embedding_id_csv = raw_data_dir / args.embedding_id_csv
    train_csv = raw_data_dir / "train.csv"
    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = core.resolve_run_dir(out_dir, str(args.results_dir))

    faiss = core.require_faiss(prefer_trace=str(args.trace_source) != "replay")
    faiss_repro = core.configure_faiss_repro(faiss, int(args.faiss_seed), int(args.faiss_threads))

    train_cache = prepare_train_cache(
        faiss,
        args,
        train_parquet,
        embedding_id_csv,
        train_csv,
        cache_root,
    )
    query_cache = prepare_query_cache(args, test_parquet, train_cache, cache_root)
    q_test_u16 = query_cache["q_test_u16"]
    q_test_real = query_cache["q_test_real"]
    q_indices = query_cache["q_indices"]

    scan_positions: list[int] | None = None
    scan_report_path: Path | None = None
    if str(args.scan_query_positions_file).strip():
        if not str(args.scan_report).strip():
            raise ValueError("--scan-report is required with --scan-query-positions-file")
        scan_positions = load_scan_query_positions(
            Path(args.scan_query_positions_file).expanduser().resolve(),
            int(q_test_u16.shape[0]),
        )
        scan_report_path = Path(args.scan_report).expanduser().resolve()
    elif int(args.trace_query_pos) < 0 or int(args.trace_query_pos) >= q_test_u16.shape[0]:
        raise ValueError(
            f"trace-query-pos={args.trace_query_pos} out of range for {q_test_u16.shape[0]} sampled queries"
        )

    graph_cache = prepare_graph_cache(faiss, args, train_cache, cache_root)
    scheme = train_cache["scheme"]
    search_q_test = core.decode_search_values(q_test_u16, scheme)

    quant_search_key, quant_search_dir, faiss_ids, faiss_distances, quant_search_cache_hit = prepare_search_cache(
        "quant_search",
        {
            "version": CACHE_VERSION,
            "kind": "quant_search",
            "graph_key": str(graph_cache["key"]),
            "query_key": str(query_cache["key"]),
            "ef_search": int(args.ef_search),
            "k": int(args.k),
            "faiss_build": faiss_build_id(faiss),
        },
        lambda: core.search_faiss_hnsw(
            index=graph_cache["index"],
            queries=search_q_test,
            k=int(args.k),
            ef_search=int(args.ef_search),
            search_bounded_queue=False,
            check_relative_distance=False,
        ),
        cache_root,
        refresh_cache=bool(args.refresh_cache),
    )
    quant_results_path = quant_search_dir / "faiss_search_results.jsonl"
    if bool(args.refresh_cache) or not quant_results_path.exists():
        core.write_search_results_jsonl(quant_results_path, q_indices, faiss_ids, faiss_distances)

    real_search_key = ""
    real_search_cache_hit: bool | None = None
    real_ids: np.ndarray | None = None
    if not bool(args.skip_real_search_comparison):
        real_search_key, real_search_dir, real_ids, _, real_search_cache_hit = prepare_search_cache(
            "real_search",
            {
                "version": CACHE_VERSION,
                "kind": "real_search",
                "train_key": str(train_cache["key"]),
                "query_key": str(query_cache["key"]),
                "metric": str(args.metric),
                "m": int(args.m),
                "ef_construction": int(args.ef_construction),
                "ef_search": int(args.ef_search),
                "k": int(args.k),
                "faiss_seed": int(args.faiss_seed),
                "faiss_build": faiss_build_id(faiss),
            },
            lambda: core.search_faiss_hnsw(
                index=build_seeded_real_hnsw_index(
                    faiss,
                    args,
                    load_real_train_vectors(
                        faiss,
                        args,
                        train_parquet,
                    ),
                ),
                queries=q_test_real,
                k=int(args.k),
                ef_search=int(args.ef_search),
                search_bounded_queue=False,
                check_relative_distance=False,
            ),
            cache_root,
            refresh_cache=bool(args.refresh_cache),
        )
        del real_search_dir

    if scan_positions is not None:
        assert scan_report_path is not None
        scan_replay_trace_shapes(
            args=args,
            report_path=scan_report_path,
            positions=scan_positions,
            train_cache=train_cache,
            query_cache=query_cache,
            graph_cache=graph_cache,
            q_test_u16=q_test_u16,
            q_indices=q_indices,
            faiss_repro=faiss_repro,
            faiss=faiss,
        )
        return

    trace_query_pos = int(args.trace_query_pos)
    query_index = int(q_indices[trace_query_pos])
    query_record = query_cache["query_records"][trace_query_pos]
    faiss_topk = [int(x) for x in faiss_ids[trace_query_pos].tolist()]
    trace_api = ""
    trace_fallback_reason = ""
    raw_trace_payload: dict[str, object] | None = None

    active_trace_cache_dir: Path | None = None
    active_trace_cache_hit = False
    search_trace: dict[str, Any] | None = None
    proof_input: dict[str, Any] | None = None
    proof_equivalence: dict[str, Any] | None = None

    if str(args.trace_source) in {"auto", "direct"}:
        direct_key = build_trace_cache_key(
            "direct",
            str(graph_cache["key"]),
            str(query_cache["key"]),
            query_index,
            trace_query_pos,
            int(args.ef_search),
            int(args.k),
        )
        direct_dir = cache_root / "trace_export" / direct_key
        direct_hit = (not bool(args.refresh_cache)) and trace_cache_complete(direct_dir)
        try:
            if direct_hit:
                cached = load_trace_cache_payload(direct_dir)
                search_trace = cached["search_trace"]
                proof_input = cached["proof_input"]
                proof_equivalence = cached["proof_equivalence"]
                manifest = cached["manifest"]
                raw_trace_path = cached["raw_trace_path"]
                trace_api = str(manifest.get("trace_api", ""))
                raw_trace_payload = read_json(raw_trace_path) if raw_trace_path.exists() else None
                active_trace_cache_dir = direct_dir
                active_trace_cache_hit = True
            else:
                direct_dir.mkdir(parents=True, exist_ok=True)
                raw_trace_payload, trace_api = core.get_faiss_hnsw_trace(
                    faiss,
                    graph_cache["index"],
                    search_q_test[trace_query_pos],
                    int(args.k),
                    int(args.ef_search),
                    prefer_prove=True,
                )
                raw_trace_payload["trace_api"] = str(trace_api)
                search_trace = core.minimal_search_trace_from_faiss_trace(raw_trace_payload)
                # Base proofs authenticate variable-length graph rows. Keep
                # their physical width equal to the rows actually expanded;
                # experiment sets a larger cap only after sentinel-padding
                # both the trace and its graph-row sidecar.
                search_trace["level0"]["round_width"] = max(
                    (
                        len(exp.get("neighbors", []))
                        for exp in search_trace["level0"]["expansions"]
                    ),
                    default=0,
                )
                direct_topk = [int(x) for x in search_trace["level0"]["final_topk"]]
                direct_topk_match = topk_ids_match_up_to_ties(
                    np.ascontiguousarray(q_test_u16[trace_query_pos].astype(np.uint16, copy=False)),
                    train_cache["x_train_u16"],
                    int(scheme.quantizer.zero_point),
                    direct_topk,
                    faiss_topk,
                )
                direct_top1_match = bool(
                    direct_topk_match or route_output_id_matches(direct_topk, faiss_topk)
                )
                if not direct_top1_match:
                    raise ValueError(
                        "direct quantized trace top-1 does not match quantized FAISS HNSW search for the traced query: "
                        f"trace={direct_topk} faiss={faiss_topk}"
                    )
                search_trace["query_index"] = int(query_index)
                search_trace["faiss_topk"] = [int(x) for x in faiss_topk]
                search_trace["faiss_distances"] = [float(x) for x in faiss_distances[trace_query_pos].tolist()]
                search_trace["faiss_topk_match_up_to_ties"] = bool(direct_topk_match)
                search_trace["faiss_top1_match"] = bool(direct_top1_match)
                proof_input, equivalence = core.build_proof_input(
                    query_index=int(query_index),
                    search_trace=search_trace,
                    query_u16=np.ascontiguousarray(q_test_u16[trace_query_pos].astype(np.uint16, copy=False)),
                    vectors_u16=train_cache["x_train_u16"],
                    metric=str(args.metric),
                    model_ids=train_cache["model_ids"],
                    quantizer=scheme.quantizer,
                    scheme_name=scheme.name,
                    projected_dim=int(scheme.projected_dim),
                    global_node_table_path=train_cache["global_nodes_path"].resolve(),
                    global_graph_rows_path=graph_cache["graph_rows_path"].resolve(),
                    commitment_randomness={
                        **graph_cache["index_commitment_randomness"],
                    },
                )
                proof_equivalence = build_equivalence_payload(
                    query_index,
                    equivalence,
                    faiss_ids[trace_query_pos],
                    search_trace,
                    proof_input,
                )
                save_trace_cache_payload(
                    direct_dir,
                    search_trace,
                    proof_input,
                    proof_equivalence,
                    {
                        "trace_source_used": str(search_trace.get("trace_source", "")),
                        "trace_api": str(trace_api),
                        "trace_fallback_reason": "",
                    },
                    raw_trace_payload,
                )
                active_trace_cache_dir = direct_dir
                active_trace_cache_hit = False
        except Exception as exc:
            trace_fallback_reason = str(exc)
            if str(args.trace_source) == "direct":
                raise
            search_trace = None
            proof_input = None
            proof_equivalence = None
            raw_trace_payload = None
            trace_api = ""

    if search_trace is None or proof_input is None or proof_equivalence is None:
        replay_key = build_trace_cache_key(
            "replay",
            str(graph_cache["key"]),
            str(query_cache["key"]),
            query_index,
            trace_query_pos,
            int(args.ef_search),
            int(args.k),
        )
        replay_dir = cache_root / "trace_export" / replay_key
        replay_hit = (not bool(args.refresh_cache)) and trace_cache_complete(replay_dir)
        if replay_hit:
            cached = load_trace_cache_payload(replay_dir)
            search_trace = cached["search_trace"]
            proof_input = cached["proof_input"]
            proof_equivalence = cached["proof_equivalence"]
            manifest = cached["manifest"]
            trace_api = str(manifest.get("trace_api", ""))
            active_trace_cache_dir = replay_dir
            active_trace_cache_hit = True
        else:
            replay_dir.mkdir(parents=True, exist_ok=True)
            replay_trace = core.replay_query_quantized(
                query_u16=np.ascontiguousarray(q_test_u16[trace_query_pos].astype(np.uint16, copy=False)),
                vectors_u16=train_cache["x_train_u16"],
                accessor=graph_cache["accessor"],
                raw=graph_cache["raw_graph"],
                ef_search=int(args.ef_search),
                k=int(args.k),
                zero_point=int(scheme.quantizer.zero_point),
            )
            replay_topk = [int(x) for x in replay_trace["final_topk"]]
            replay_topk_match = topk_ids_match_up_to_ties(
                np.ascontiguousarray(q_test_u16[trace_query_pos].astype(np.uint16, copy=False)),
                train_cache["x_train_u16"],
                int(scheme.quantizer.zero_point),
                replay_topk,
                faiss_topk,
            )
            replay_top1_match = bool(
                replay_topk_match or route_output_id_matches(replay_topk, faiss_topk)
            )
            replay_mismatch_warning = ""
            if not replay_top1_match:
                replay_mismatch_warning = (
                    "quantized replay top-1 does not match quantized FAISS HNSW search for the traced query: "
                    f"replay={replay_topk} faiss={faiss_topk}"
                )
            search_trace = core.minimal_search_trace_from_quantized_replay(replay_trace)
            search_trace["query_index"] = int(query_index)
            search_trace["faiss_topk"] = [int(x) for x in faiss_topk]
            search_trace["faiss_distances"] = [float(x) for x in faiss_distances[trace_query_pos].tolist()]
            search_trace["faiss_topk_match_up_to_ties"] = bool(replay_topk_match)
            search_trace["faiss_top1_match"] = bool(replay_top1_match)
            if replay_mismatch_warning:
                search_trace["faiss_mismatch_warning"] = str(replay_mismatch_warning)
            proof_input, equivalence = core.build_proof_input(
                query_index=int(query_index),
                search_trace=search_trace,
                query_u16=np.ascontiguousarray(q_test_u16[trace_query_pos].astype(np.uint16, copy=False)),
                vectors_u16=train_cache["x_train_u16"],
                metric=str(args.metric),
                model_ids=train_cache["model_ids"],
                quantizer=scheme.quantizer,
                scheme_name=scheme.name,
                projected_dim=int(scheme.projected_dim),
                global_node_table_path=train_cache["global_nodes_path"].resolve(),
                global_graph_rows_path=graph_cache["graph_rows_path"].resolve(),
                commitment_randomness={
                    **graph_cache["index_commitment_randomness"],
                },
            )
            proof_equivalence = build_equivalence_payload(
                query_index,
                equivalence,
                faiss_ids[trace_query_pos],
                search_trace,
                proof_input,
            )
            save_trace_cache_payload(
                replay_dir,
                search_trace,
                proof_input,
                proof_equivalence,
                {
                    "trace_source_used": str(search_trace.get("trace_source", "")),
                    "trace_api": "",
                    "trace_fallback_reason": str(trace_fallback_reason),
                },
                None,
            )
            active_trace_cache_dir = replay_dir
            active_trace_cache_hit = False

    assert search_trace is not None
    assert proof_input is not None
    assert proof_equivalence is not None
    assert active_trace_cache_dir is not None

    scheme_dir = run_dir / str(args.scheme)
    scheme_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "graph_meta": scheme_dir / "graph_meta.json",
        "graph_nodes": scheme_dir / "graph_nodes.jsonl",
        "faiss_search_results": scheme_dir / "faiss_search_results.jsonl",
        "search_trace": scheme_dir / f"search_trace_query_{query_index}.json",
        "faiss_prove_trace": scheme_dir / f"faiss_prove_trace_query_{query_index}.json",
        "proof_input": scheme_dir / f"proof_input_linear_query_{query_index}.json",
        "proof_equivalence": scheme_dir / f"proof_equivalence_query_{query_index}.json",
        "global_quantized_nodes": scheme_dir / "global_quantized_nodes.jsonl",
        "global_graph_rows": scheme_dir / "global_graph_rows.jsonl",
    }

    materialize_file(graph_cache["graph_meta_path"], outputs["graph_meta"])
    materialize_file(graph_cache["graph_nodes_path"], outputs["graph_nodes"])
    materialize_file(quant_results_path, outputs["faiss_search_results"])
    materialize_file(train_cache["global_nodes_path"], outputs["global_quantized_nodes"])
    materialize_file(graph_cache["graph_rows_path"], outputs["global_graph_rows"])
    materialize_file(active_trace_cache_dir / "search_trace.json", outputs["search_trace"])
    materialize_file(proof_input_cache_path(active_trace_cache_dir), outputs["proof_input"])
    materialize_file(active_trace_cache_dir / "proof_equivalence.json", outputs["proof_equivalence"])
    faiss_trace_cache_path = active_trace_cache_dir / "faiss_prove_trace.json"
    if faiss_trace_cache_path.exists():
        materialize_file(faiss_trace_cache_path, outputs["faiss_prove_trace"])
    else:
        clear_optional_output(outputs["faiss_prove_trace"])

    comparison_payload: dict[str, Any] = {
        "real_search_skipped": bool(args.skip_real_search_comparison),
        "trace_query_quantized_topk": [int(x) for x in faiss_ids[trace_query_pos].tolist()],
        "route_output_match": bool(
            route_output_id_matches(
                [int(x) for x in search_trace["level0"]["final_topk"]],
                [int(x) for x in faiss_ids[trace_query_pos].tolist()],
            )
        ),
        "trace_query_topk_match_up_to_ties": bool(search_trace.get("faiss_topk_match_up_to_ties", False)),
        "trace_query_top1_match": bool(search_trace.get("faiss_top1_match", False)),
        "trace_query_mismatch_warning": str(search_trace.get("faiss_mismatch_warning", "")),
    }
    if real_ids is None:
        comparison_payload.update(
            {
                "trace_query_real_topk": [],
                "trace_query_exact_match": None,
                "all_query_exact_match_rate": None,
            }
        )
    else:
        comparison_payload.update(
            {
                "trace_query_real_topk": [int(x) for x in real_ids[trace_query_pos].tolist()],
                "trace_query_exact_match": bool(
                    np.array_equal(
                        np.ascontiguousarray(real_ids[trace_query_pos]),
                        np.ascontiguousarray(faiss_ids[trace_query_pos]),
                    )
                ),
                "all_query_exact_match_rate": float(
                    np.mean(
                        np.asarray(
                            [
                                np.array_equal(
                                    np.ascontiguousarray(real_ids[i]),
                                    np.ascontiguousarray(faiss_ids[i]),
                                )
                                for i in range(real_ids.shape[0])
                            ],
                            dtype=np.float64,
                        )
                    )
                ),
            }
        )

    report = {
        "raw_data_dir": str(raw_data_dir),
        "train_parquet": core.file_fingerprint(train_parquet),
        "test_parquet": core.file_fingerprint(test_parquet),
        "embedding_id_csv_exists": bool(embedding_id_csv.exists()),
        "metric": str(args.metric),
        "normalize": bool(args.normalize),
        "n_train_used": int(train_cache["meta"]["n_train_used"]),
        "synthetic_train": dict(train_cache["meta"].get("synthetic_train", {})),
        "n_test_used": int(query_cache["meta"]["n_test_used"]),
        "heldout_query_count": int(
            query_cache["meta"].get("heldout_query_count", 0)
        ),
        "coverage_probe_count": int(
            query_cache["meta"].get("coverage_probe_count", 0)
        ),
        "trace_query_pos": int(trace_query_pos),
        "trace_query_index": int(query_index),
        "trace_query_source": str(query_record.get("query_source", "heldout")),
        "trace_query_family": str(query_record.get("query_family", "heldout")),
        "trace_query_vector_sha256": str(
            query_record.get("query_vector_sha256", "")
        ),
        "m": int(args.m),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "k": int(args.k),
        "faiss_repro": faiss_repro,
        "hnsw_rng_seed": int(args.faiss_seed),
        "hnsw_rng_binding": str(graph_cache["hnsw_rng_binding"]),
        "faiss_build": faiss_build_id(faiss),
        "search_mode": {
            "search_bounded_queue": False,
            "check_relative_distance": False,
        },
        "trace_source_requested": str(args.trace_source),
        "trace_source_used": str(search_trace.get("trace_source", "")),
        "trace_api": str(trace_api),
        "trace_fallback_reason": str(trace_fallback_reason),
        "cache": {
            "root": str(cache_root),
            "refresh_cache": bool(args.refresh_cache),
            "train_key": str(train_cache["key"]),
            "query_key": str(query_cache["key"]),
            "graph_key": str(graph_cache["key"]),
            "quant_search_key": str(quant_search_key),
            "real_search_key": str(real_search_key),
            "trace_cache_dir": str(active_trace_cache_dir),
            "hits": {
                "train_quant": bool(train_cache["cache_hit"]),
                "query_sample": bool(query_cache["cache_hit"]),
                "quant_graph": bool(graph_cache["cache_hit"]),
                "quant_search": bool(quant_search_cache_hit),
                "real_search": real_search_cache_hit,
                "trace_export": bool(active_trace_cache_hit),
            },
        },
        "files": {name: str(path.resolve()) for name, path in outputs.items() if name != "faiss_prove_trace"},
        "scheme": {
            "name": str(scheme.name),
            "scale": float(scheme.quantizer.scale),
            "zero_point": int(scheme.quantizer.zero_point),
            "projected_dim": int(scheme.projected_dim),
            "projection_seed": int(scheme.projection_seed),
            "projection_percentile": float(scheme.projection_percentile),
        },
        "comparison": comparison_payload,
    }
    if faiss_trace_cache_path.exists():
        report["files"]["faiss_prove_trace"] = str(outputs["faiss_prove_trace"].resolve())
    else:
        report["files"]["faiss_prove_trace"] = ""

    core.dump_json(run_dir / "export_report.json", report)
