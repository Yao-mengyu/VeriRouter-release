from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SIGNATURE_PY_ROOT = Path(__file__).resolve().parents[1] / "data-plane" / "signature" / "python"
if str(SIGNATURE_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGNATURE_PY_ROOT))

from execution_certificate import (  # noqa: E402
    VERSION as SERVING_CERTIFICATE_VERSION,
    compute_request_commitment_report,
    hash_protocol_value,
    resolve_payload,
    openssl_verify,
    public_key_fingerprint,
    raw_digest,
    recompute_serving_digest_from_statement,
)

GOLDILOCKS_MODULUS = (1 << 64) - (1 << 32) + 1


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_report_proof_path(report_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("proof report is missing a proof path")
    path = Path(value)
    return (path if path.is_absolute() else report_path.parent / path).resolve()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Replace verifier state atomically after all checks have passed."""

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
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


@contextmanager
def session_state_lock(state_path: Path):
    """Serialize one client's state transition across concurrent verifier calls."""

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def public_hashout_value(public_inputs: dict[str, Any], *names: str) -> list[int]:
    for name in names:
        if name in public_inputs:
            value = parse_hashout(public_inputs[name])
            if value is not None:
                return value
    raise ValueError(f"public inputs are missing one of: {', '.join(names)}")


def length_prefixed_utf8(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def uint64_be(value: int) -> bytes:
    if value < 0 or value >= (1 << 64):
        raise ValueError(f"expected uint64 value, got {value!r}")
    return int(value).to_bytes(8, "big")


def uint64_list_be(values: list[int]) -> bytes:
    return b"".join(uint64_be(int(value)) for value in values)


def parse_hashout(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        parts = value
    else:
        text = str(value).strip()
        if text.startswith("["):
            parts = json.loads(text)
        else:
            parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"expected 4 HashOut limbs, got {len(parts)}")
    return [int(str(part), 0) for part in parts]


def prompt_hash_binding_bytes(scalar_value: int, hashout_value: list[int] | None) -> bytes:
    if hashout_value is not None:
        return b"hashout" + uint64_list_be(hashout_value)
    return b"scalar" + uint64_be(scalar_value)


def hashout_binding_bytes(value: Any) -> bytes:
    hashout = parse_hashout(value)
    if hashout is None:
        raise ValueError("expected non-empty HashOut")
    return uint64_list_be(hashout)


def tuple_digest(parts: list[bytes]) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(b''.join(parts)).hexdigest()}"


def encoder_binding_digest(
    *,
    rid: str,
    encoder_id: str,
    client_request_hash: int,
    client_request_hashout: list[int] | None,
    route_query_hash: Any,
) -> str:
    return tuple_digest(
        [
            b"VeriRoute.public_encoder_binding.v1",
            length_prefixed_utf8(rid),
            length_prefixed_utf8(encoder_id),
            prompt_hash_binding_bytes(client_request_hash, client_request_hashout),
            hashout_binding_bytes(route_query_hash),
        ]
    )


def role_value(value: Any) -> Any:
    if type(value) is int and 0 <= value < GOLDILOCKS_MODULUS:
        return value
    if isinstance(value, str) and value in {"system", "user", "assistant"}:
        return value
    raise ValueError(f"unsupported role value: {value!r}")


def resolve_msg_item(item: Any, commit: dict[str, Any]) -> dict[str, Any]:
    if (
        isinstance(item, dict)
        and "ref" in item
        and ("role" in item or "content" in item)
    ):
        raise ValueError("message insertion cannot mix ref with inline role/content")
    if isinstance(item, dict) and "ref" in item:
        index = int(item["ref"])
        msg_list = commit.get("msg_list", [])
        if index < 0 or index >= len(msg_list):
            raise IndexError(f"message ref {index} out of range")
        resolved = msg_list[index]
        return {
            "role": role_value(resolved["role"]),
            "content": str(resolved["content"]),
        }
    if isinstance(item, dict) and "role" in item and "content" in item:
        return {
            "role": role_value(item["role"]),
            "content": str(item["content"]),
        }
    raise ValueError(f"invalid message item: {item!r}")


def resolve_frag_item(item: Any, commit: dict[str, Any]) -> str:
    if isinstance(item, dict) and "ref" in item:
        index = int(item["ref"])
        frag_list = commit.get("frag_list", [])
        if index < 0 or index >= len(frag_list):
            raise IndexError(f"fragment ref {index} out of range")
        return str(frag_list[index])
    if isinstance(item, str):
        return item
    raise ValueError(f"invalid fragment item: {item!r}")


def resolve_rewrite_input(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    commit = payload.get("commit") or {}
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("rewrite input is missing messages[]")

    original: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"invalid message row: {message!r}")

        for gap in message.get("gap_before", []):
            inserted = resolve_msg_item(gap, commit)
            executed.append(inserted)

        present = bool(message.get("present", True))
        if not present:
            continue

        role = role_value(message["role"])
        content = str(message["content"])
        original.append({"role": role, "content": content})

        prefix = "".join(resolve_frag_item(item, commit) for item in message.get("prefixes", []))
        suffix = "".join(resolve_frag_item(item, commit) for item in message.get("suffixes", []))
        executed.append({"role": role, "content": f"{prefix}{content}{suffix}"})

    for gap in payload.get("trailing_gap", []):
        executed.append(resolve_msg_item(gap, commit))

    return original, executed


def normalize_client_request(payload: Any) -> list[dict[str, Any]]:
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(messages, list):
        raise ValueError("client request must be a message array or contain messages[]")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("each client request message must contain role and content")
        normalized.append(
            {"role": role_value(message["role"]), "content": str(message["content"])}
        )
    return normalized


def compute_client_request_hash(
    client_request_path: Path, *, chunk_bytes: int, prompt_hash_bin: str
) -> list[int]:
    payload = json.loads(client_request_path.read_text(encoding="utf-8"))
    report = compute_request_commitment_report(
        normalize_client_request(payload),
        request_randomness=[0, 0, 0, 0],
        chunk_bytes=chunk_bytes,
        request_commitment_bin=prompt_hash_bin,
    )
    digest = parse_hashout(report.get("request_hash_hashout"))
    if digest is None:
        raise ValueError("request helper did not return request_hash_hashout")
    return digest


def verify_rewrite_root(
    report_path: Path,
    policy_path: Path,
    *,
    rewrite_bin: str,
    verifier_data_path: Path | None = None,
) -> dict[str, Any]:
    default_binary = (
        Path(__file__).resolve().parents[1]
        / "data-plane"
        / "zkrewrite"
        / "rust"
        / "target"
        / "release"
        / "rewrite_demo"
    )
    binary = Path(rewrite_bin).resolve() if rewrite_bin else default_binary
    verifier_args = [
        "--verify-report",
        str(report_path),
        "--verify-policy",
        str(policy_path),
    ]
    if verifier_data_path is not None:
        verifier_args.extend(["--verifier-data", str(verifier_data_path)])
    if binary.exists():
        command = [str(binary), *verifier_args]
        cwd = None
    else:
        rust_dir = default_binary.parents[2]
        command = [
            "cargo",
            "+nightly",
            "run",
            "--release",
            "--locked",
            "--bin",
            "rewrite_demo",
            "--",
            *verifier_args,
        ]
        cwd = rust_dir
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if payload.get("verified") is not True:
        raise ValueError("rewrite root proof verifier did not accept")
    return payload


def rewrite_policy_from_statement(statement: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the caller-pinned root statement; never trust a stored policy."""

    return {
        "version": "rewrite_verification_policy_v1",
        "public_inputs": {
            "message_bank_commitment": public_hashout_value(
                statement, "message_bank_commitment"
            ),
            "fragment_bank_commitment": public_hashout_value(
                statement, "fragment_bank_commitment"
            ),
            "client_message_count": int(statement["client_message_count"]),
            "client_request_hash": public_hashout_value(
                statement, "client_request_hash"
            ),
            "rewritten_message_count": int(statement["rewritten_message_count"]),
            "request_commitment": public_hashout_value(
                statement, "request_commitment"
            ),
        },
    }


def verify_rewrite_certificate_artifact(
    certificate: dict[str, Any],
    *,
    rewrite_policy_path: Path,
    trusted_profile_path: Path,
    rewrite_bin: str = "",
) -> dict[str, Any]:
    """Actually verify the serialized proof against the certificate statement."""

    artifacts = certificate.get("proof_artifacts") or {}
    report_value = artifacts.get("rewrite_report_path")
    if not report_value:
        raise ValueError("rewrite certificate is missing rewrite_report_path")
    statement = certificate.get("rewrite_statement")
    if not isinstance(statement, dict):
        raise ValueError("rewrite certificate is missing rewrite_statement")
    expected_policy = rewrite_policy_from_statement(statement)
    if load_json(rewrite_policy_path) != expected_policy:
        raise ValueError("caller-pinned rewrite policy does not match the certificate statement")
    with approved_verifier_key(trusted_profile_path, "rewrite") as vk_path:
        return verify_rewrite_root(
            Path(str(report_value)).resolve(), rewrite_policy_path,
            rewrite_bin=rewrite_bin, verifier_data_path=vk_path,
        )


def build_rewrite_certificate(args: argparse.Namespace) -> None:
    report_path = Path(args.rewrite_report).resolve()
    input_path = Path(args.rewrite_input).resolve()
    out_path = Path(args.out).resolve()

    report = load_json(report_path)
    rewrite_input = load_json(input_path)
    original_conversation, executed_conversation = resolve_rewrite_input(rewrite_input)
    if args.client_request_file:
        supplied_client_request = normalize_client_request(
            json.loads(
                Path(args.client_request_file)
                .resolve()
                .read_text(encoding="utf-8")
            )
        )
        if supplied_client_request != original_conversation:
            raise ValueError(
                "rewrite input does not exactly match the caller-supplied client request"
            )

    root_report = report.get("root_report")
    if not isinstance(root_report, dict):
        raise ValueError("paper-aligned rewrite certificate requires an aggregated root_report")
    public_inputs = root_report.get("public_inputs")
    if not isinstance(public_inputs, dict):
        raise ValueError("rewrite root_report is missing public_inputs")
    root_proof_path = str(resolve_report_proof_path(report_path, root_report.get("proof_path")))
    root_proof_bytes = root_report.get("proof_size_bytes")

    message_bank_commitment = public_hashout_value(
        public_inputs, "message_bank_commitment", "message_commitment_hash"
    )
    fragment_bank_commitment = public_hashout_value(
        public_inputs, "fragment_bank_commitment", "fragment_commitment_hash"
    )
    client_request_hash = public_hashout_value(
        public_inputs, "client_request_hash", "user_request_hash"
    )
    request_commitment = public_hashout_value(public_inputs, "request_commitment")
    client_message_count = int(public_inputs["client_message_count"])
    rewritten_message_count = int(public_inputs["rewritten_message_count"])

    if client_message_count != len(original_conversation):
        raise ValueError(
            "rewrite proof client_message_count does not match the supplied client request"
        )
    if rewritten_message_count != len(executed_conversation):
        raise ValueError(
            "rewrite proof rewritten_message_count does not match the resolved model request"
        )

    chunk_bytes = int(args.chunk_bytes)
    client_hash_report = compute_request_commitment_report(
        original_conversation,
        request_randomness=[0, 0, 0, 0],
        chunk_bytes=chunk_bytes,
        request_commitment_bin=args.prompt_hash_bin,
    )
    independently_computed_client_hash = parse_hashout(
        client_hash_report.get("request_hash_hashout")
    )
    if independently_computed_client_hash != client_request_hash:
        raise ValueError("rewrite proof does not bind the supplied client request")
    request_randomness = (rewrite_input.get("commit") or {}).get("request_randomness")
    if not isinstance(request_randomness, list) or len(request_randomness) != 4:
        raise ValueError("rewrite input must contain four-limb commit.request_randomness")
    provider_report = compute_request_commitment_report(
        executed_conversation,
        request_randomness=request_randomness,
        chunk_bytes=chunk_bytes,
        request_commitment_bin=args.prompt_hash_bin,
    )
    if parse_hashout(provider_report.get("request_commitment")) != request_commitment:
        raise ValueError(
            "provider-side recomputation disagrees with the rewrite request commitment"
        )

    expected_policy = rewrite_policy_from_statement(
        {
            "message_bank_commitment": message_bank_commitment,
            "fragment_bank_commitment": fragment_bank_commitment,
            "client_message_count": client_message_count,
            "client_request_hash": client_request_hash,
            "rewritten_message_count": rewritten_message_count,
            "request_commitment": request_commitment,
        }
    )
    if args.rewrite_policy:
        policy_path = Path(args.rewrite_policy).resolve()
        if load_json(policy_path) != expected_policy:
            raise ValueError("caller-supplied rewrite policy does not match this statement")
    else:
        policy_path = out_path.with_name(f"{out_path.stem}.rewrite_verify_policy.json")
        write_json(policy_path, expected_policy)
    proof_verification = verify_rewrite_root(
        report_path,
        policy_path,
        rewrite_bin=args.rewrite_bin,
    )

    # The wrapper does not reveal r_m. The circuit report already binds the
    # request commitment; provider-side recomputation is checked by the serving
    # certificate and final chain.
    binding_digest = tuple_digest(
        [
            b"VeriRouter.rewrite-statement.v2",
            length_prefixed_utf8(args.rid),
            hashout_binding_bytes(client_request_hash),
            hashout_binding_bytes(request_commitment),
            hashout_binding_bytes(message_bank_commitment),
            hashout_binding_bytes(fragment_bank_commitment),
        ]
    )

    artifacts_dir = (
        Path(args.artifacts_dir).resolve()
        if args.artifacts_dir
        else out_path.parent
    )
    user_conv_path = artifacts_dir / "user_conversation.json"
    write_json(user_conv_path, original_conversation)

    public_report_path = out_path.with_name(f"{out_path.stem}.public_report.json")
    write_json(public_report_path, {
        "version": report["version"], "aggregated": True,
        "root_report": {"public_inputs": public_inputs, "proof_path": root_proof_path},
    })
    proof_verification["report_path"] = str(public_report_path)

    cert = {
        "certificate_type": "rewrite_certificate",
        "version": "rewrite_certificate_v2_request_commitment",
        "rid": args.rid,
        "binding_digest": binding_digest,
        "rewrite_statement": {
            "message_bank_commitment": message_bank_commitment,
            "fragment_bank_commitment": fragment_bank_commitment,
            "client_message_count": client_message_count,
            "client_request_hash": client_request_hash,
            "rewritten_message_count": rewritten_message_count,
            "request_commitment": request_commitment,
        },
        "proof_artifacts": {
            "rewrite_report_path": str(public_report_path),
            "root_proof_path": root_proof_path,
            "root_proof_bytes": root_proof_bytes,
            "aggregated": bool(report.get("aggregated", False)),
            "batch_count": int(report.get("batch_count", 0)),
            "rewrite_verification_policy_path": str(policy_path),
        },
        "proof_verification": proof_verification,
        "conversation_artifacts": {
            "client_request_path": str(user_conv_path),
        },
    }
    write_json(out_path, cert)


def route_policy_from_statement(statement: dict[str, Any]) -> dict[str, Any]:
    internal_score_fields = {
        "score_table_hash",
        "score_table_commit",
        "score_table_commitment",
        "h_dtable",
    }
    leaked = sorted(internal_score_fields.intersection(statement))
    if leaked:
        raise ValueError(
            "route public statement exposes recursive-only score-table fields: "
            + ", ".join(leaked)
        )
    return {
        "version": "route_verification_policy_v2_commitments",
        "reference_commitment": public_hashout_value(
            statement, "reference_commitment"
        ),
        "q_hash": public_hashout_value(statement, "q_hash"),
        "graph_commitment": public_hashout_value(statement, "graph_commitment"),
        "entry_point": int(statement["entry_point"]),
        "entry_layer": int(statement["entry_layer"]),
        "ef_search": int(statement["ef_search"]),
        "top_k": int(statement["top_k"]),
    }


def verify_route_root(
    report_path: Path,
    proof_input: Path | None,
    policy_path: Path,
    *,
    route_bin: str,
    verifier_data_path: Path | None = None,
) -> dict[str, Any]:
    default_binary = (
        Path(__file__).resolve().parents[1]
        / "control-plane"
        / "zkroute"
        / "rust"
        / "target"
        / "release"
        / "veriroute_route"
    )
    binary = Path(route_bin).resolve() if route_bin else default_binary
    verifier_args = [
        "--verify-report",
        str(report_path),
        "--verify-policy",
        str(policy_path),
    ]
    if verifier_data_path is None:
        if proof_input is None:
            raise ValueError("offline diagnostic route verification requires a trace")
        verifier_args.extend(["--trace", str(proof_input)])
    if verifier_data_path is not None:
        verifier_args.extend(["--verifier-data", str(verifier_data_path)])
    if binary.exists():
        command = [str(binary), *verifier_args]
        cwd = None
    else:
        rust_dir = default_binary.parents[2]
        command = [
            "cargo",
            "+nightly",
            "run",
            "--release",
            "--locked",
            "--bin",
            "veriroute_route",
            "--",
            *verifier_args,
        ]
        cwd = rust_dir
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if payload.get("verified") is not True:
        raise ValueError("route root proof verifier did not accept")
    return payload


def verify_route_certificate_artifact(
    certificate: dict[str, Any],
    *,
    route_policy_path: Path,
    trusted_profile_path: Path,
    route_bin: str = "",
) -> dict[str, Any]:
    statement = certificate.get("route_statement")
    if not isinstance(statement, dict):
        raise ValueError("route certificate is missing route_statement")
    expected_policy = route_policy_from_statement(statement)
    if load_json(route_policy_path) != expected_policy:
        raise ValueError("caller-pinned route policy does not match the certificate statement")
    artifacts = certificate.get("proof_artifacts") or {}
    report_value = artifacts.get("route_report_path")
    if not report_value:
        raise ValueError("route certificate is missing public proof report")
    with approved_verifier_key(trusted_profile_path, "route") as vk_path:
        verification = verify_route_root(
            Path(str(report_value)).resolve(), None,
            route_policy_path, route_bin=route_bin, verifier_data_path=vk_path,
        )
    verified_statement = verification.get("final_statement")
    if not isinstance(verified_statement, dict):
        raise ValueError("route proof verifier did not return a verified final_statement")
    if statement != verified_statement:
        raise ValueError("route certificate statement does not match the verified route proof")
    verified_model = verified_statement.get("output_model_id")
    if verified_model is None or str(certificate.get("selected_backend_id")) != str(verified_model):
        raise ValueError("route certificate selected model does not match the verified route proof")
    return verification


def build_route_certificate(args: argparse.Namespace) -> None:
    report_path = Path(args.route_report).resolve()
    out_path = Path(args.out).resolve()

    report = load_json(report_path)
    statement = report.get("final_statement")
    if not isinstance(statement, dict):
        raise ValueError("route report is missing final_statement")
    # Also validates that the recursion-only h_dtable interface is absent from
    # the client-visible statement.
    route_policy_from_statement(statement)
    required_statement_fields = ("entry_point", "entry_layer", "ef_search", "top_k")
    if any(field not in statement for field in required_statement_fields):
        raise ValueError(
            "route report lacks the complete public HNSW policy statement; regenerate it with the current route circuit"
        )

    selected_backend = str(statement["output_model_id"])
    if args.backend_id and str(args.backend_id) != selected_backend:
        raise ValueError(
            f"backend_id mismatch: route proof selected {selected_backend}, "
            f"but wrapper was asked to bind {args.backend_id}"
        )

    client_request_hash: int | None = None
    client_request_hashout: list[int] | None = parse_hashout(
        args.client_request_hashout or args.user_prompt_hashout
    )
    request_hash_arg = args.client_request_hash or args.user_prompt_hash
    if request_hash_arg:
        client_request_hash = int(request_hash_arg, 0)
        if client_request_hashout is None:
            raise ValueError(
                "the route protocol requires the complete --client-request-hashout"
            )
        if client_request_hash != client_request_hashout[0]:
            raise ValueError(
                "client request scalar does not match the first HashOut limb"
            )
    elif client_request_hashout is not None:
        client_request_hash = client_request_hashout[0]
    elif args.user_conversation_file:
        user_payload = json.loads(Path(args.user_conversation_file).read_text(encoding="utf-8"))
        client_report = compute_request_commitment_report(
            user_payload,
            request_randomness=[0, 0, 0, 0],
            chunk_bytes=int(args.chunk_bytes),
            request_commitment_bin=args.prompt_hash_bin,
        )
        client_request_hashout = parse_hashout(client_report.get("request_hash_hashout"))
        if client_request_hashout is None:
            raise ValueError("request helper did not return request_hash_hashout")
        client_request_hash = client_request_hashout[0]

    binding_digest = None
    encoder_statement = None
    if client_request_hash is not None:
        binding_digest = tuple_digest(
            [
                length_prefixed_utf8(args.rid),
                prompt_hash_binding_bytes(client_request_hash, client_request_hashout),
                length_prefixed_utf8(selected_backend),
            ]
        )

    route_query_hash = public_hashout_value(statement, "q_hash")
    reference_commitment = public_hashout_value(statement, "reference_commitment")
    graph_commitment = public_hashout_value(statement, "graph_commitment")
    if client_request_hash is not None:
        if not args.encoder_id:
            raise ValueError("--encoder-id is required when binding a client request hash to route_query_hash")
        encoder_digest = encoder_binding_digest(
            rid=args.rid,
            encoder_id=args.encoder_id,
            client_request_hash=client_request_hash,
            client_request_hashout=client_request_hashout,
            route_query_hash=route_query_hash,
        )
        encoder_statement = {
            "encoder_id": args.encoder_id,
            "encoder_public": True,
            "client_request_hash": client_request_hashout,
            "route_query_hash": route_query_hash,
            "encoder_binding_digest": encoder_digest,
        }
    route_statement = {
        "reference_commitment": reference_commitment,
        "q_hash": route_query_hash,
        "graph_commitment": graph_commitment,
        "output_model_id": int(statement["output_model_id"]),
        "entry_point": int(statement["entry_point"]),
        "entry_layer": int(statement["entry_layer"]),
        "ef_search": int(statement["ef_search"]),
        "top_k": int(statement["top_k"]),
    }
    optional_map = {
        "nbr_commit": "neighbor_visit_hash",
        "neighbor_visit_hash": "neighbor_visit_hash",
        "upper_len": "upper_len",
        "rounds_len": "rounds_len",
    }
    for source_key, public_key in optional_map.items():
        if source_key in statement:
            route_statement[public_key] = statement[source_key]

    expected_route_policy = route_policy_from_statement(route_statement)
    if args.route_policy:
        policy_path = Path(args.route_policy).resolve()
        if load_json(policy_path) != expected_route_policy:
            raise ValueError("caller-pinned route policy does not match this statement")
    else:
        policy_path = out_path.with_name(f"{out_path.stem}.route_verify_policy.json")
        write_json(policy_path, expected_route_policy)
    if not args.proof_input:
        raise ValueError("--proof-input is required for client-side route proof verification")
    route_proof_verification = verify_route_root(
        report_path,
        Path(args.proof_input).resolve(),
        policy_path,
        route_bin=args.route_bin,
    )

    final_proof_path = str(resolve_report_proof_path(report_path, report.get("final_proof_path")))
    public_report_path = out_path.with_name(f"{out_path.stem}.public_report.json")
    write_json(public_report_path, {
        "final_statement": route_statement, "final_proof_path": final_proof_path,
    })
    route_proof_verification["report_path"] = str(public_report_path)

    cert = {
        "certificate_type": "route_certificate",
        "version": "route_certificate_v6_commitments",
        "rid": args.rid,
        "binding_digest": binding_digest,
        "client_request_hash": client_request_hashout,
        "encoder_statement": encoder_statement,
        "selected_backend_id": selected_backend,
        "route_statement": route_statement,
        "proof_artifacts": {
            "route_report_path": str(public_report_path),
            "route_verification_policy_path": str(policy_path),
            "final_proof_path": final_proof_path,
            "final_proof_bytes": report.get("final_proof_bytes"),
            "final_output_model_id": int(report["final_output_model_id"]),
            "final_public_inputs_len": report.get("final_public_inputs_len"),
        },
        "consistency_checks": {
            "backend_matches_report": int(report["final_output_model_id"]) == int(statement["output_model_id"]),
        },
        "proof_verification": route_proof_verification,
    }
    write_json(out_path, cert)


def resolve_backend_registry_entry(registry_path: str, backend_id: str) -> dict[str, Any]:
    if not registry_path:
        raise ValueError("--backend-registry is required for backend/public-key binding")
    path = Path(registry_path).resolve()
    registry = load_json(path)
    backends = registry.get("backends", registry)
    if not isinstance(backends, dict):
        raise ValueError("backend registry must be an object or contain a backends object")
    if backend_id not in backends:
        raise ValueError(f"backend {backend_id!r} is not present in backend registry {path}")

    raw_entry = backends[backend_id]
    if isinstance(raw_entry, str):
        entry = (
            {"public_key_fingerprint": raw_entry}
            if raw_entry.startswith("sha256:")
            else {"public_key_path": raw_entry}
        )
    elif isinstance(raw_entry, dict):
        entry = dict(raw_entry)
    else:
        raise ValueError(f"invalid registry entry for backend {backend_id!r}: {raw_entry!r}")

    if entry.get("public_key_path"):
        key_path = Path(str(entry["public_key_path"]))
        if not key_path.is_absolute():
            key_path = path.parent / key_path
        key_path = key_path.resolve()
        entry["public_key_path"] = str(key_path)
        computed_fp = public_key_fingerprint(key_path)
        if entry.get("public_key_fingerprint") and entry["public_key_fingerprint"] != computed_fp:
            raise ValueError(
                f"registry fingerprint mismatch for backend {backend_id}: "
                f"{entry['public_key_fingerprint']} != {computed_fp}"
            )
        entry["public_key_fingerprint"] = computed_fp
    if not entry.get("public_key_fingerprint"):
        raise ValueError(f"registry entry for backend {backend_id!r} lacks public_key_fingerprint")
    return entry


def verify_execution_signature_with_registry_key(
    execution_cert: dict[str, Any],
    registry_entry: dict[str, Any],
    response_payload: Any,
) -> tuple[bool | None, str | None, str | None, bool]:
    public_key_path = registry_entry.get("public_key_path")
    if not public_key_path:
        return (
            None,
            None,
            "registry has only a fingerprint; signature re-verification needs public_key_path",
            False,
        )
    if execution_cert.get("version") != SERVING_CERTIFICATE_VERSION:
        return None, None, "unsupported serving certificate version", False
    if execution_cert.get("certificate_type") != "serving_certificate":
        return None, None, "unexpected serving certificate type", False
    if execution_cert.get("signature_algorithm") != "ed25519":
        return None, None, "unsupported signature algorithm", False
    if execution_cert.get("digest_algorithm") != "sha256":
        return None, None, "unsupported digest algorithm", False
    statement = execution_cert.get("statement", {})
    if (
        statement.get("request_commitment_scheme")
        != "poseidon-hashout4-request-commitment-v1"
    ):
        return None, None, "unsupported request commitment scheme", False
    recomputed_digest = recompute_serving_digest_from_statement(statement)
    signature_bytes = base64.b64decode(
        execution_cert.get("signature", {})["value"], validate=True
    )
    verified, details = openssl_verify(
        Path(public_key_path).resolve(),
        raw_digest(recomputed_digest),
        signature_bytes,
    )
    digest_matches_statement = recomputed_digest == statement.get("serving_digest")
    response_matches = hash_protocol_value(response_payload) == statement.get("response_hash")
    return (
        bool(verified and digest_matches_statement and response_matches),
        recomputed_digest,
        details or None,
        response_matches,
    )


@contextmanager
def approved_verifier_key(profile_path: Path, kind: str):
    """Profile comes from authenticated client provisioning, never a report."""
    profile = load_json(profile_path)
    if profile.get("version") != "verirouter_trusted_profile_v2":
        raise ValueError("unsupported trusted profile version")
    entry = profile.get(kind)
    if not isinstance(entry, dict) or entry.get("zero_knowledge") is not True:
        raise ValueError(f"missing approved ZK {kind} profile")
    depth = entry.get("recursion_depth")
    max_depth = entry.get("max_depth")
    if type(depth) is not int or type(max_depth) is not int or not 1 <= depth <= max_depth <= 8:
        raise ValueError("approved recursion depth must satisfy 1 <= depth <= max_depth <= 8")
    if kind == "rewrite" and (type(entry.get("max_leaf_count")) is not int or not 1 <= entry["max_leaf_count"] <= 65536):
        raise ValueError("rewrite profile must bound leaf count by 65536")
    key = bytes.fromhex(entry["verifier_data_hex"])
    if not key:
        raise ValueError("empty approved verifier key")
    with tempfile.NamedTemporaryFile(suffix=".verifier.bin") as handle:
        handle.write(key)
        handle.flush()
        yield Path(handle.name)


def build_chain_manifest(args: argparse.Namespace) -> None:
    if Path(args.out).resolve() == Path(args.client_state).resolve():
        raise ValueError("--out must not overwrite --client-state")
    with session_state_lock(Path(args.client_state).resolve()):
        _build_chain_manifest_locked(args)


def _build_chain_manifest_locked(args: argparse.Namespace) -> None:
    rewrite_cert = load_json(Path(args.rewrite_cert).resolve())
    route_cert = load_json(Path(args.route_cert).resolve())
    execution_cert = load_json(Path(args.execution_cert).resolve())
    out_path = Path(args.out).resolve()
    response_payload = resolve_payload(
        args.response_file,
        args.response_text,
        args.response_format,
        "response",
    )

    rewrite_statement = rewrite_cert.get("rewrite_statement", {})
    execution_statement = execution_cert.get("statement", {})

    rewrite_rid = rewrite_cert.get("rid")
    route_rid = route_cert.get("rid")
    execution_rid = execution_statement.get("rid")
    state_path = Path(args.client_state).resolve()
    client_state = load_json(state_path) if state_path.exists() else {"completed_rids": []}
    rid_match = bool(execution_rid) and rewrite_rid == route_rid == execution_rid == args.expected_rid
    rid_unused = execution_rid not in client_state.get("completed_rids", [])

    rewrite_client_hashout = parse_hashout(rewrite_statement.get("client_request_hash"))
    route_client_hashout = parse_hashout(route_cert.get("client_request_hash"))
    caller_client_hashout = compute_client_request_hash(
        Path(args.client_request_file).resolve(),
        chunk_bytes=int(args.chunk_bytes),
        prompt_hash_bin=args.prompt_hash_bin,
    )
    client_hashout_match = (
        rewrite_client_hashout is not None
        and route_client_hashout is not None
        and rewrite_client_hashout == route_client_hashout
        and rewrite_client_hashout == caller_client_hashout
    )
    encoder_statement = route_cert.get("encoder_statement") or {}
    encoder_binding_match = None
    if encoder_statement:
        recomputed_encoder_digest = encoder_binding_digest(
            rid=str(route_rid),
            encoder_id=str(encoder_statement["encoder_id"]),
            client_request_hash=int(encoder_statement["client_request_hash"][0]),
            client_request_hashout=parse_hashout(encoder_statement.get("client_request_hash")),
            route_query_hash=encoder_statement["route_query_hash"],
        )
        encoder_binding_match = (
            recomputed_encoder_digest == encoder_statement.get("encoder_binding_digest")
            and parse_hashout(encoder_statement["client_request_hash"]) == route_client_hashout
            and parse_hashout(encoder_statement["route_query_hash"])
            == parse_hashout((route_cert.get("route_statement") or {}).get("q_hash"))
        )

    rewrite_request_commitment = parse_hashout(rewrite_statement.get("request_commitment"))
    serving_request_commitment = parse_hashout(execution_statement.get("request_commitment"))
    request_commitment_match = (
        rewrite_request_commitment is not None
        and serving_request_commitment is not None
        and rewrite_request_commitment == serving_request_commitment
    )

    route_backend = route_cert.get("selected_backend_id")
    execution_backend = execution_statement.get("model")
    backend_id_match = str(route_backend) == str(execution_backend)
    serving_public_key_fingerprint = execution_cert.get("key_metadata", {}).get("public_key_fingerprint")
    registry_entry = resolve_backend_registry_entry(args.backend_registry, str(route_backend))
    registry_fingerprint = registry_entry.get("public_key_fingerprint")
    registry_backend_key_match = (
        str(registry_fingerprint) == str(serving_public_key_fingerprint)
    )
    client_signature_verified, recomputed_execution_digest, signature_details, response_hash_match = (
        verify_execution_signature_with_registry_key(
            execution_cert, registry_entry, response_payload
        )
    )

    signature_verified = execution_cert.get("verification", {}).get("verified")
    try:
        rewrite_proof_verification = verify_rewrite_certificate_artifact(
            rewrite_cert,
            rewrite_policy_path=Path(args.rewrite_policy).resolve(),
            trusted_profile_path=Path(args.trusted_profile).resolve(),
            rewrite_bin=getattr(args, "rewrite_bin", ""),
        )
    except Exception as exc:
        rewrite_proof_verification = {"verified": False, "error": str(exc)}
    rewrite_proof_verified = rewrite_proof_verification.get("verified") is True
    try:
        route_proof_verification = verify_route_certificate_artifact(
            route_cert,
            route_policy_path=Path(args.route_policy).resolve(),
            trusted_profile_path=Path(args.trusted_profile).resolve(),
            route_bin=getattr(args, "route_bin", ""),
        )
    except Exception as exc:
        route_proof_verification = {"verified": False, "error": str(exc)}
    route_proof_verified = route_proof_verification.get("verified") is True
    required_checks = {
        "rid_match": rid_match,
        "rid_unused": rid_unused,
        "client_request_hash_match": client_hashout_match,
        "request_commitment_match": request_commitment_match,
        "encoder_binding_digest_match": encoder_binding_match,
        "backend_id_match": backend_id_match,
        "backend_registry_public_key_match": registry_backend_key_match,
        "execution_signature_verified_by_client": client_signature_verified,
        "rewrite_proof_verified_by_client": rewrite_proof_verified,
        "route_proof_verified_by_client": route_proof_verified,
    }
    accept = all(required_checks.values())
    if accept:
        client_state.setdefault("completed_rids", []).append(execution_rid)
        atomic_write_json(state_path, client_state)

    manifest = {
        "manifest_type": "certificate_chain",
        "version": "certificate_chain_v2_request_commitment",
        "rewrite_certificate_path": str(Path(args.rewrite_cert).resolve()),
        "route_certificate_path": str(Path(args.route_cert).resolve()),
        "execution_certificate_path": str(Path(args.execution_cert).resolve()),
        "checks": {
            "rid_match": rid_match,
            "rid_unused": rid_unused,
            "client_request_hash_match": client_hashout_match,
            "encoder_binding_digest_match": encoder_binding_match,
            "request_commitment_match": request_commitment_match,
            "backend_id_match": backend_id_match,
            "backend_registry_public_key_match": registry_backend_key_match,
            "execution_signature_verified": signature_verified,
            "execution_signature_verified_by_client": client_signature_verified,
            "rewrite_proof_verified_by_client": rewrite_proof_verified,
            "route_proof_verified_by_client": route_proof_verified,
            "response_hash_match": response_hash_match,
        },
        "accept": accept,
        "summary": {
            "rid": execution_rid,
            "client_request_hash": rewrite_client_hashout,
            "caller_client_request_hash": caller_client_hashout,
            "request_commitment": rewrite_request_commitment,
            "response_hash": execution_statement.get("response_hash"),
            "selected_backend_id": route_backend,
            "execution_backend_id": execution_backend,
            "serving_public_key_fingerprint": serving_public_key_fingerprint,
            "registry_public_key_fingerprint": registry_fingerprint,
            "recomputed_execution_digest": recomputed_execution_digest,
            "client_signature_details": signature_details,
            "rewrite_proof_verification": rewrite_proof_verification,
            "route_proof_verification": route_proof_verification,
            "encoder_statement": encoder_statement,
        },
        "client_verification_bundle": {
            "rewrite_proof_artifacts": rewrite_cert.get("proof_artifacts"),
            "route_proof_artifacts": route_cert.get("proof_artifacts"),
            "encoder_statement": encoder_statement,
            "execution_statement": execution_statement,
            "backend_registry_entry": registry_entry,
            "trusted_profile_path": str(Path(args.trusted_profile).resolve()),
            "verification_interface": "verirouter.py chain verify with the authenticated profile, current policy and client state",

        },
    }
    write_json(out_path, manifest)


def canonical_object_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def initialize_session_pin(args: argparse.Namespace) -> None:
    """Persist a model pin only from an already accepted initial full chain."""

    manifest_path = Path(args.chain_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("accept") is not True:
        raise ValueError("cannot pin a model from a rejected initial chain")
    initial_checks = manifest.get("checks") or {}
    required_initial_checks = (
        "rid_match",
        "rid_unused",
        "client_request_hash_match",
        "encoder_binding_digest_match",
        "request_commitment_match",
        "backend_id_match",
        "backend_registry_public_key_match",
        "execution_signature_verified_by_client",
        "rewrite_proof_verified_by_client",
        "route_proof_verified_by_client",
        "response_hash_match",
    )
    failed_checks = [
        name for name in required_initial_checks if initial_checks.get(name) is not True
    ]
    if failed_checks:
        raise ValueError(
            "initial chain is not a complete client acceptance; failed checks: "
            + ", ".join(failed_checks)
        )
    summary = manifest.get("summary") or {}
    initial_rid = str(summary.get("rid", ""))
    if not initial_rid or initial_rid != args.expected_rid:
        raise ValueError("initial chain rid does not equal the client-issued expected rid")
    model_pin = str(summary.get("selected_backend_id", ""))
    if not model_pin or model_pin != str(summary.get("execution_backend_id", "")):
        raise ValueError("initial chain does not agree on the selected model")

    # The local accepted manifest records verified statements. Referenced
    # certificate files may have changed since acceptance and are not trusted.
    route_verification = summary.get("route_proof_verification") or {}
    rewrite_verification = summary.get("rewrite_proof_verification") or {}
    route_statement = route_verification.get("final_statement")
    rewrite_statement = rewrite_verification.get("public_inputs") or {}
    if route_verification.get("verified") is not True or rewrite_verification.get("verified") is not True:
        raise ValueError("initial chain lacks verified proof statements")
    if not isinstance(route_statement, dict) or str(route_statement.get("output_model_id")) != model_pin:
        raise ValueError("initial verified route statement does not match the selected model")
    request_commitment = parse_hashout(rewrite_statement.get("request_commitment"))
    if request_commitment is None:
        raise ValueError("initial rewrite statement lacks request_commitment")
    if request_commitment != parse_hashout(summary.get("request_commitment")):
        raise ValueError("initial verified request commitment does not match the accepted summary")
    message_bank_commitment = parse_hashout(
        rewrite_statement.get("message_bank_commitment")
    )
    fragment_bank_commitment = parse_hashout(
        rewrite_statement.get("fragment_bank_commitment")
    )
    if message_bank_commitment is None or fragment_bank_commitment is None:
        raise ValueError("initial rewrite statement lacks pinned bank commitments")

    context = {
        "route_statement": route_statement,
        "encoder_id": (summary.get("encoder_statement") or {}).get("encoder_id"),
        "registry_public_key_fingerprint": summary.get(
            "registry_public_key_fingerprint"
        ),
        "message_bank_commitment": message_bank_commitment,
        "fragment_bank_commitment": fragment_bank_commitment,
    }
    state = {
        "state_type": "session_pin_state",
        "version": "session_pin_state_v1",
        "session_id": args.session_id,
        "context_digest": canonical_object_digest(context),
        "context": context,
        "model_pin": model_pin,
        "initial_accepted_chain": str(manifest_path),
        "completed_rids": [initial_rid],
        "completed_request_commitments": [request_commitment],
    }
    state_path = Path(args.state).resolve()
    if args.out and Path(args.out).resolve() == state_path:
        raise ValueError("--out must not overwrite --state")
    with session_state_lock(state_path):
        if state_path.exists() and not args.force:
            raise FileExistsError("session state already exists; pass --force to replace it")
        atomic_write_json(state_path, state)
    result = {
        "accepted": True,
        "session_id": args.session_id,
        "model_pin": model_pin,
        "completed_rid": initial_rid,
        "state_path": str(state_path),
    }
    if args.out:
        write_json(Path(args.out).resolve(), result)
    else:
        print(json.dumps(result, indent=2))


def _verify_pinned_turn_locked(args: argparse.Namespace, state_path: Path) -> None:
    """Verify and consume one turn while the session state lock is held."""

    state = load_json(state_path)
    if state.get("version") != "session_pin_state_v1":
        raise ValueError("unsupported session pin state")
    if args.session_id and args.session_id != state.get("session_id"):
        raise ValueError("session_id does not match the verifier state")
    if canonical_object_digest(state.get("context") or {}) != state.get("context_digest"):
        raise ValueError("session context digest mismatch")

    rewrite_cert = load_json(Path(args.rewrite_cert).resolve())
    serving_cert = load_json(Path(args.execution_cert).resolve())
    rewrite_statement = rewrite_cert.get("rewrite_statement") or {}
    serving_statement = serving_cert.get("statement") or {}
    rid = str(serving_statement.get("rid", ""))
    response_payload = resolve_payload(
        args.response_file,
        args.response_text,
        args.response_format,
        "response",
    )
    model = str(serving_statement.get("model", ""))
    registry_entry = resolve_backend_registry_entry(args.backend_registry, model)
    signature_verified, serving_digest_value, signature_details, response_hash_match = (
        verify_execution_signature_with_registry_key(
            serving_cert, registry_entry, response_payload
        )
    )

    rewrite_commitment = parse_hashout(rewrite_statement.get("request_commitment"))
    serving_commitment = parse_hashout(serving_statement.get("request_commitment"))
    expected_message_bank = parse_hashout(
        (state.get("context") or {}).get("message_bank_commitment")
    )
    expected_fragment_bank = parse_hashout(
        (state.get("context") or {}).get("fragment_bank_commitment")
    )
    caller_client_hashout = compute_client_request_hash(
        Path(args.client_request_file).resolve(),
        chunk_bytes=int(args.chunk_bytes),
        prompt_hash_bin=args.prompt_hash_bin,
    )
    try:
        rewrite_proof_verification = verify_rewrite_certificate_artifact(
            rewrite_cert,
            rewrite_policy_path=Path(args.rewrite_policy).resolve(),
            trusted_profile_path=Path(args.trusted_profile).resolve(),
            rewrite_bin=getattr(args, "rewrite_bin", ""),
        )
    except Exception as exc:
        rewrite_proof_verification = {"verified": False, "error": str(exc)}

    checks = {
        "client_issued_rid_matches": bool(rid) and rid == args.expected_rid,
        "rid_matches_rewrite": rid == str(rewrite_cert.get("rid", "")),
        "rid_unused": rid not in set(state.get("completed_rids") or []),
        "model_matches_pin": model == str(state.get("model_pin", "")),
        "request_commitment_matches": (
            rewrite_commitment is not None
            and rewrite_commitment == serving_commitment
        ),
        "request_commitment_fresh": (
            rewrite_commitment is not None
            and rewrite_commitment
            not in (state.get("completed_request_commitments") or [])
        ),
        "message_bank_commitment_matches_pin": parse_hashout(
            rewrite_statement.get("message_bank_commitment")
        )
        == expected_message_bank,
        "fragment_bank_commitment_matches_pin": parse_hashout(
            rewrite_statement.get("fragment_bank_commitment")
        )
        == expected_fragment_bank,
        "client_request_hash_matches_caller": parse_hashout(
            rewrite_statement.get("client_request_hash")
        )
        == caller_client_hashout,
        "registry_key_matches_pin": registry_entry.get("public_key_fingerprint")
        == (state.get("context") or {}).get("registry_public_key_fingerprint"),
        "serving_signature_verified": signature_verified is True,
        "response_hash_matches": response_hash_match,
        "rewrite_proof_verified": rewrite_proof_verification.get("verified") is True,
    }
    accepted = all(checks.values())
    if accepted:
        completed_rids = list(state.get("completed_rids") or [])
        completed_rids.append(rid)
        completed_commitments = list(state.get("completed_request_commitments") or [])
        completed_commitments.append(rewrite_commitment)
        state["completed_rids"] = completed_rids
        state["completed_request_commitments"] = completed_commitments
        atomic_write_json(state_path, state)

    result = {
        "manifest_type": "session_pinned_turn",
        "version": "session_pinned_turn_v1",
        "session_id": state.get("session_id"),
        "rid": rid,
        "model": model,
        "model_pin": state.get("model_pin"),
        "checks": checks,
        "serving_digest": serving_digest_value,
        "signature_details": signature_details,
        "rewrite_proof_verification": rewrite_proof_verification,
        "caller_client_request_hash": caller_client_hashout,
        "accept": accepted,
        "state_updated": accepted,
    }
    write_json(Path(args.out).resolve(), result)


def verify_pinned_turn(args: argparse.Namespace) -> None:
    """Verify a later turn, then consume its rid as one serialized transition."""

    state_path = Path(args.state).resolve()
    if Path(args.out).resolve() == state_path:
        raise ValueError("--out must not overwrite --state")
    with session_state_lock(state_path):
        _verify_pinned_turn_locked(args, state_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verirouter.py chain",
        description="Build wrapper certificates and a final chain manifest around zkroute, zkrewrite, and execution signatures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite = subparsers.add_parser("rewrite-cert", help="Build a rewrite certificate wrapper from rewrite inputs and JSON report.")
    rewrite.add_argument("--rewrite-report", required=True)
    rewrite.add_argument("--rewrite-input", required=True)
    rewrite.add_argument("--client-request-file", default="")
    rewrite.add_argument("--rid", required=True)
    rewrite.add_argument("--out", required=True)
    rewrite.add_argument("--artifacts-dir", default="")
    rewrite.add_argument("--chunk-bytes", default="8")
    rewrite.add_argument("--prompt-hash-bin", default="")
    rewrite.add_argument("--rewrite-bin", default="")
    rewrite.add_argument(
        "--rewrite-policy",
        default="",
        help="Caller-pinned rewrite policy. The demo creates an event policy when omitted.",
    )
    rewrite.set_defaults(func=build_rewrite_certificate)

    route = subparsers.add_parser("route-cert", help="Build a route certificate wrapper from zkroute JSON report.")
    route.add_argument("--route-report", required=True)
    route.add_argument("--proof-input", default="")
    route.add_argument("--rid", required=True)
    route.add_argument("--out", required=True)
    route.add_argument("--backend-id", default="")
    route.add_argument("--user-conversation-file", default="")
    route.add_argument("--client-request-hash", default="")
    route.add_argument("--client-request-hashout", default="")
    route.add_argument("--user-prompt-hash", default="")
    route.add_argument("--user-prompt-hashout", default="")
    route.add_argument("--encoder-id", default="public-query-encoder-v1")
    route.add_argument("--chunk-bytes", default="8")
    route.add_argument("--prompt-hash-bin", default="")
    route.add_argument("--route-policy", default="")
    route.add_argument("--route-bin", default="")
    route.set_defaults(func=build_route_certificate)

    chain = subparsers.add_parser("verify", help="Assemble the final certificate chain manifest.")
    chain.add_argument("--trusted-profile", required=True)
    chain.add_argument("--expected-rid", required=True)
    chain.add_argument("--client-state", required=True)
    chain.add_argument("--rewrite-cert", required=True)
    chain.add_argument("--route-cert", required=True)
    chain.add_argument("--execution-cert", required=True)
    chain.add_argument("--backend-registry", required=True)
    chain.add_argument("--rewrite-policy", required=True)
    chain.add_argument("--route-policy", required=True)
    chain.add_argument("--client-request-file", required=True)
    chain.add_argument("--chunk-bytes", type=int, default=8)
    chain.add_argument("--prompt-hash-bin", default="")
    chain.add_argument("--rewrite-bin", default="")
    chain.add_argument("--route-bin", default="")
    chain.add_argument("--response-file", default="")
    chain.add_argument("--response-text", default="")
    chain.add_argument(
        "--response-format", default="auto", choices=["auto", "json", "text"]
    )
    chain.add_argument("--out", required=True)
    chain.set_defaults(func=build_chain_manifest)

    session_init = subparsers.add_parser(
        "session-init",
        help="Pin the model from an accepted initial full certificate chain.",
    )
    session_init.add_argument("--chain-manifest", required=True)
    session_init.add_argument("--session-id", required=True)
    session_init.add_argument("--expected-rid", required=True)
    session_init.add_argument("--state", required=True)
    session_init.add_argument("--out", default="")
    session_init.add_argument("--force", action="store_true")
    session_init.set_defaults(func=initialize_session_pin)

    pinned = subparsers.add_parser(
        "session-verify-turn",
        help="Verify a later session-pinned turn without a new route proof.",
    )
    pinned.add_argument("--trusted-profile", required=True)
    pinned.add_argument("--state", required=True)
    pinned.add_argument("--session-id", default="")
    pinned.add_argument("--expected-rid", required=True)
    pinned.add_argument("--rewrite-cert", required=True)
    pinned.add_argument("--execution-cert", required=True)
    pinned.add_argument("--backend-registry", required=True)
    pinned.add_argument("--rewrite-policy", required=True)
    pinned.add_argument("--client-request-file", required=True)
    pinned.add_argument("--chunk-bytes", type=int, default=8)
    pinned.add_argument("--prompt-hash-bin", default="")
    pinned.add_argument("--rewrite-bin", default="")
    pinned.add_argument("--response-file", default="")
    pinned.add_argument("--response-text", default="")
    pinned.add_argument(
        "--response-format", default="auto", choices=["auto", "json", "text"]
    )
    pinned.add_argument("--out", required=True)
    pinned.set_defaults(func=verify_pinned_turn)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
