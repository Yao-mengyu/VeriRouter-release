"""Component and multi-turn demo orchestration for the unified CLI."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

from . import chain
from .chain import GOLDILOCKS_MODULUS, resolve_rewrite_input


ROOT = Path(__file__).resolve().parents[1]
REWRITE_RUST_ROOT = ROOT / "data-plane" / "zkrewrite" / "rust"
ROUTE_RUST_MANIFEST = ROOT / "control-plane" / "zkroute" / "rust" / "Cargo.toml"
SIGNATURE_RUST_ROOT = ROOT / "data-plane" / "signature" / "rust"
SIGNATURE_PY_ROOT = ROOT / "data-plane" / "signature" / "python"
if str(SIGNATURE_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGNATURE_PY_ROOT))

import execution_certificate as signature
from execution_certificate import (  # noqa: E402
    certificate_payload,
    compute_request_commitment_report,
    openssl_verify,
    public_key_fingerprint,
    raw_digest,
)


def cargo_release_binary(crate_root: Path, name: str) -> Path:
    """Resolve a Cargo binary exactly where the corresponding build writes it."""
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        target_dir = Path(configured).expanduser()
        if not target_dir.is_absolute():
            target_dir = ROOT / target_dir
    else:
        target_dir = crate_root / "target"
    return target_dir.resolve() / "release" / name


REWRITE_BINARY = cargo_release_binary(
    REWRITE_RUST_ROOT, "rewrite_demo"
)
ROUTE_BINARY = cargo_release_binary(
    ROUTE_RUST_MANIFEST.parent, "veriroute_route"
)
SIGNATURE_BINARY = cargo_release_binary(
    SIGNATURE_RUST_ROOT, "veriroute-signature-tools"
)
PUBLIC_ENCODER_ID = "public-query-encoder-v1"


def banner(text: str) -> None:
    print(f"============== {text} ================", flush=True)


def issue_execution_certificate_inline(
    *,
    private_key: Path,
    public_key: Path,
    rid: str,
    model: str,
    received_request: list[dict],
    request_randomness: list[int],
    chunk_bytes: int,
    expected_request_commitment: list[int],
    response_text: str,
    out: Path,
) -> float:
    banner("Signing Serving Certificate")
    provider_report = compute_request_commitment_report(
        received_request,
        request_randomness=request_randomness,
        chunk_bytes=chunk_bytes,
        request_commitment_bin=str(SIGNATURE_BINARY),
    )
    request_commitment = provider_report["request_commitment"]
    if request_commitment != expected_request_commitment:
        raise ValueError(
            "provider-side recomputation disagrees with the router's request commitment"
        )
    payload, sign_ms = certificate_payload(
        argparse.Namespace(private_key=str(private_key), public_key=str(public_key),
                           rid=rid, model=model, verify=False),
        request_commitment, response_text,
    )
    print(f"signing total time: {sign_ms / 1000.0:.6f} s", flush=True)
    verify_start = time.monotonic()
    verified, details = openssl_verify(
        public_key.resolve(), raw_digest(payload["statement"]["serving_digest"]),
        base64.b64decode(payload["signature"]["value"]),
    )
    verify_elapsed = time.monotonic() - verify_start
    payload["verification"] = {"verified": verified, "details": details or None}
    write_json(out, payload)
    return verify_elapsed


def print_verify_time(label: str, seconds: float) -> None:
    banner(label)
    print(f"verify total time: {seconds:.6f} s", flush=True)


def rewrite_verify_seconds(report: dict) -> float:
    total = 0.0
    root_report = report.get("root_report")
    if isinstance(root_report, dict):
        total += float(root_report.get("verify_time_sec") or 0.0)
    totals = report.get("totals")
    if isinstance(totals, dict):
        total += float(totals.get("leaf_verify_wall_time_sec") or 0.0)
    return total


def build_rust_binaries() -> None:
    banner("compile Rust binaries")
    for manifest, name in [
        (REWRITE_RUST_ROOT / "Cargo.toml", "rewrite_demo"),
        (ROUTE_RUST_MANIFEST, "veriroute_route"),
        (SIGNATURE_RUST_ROOT / "Cargo.toml", "veriroute-signature-tools"),
    ]:
        subprocess.run(
            ["cargo", "+nightly", "build", "--release", "--locked",
             "--manifest-path", str(manifest), "--bin", name],
            cwd=ROOT, check=True,
        )
    banner("compile Rust binaries done")


def run_prover(
    cmd: list[str], *, parallelism: int, label: str, tag: str,
    start_event: str, end_event: str,
) -> None:
    banner(f"Setup {label} Proof")
    setup_start = time.monotonic()
    prove_start: float | None = None
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(int(parallelism))
    process = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )
    assert process.stdout is not None
    captured: list[str] = []
    for raw_line in process.stdout:
        captured.append(raw_line)
        if tag not in raw_line:
            continue
        event = raw_line.split("] ", 1)[-1].strip()
        finished = event.startswith(end_event)
        if prove_start is None and (event == start_event or finished):
            print(f"setup in {time.monotonic() - setup_start:.3f} s", flush=True)
            banner(f"Proving {label} Proof")
            prove_start = time.monotonic()
        if finished and prove_start is not None:
            print(f"proving total time: {time.monotonic() - prove_start:.3f} s", flush=True)
    returncode = process.wait()
    if returncode:
        print("$ " + " ".join(cmd))
        print("".join(captured), end="")
        raise subprocess.CalledProcessError(returncode, cmd, "".join(captured), None)


def run_rewrite_prover(cmd: list[str], *, parallelism: int) -> None:
    run_prover(cmd, parallelism=parallelism, label="Request-rewrite", tag="[VeriRouteRewrite]",
               start_event="prove rewrite leaves in parallel", end_event="rewrite proof done")


def run_route_prover(cmd: list[str], *, parallelism: int) -> None:
    run_prover(cmd, parallelism=parallelism, label="Model-selection", tag="[VeriRouteRoute]",
               start_event="parallel_subproofs enabled: proving subproofs in parallel",
               end_event="all proving stages finished")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_bin(path: Path):
    return json.loads(zlib.decompress(path.read_bytes()).decode("utf-8"))


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def default_route_input(size: str) -> Path:
    return ROOT / "examples" / f"route_input_{size}.bin"


def default_rewrite_input(size: str) -> Path:
    if size == "large":
        return ROOT / "examples" / "rewrite_input_large.bin"
    return ROOT / "examples" / "rewrite_input.bin"


def request_hashes_from_rewrite_input(
    rewrite_input: dict,
) -> tuple[list[int], list[int]]:
    user_conversation, executed_conversation = resolve_rewrite_input(rewrite_input)
    chunk_bytes = int((rewrite_input.get("params") or {}).get("chunk_bytes", 8))
    client_report = compute_request_commitment_report(
        user_conversation,
        request_randomness=[0, 0, 0, 0],
        chunk_bytes=chunk_bytes,
        request_commitment_bin=str(SIGNATURE_BINARY),
    )
    request_randomness = (rewrite_input.get("commit") or {}).get("request_randomness")
    if not isinstance(request_randomness, list) or len(request_randomness) != 4:
        raise ValueError("rewrite input must contain four-limb commit.request_randomness")
    rewritten_report = compute_request_commitment_report(
        executed_conversation,
        request_randomness=request_randomness,
        chunk_bytes=chunk_bytes,
        request_commitment_bin=str(SIGNATURE_BINARY),
    )
    client_request_hash = client_report.get("request_hash_hashout")
    if not isinstance(client_request_hash, list) or len(client_request_hash) != 4:
        raise ValueError("request helper did not return request_hash_hashout")
    return client_request_hash, rewritten_report["request_commitment"]


def fresh_poseidon_randomness() -> list[int]:
    # Rejection sampling gives independent uniform canonical field elements.
    limbs: list[int] = []
    while len(limbs) < 4:
        candidate = secrets.randbits(64)
        if candidate < GOLDILOCKS_MODULUS:
            limbs.append(candidate)
    return limbs


def ensure_rewrite_randomness(rewrite_input: dict) -> None:
    commit = rewrite_input.setdefault("commit", {})
    for name in ("msg_randomness", "frag_randomness", "request_randomness"):
        value = commit.get(name)
        if value is None:
            commit[name] = fresh_poseidon_randomness()
        elif (
            not isinstance(value, list)
            or len(value) != 4
            or any(
                not isinstance(limb, int)
                or isinstance(limb, bool)
                or limb < 0
                or limb >= GOLDILOCKS_MODULUS
                for limb in value
            )
        ):
            raise ValueError(f"commit.{name} must contain four canonical field limbs")


def run_component(*, out_dir: Path, rid: str, rewrite_input: dict,
                  route_input: dict, response_payload: dict, parallelism: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_rewrite_randomness(rewrite_input)
    rewrite_json = out_dir / "rewrite_input.json"
    response_json = out_dir / "response.json"
    rewrite_report = out_dir / "rewrite_report.json"
    route_report = out_dir / "route_report.json"
    write_json(rewrite_json, rewrite_input)
    write_json(response_json, response_payload)

    client_request_hash, request_commitment = request_hashes_from_rewrite_input(
        rewrite_input
    )
    client_request, received_request = resolve_rewrite_input(rewrite_input)
    client_request_json = out_dir / "client_request.json"
    write_json(client_request_json, client_request)
    request_randomness = (rewrite_input.get("commit") or {}).get(
        "request_randomness"
    )
    if not isinstance(request_randomness, list) or len(request_randomness) != 4:
        raise ValueError(
            "rewrite input must contain four-limb commit.request_randomness"
        )
    request_chunk_bytes = int(
        (rewrite_input.get("params") or {}).get("chunk_bytes", 8)
    )
    proof_input_source = resolve_repo_path(str(route_input["proof_input_path"]))
    route_input_bundle = out_dir / "route_input_bundle"
    shutil.copytree(proof_input_source.parent, route_input_bundle, dirs_exist_ok=True)
    proof_input = route_input_bundle / proof_input_source.name
    route_cmd = [
        str(ROUTE_BINARY),
        "--trace",
        str(proof_input),
        "--zk",
        "--parallel-subproofs",
        "--report",
        str(route_report),
    ]
    run_route_prover(
        route_cmd,
        parallelism=parallelism,
    )
    raw_route_report = json.loads(route_report.read_text(encoding="utf-8"))
    route_verify_sec = float(raw_route_report.get("verify_sec") or 0.0)
    print_verify_time("Verify Model-selection Proof", route_verify_sec)
    route_report_payload = json.loads(route_report.read_text(encoding="utf-8"))

    rewrite_cert = out_dir / "rewrite_certificate.json"
    route_cert = out_dir / "route_certificate.json"
    rewrite_policy = out_dir / "rewrite_certificate.rewrite_verify_policy.json"
    route_policy = out_dir / "route_certificate.route_verify_policy.json"
    execution_cert = out_dir / "execution_certificate.json"
    chain_manifest = out_dir / "chain_manifest.json"
    artifacts = out_dir / "artifacts"
    backend_id = str(route_report_payload["final_statement"]["output_model_id"])
    chain.main(
        [
            'route-cert',
            '--route-report', str(route_report),
            '--proof-input', str(proof_input),
            '--rid', rid,
            '--backend-id', backend_id,
            '--client-request-hash', str(client_request_hash[0]),
            '--client-request-hashout', json.dumps(client_request_hash),
            '--encoder-id', PUBLIC_ENCODER_ID,
            '--out', str(route_cert),
            '--route-bin', str(ROUTE_BINARY),
        ]
    )

    private_key = out_dir / "backend_ed25519.pem"
    public_key = out_dir / "backend_ed25519.pub.pem"
    backend_registry = out_dir / "backend_registry.json"
    banner("Setup Serving Certificate")
    key_start = time.monotonic()
    signature.init_keypair(
        argparse.Namespace(private_key=str(private_key), public_key=str(public_key), force=True, out="")
    )
    print(f"generate key time: {time.monotonic() - key_start:.3f} s", flush=True)
    write_json(
        backend_registry,
        {
            "version": "backend_registry_v1",
            "backends": {
                backend_id: {
                    "public_key_path": str(public_key.resolve()),
                    "public_key_fingerprint": public_key_fingerprint(public_key.resolve()),
                }
            },
        },
    )
    serving_verify_sec = issue_execution_certificate_inline(
        private_key=private_key,
        public_key=public_key,
        rid=rid,
        model=backend_id,
        received_request=received_request,
        request_randomness=request_randomness,
        chunk_bytes=request_chunk_bytes,
        expected_request_commitment=request_commitment,
        response_text=str(response_payload.get("response", "")),
        out=execution_cert,
    )
    print_verify_time("Verify Serving Certificate", serving_verify_sec)
    run_rewrite_prover(
        [
            str(REWRITE_BINARY),
            "--input",
            str(rewrite_json),
            "--report-json",
            str(rewrite_report),
            "--agg",
            "--fan-in",
            "4",
            "--dedup-merkle",
            "--zk",
            "--strict-commit",
        ],
        parallelism=parallelism,
    )
    raw_rewrite_report = json.loads(rewrite_report.read_text(encoding="utf-8"))
    rewrite_verify_sec = rewrite_verify_seconds(raw_rewrite_report)
    print_verify_time("Verify Request-rewrite Proof", rewrite_verify_sec)
    chain.main(
        [
            'rewrite-cert',
            '--rewrite-report', str(rewrite_report),
            '--rewrite-input', str(rewrite_json),
            '--client-request-file', str(client_request_json),
            '--chunk-bytes', str(request_chunk_bytes),
            '--rid', rid,
            '--out', str(rewrite_cert),
            '--artifacts-dir', str(artifacts),
            '--prompt-hash-bin', str(SIGNATURE_BINARY),
            '--rewrite-bin', str(REWRITE_BINARY),
        ]
    )
    # Offline fixture setup: this process owns the source/assets and just
    # rebuilt the approved circuits. Production clients provision this
    # profile through their authenticated endorsement channel instead.
    trusted_profile = out_dir / "fixture_trusted_profile.json"
    profile = {"version": "verirouter_trusted_profile_v2"}
    for kind, certificate_path in [("route", route_cert), ("rewrite", rewrite_cert)]:
        verified = json.loads(certificate_path.read_text())["proof_verification"]
        profile[kind] = {
            "verifier_data_hex": verified["verifier_data_hex"],
            "zero_knowledge": True,
            "recursion_depth": verified["recursion_depth"],
            "max_depth": 8,
            "max_leaf_count": 65536,
        }
    write_json(trusted_profile, profile)
    chain.main(
        [
            'verify',
            '--trusted-profile', str(trusted_profile),
            '--expected-rid', rid,
            '--client-state', str(out_dir / 'client_state.json'),
            '--rewrite-cert', str(rewrite_cert),
            '--route-cert', str(route_cert),
            '--execution-cert', str(execution_cert),
            '--backend-registry', str(backend_registry),
            '--rewrite-policy', str(rewrite_policy),
            '--route-policy', str(route_policy),
            '--client-request-file', str(client_request_json),
            '--chunk-bytes', str(request_chunk_bytes),
            '--prompt-hash-bin', str(SIGNATURE_BINARY),
            '--response-text', str(response_payload.get('response', '')),
            '--response-format', 'text',
            '--out', str(chain_manifest),
            '--route-bin', str(ROUTE_BINARY),
            '--rewrite-bin', str(REWRITE_BINARY),
        ]
    )

    print(f"Result written to {out_dir}", flush=True)


def prepare_rewrite_template(template: dict) -> dict:
    payload = copy.deepcopy(template)
    ensure_rewrite_randomness(payload)
    return payload

def make_turn_rewrite_input(template: dict, user_text: str) -> dict:
    """Bind the current request before the fixture response is produced."""
    payload = copy.deepcopy(template)
    payload.setdefault("commit", {})["request_randomness"] = fresh_poseidon_randomness()
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("rewrite input template is missing messages[]")
    messages[0]["role"] = "user"
    messages[0]["content"] = user_text
    messages[0]["present"] = True
    for padding in messages[1:]:
        padding["present"] = False
        padding["gap_before"] = []
        padding["prefixes"] = []
        padding["suffixes"] = []
    return payload


def run_component_turn(*, turn_dir: Path, rid: str, rewrite_input: dict,
                       route_input: dict, response: str, backend_id: str,
                       parallelism: int) -> dict:
    run_component(out_dir=turn_dir, rid=rid, rewrite_input=rewrite_input,
                  route_input=route_input, response_payload={"backend_id": backend_id, "response": response},
                  parallelism=parallelism)
    manifest = chain.load_json(turn_dir / "chain_manifest.json")
    route = chain.load_json(turn_dir / "route_report.json")
    return {
        "rid": rid, "mode": "route_each_turn",
        "chain_manifest": str(turn_dir / "chain_manifest.json"),
        "accept": bool(manifest.get("accept")), "route_query_index": route.get("query_index"),
        "selected_backend_id": manifest.get("summary", {}).get("selected_backend_id"),
        "backend_registry": str(turn_dir / "backend_registry.json"),
    }

def run_rewrite_and_execution_only(
    *,
    turn_dir: Path,
    rid: str,
    rewrite_input: dict,
    response: str,
    private_key: Path,
    public_key: Path,
    selected_backend_id: str,
    backend_registry: Path,
    session_state: Path,
    session_id: str,
    parallelism: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="veriroute_e2e_amortized_") as tmp:
        tmpdir = Path(tmp)
        rewrite_json = tmpdir / "rewrite_input.json"
        write_json(rewrite_json, rewrite_input)
        client_request, received_request = resolve_rewrite_input(rewrite_input)
        client_request_path = tmpdir / "client_request.json"
        write_json(client_request_path, client_request)
        request_chunk_bytes = int(
            (rewrite_input.get("params") or {}).get("chunk_bytes", 8)
        )
        rewrite_report = turn_dir / "rewrite_report.json"
        run_rewrite_prover(
            [str(REWRITE_BINARY), "--input", str(rewrite_json), "--report-json", str(rewrite_report),
             "--fan-in", "4", "--dedup-merkle", "--zk", "--strict-commit"],
            parallelism=parallelism,
        )
        rewrite_cert = turn_dir / "rewrite_certificate.json"
        rewrite_policy = turn_dir / "rewrite_certificate.rewrite_verify_policy.json"
        artifacts = turn_dir / "artifacts"
        chain.main(
            [
                'rewrite-cert',
                '--rewrite-report', str(rewrite_report),
                '--rewrite-input', str(rewrite_json),
                '--client-request-file', str(client_request_path),
                '--chunk-bytes', str(request_chunk_bytes),
                '--rid', rid,
                '--out', str(rewrite_cert),
                '--artifacts-dir', str(artifacts),
                '--rewrite-bin', str(REWRITE_BINARY),
                '--prompt-hash-bin', str(SIGNATURE_BINARY),
            ]
        )
        rewrite_cert_payload = json.loads(rewrite_cert.read_text(encoding="utf-8"))
        request_commitment = rewrite_cert_payload["rewrite_statement"]["request_commitment"]
        received_request_path = tmpdir / "received_request.json"
        write_json(received_request_path, received_request)
        request_randomness = (rewrite_input.get("commit") or {}).get("request_randomness")
        if not isinstance(request_randomness, list) or len(request_randomness) != 4:
            raise ValueError("rewrite input must contain four-limb commit.request_randomness")
        execution_cert = turn_dir / "execution_certificate.json"
        signature.main(
            [
                'issue',
                '--private-key', str(private_key),
                '--public-key', str(public_key),
                '--rid', rid,
                '--model', selected_backend_id,
                '--received-request-file', str(received_request_path),
                '--request-randomness', json.dumps(request_randomness),
                '--chunk-bytes', str(request_chunk_bytes),
                '--response-text', response,
                '--verify',
                '--out', str(execution_cert),
                '--request-commitment-bin', str(SIGNATURE_BINARY),
            ]
        )
        provider_commitment = (
            json.loads(execution_cert.read_text(encoding="utf-8"))
            .get("statement", {})
            .get("request_commitment")
        )
        if provider_commitment != request_commitment:
            raise ValueError("provider recomputation disagrees with rewrite request commitment")
        turn_manifest_path = turn_dir / "turn_manifest.json"
        chain.main(
            [
                'session-verify-turn',
                '--trusted-profile', str(backend_registry.parent / 'fixture_trusted_profile.json'),
                '--state', str(session_state),
                '--session-id', session_id,
                '--expected-rid', rid,
                '--rewrite-cert', str(rewrite_cert),
                '--execution-cert', str(execution_cert),
                '--backend-registry', str(backend_registry),
                '--rewrite-policy', str(rewrite_policy),
                '--client-request-file', str(client_request_path),
                '--chunk-bytes', str(request_chunk_bytes),
                '--response-text', response,
                '--response-format', 'text',
                '--out', str(turn_manifest_path),
                '--rewrite-bin', str(REWRITE_BINARY),
                '--prompt-hash-bin', str(SIGNATURE_BINARY),
            ]
        )
    turn_manifest = json.loads(turn_manifest_path.read_text(encoding="utf-8"))
    return {
        "rid": rid,
        "mode": "session_pinned_later_turn",
        "turn_manifest": str(turn_manifest_path),
        "accept": bool(turn_manifest["accept"]),
        "selected_backend_id": selected_backend_id,
    }


def run_session(args: argparse.Namespace) -> None:
    setting = read_bin(Path(args.setting_bin))
    route_input = read_bin(Path(args.route_input_bin) if args.route_input_bin else default_route_input(args.demo_size))
    name = str(setting.get("name", Path(args.setting_bin).stem))
    session_rid = args.rid or str(setting.get("rid", f"demo-{name}"))
    mode = str(setting.get("mode", "route_each_turn"))
    if mode not in {"route_each_turn", "session_pinned"}:
        raise ValueError(f"unsupported e2e mode: {mode}")
    turns = setting.get("turns") or []
    if not turns:
        raise ValueError("e2e setting is missing turns[]")
    backend_id = str(route_input.get("expected_output_model_id", setting.get("backend_id", "")))
    if not backend_id:
        raise ValueError("the route fixture does not declare an expected output model")
    out_dir = Path(args.out_dir).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    template = prepare_rewrite_template(setting["rewrite_input_template"])
    turn_summaries = []
    private_key: Path | None = None
    public_key: Path | None = None
    selected_backend_id: str | None = None
    backend_registry: Path | None = None
    session_state = out_dir / "session_pin_state.json"

    for idx, turn in enumerate(turns, start=1):
        user_text = str(turn["user"])
        response = str(
            turn.get("fixture_response", "This is a deterministic fixture response.")
        )
        rewrite_input = make_turn_rewrite_input(template, user_text)
        turn_dir = out_dir / f"turn_{idx:02d}"
        turn_rid = f"{session_rid}-turn-{idx:02d}"
        if mode == "route_each_turn" or idx == 1:
            summary = run_component_turn(
                turn_dir=turn_dir,
                rid=turn_rid,
                rewrite_input=rewrite_input,
                route_input=route_input,
                response=response,
                backend_id=backend_id,
                parallelism=args.parallelism,
            )
            if idx == 1:
                private_key = turn_dir / "backend_ed25519.pem"
                public_key = turn_dir / "backend_ed25519.pub.pem"
                selected_backend_id = str(summary.get("selected_backend_id") or backend_id)
                if not summary.get("accept"):
                    raise RuntimeError("initial full certificate chain was rejected")
                backend_registry = Path(str(summary["backend_registry"])).resolve()
                if mode == "session_pinned":
                    summary["mode"] = "session_pinned_initial"
                    chain.main(
                        [
                            'session-init',
                            '--chain-manifest', str(turn_dir / 'chain_manifest.json'),
                            '--session-id', session_rid,
                            '--expected-rid', turn_rid,
                            '--state', str(session_state),
                        ]
                    )
        else:
            if (
                private_key is None
                or public_key is None
                or selected_backend_id is None
                or backend_registry is None
            ):
                raise RuntimeError("first-turn route certificate is missing")
            summary = run_rewrite_and_execution_only(
                turn_dir=turn_dir,
                rid=turn_rid,
                rewrite_input=rewrite_input,
                response=response,
                private_key=private_key,
                public_key=public_key,
                selected_backend_id=selected_backend_id,
                backend_registry=backend_registry,
                session_state=session_state,
                session_id=session_rid,
                parallelism=args.parallelism,
            )
        turn_summaries.append(summary)

    summary = {
        "setting": name,
        "mode": mode,
        "session_rid": session_rid,
        "turn_count": len(turns),
        "serving_mode": "offline_fixture",
        "route_policy": (
            "prove route for every turn"
            if mode == "route_each_turn"
            else "pin the model from the accepted initial selection; verify current rewrite and serving evidence on later turns"
        ),
        "turns": turn_summaries,
        "accept": all(bool(turn.get("accept")) for turn in turn_summaries),
    }
    write_json(out_dir / "e2e_summary.json", summary)
    print(f"Result written to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="verirouter.py demo", description="Run a component or multi-turn VeriRouter demo.")
    parser.add_argument("--setting-bin", default="", help="Multi-turn setting; omit for the component demo.")
    parser.add_argument("--rewrite-input-bin", default="")
    parser.add_argument("--route-input-bin", default="")
    parser.add_argument("--demo-size", choices=["small", "large"], default="small")
    parser.add_argument("--response-bin", default=str(ROOT / "examples/response.bin"))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--rid", default="")
    parser.add_argument("--parallelism", type=int, default=32)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or str(
        ROOT / "out" / ("e2e_demo" if args.setting_bin else "component_demo"))
    build_rust_binaries()
    if args.setting_bin:
        run_session(args)
    else:
        run_component(
            out_dir=Path(args.out_dir).resolve(), rid=args.rid or "demo-rid-001",
            rewrite_input=read_bin(
                Path(args.rewrite_input_bin) if args.rewrite_input_bin else default_rewrite_input(args.demo_size)),
            route_input=read_bin(
                Path(args.route_input_bin) if args.route_input_bin else default_route_input(args.demo_size)),
            response_payload=read_bin(Path(args.response_bin)), parallelism=args.parallelism,
        )
