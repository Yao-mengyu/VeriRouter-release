"""Issue and verify the paper's provider serving certificate.

The provider signs a commitment to the complete request it received, never the
raw request hash. Request randomness is accepted only by the issuing side and
is deliberately omitted from the certificate.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


VERSION = "serving_certificate_v1"
SIGNATURE_ALGORITHM = "ed25519"
DIGEST_ALGORITHM = "sha256"
DOMAIN_SEPARATOR = b"VeriRouter.serving-certificate.v1"
REQUEST_COMMITMENT_SCHEME = "poseidon-hashout4-request-commitment-v1"


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def tagged_sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def raw_digest(tagged_hex: str) -> bytes:
    prefix = "sha256:"
    if not tagged_hex.startswith(prefix):
        raise ValueError(f"expected sha256:... digest, got {tagged_hex!r}")
    value = bytes.fromhex(tagged_hex[len(prefix) :])
    if len(value) != 32:
        raise ValueError("SHA-256 digest must contain exactly 32 bytes")
    return value


def uint64_be(value: int) -> bytes:
    if value < 0 or value >= (1 << 64):
        raise ValueError(f"expected uint64 value, got {value!r}")
    return int(value).to_bytes(8, "big")


def parse_hashout(value: Any) -> list[int]:
    if value is None or value == "":
        raise ValueError("empty HashOut value")
    if isinstance(value, list):
        parts = value
    else:
        text = str(value).strip()
        parts = json.loads(text) if text.startswith("[") else text.split(",")
    if len(parts) != 4:
        raise ValueError(f"expected 4 HashOut limbs, got {len(parts)}")
    limbs = [int(str(part).strip(), 0) for part in parts]
    for limb in limbs:
        uint64_be(limb)
    return limbs


def encode_field(value: bytes) -> bytes:
    """Protocol-fixed injective framing for one tuple field."""

    if len(value) >= (1 << 32):
        raise ValueError("tuple field is too long")
    return len(value).to_bytes(4, "big") + value


def serving_digest(
    rid: str,
    model: str,
    request_commitment: list[int],
    response_hash: str,
) -> str:
    commitment_bytes = b"".join(uint64_be(value) for value in parse_hashout(request_commitment))
    encoded = b"".join(
        [
            DOMAIN_SEPARATOR,
            encode_field(rid.encode("utf-8")),
            encode_field(model.encode("utf-8")),
            encode_field(commitment_bytes),
            encode_field(raw_digest(response_hash)),
        ]
    )
    return tagged_sha256(encoded)


def recompute_serving_digest_from_statement(statement: dict[str, Any]) -> str:
    return serving_digest(
        rid=str(statement["rid"]),
        model=str(statement["model"]),
        request_commitment=parse_hashout(statement["request_commitment"]),
        response_hash=str(statement["response_hash"]),
    )


def load_json_or_text(path: Path, mode: str) -> Any:
    raw = path.read_text(encoding="utf-8")
    if mode == "json":
        return json.loads(raw)
    if mode == "text":
        return raw
    stripped = raw.strip()
    if path.suffix.lower() == ".json" or stripped.startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def resolve_payload(file_path: str, inline_value: str, mode: str, label: str) -> Any:
    if bool(file_path) == bool(inline_value):
        raise ValueError(f"exactly one of --{label}-file or --{label}-text is required")
    return load_json_or_text(Path(file_path).resolve(), mode) if file_path else inline_value


def hash_protocol_value(payload: Any) -> str:
    return tagged_sha256(canonical_json_bytes(payload))


def default_request_commitment_bin() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "rust"
        / "target"
        / "release"
        / "veriroute-signature-tools"
    )


def compute_request_commitment_report(
    received_request: Any,
    *,
    request_randomness: list[int],
    chunk_bytes: int,
    request_commitment_bin: str = "",
) -> dict[str, Any]:
    """Provider-side recomputation of c_m from (received request, r_m)."""

    randomness = parse_hashout(request_randomness)
    with tempfile.TemporaryDirectory(prefix="verirouter_provider_request_") as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "received_request.json"
        output_path = tmp / "request_commitment.json"
        input_path.write_text(
            json.dumps(received_request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        binary = (
            Path(request_commitment_bin).resolve()
            if request_commitment_bin
            else default_request_commitment_bin()
        )
        tool_args = [
            "--input",
            str(input_path),
            "--chunk-bytes",
            str(chunk_bytes),
            "--request-randomness",
            json.dumps(randomness),
            "--out",
            str(output_path),
        ]
        if binary.exists():
            command = [str(binary), *tool_args]
            cwd = None
        else:
            rust_dir = Path(__file__).resolve().parents[1] / "rust"
            command = [
                "cargo",
                "+nightly",
                "run",
                "--release",
                "--locked",
                "--bin",
                "veriroute-signature-tools",
                "--",
                *tool_args,
            ]
            cwd = rust_dir
        subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
        report = json.loads(output_path.read_text(encoding="utf-8"))
    if "request_commitment" not in report:
        raise ValueError("request commitment helper output is missing request_commitment")
    report["request_commitment"] = parse_hashout(report["request_commitment"])
    return report


def resolve_request_commitment(args: argparse.Namespace) -> list[int]:
    manual = bool(args.request_commitment)
    received = bool(args.received_request_file) or bool(args.received_request_text)
    if manual == received:
        raise ValueError(
            "provide either --request-commitment or a received request plus --request-randomness"
        )
    if manual:
        return parse_hashout(args.request_commitment)
    if not args.request_randomness:
        raise ValueError("provider recomputation requires --request-randomness")
    payload = resolve_payload(
        args.received_request_file,
        args.received_request_text,
        "json",
        "received-request",
    )
    return compute_request_commitment_report(
        payload,
        request_randomness=parse_hashout(args.request_randomness),
        chunk_bytes=int(args.chunk_bytes),
        request_commitment_bin=args.request_commitment_bin,
    )["request_commitment"]


def public_key_fingerprint(public_key_path: Path) -> str:
    return tagged_sha256(public_key_path.read_bytes())


def openssl_sign(private_key_path: Path, message: bytes) -> tuple[bytes, float]:
    with tempfile.TemporaryDirectory(prefix="verirouter_sign_") as tmpdir:
        message_path = Path(tmpdir) / "message.bin"
        signature_path = Path(tmpdir) / "signature.bin"
        message_path.write_bytes(message)
        started = time.perf_counter()
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key_path),
                "-in",
                str(message_path),
                "-out",
                str(signature_path),
            ],
            check=True,
            capture_output=True,
        )
        return signature_path.read_bytes(), (time.perf_counter() - started) * 1000.0


def openssl_verify(public_key_path: Path, message: bytes, signature: bytes) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="verirouter_verify_") as tmpdir:
        message_path = Path(tmpdir) / "message.bin"
        signature_path = Path(tmpdir) / "signature.bin"
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(public_key_path),
                "-in",
                str(message_path),
                "-sigfile",
                str(signature_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        details = "\n".join(
            item.strip() for item in (result.stdout, result.stderr) if item and item.strip()
        )
        return result.returncode == 0, details


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def init_keypair(args: argparse.Namespace) -> None:
    private_key = Path(args.private_key).resolve()
    public_key = Path(args.public_key).resolve()
    if not args.force and (private_key.exists() or public_key.exists()):
        raise FileExistsError("key file already exists; pass --force to overwrite")
    private_key.parent.mkdir(parents=True, exist_ok=True)
    public_key.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_key)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
        check=True,
        capture_output=True,
    )
    payload = {
        "version": VERSION,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "private_key_path": str(private_key),
        "public_key_path": str(public_key),
        "public_key_fingerprint": public_key_fingerprint(public_key),
    }
    if args.out:
        write_json(Path(args.out).resolve(), payload)
    else:
        print(json.dumps(payload, indent=2))


def certificate_payload(
    args: argparse.Namespace,
    request_commitment: list[int],
    response_payload: Any,
) -> tuple[dict[str, Any], float]:
    private_key = Path(args.private_key).resolve()
    public_key = Path(args.public_key).resolve()
    response_hash = hash_protocol_value(response_payload)
    digest = serving_digest(args.rid, args.model, request_commitment, response_hash)
    signature, sign_ms = openssl_sign(private_key, raw_digest(digest))
    verified: bool | None = None
    details: str | None = None
    if args.verify:
        verified, raw_details = openssl_verify(public_key, raw_digest(digest), signature)
        details = raw_details or None
    payload = {
        "certificate_type": "serving_certificate",
        "version": VERSION,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "digest_algorithm": DIGEST_ALGORITHM,
        "issued_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "statement": {
            "rid": args.rid,
            "model": args.model,
            "request_commitment_scheme": REQUEST_COMMITMENT_SCHEME,
            "request_commitment": request_commitment,
            "response_hash": response_hash,
            "serving_digest": digest,
        },
        "signature": {
            "encoding": "base64",
            "value": base64.b64encode(signature).decode("ascii"),
        },
        "key_metadata": {
            "public_key_path": str(public_key),
            "public_key_fingerprint": public_key_fingerprint(public_key),
        },
        "verification": {"verified": verified, "details": details},
    }
    return payload, sign_ms


def issue_certificate(args: argparse.Namespace) -> None:
    request_commitment = resolve_request_commitment(args)
    response_payload = resolve_payload(
        args.response_file,
        args.response_text,
        args.response_format,
        "response",
    )
    payload, _ = certificate_payload(args, request_commitment, response_payload)
    if args.out:
        write_json(Path(args.out).resolve(), payload)
    else:
        print(json.dumps(payload, indent=2))


def verify_certificate(args: argparse.Namespace) -> None:
    cert = json.loads(Path(args.certificate).resolve().read_text(encoding="utf-8"))
    if cert.get("version") != VERSION:
        raise ValueError(f"unsupported serving certificate version: {cert.get('version')!r}")
    labels_match = (
        cert.get("certificate_type") == "serving_certificate"
        and cert.get("signature_algorithm") == SIGNATURE_ALGORITHM
        and cert.get("digest_algorithm") == DIGEST_ALGORITHM
    )
    statement = cert["statement"]
    scheme_matches = statement.get("request_commitment_scheme") == REQUEST_COMMITMENT_SCHEME
    public_key = Path(args.public_key).resolve()
    digest = recompute_serving_digest_from_statement(statement)
    response_payload = resolve_payload(
        args.response_file,
        args.response_text,
        args.response_format,
        "response",
    )
    response_matches = hash_protocol_value(response_payload) == statement.get("response_hash")
    digest_matches = digest == statement.get("serving_digest")
    key_fp = public_key_fingerprint(public_key)
    metadata_matches = cert.get("key_metadata", {}).get("public_key_fingerprint") == key_fp
    signature = base64.b64decode(cert["signature"]["value"], validate=True)
    signature_valid, details = openssl_verify(public_key, raw_digest(digest), signature)
    result = {
        "certificate_type": cert.get("certificate_type"),
        "version": cert.get("version"),
        "verified": bool(
            signature_valid
            and digest_matches
            and response_matches
            and metadata_matches
            and labels_match
            and scheme_matches
        ),
        "checks": {
            "protocol_labels_match": labels_match,
            "request_commitment_scheme_matches": scheme_matches,
            "signature_valid": signature_valid,
            "digest_matches_statement": digest_matches,
            "response_hash_matches": response_matches,
            "registry_key_matches_metadata": metadata_matches,
        },
        "details": details or None,
        "statement": statement,
    }
    if args.out:
        write_json(Path(args.out).resolve(), result)
    else:
        print(json.dumps(result, indent=2))


def benchmark_signing(args: argparse.Namespace) -> None:
    commitment = resolve_request_commitment(args)
    response_payload = resolve_payload(
        args.response_file,
        args.response_text,
        args.response_format,
        "response",
    )
    digest = serving_digest(args.rid, args.model, commitment, hash_protocol_value(response_payload))
    digest_bytes = raw_digest(digest)
    private_key = Path(args.private_key).resolve()
    for _ in range(args.warmup):
        openssl_sign(private_key, digest_bytes)
    samples = [openssl_sign(private_key, digest_bytes)[1] for _ in range(args.iterations)]
    ordered = sorted(samples)
    p95 = ordered[round(0.95 * (len(ordered) - 1))]
    payload = {
        "benchmark_type": "serving_certificate_signing_latency",
        "version": VERSION,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "summary_ms": {
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "p95": p95,
        },
        "samples_ms": samples,
    }
    if args.out:
        write_json(Path(args.out).resolve(), payload)
    else:
        print(json.dumps(payload, indent=2))


def add_statement_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rid", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-commitment", default="")
    parser.add_argument("--received-request-file", default="")
    parser.add_argument("--received-request-text", default="")
    parser.add_argument("--request-randomness", default="")
    parser.add_argument("--chunk-bytes", type=int, default=8)
    parser.add_argument("--request-commitment-bin", default="")
    parser.add_argument("--response-file", default="")
    parser.add_argument("--response-text", default="")
    parser.add_argument("--response-format", default="auto", choices=["auto", "json", "text"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verirouter.py signature", description="Issue and verify VeriRouter serving certificates.")
    commands = parser.add_subparsers(dest="command", required=True)

    init_key = commands.add_parser("init-key")
    init_key.add_argument("--private-key", required=True)
    init_key.add_argument("--public-key", required=True)
    init_key.add_argument("--out", default="")
    init_key.add_argument("--force", action="store_true")
    init_key.set_defaults(func=init_keypair)

    issue = commands.add_parser("issue")
    issue.add_argument("--private-key", required=True)
    issue.add_argument("--public-key", required=True)
    add_statement_args(issue)
    issue.add_argument("--verify", action="store_true")
    issue.add_argument("--out", default="")
    issue.set_defaults(func=issue_certificate)

    verify = commands.add_parser("verify")
    verify.add_argument("--certificate", required=True)
    verify.add_argument("--public-key", required=True)
    verify.add_argument("--response-file", default="")
    verify.add_argument("--response-text", default="")
    verify.add_argument("--response-format", default="auto", choices=["auto", "json", "text"])
    verify.add_argument("--out", default="")
    verify.set_defaults(func=verify_certificate)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--private-key", required=True)
    benchmark.add_argument("--public-key", required=True)
    add_statement_args(benchmark)
    benchmark.add_argument("--iterations", type=int, default=20)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--out", default="")
    benchmark.set_defaults(func=benchmark_signing, verify=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "iterations", 1) <= 0:
        raise ValueError("--iterations must be positive")
    args.func(args)
