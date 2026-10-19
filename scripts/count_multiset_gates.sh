#!/usr/bin/env bash
# Compare model-selection circuit sizes for 2 vs 3 multiset challenge pairs.
# Only builds circuits (no proving), via route_experiment_bench --setup-only.
#
# Usage: scripts/count_multiset_gates.sh PATH [PATH ...]
#   PATH is a route_experiment_manifest.json or a directory searched for them
#   (the pad directories produced by `verirouter.py experiment`).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UTIL="$ROOT/control-plane/zkroute/rust/src/util.rs"
CARGO_TOML="$ROOT/control-plane/zkroute/rust/Cargo.toml"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/control-plane/zkroute/rust/target}"
OUT_DIR="$ROOT/out/multiset_gate_count"
mkdir -p "$OUT_DIR"

MANIFESTS=()
for p in "$@"; do
  if [ -d "$p" ]; then
    while IFS= read -r f; do MANIFESTS+=("$f"); done < <(find "$p" -name route_experiment_manifest.json | sort)
  else
    MANIFESTS+=("$p")
  fi
done
[ "${#MANIFESTS[@]}" -gt 0 ] || { echo "no route_experiment_manifest.json found" >&2; exit 1; }

ORIG_COUNT="$(sed -nE 's/^pub const MULTISET_CHALLENGE_COUNT: usize = ([0-9]+);/\1/p' "$UTIL")"
restore() { sed -i.bak -E "s/^(pub const MULTISET_CHALLENGE_COUNT: usize = )[0-9]+;/\1${ORIG_COUNT};/" "$UTIL"; rm -f "$UTIL.bak"; }
trap restore EXIT

for n in 2 3; do
  sed -i.bak -E "s/^(pub const MULTISET_CHALLENGE_COUNT: usize = )[0-9]+;/\1${n};/" "$UTIL"; rm -f "$UTIL.bak"
  cargo +nightly build --release --locked --manifest-path "$CARGO_TOML" --bin route_experiment_bench
  i=0
  for m in "${MANIFESTS[@]}"; do
    i=$((i + 1))
    echo "[count=$n] $m" >&2
    "$TARGET_DIR/release/route_experiment_bench" --manifest "$m" --zk \
      --score-node-id-mode wide --setup-only > "$OUT_DIR/c${n}_${i}.json"
  done
done

python3 - "$OUT_DIR" "${MANIFESTS[@]}" <<'PY'
import json, sys
out, manifests = sys.argv[1], sys.argv[2:]
print(f"{'node_count':>10} {'visit_len':>9} | {'circuit':<10} {'gates(2)':>10} {'gates(3)':>10} {'+%':>6} | {'2^k(2)':>6} {'2^k(3)':>6}")
for i, m in enumerate(manifests, 1):
    a = json.load(open(f"{out}/c2_{i}.json")); b = json.load(open(f"{out}/c3_{i}.json"))
    for c in ("score_open", "upper", "ef_search"):
        g2, g3 = a[c]["pre_padding_gates"], b[c]["pre_padding_gates"]
        d2, d3 = a[c]["degree_bits"], b[c]["degree_bits"]
        flag = "  <-- crosses a power of two" if d3 > d2 else ""
        print(f"{a['node_count']:>10} {a['visit_len']:>9} | {c:<10} {g2:>10} {g3:>10} {100*(g3-g2)/g2:>5.1f}% | {d2:>6} {d3:>6}{flag}")
PY
