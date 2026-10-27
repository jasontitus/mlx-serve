#!/bin/bash
# Run locally after disconnecting. Only the Python drivers' owned servers are stopped.
set -u -o pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="${1:-$HOME/.mlx-serve/bench/lookup-staircut/runs/verified-$(date +%Y%m%d-%H%M%S)}"
PHASE="${2:-all}"
case "$PHASE" in all|fast|canonical|economics) ;; *) echo "phase must be all, fast, canonical or economics" >&2; exit 2 ;; esac
if [ -e "$OUT" ]; then echo "Output must be new: $OUT" >&2; exit 2; fi
mkdir -p "$OUT" || exit 1
printf 'Offline validation started\nEvidence: %s\nPhase: %s\n' "$OUT" "$PHASE"
printf 'phase\texit_code\n' > "$OUT/status.tsv"
FAILED=0
run() {
    local label="$1"; shift
    printf '\n=== %s ===\n' "$label"
    "$@" 2>&1 | tee "$OUT/$label.log"
    local code=${PIPESTATUS[0]}
    printf '%s\t%s\n' "$label" "$code" >> "$OUT/status.tsv"
    if [ "$code" -ne 0 ]; then FAILED=1; fi
    return "$code"
}
run build .zig-toolchain/zig build -Doptimize=ReleaseFast || exit 1
run python-regressions python3 -m unittest discover -s tests -p 'test_lookup_*.py' || exit 1
run policy-regressions .zig-toolchain/zig test src/lookup_adapter.zig || exit 1
run store-regressions .zig-toolchain/zig test src/token_history.zig || exit 1
run lookup-regressions .zig-toolchain/zig build test -Doptimize=ReleaseFast '-Dtest-filter=lookup' || exit 1
run prepare python3 tests/prepare_lookup_offline.py --require-probe \
    --probe "$HOME/.mlx-serve/bench/lookup-staircut/tools/llmprobe/node_modules/llmprobe/bin/dist/llmprobe.mjs" || exit 1
if [ "$PHASE" = all ] || [ "$PHASE" = canonical ]; then
    run canonical python3 tests/bench_lookup_staircut.py --model qwen27b \
        --modes dflash-local --cases copy,budget --boot-repetitions 2 \
        --repetitions 3 --warmup 1 --max-tokens 32 --ctx-size 16384 \
        --output "$OUT/canonical" || true
fi
if [ "$PHASE" = all ] || [ "$PHASE" = fast ]; then
    run controls python3 tests/validate_lookup_controls.py --model qwen27b --output "$OUT/controls" || true
    run runtime python3 tests/validate_lookup_runtime.py --model qwen27b --output "$OUT/runtime" || true
    run frozen-tiers python3 tests/bench_lookup_staircut.py --model qwen27b \
        --modes dflash-all --cases copy,corpus,budget,keyed --seeds 0,7,19 \
        --boot-repetitions 2 --repetitions 1 --warmup 1 --max-tokens 32 \
        --ctx-size 2048 --skip-probe --output "$OUT/frozen-tiers" || true
fi
if [ "$PHASE" = all ] || [ "$PHASE" = economics ]; then
    run economics python3 tests/bench_lookup_economics.py --model qwen27b \
        --modes dflash-local --cases lookup-hit,lookup-edit,prose --contexts 0 --train-repetitions 2 \
        --heldout-repetitions 2 --priming 1 --max-tokens 65 --ctx-size 2048 \
        --output "$OUT/economics" || true
fi
printf '\nFinished. Failure flag: %s\nInspect %s/status.tsv and each phase summary.\n' "$FAILED" "$OUT"
printf 'Runs are contended diagnostics by default; no isolated speedup is certified.\n'
exit "$FAILED"
