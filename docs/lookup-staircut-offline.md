# Offline lookup / StairCut validation

Run from the **mlx-serve repository root**. The prepared machine has targets, drafters, fixtures, Node and pinned llmprobe 0.6.17 in `~/.mlx-serve/bench/lookup-staircut`. Offline operation does not require package installation or `npx` downloads. A new machine must prepare its own manifest/assets before disconnecting.

The independently authored implementation and authoring limitations are described in [the design note](lookup-staircut-port.md). The [results note](lookup-staircut-results.md) separates measured revisions by binary identity; an earlier measurement does not certify the current working tree.

## Build and CPU/in-process checks

```sh
.zig-toolchain/zig build -Doptimize=ReleaseFast
.zig-toolchain/zig test src/lookup_adapter.zig
python3 -m unittest discover -s tests -p 'test_lookup_*.py'
.zig-toolchain/zig build test -Doptimize=ReleaseFast -Dtest-filter=lookup
```

Compilation is not a GPU benchmark. The integrated tests include short real hybrid verification/commit scenarios. Keep model benchmarks separate from builds so an interrupted build is not mistaken for an inference result. Never reuse an old installed binary after a source-only fix.

## One bounded GPU window at a time

The repository owns the authoritative drivers:

- `tests/lookup_similar_requests.py`: unseen related requests and conflict/miss controls; both native baselines; optional serial correctness oracle.
- `tests/lookup_confirmation_window.py`: predeclared four-pair confirmation split into independently launched full-workload windows.
- `tests/lookup_counterfactual_window.py`: first-offer same-state width/action diagnostics, with full learner reset and strict-authenticated owned server.
- `tests/lookup_canonical_window.py`: one complete pinned canonical arm, or its separate exact replay.
- `tests/lookup_route_diagnostic.py`: existing exact-reuse/route mechanism controls.
- `tests/lookup_transfer_contract.json`: predeclared training/evaluation request contract.

Each window has a **290-second process-group bound**, owns its server, and never automatically starts another window. A deadline-truncated schedule fails; completed subsets are not silently promoted. Use a new output directory for every run. The existing drivers support `--plan-only`; the confirmation wrapper instead has a CPU-only `--declare` operation and fixed, non-reusable output slots. Inspect the result before selecting another action.

Private `tools/lookup-transfer` launchers on the prepared machine delegate to these repository files. `MLX_SERVE_REPO` can select another checkout for those launchers. Neither historical run artifacts nor frozen seeds are rewritten by the migration.

**Do not use `tests/run_lookup_offline_validation.sh ... all` for a five-minute session.** That older opt-in suite and the general economics/canonical runners can execute many windows and are not bounded short-benchmark substitutes. AC power is preferable; no driver stops unrelated user processes or bypasses memory admission.

## Held-out shared-history comparison

```sh
kit="$HOME/.mlx-serve/bench/lookup-staircut"
python3 tests/lookup_similar_requests.py \
  --seed "$kit/runs/20261025-transfer-training-literal" \
  --family mtp --row-exact --both-native-baselines \
  --cases related-config,related-code \
  --output "$kit/runs/independent-mtp-related"
```

This declares 24 requests in two balanced pairs: both native families and the selected feature, with first exposure and a timed repeat for each case. Use a separate invocation with `--family dflash` for DFlash feature performance; another with `--cases conflicting-config,unrelated-prose` checks changed values and misses. Do not mix binary hashes, request contracts, numerical modes or seed identities across comparisons.

All arms use explicit row-exact arithmetic, not unchanged stock MTP arithmetic. Exact text, published token IDs, usage, finish state and semantic controls must match. Training requests are distinct from evaluation; accepted shared-output spans, not merely a nonempty cache, establish the reuse mechanism. A feature win against MTP alone is not a fastest-decoder win when native DFlash is faster.

Default isolation is `contended`. Only use `--isolation declared` when the operator actually knows other GPU inference has stopped. Free memory is not isolation evidence. Confirmation requires complete unchanged power/settings, thermal limits, normal pressure and swap counters/usage. Two pairs remain diagnostic; four qualified pairs are needed for transfer confirmation. Cold and timed-repeat gates remain separate. Schema-three aggregation refuses historical schema-two evidence rather than relabeling it.

For a correctness-only reference window, use `--row-exact --serial-oracle --cases related-config` (or `related-code`) instead of `--both-native-baselines`. It executes eight requests spanning serial, both native families and one feature. `--feature-selection cut|fixed:8|fixed:16` is restricted to that oracle; forced-control timings are not adaptive performance evidence.

## Confirmation without a long GPU sweep

Do not put four full-workload pairs into one five-minute invocation. Declare a new cohort on CPU:

```sh
python3 tests/lookup_confirmation_window.py \
  --declare "$kit/runs/mtp-confirmation" \
  --seed "$kit/runs/20261025-transfer-training-literal" --family mtp
```

Default isolation remains `contended`. Add `--isolation declared` **at declaration only** if exclusive inference is actually known. Each slot keeps all four cases in their declared order within each server boot, both native references, and first exposure plus repeat: 24 requests. This preserves cross-request learner state; splitting into isolated single-case runs would test a different workload.

Launch just one slot:

```sh
python3 tests/lookup_confirmation_window.py \
  --cohort "$kit/runs/mtp-confirmation" --pair 0
```

Later, explicitly invoke slots `1`, `2`, and `3` individually; do not use an automatic loop. Their forward/reverse arm orders were frozen before any results. An existing slot is refused, including a failed or interrupted one; do not replace a failed slot with a more favorable run. A changed binary, helper, tokenizer, seed or declaration rejects the cohort. To evaluate a revised implementation, declare a new cohort and retain the old result.

CPU-only aggregation reads exactly the four fixed slots:

```sh
python3 tests/lookup_confirmation_window.py \
  --aggregate "$kit/runs/mtp-confirmation" \
  --output "$kit/runs/mtp-confirmation-summary"
```

Missing slots fail closed. Complete execution and measurement qualification are separate: a complete contended cohort is still not performance evidence. Every cold/repeat, related/miss, source-activation and exactness gate remains; machine stability must also hold across slots. A family win does not imply beating native DFlash. Artifact hashes detect accidental changes, not deliberate receipt forgery. Aggregation recomputes summaries without modifying original evidence.

Use a separately declared `--family dflash` cohort for DFlash feature confirmation. This protocol fixes the held-out confirmation scheduling problem; it does **not** shorten or certify the separate canonical benchmark.

## Same-state first-offer diagnostic

Run these as **separate invocations**, inspecting each result:

```sh
python3 tests/lookup_counterfactual_window.py run \
  --action serial --binary zig-out/bin/mlx-serve \
  --output "$kit/runs/independent-cf-serial"
```

```sh
python3 tests/lookup_counterfactual_window.py run \
  --action assistant8 --binary zig-out/bin/mlx-serve \
  --oracle "$kit/runs/independent-cf-serial" \
  --output "$kit/runs/independent-cf-assistant8"
```

The other actions are `assistant16`, `lookup8` and `lookup16`. Each action gets one excluded warmup and one measured identical request. The same binary/model contents, full baseline learner, prompt/control/state fingerprints and actual dispatch geometry are required. Width contrasts also require identical full offers and prepared state; a later convenient offer cannot substitute for the declared first offer.

```sh
python3 tests/lookup_counterfactual_window.py compare \
  "$kit/runs/independent-cf-assistant8" "$kit/runs/independent-cf-assistant16" \
  --oracle "$kit/runs/independent-cf-serial"
```

These are readback-perturbed **verification/commit** diagnostics. They do not measure complete round time or certify throughput. Strict authentication, disabled prompt caches and single-slot admission are observed at runtime. A third request must return the exact HTTP 409 diagnostic-limit error; a server crash or timeout does not count as rejection proof.

## Canonical predictable/novel workload

Declare a new cohort **after rebuilding**:

```sh
python3 tests/lookup_canonical_window.py --declare \
  --config "$kit/independent-canonical.json" \
  --seed "$kit/runs/20261025-transfer-training-literal"
```

Execute one explicitly selected arm:

```sh
python3 tests/lookup_canonical_window.py \
  --config "$kit/independent-canonical.json" --mode mtp-legacy --pair 0 \
  --output "$kit/runs/independent-canonical-mtp-0"
```

If complete, its separate replay uses `--source "$kit/runs/independent-canonical-mtp-0"` and a new output directory. Other declared arms are `dflash-legacy`, `mtp-persistent`, `dflash-persistent`; the second pair reverses order. The driver never schedules them automatically. CPU aggregation requires all declared measurement and replay slots.

The pinned protocol retains three runs, literal rung `512`, fixed prefill, prefix, batching and drift subscenarios. Context is 16,384 to fit the client's fixed long prefill. It does not silently reduce output lengths or omit expensive scenarios to meet the deadline. If the complete protocol cannot fit, report that limitation; a shortened protocol is not the standard benchmark.

Defined sequential metric requests require exact token replay. Setup/batching diagnostics remain outside the sequential verifier's correctness contract. Predictable-copy, novel-prose and ordinary decode rates are compared separately against **both** native families. Cross-window aggregation is descriptive, not jointly balanced promotion. Missing hits cannot be credited as a lookup benefit, even if a metric happens to improve.

## Storage, diagnostics and privacy

- Frozen seed documents contain only completed training prompt/output IDs. Evaluation never appends answers. Hashes, tokenizer identity and actual store-open receipts are checked; read-only leases do not chmod or rewrite files.
- `MLX_SERVE_LOOKUP_RECEIPTS=1` exposes publication and round evidence; `[lookup-setup]` identifies index-cache reuse, and `[lookup-native-cost]` counts clean MTP calibration opportunities/gaps.
- `MLX_SERVE_LOOKUP_PRICE_TRACE=1` is timing-perturbing and reports only evaluated widths. It does not measure unchosen actions.
- Price files require schema two and `floor-power-of-two-2048` context labeling. Historical class tables must not be relabeled as new priors.
- Raw text, token IDs, environment captures and temporary diagnostic API keys are private. Keep 0700/0600 output permissions; publish only redacted reports. Token IDs are reversible, not anonymized.
- The drivers ignore proxies and set offline variables. For enforced nonlocal network denial on macOS, prefix an individual command with `sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'`. Loopback must remain allowed.

## Local model limits

The Qwen3.8-27B target and explicit incoai DFlash2 sidecar are the exercised model family for these controls. Flash-Next assets are staged, but `qwen4_exp` is excluded from the merged row-exact DFlash tree path, and the larger model's live memory fit is not established by a successful 27B run. Do not weaken architecture or memory guards to produce a benchmark.
