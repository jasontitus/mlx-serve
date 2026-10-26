# Offline lookup / StairCut validation

Run from the **mlx-serve repository root**. The prepared machine has targets, drafters, fixtures, Node and pinned llmprobe 0.6.17 in `~/.mlx-serve/bench/lookup-staircut`. Offline operation does not require package installation or `npx` downloads. A new machine must prepare its own manifest/assets before disconnecting.

The independently authored implementation and authoring limitations are described in [the design note](lookup-staircut-port.md). The [results note](lookup-staircut-results.md) separates measured revisions by binary identity; an earlier measurement does not certify the current working tree.

## Build and CPU/in-process checks

```sh
.zig-toolchain/zig build -Doptimize=ReleaseFast
.zig-toolchain/zig test src/lookup_adapter.zig -lc
python3 -m unittest discover -s tests -p 'test_lookup_*.py'
.zig-toolchain/zig build test -Doptimize=ReleaseFast -Dtest-filter=lookup
```

Compilation is not a GPU benchmark. The integrated tests include short real hybrid verification/commit scenarios. Keep model benchmarks separate from builds so an interrupted build is not mistaken for an inference result. Never reuse an old installed binary after a source-only fix.

## Explicit short windows or complete experiments

The repository owns the authoritative drivers:

- `tests/lookup_similar_requests.py`: unseen related requests and conflict/miss controls; both native baselines; optional serial correctness oracle.
- `tests/lookup_confirmation_window.py`: predeclared four-pair confirmation split into independently launched full-workload windows.
- `tests/lookup_counterfactual_window.py`: first-offer same-state width/action diagnostics, with full learner reset and strict-authenticated owned server.
- `tests/lookup_canonical_window.py`: one complete pinned canonical arm, or its separate exact replay.
- `tests/lookup_component_bench.py`: frozen independent mechanism contrasts over complete four-case workloads and at least four balanced blocks, without a whole-run deadline.
- `tests/lookup_route_diagnostic.py`: existing exact-reuse/route mechanism controls.
- `tests/lookup_transfer_contract.json`: predeclared training/evaluation request contract.

The short-window drivers retain their **290-second process-group bound** and never automatically start another window. The component driver has no whole-run deadline; canonical declarations can opt into `--unbounded`, retaining per-request/startup and full-probe hang safeguards. These modes preserve complete workloads rather than trimming them to five minutes. A truncated schedule fails; completed subsets are not silently promoted. Use a new output directory for every run. Execution modes support `--plan-only`; declarations freeze the protocol before inference.

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
  --preparation merged16 --action serial --binary zig-out/bin/mlx-serve \
  --output "$kit/runs/independent-cf-serial"
```

```sh
python3 tests/lookup_counterfactual_window.py run \
  --preparation merged16 --action assistant8 --binary zig-out/bin/mlx-serve \
  --oracle "$kit/runs/independent-cf-serial" \
  --output "$kit/runs/independent-cf-assistant8"
```

Actions cover `assistant1` through `assistant16` and `lookup1` through `lookup16`. Each action gets one excluded warmup and one measured identical request. The same binary/model contents, full baseline learner, prompt/control/state fingerprints and actual dispatch geometry are required. Width contrasts also require identical full offers and prepared state; a later convenient offer cannot substitute for the declared first offer. These DFlash actions are priced as the executed **tree verifier**, including a linear-GDN dispatch; they are not ordinary MTP/PLD chain prices.

`--preparation` is required on both the oracle and action declarations. `merged16` retains the diagnostic's 16-position assistant horizon and suffix merging. For neural-only production-horizon calibration, use `--preparation production-neural` with `serial` or an `assistantN` action: it preserves the configured production horizon (normally seven positions here), offers up to 15 neural nodes, and disables suffix preparation. Lookup actions are rejected in this mode. Requested widths must actually execute; insufficient candidate offers fail rather than being silently shortened. Preparation groups need separate oracles and exports and cannot be mixed. Receipts retain actual horizon, context length, selected ancestry, committed path and compaction geometry.

For the production horizon with suffix merging, use `--preparation production-combined` on the serial oracle and all actions. Both `assistantN` and `lookupN` are supported. Actual selected geometry determines the verifier variant; `lookup16` does not guarantee a linear tree. Request-local trust is reset, so this measures cold first-offer verification, not learned-state bypass economics.

```sh
python3 tests/lookup_counterfactual_window.py compare \
  "$kit/runs/independent-cf-assistant8" "$kit/runs/independent-cf-assistant16" \
  --oracle "$kit/runs/independent-cf-serial"
```

These are readback-perturbed **verification/commit** diagnostics. They do not measure complete round time or certify throughput. Strict authentication, disabled prompt caches and single-slot admission are observed at runtime. A third request must return the exact HTTP 409 diagnostic-limit error; a server crash or timeout does not count as rejection proof.

For reusable price calibration, declare the serial oracle and every action with `--case train-config-a` (or another contract training case). Export validated windows on CPU:

```sh
python3 tests/lookup_counterfactual_window.py export-prices \
  "$kit/runs/train-assistant8" "$kit/runs/train-assistant16" \
  --oracle "$kit/runs/train-serial" --output "$kit/training-prices.json"
```

The exporter creates the schema-three table and adjacent `training-prices.json.source.json`, binding the exact table bytes with `prices_sha256`. It checks the frozen training contract, current-run serial oracle, runtime identity, same-state receipts and actual kernel variant. Evaluation cases, historical declarations without the required contract/preparation, duplicate price cells and existing outputs are rejected. Only measured verification/commit time is exported; warmup, diagnostic readback duration and full-round timing are excluded. Host work and readback-induced cache/synchronization effects are not removed by that exclusion. A one-row tree price is not relabeled as a plain-decoding price.

## Canonical predictable/novel workload

Declare a new cohort **after rebuilding**:

```sh
python3 tests/lookup_canonical_window.py --declare --unbounded \
  --config "$kit/independent-canonical.json" --pairs 4 --isolation declared \
  --seed "$kit/runs/20261025-transfer-training-literal"
```

Execute one explicitly selected arm:

```sh
python3 tests/lookup_canonical_window.py \
  --config "$kit/independent-canonical.json" --mode mtp-legacy --pair 0 \
  --output "$kit/runs/independent-canonical-mtp-0"
```

If complete, its separate replay uses `--source "$kit/runs/independent-canonical-mtp-0"` and a new output directory. Other declared arms are `dflash-legacy`, `mtp-persistent`, `dflash-persistent`. Follow the frozen forward/reverse order for all four pairs; aggregation verifies actual nonoverlapping measurement-boot timestamps. The driver never schedules the other arms automatically. CPU aggregation requires every declared measurement and replay slot. `--pairs 2` is diagnostic-only; `--isolation undeclared` cannot qualify performance claims.

The pinned protocol retains all 34 requests, three runs, literal rung `512`, fixed prefill, prefix, batching and drift subscenarios. Context is 16,384 to fit the client's fixed long prefill. It does not reduce output lengths or omit expensive scenarios. `--unbounded` removes the worker/process-group deadline and raises the full-probe hang safeguard to 3,600 seconds; bounded mode remains available explicitly by omitting that declaration flag.

Defined sequential metric requests require exact token replay. Setup/batching diagnostics remain outside the sequential verifier's correctness contract. Predictable-copy, novel-prose and ordinary decode rates are compared separately against **both** native families. Four complete isolated balanced pairs, exact replay, stable machine/load evidence and a lower paired-block confidence bound above one can qualify **pointwise metric** claims. Full-workload regressions remain separately visible; no automatic promotion or blanket correctness claim follows. Missing hits cannot be credited as a lookup benefit. Executable, resolved non-system libraries, Metal assets and loading configuration are frozen and checked before/after runs and at aggregation.

## Storage, diagnostics and privacy

- Frozen seed documents contain only completed training prompt/output IDs. Evaluation never appends answers. Hashes, tokenizer identity and actual store-open receipts are checked; read-only leases do not chmod or rewrite files.
- `MLX_SERVE_LOOKUP_RECEIPTS=1` exposes publication and round evidence; `[lookup-setup]` identifies index-cache reuse, and `[lookup-native-cost]` counts clean MTP calibration opportunities/gaps.
- `MLX_SERVE_LOOKUP_PRICE_TRACE=1` is timing-perturbing and reports only evaluated widths. `[lookup-route-price]` includes losing ranked-prefix lookup alternatives and null costs for unpriced variants. These are predictions from historical prices, not new measurements of unchosen actions.
- Price files require schema three and `floor-power-of-two-2048` context labeling, with separate `generic`/`linear_gdn` width tables. Schema-two pooled prices must not be relabeled.
- Raw text, token IDs, environment captures and temporary diagnostic API keys are private. Keep POSIX output permissions at 0700/0600. Windows stores require an already protected parent with a restrictive inheritable ACL; existing ACLs are not tightened or validated. Publish only redacted reports. Token IDs are reversible, not anonymized.
- New launch-environment and configuration evidence uses an explicit reproducibility-control allowlist, excluding arbitrary inherited credentials. Older private artifacts may contain inherited provider credentials and remain unchanged for provenance. Generated diagnostic API keys can still appear in command evidence; the allowlist does not make raw artifacts safe to publish.
- The drivers ignore proxies and set offline variables. For enforced nonlocal network denial on macOS, prefix an individual command with `sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'`. Loopback must remain allowed.

## Local model limits

The Qwen3.8-27B target and explicit incoai DFlash2 sidecar are the exercised model family for these controls. Flash-Next assets are staged, but `qwen4_exp` is excluded from the merged row-exact DFlash tree path. A guarded v10 stock-serial launch with context 512 was refused before loading: estimated weights 48.04 GB, required free memory about 55.0 GB including headroom, available 42.14 GB. This is an observed current-memory refusal, not proof that the pack can never run on a 64 GB machine. No architecture or memory guard was bypassed; full-model inference and performance remain unverified.

## Adjacent native/adaptive confirmation

Use `production-routing` for the MTP or DFlash native/adaptive comparison, independently of forced-width component ablations:

```sh
python3 tests/lookup_component_bench.py --declare \
  --config "$kit/mtp-abba.json" \
  --seed "$kit/runs/20261025-transfer-training-literal" \
  --family mtp --component production-routing --blocks 6 --isolation declared
python3 tests/lookup_component_bench.py \
  --config "$kit/mtp-abba.json" --output "$kit/runs/mtp-abba"
```

For DFlash, declare a separate configuration with `--family dflash` and new paths. Run the families sequentially. Six quartets mean **24 fresh server boots and 192 requests per family**, retaining all four cases and first/repeat exposure. There is no automatic runtime rebuild, model sweep or shortened-workload fallback.

Each quartet runs **ABBA or BAAB**, alternating orientations. A is native, B is combined adaptive lookup; each occurrence has a distinct physical ID, process, HOME, read-only history clone and fresh controller. Both replicas' durations are added before forming one A/B ratio. The estimator is the geometric mean of complete-quartet ratios; six quartets supply six bootstrap observations, not 24 boots or 192 independent samples. `physical_boot_seconds` retains each replica's summed workload duration.

The five-percent case/exposure guard applies to **paired totals within each quartet**, not each individual physical boot. This new unit is not the old single-boot worst-observation guarantee. Exactness, full physical schedule, source activation, telemetry and machine-state gates still apply. Partial quartets cannot qualify; interrupted permutation evidence cannot be reused as ABBA slots.

ABBA balances positions and helps with simple smooth drift. It does not establish A/A equivalence or eliminate paging, nonlinear thermal/frequency changes, unequal elapsed-time spacing, predecessor effects or correlation across quartets. Intervals remain descriptive under the block-independence assumption. Preserve physical variability and failed gates rather than selecting the favorable orientation.

## Full-workload component ablations

```sh
python3 tests/lookup_component_bench.py --declare \
  --config "$kit/dflash-neural-width.json" \
  --seed "$kit/runs/20261025-transfer-training-literal" \
  --family dflash --component neural-width --isolation declared
python3 tests/lookup_component_bench.py \
  --config "$kit/dflash-neural-width.json" \
  --output "$kit/runs/dflash-neural-width"
```

This is an intentionally complete experiment, not a short battery window. All four held-out cases have measured first and repeat exposures in every boot. Available contrasts are DFlash `neural-width`, `combined-width`, `lookup-increment`; MTP `lookup`; and standalone PLD `suffix-backoff`. The last compares identical frozen-history/cut controls with backoff off/on plus a serial reference. Neural-only MTP StairCut is not implemented. Protocol version two runs all six arm permutations for MTP lookup (144 requests). Its adaptive/native contrast is the production gate; forced-cut controls remain visible but do not decide that gate. Other defining component contrasts remain separate. Native-family wins are not automatically fastest-decoder wins.

Declare `--prices FILE --prices-source SOURCE.json` for shared offline priors; all nonnative arms receive the same content-frozen schema-three table. The consumer requires exact table-byte binding, the same training/evaluation contract, and matching binary/runtime/model identities. It hashes target and assistant content once at declaration and compares that with calibration; subsequent execution uses filename/size/mtime plus JSON-content inventory checks, not continuous weight hashing. Same-size payload modification with restored timestamps is outside that ongoing guard. Calibration sources must be independent training cases. Without this option, pricing is explicitly online-only. Both protocols freeze the linked runtime, sanitize loader overrides, retain raw observations and refuse changed identities. No qualified result follows merely from a completed run.

The component protocol requires passive `macmon` collection, with the executable identity, command and cadence frozen at declaration. `--macmon` selects its path; `--telemetry-interval-ms` and `--telemetry-max-gap-ms` declare sampling and coverage limits. Raw GPU frequency, utilization, temperature, power and fan evidence are retained and joined to request intervals; actual active ratio is separate when available. Machine-state checks include boot phases and request endpoints. Missing coverage or unstable state blocks qualification. Collection gaps lie outside HTTP latency, so summed request times are not uninterrupted offered-load throughput or boot-to-answer measurements. Telemetry coverage alone does not establish thermal stabilization.

For a CPU-only learned-embedding component run:

```sh
.zig-toolchain/zig build ple-bench-build ple-bench-test -Doptimize=ReleaseFast
zig-out/bin/ple-bench \
  --model "$HOME/.mlx-serve/models/TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP" \
  --output "$kit/runs/flashnext-ple.json" --samples 256 --warm-repeats 8
```

Run this separately from inference and compilation. It uses actual staged table bytes, not synthetic table performance, and checks exact vectors. Its first-access/warm comparisons exclude full-target compute, GPU transfer and embedding projection; they cannot establish end-to-end speedup. Same-state diagnostics accept widths 1–16 and `--case train-config-a` (or another contract training case) for independent calibration; retain their diagnostic timing limitations.
