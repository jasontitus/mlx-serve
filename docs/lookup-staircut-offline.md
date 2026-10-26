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
  --config "$kit/independent-canonical.json" --quartets 4 --isolation declared \
  --seed "$kit/runs/20261025-transfer-training-literal"
```

Execute one physical slot at a time, in the config's frozen `schedule` order:

```sh
python3 tests/lookup_canonical_window.py \
  --config "$kit/independent-canonical.json" --slot 0 \
  --output "$kit/runs/independent-canonical-slot-0"
```

Each family has adjacent **A B B A / B A A B**, where A is its native `legacy` arm and B its adaptive `persistent` arm. Every letter is a fresh physical boot running the entire canonical workload, not a repeated request in one boot. Four quartets per family mean **32 measurement windows and 32 separate exact replays**. Quartets 0–3 use ABBA, BAAB, BAAB, ABBA; family precedence alternates MTP-first, DFlash-first, MTP-first, DFlash-first, crossing orientation with precedence. Slot IDs 0–31 are unique; repeated logical arms cannot overwrite each other. All eight boots in a quartet share the same client request/cache tag. The runner launches only the selected slot; it does not sweep automatically.

After all measurement slots finish in schedule order, replay each complete slot separately (replays do not belong between the adjacent measurement boots):

```sh
python3 tests/lookup_canonical_window.py \
  --config "$kit/independent-canonical.json" \
  --source "$kit/runs/independent-canonical-slot-0" \
  --output "$kit/runs/independent-canonical-replay-0"
```

Repeat explicitly for every frozen slot, then join all measurement and replay directories on CPU:

```sh
python3 tests/lookup_canonical_window.py \
  --aggregate "$kit"/runs/independent-canonical-slot-* \
              "$kit"/runs/independent-canonical-replay-* \
  --output "$kit/runs/independent-canonical-aggregate"
```

Aggregation verifies real nonoverlapping boot timestamps rather than input path ordering, and requires nonoverlapping exact replays after all measurement slots. Missing, duplicate, reordered or mutated slots and replay/source identity mismatches cannot qualify. Replay plans bind the source plan, probe and client-report bytes; changed sources fail rather than being silently reused. `--quartets 2` is diagnostic-only; `--isolation undeclared` cannot qualify claims. Protocol version two and helper hashes reject historical `--pairs`/`--mode` plans/configs; never rewrite or upgrade historical evidence.

The pinned protocol retains the full **34-request benchmark body plus every setup inference request**, three runs, literal rung `512`, fixed prefill, prefix, batching and drift subscenarios. The client source defines the body as decode length-probe/warmup/measured (5), prefill (4), predictable (4), novel (4), prefix cold/warm (2), batch single/four-stream (5), context calibration (1), context code/count pairs (6), and final drift (3). Before that body, `detectReasoning` makes one request or two with the opt-in tool probe; `resolveReasoning` makes one request or two with vendor thinking-disable. Thus the supported chat path has 36–38 total captures, not an unconditional 34 or 37. All actual captures remain in exact-replay accounting; auxiliary requests are not discarded or relabeled as metric-certified.

The frozen coverage contract checks the supported setup request shapes and response-dependent branches separately from every benchmark stage, request limit, tag, and actual metric assignment. Prefix cold/warm intentionally have identical requests/IDs, and context code/count pairs share tags; neither IDs nor tags must be globally unique. The worker saves `workload-coverage.json` and rejects incomplete structure before writing its completion marker; measurement summarization independently rechecks the saved probe, so a structurally invalid first slot cannot appear complete until late aggregation. Changed coverage/helper identity requires a new cohort; preserve prior configs and artifacts without upgrading them.

Context is 16,384 to fit the client's fixed long prefill. The driver does not reduce output lengths or omit expensive scenarios. `--unbounded` removes the worker/process-group deadline and raises the full-probe hang safeguard to 3,600 seconds; bounded mode remains available explicitly by omitting that declaration flag.

Defined sequential metric requests require exact token replay across both replicas of the native and adaptive arm **within each family quartet**. Setup/batching diagnostics remain outside the sequential verifier's correctness contract. Each metric contributes **one effect per complete family quartet**, and uncertainty resamples those quartet effects, never the physical boots, 34 requests or three within-client repeats. Latency uses arithmetic means of the two boot summaries (primary TTFT first averages its three client samples; context/prefix use the client's reported boot summaries). Throughput uses a **geometric estimate of the two reported boot rates**, not summed rates or a claim of pooled tokens divided by elapsed duration. The paired effect is adaptive/native for rates and native/adaptive for latency.

Four complete isolated balanced quartets per family, exact replay, stable machine/load evidence and a lower quartet-bootstrap confidence bound above one can qualify **pointwise family-native/adaptive metric** claims. This adjacent schedule does **not** justify a drift-matched cross-family fastest-native comparison, so that comparison is explicitly unsupported, not relabeled as a family result. Physical observations, quartet membership and full-workload regressions remain visible. Four quartets still give limited uncertainty resolution and do not remove nonlinear drift, carryover or between-quartet dependence. No automatic promotion, overall-workload win or blanket correctness claim follows. Missing hits cannot be credited as a lookup benefit. Executable, resolved non-system libraries, Metal assets and loading configuration remain frozen and checked before/after runs and at aggregation.

## Storage, diagnostics and privacy

- Frozen seed documents contain only completed training prompt/output IDs. Evaluation never appends answers. Hashes, tokenizer identity and actual store-open receipts are checked; read-only leases do not chmod or rewrite files.
- `MLX_SERVE_LOOKUP_RECEIPTS=1` exposes publication and round evidence; `[lookup-setup]` identifies index-cache reuse, and `[lookup-native-cost]` counts clean MTP calibration opportunities/gaps.
- `MLX_SERVE_LOOKUP_PRICE_TRACE=1` is timing-perturbing and reports only evaluated widths. `[lookup-route-price]` includes losing ranked-prefix lookup alternatives and null costs for unpriced variants. These are predictions from historical prices, not new measurements of unchosen actions.
- Price files require schema three and `floor-power-of-two-2048` context labeling, with separate `generic`/`linear_gdn` width tables. Schema-two pooled prices must not be relabeled.
- Raw text, token IDs, environment captures and temporary diagnostic API keys are private. Keep POSIX output permissions at 0700/0600. Windows stores require an already protected parent with a restrictive inheritable ACL; existing ACLs are not tightened or validated. Publish only redacted reports. Token IDs are reversible, not anonymized.
- New launch-environment and configuration evidence uses an explicit reproducibility-control allowlist, excluding arbitrary inherited credentials. Older private artifacts may contain inherited provider credentials and remain unchanged for provenance. Generated diagnostic API keys can still appear in command evidence; the allowlist does not make raw artifacts safe to publish.
- The drivers ignore proxies and set offline variables. For enforced nonlocal network denial on macOS, prefix an individual command with `sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'`. Loopback must remain allowed.

## Incremental resource cost and upstream attribution

The comparison is **feature-enabled minus the same native MTP/DFlash2 setup**, not total model size. Existing target/MTP/DFlash2 weights cancel out. Report learned-table optimizations separately: model-required embedding weights are not new lookup-drafter weights.

- Owned boots record `memory-loaded.json` and `memory-final.json`; diagnostic request rows also contain memory endpoints. On macOS, `proc_pid_rusage` reports RSS, physical footprint and the kernel's process-lifetime maximum physical footprint. The maximum is **not a per-request peak** and may be dominated by loading. Do not add RSS and footprint together or interpret either as lookup-only allocation.
- `allocator-loaded.json` and `allocator-final.json` query the existing owned, authenticated `/props` endpoint outside HTTP measurements. Active, reclaimable-cache, lifetime-peak, weight and KV byte counters remain separate from process footprint. Missing or malformed fields are incomplete, never zero; check resource completeness separately from latency qualification. No peak reset or cache clear is performed.
- Memory queries occur outside timed HTTP intervals, but still add between-request work. Compare identical instrumentation and workload order. Report loaded, post-workload and lifetime-peak deltas separately; retained allocator/KV state can explain differences larger than the lookup index itself.
- `storage-before.json` / `storage-after.json` count unique regular-file inodes in runtime lookup assets, separating logical content from filesystem-allocated bytes. Symlinks, missing roots and read errors make accounting incomplete. Allocated blocks are not exclusive APFS extent usage. Model weights, benchmark logs and build artifacts are excluded.
- Frozen history is write-disabled. Its size does not measure production append/trim latency, transient replacement space or usefulness at larger capacities. The persistent payload is bounded to 262,144 token/separator entries: **1 MiB + 40 bytes**. Its token/position arrays require up to **2 MiB** per retained index, excluding allocator metadata and simultaneous old/new index construction. The optional corpus permits 16,777,216 entries, up to **128 MiB** for those two arrays; it is not required by the persistent-only comparison.
- The local index allocates a 65,536-slot ring per armed request. On the measured v15 arm64 build, its slots occupy **1 MiB**, plus **36,440 bytes** of inline drafter state; the inline policy was **8,608 bytes**. The v19 residual-calibration experiment added 32 bytes of prediction sums. The v21 raw-neural ablation removes those sums and the 64-byte neural hit/sample population: 96 inline bytes less than v20, before padding. Native-yield tracking adds 24 bytes of request-state arrays. These figures exclude dynamic price cells, other request/tree scratch and GPU verification buffers. They are not a claim that process memory rises by only those amounts.
- Order-specific request trust in v22 occupies **8,320 inline bytes**, versus 128 bytes previously: **8,192 additional bytes per request-trust instance**. It adds no heap/GPU allocation or persistent storage. Initialization/reset and diagnostic hashing touch more state; sparse order-specific evidence may delay profitable reuse. This is a separate request-state cost, not part of the policy-size figures above.
- The v25 candidate restores pooled request support, removing that extra 8 KiB while retaining exact-order model calibration. Full-only MTP preparation diagnostics use a 96-byte optional inline flight field on the exercised arm64 build. Off/publication-only modes skip its clock reads and logging but still carry the field layout; no heap/GPU allocation or disk payload is added by the diagnostic.
- Attribute speed with matched component contrasts: fixed/max versus cut with identical candidate sources; neural-only versus neural-plus-lookup under the same width policy; adaptive admission versus the same offered candidates; and backoff on/off on misses. Report interactions rather than adding percentages. A pruning counter proves execution, not a speed contribution.

Upstream scope must separate production integration, permanent correctness tests, reproducibility tools and experimental diagnostics. The working branch is not presumed minimally invasive. A PR series should retain each demonstrated capability while avoiding a dependency on unrelated experiments or unsupported performance claims.

## Local model limits

The Qwen3.8-27B target and explicit incoai DFlash2 sidecar are the exercised model family for lookup controls. Flash-Next's `qwen4_exp` remains excluded from merged row-exact DFlash trees. A historical guarded v10 server launch at context 512 was refused before loading: estimated weights 48.04 GB, required free memory about 55.0 GB including headroom, available 42.14 GB. This was a memory-state refusal, not a permanent model limit.

The v22 executable subsequently completed a **standalone CLI** Flash-Next smoke at context 512, with vision, MTP drafting, external drafters and PLD disabled: eight generated tokens, **27.426 tok/s**, **45.905 GB reported peak memory**. The log confirms CPU learned-table gather and the normal 17.9-GB table warmer. No memory-bypass option was supplied. This establishes a short full-model execution, not server preflight admission, sustained throughput, a cache-off/on speedup, or native-MTP correctness. The eight-token response was capped, not a complete task-quality evaluation. Keep the distinct launch paths and memory conditions explicit.

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

Standalone `suffix-backoff` now also requires an adjacent primary quartet: four blocks produce **24 boots / 192 requests**, with 16 primary off/on boots and eight serial-reference boots outside those quartets. Each occurrence has independent process/history/controller state. The defining claim is backoff-on versus backoff-off at identical cut/history settings; serial comparisons are descriptive, not a substitute when backoff regresses. Older three-arm declarations are rejected rather than silently upgraded. The completed v22 cohort shows why: miss-heavy prose improves while useful-reuse cases regress, leaving full-workload performance flat.

For an isolated dynamic-width claim, use an adjacent primary quartet rather than the descriptive multiarm width schedule:

```sh
python3 tests/lookup_component_bench.py --declare \
  --config "$kit/dflash-combined-fixed8-cut.json" \
  --seed "$kit/runs/20261025-transfer-training-literal" \
  --family dflash --component combined-fixed-cut \
  --fixed-width 8 --blocks 4 --isolation declared
python3 tests/lookup_component_bench.py \
  --config "$kit/dflash-combined-fixed8-cut.json" \
  --output "$kit/runs/dflash-combined-fixed8-cut"
```

`combined-fixed-cut`, `combined-max-cut`, `neural-fixed-cut` and `neural-max-cut` each isolate one contrast. Four blocks contain 28 boots and 224 requests: 16 boots/128 requests in the primary adjacent quartets, with native references and serial correctness outside them. Both primary arms pay for the same candidate-source preparation; cut additionally pays its online width-learning trials. All first/repeat requests count. This measures the whole width policy, including learning—not a separately isolated kernel or pruning operation. Native references remain descriptive and cannot inherit a primary performance pass.

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

### Keep timing receipts separate from full diagnostics

`MLX_SERVE_LOOKUP_RECEIPTS=1` includes per-round diagnostics; native MTP diagnostics can introduce an additional GPU-settlement barrier. `MLX_SERVE_LOOKUP_RECEIPTS=commit` retains exact published token IDs and store identity without those diagnostic rounds. Real lookup and production price-measurement barriers remain. `MLX_SERVE_LOOKUP_TRACE=0` independently disables lookup traces; receipt mode does not imply trace mode.

Declare both controls in the frozen experiment. Missing round receipts in publication-only mode mean **mechanism observations unavailable**, not zero lookup activity. Keep a separate full-diagnostic correctness/mechanism run. Existing diagnostic protocols that require round receipts must not be silently switched to publication-only mode or have those gates waived. The v23 MTP smoke exercised exact serial agreement, native engagement and store evidence with traces disabled; the isolated full-versus-publication-only coding contrast retains identical tracing in both arms.

### Learned-table cache-only control

`MLX_SERVE_PLE_ROW_CACHE=0` disables only decoded-row payload reuse; unset or `1` retains it. It does not disable the worker pool, calibration, warming or wide-gather dispatch. Do not use `QWEN4_PLE_PREFETCH=0` as this control: that also changes worker execution.

For a 160-dimensional table the on/off payload difference is **655,360 bytes (640 KiB)**. The embedded 8,480-byte metadata remains in both arms; the full cache feature is 663,840 bytes (648.28125 KiB), with no additional disk data. Startup receipts show actual payload and worker count. Close-time counters cover the entire table lifetime, including warmup/calibration; positive hits do not establish measured-request hits. Process footprint and MLX allocator readings have different scopes and must not be added.

The v23 full-model server attempt failed ordinary memory admission before any request: 55.0 GB needed, 46.89 GB available. Retain that failed experiment. Do not disable the guard, resume its partial schedule, or substitute the earlier standalone CLI smoke for its missing repeated-request comparison.

### Production MTP chain-price acquisition

`tests/lookup_chain_profile.py` declares a separate training-only production population, not a same-state staircase. It uses the frozen coding contract and completed training seed, a complete serial oracle, and two opposite-order sweeps of actual widths 2–15. Four training prompts run twice per boot: 232 complete requests including the oracle. Width 16 is unsupported by the current 14-draft MTP lookup cap and is never manufactured by clipping.

```sh
python3 tests/lookup_chain_profile.py declare \
  --binary "$PWD/zig-out/bin/mlx-serve-v25" \
  --contract "$kit/coding-transfer-v2.json" \
  --seed "$kit/runs/independent-v18-coding-training" \
  --output "$kit/runs/mtp-chain-training"
python3 tests/lookup_chain_profile.py acquire --run "$kit/runs/mtp-chain-training"
python3 tests/lookup_chain_profile.py export \
  --run "$kit/runs/mtp-chain-training" --output "$kit/mtp-chain-prices.json"
```

Declaration is CPU-only. Acquisition performs real inference; do not run it beside builds, component CPU benchmarks or other inference. A failed or hardware-unqualified cohort cannot resume or export. Each exact context/kind/variant/width cell excludes its first actual valid execution per process, requires post-cold observations from both sweeps, and exports their median settled verification/commit duration. Missing offers and clipped/unpriced executions remain explicit exclusions. No cell is interpolated.

The `.source.json` sidecar binds training membership, raw artifacts, exact runtime/model inventory, numerical controls, seed and contract. Inventory/config hashes are not full weight-payload hashes. The profile consumer freezes validated bytes to absolute paths before evaluation; it rejects changed runtime, source, seed or evaluation-derived provenance. The production JSON loader alone does not enforce those bindings. Keep source artifacts and sidecars private.

Compare profile off/on with otherwise identical adaptive controls before comparing the profiled policy with native. Report acquisition time, price coverage and storage separately. Real exported intermediate-width selection must be observed before claiming that the profile improved decisions; a configured filename is not sufficient evidence.
