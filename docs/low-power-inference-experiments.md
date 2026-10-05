# Low-power inference: laptop experiment runbook

Use with the [review and optimization plan](low-power-inference-plan.md).
All E0–E8 rows below are **planned, not run**. This is a protocol for the target
laptop, not a ready-made energy benchmark runner or a list of measured wins.

## Hardware allocation and M5 follow-ups

| Machine | Role and power modes | Work allocation |
| --- | --- | --- |
| **M5 Max MacBook Pro — primary** | Low versus High; Automatic optional, particularly for the sustained comparison | Run E0–E4 and the M5 qualification below. Select E5/E6 from its traces. Add a larger dense and a MoE checkpoint only when actual RAM admits them with reserve. |
| Base M5 MacBook — secondary | Low versus Automatic; record actual available modes | Repeat the small-model baseline and shortlisted candidates, including their reachable NAX/fallback paths. Check for regressions and changed crossovers; do not repeat every Max sweep. |

Apple lists High Power support for recent MacBook Pro models with Pro/Max chips;
verify the settings on each machine and power source.
([Supported power modes](https://support.apple.com/en-us/101613))
Record exact chip/core count, RAM, chassis size and OS build. Use the same small
checkpoint revision and scenarios for the portability check, but judge each
candidate against that machine's own baseline in the same mode. Different memory
capacity, cooling and dispatch choices prevent pooling raw rates or energy.

After qualifying the Max, run its baseline, prefill and speculation comparisons
first, then contention/cache and the shortlisted follow-ups. Transfer the useful
candidates to the base M5 before considering general defaults.

| M5 follow-up | Concrete test and decision | Adversarial guard |
| --- | --- | --- |
| Correctness qualification, before timing | Run the new DQ oracle, reachable NAX kernel oracles and the GLM hcPre test in Low and High on the Max; repeat relevant checks in Low/Auto on base M5 | A staged metallib is not execution proof. Preserve tolerances; passing on M5 does not resolve the M1 baseline failure. |
| NAX crossover map, E1/E2/E5 | Probe actual verification widths around 4/8/16 rows and prefill widths 256–8192, then narrow around observed dispatch boundaries | Compare one reachable NAX path with its valid fallback, preserving weight layout. Per-kernel switches do not disable all MLX NAX use. Record engagement; decline synthetic wins absent from full requests. |
| Interactive bursts, E0/E2/E6 | Repeat a fixed short conversation after 5/30/120 seconds idle; measure TTFT and energy over the request plus a fixed post-request window | Warm shaders separately; preserve the same idle duration and cache state within pairs. Report complete-cycle energy and request latency separately so a long idle interval cannot hide a regression. Requires a tested burst scenario and aligned energy windows. |
| Bottleneck attribution, E5/E6 | Compare production-sized weight-streaming operations, prefill GEMMs and small dependent chains in each mode | Use working sets that do not fit entirely in cache. Diagnostic timings identify the limiting resource; llmprobe request results remain the performance gate. |
| Realistic memory headroom, E4 | Repeat a shortlisted workload with a recorded browser/IDE workload, then grow context/cache within admission limits | Background work is a separate scenario, not noise to mix into isolated A/B. Record compression/swap and repeat the background load; never bypass reserve or preflight. |
| Sleep/wake and charger transitions, E0/E2/E8 | With the model resident, resume and test a cached conversation and speculative decode against a fresh-server reference | Separate source changes from mode changes, record transition markers and settling. Check restored state and timing-controller recovery; do not count sleep time as request service time. |

ANE remains a separate optional E7 follow-up after the GPU baseline. A bounded
M5 Low Power trial may revisit its energy tradeoff despite prior throughput losses;
it requires explicit force-path engagement, sufficient memory, lossy-quality
validation and cold/warm accounting. It is not part of the initial default matrix.

## First session: establish what changes

1. Start on the M5 Max MacBook Pro and verify the exposed power modes. Record
   model, chip, CPU/GPU cores, RAM, OS build, battery health/charge and adapter.
   Read active state as well as configured AC/battery settings. Do not emulate
   Low Power Mode on the Studio with sleeps, concurrency limits or QoS changes.
2. Build once, pin the model and llmprobe versions, and predownload everything.
   No downloads, shader builds, Spotlight indexing or unrelated inference during
   timed runs. Use the same display brightness, peripherals and fan policy.
3. Start with one small dense/hybrid pack and one small attention pack. Run
   High versus Low on AC, then repeat on battery as separate strata. Automatic
   is an optional third arm. If a mode/source combination is unavailable, mark it
   unavailable; do not relabel another mode or treat unplugging as the toggle.
   No finer-grained power/frequency settings are part of this campaign.
4. Measure a warm resident model, no prefix reuse, plain decode: short prompt,
   long prefill, long-context decode. Then repeat shipping speculation and a
   warm conversation. Take separate phase traces for the cases that change most.
5. Choose the next experiment from the observed critical path. Do not start with
   the full cross-product of models, quantizations, contexts and all knobs.

## Build and freeze the inputs

Follow [building.md](building.md), including submodules, the pinned Zig, staged
MLX/MLX-C and llama dependencies. Run these from the repo in a **bash** shell once
dependencies are staged:

```bash
export PATH="$PWD/.zig-toolchain:$PATH"
zig build -Doptimize=ReleaseFast
bash tests/test_mlx_staged_nax.sh
git rev-parse HEAD
git submodule status
shasum -a 256 zig-out/bin/mlx-serve lib/mlx/lib/libmlx.dylib
```

Check the actual staged dylib path before hashing; if the stage layout changes,
use the dependency path shown by `otool -L zig-out/bin/mlx-serve`.
Record dirty diffs and hashes for locally rebuilt libraries/kernels, not just the
top-level SHA. Do not reuse the baseline binary after changing source.

Set these values for each arm; the missing-value checks deliberately stop an
unconfigured copy/paste:

```bash
: "${MODEL:?absolute path to a memory-fitting checkpoint}"
: "${LLMPROBE_VERSION:?exact installed/tested llmprobe version}"
: "${RUN_TAG:?unique tag including chip, source, mode, experiment, arm and repeat}"
RUN_DIR="$HOME/claude-tmp/bench-$RUN_TAG"
mkdir -p "$RUN_DIR"
export LLMPROBE="npx --offline llmprobe@$LLMPROBE_VERSION"
export MLX_SERVE_ROUND_COST_PERSIST=0
```

Resolve/download that exact llmprobe package before the session, run its `--help`,
and verify the report schema. `tests/bench.sh` otherwise defaults to `@latest`.
Inspect its installed request payload/options: timing controls must not silently
enable speculation, logprobs, penalties or grammar. If an intended control cannot
be expressed, add a tested llmprobe scenario before measuring it; do not invent a
probe flag. Prefix-cold and prefix-warm cases must be distinguished by reported
`cached_tokens`.

Record checkpoint revision, shard hashes, config, tokenizer/template hashes,
quant mode/bits/group sizes and sidecar revision. Inspect the relevant entry in
`~/.mlx-serve/model-settings.json`: per-model overrides can outrank launch
settings. Record the effective load log and `/props`; make a deliberate,
reversible correction to that entry if needed. Do not assume CLI spelling proves
the effective KV, MTP acceptance, drafter or int8-prefill state.

## Measurement and telemetry

Keep four kinds of run distinct:

| Run | Purpose | Exclusions |
| --- | --- | --- |
| Uninstrumented llmprobe | Primary TTFT, prefill/decode rate, context ladder and task time | No Metal capture or barrier-inserting profiler |
| Energy pass with identical scenarios | Task joules, average power, sustained behavior | No shader trace; compare collector-on/off overhead first |
| Metal / phase trace | Kernel routes, gaps, dispatches, memory and CPU submission attribution | Never quote its tok/s as the performance result |
| Correctness / quality run | Kernel/state equivalence and task acceptance | Not a throughput sample |

Log time-stamped mode, source, charge, thermal state and process identity before,
during and after a block. `pmset -g custom` records configured policies;
`pmset -g batt` records source/charge; neither replaces a Foundation active-state
sample. The branch includes a native state sampler with wall and boot-relative
monotonic timestamps. Compile it before timed runs:

```bash
bash tests/test_low_power_state.sh
swiftc -O -parse-as-library scripts/low-power-state.swift -o /tmp/mlx-low-power-state
: "${POWER_MODE:?operator-selected low, high or auto}"
/tmp/mlx-low-power-state --mode "$POWER_MODE" --samples 120 --interval-ms 1000 \
  > "$RUN_DIR/power-state.jsonl"
```

Run that last command in a second terminal so samples bracket the workload;
choose enough samples for the entire block. `--mode` declares the setting; it
does not change it. Low flag false yields `low_flag_clear_only` for High/Auto:
verify those choices in OS settings and archive configured policies separately.
`mismatch`, source changes, missing coverage or thermal changes invalidate a
steady-state block. The sampler does not record battery charge, task phase markers,
server identity or joules; collect those separately. Polling can miss brief mode
changes, so retain explicit operator transition markers and settling windows.

Inspect `powermetrics --help` and the target's available samplers/units first.
Collect only supported CPU/GPU/ANE/thermal counters with the needed permission;
keep raw output and sample timestamps. Missing/empty ANE samples mean unknown,
not zero. Use a sufficiently long repeated-work window (initially 60+ seconds)
for coarse power sampling, and compare a second sampling interval for overhead.
Short request phase joules need synchronized phase markers; until available,
report whole-block/task energy instead of assigning samples to guessed phases.

For a power series in watts, integrate over the measured task interval:
`E = sum(0.5 * (P[i] + P[i+1]) * (t[i+1] - t[i]))` in joules. Trim/interpolate
the boundary samples and retain gaps as invalid data, not zeros. Sum only
non-overlapping domains; do not add package power to its CPU/GPU components.
Record gross energy and `E - idle_power * duration` separately. Divide by actual
committed output tokens only for comparable decode workloads; task joules is the
primary metric when output lengths differ. Never divide full request energy by
decode tokens and call it decode-only energy.

For AC wall measurements, use a stable charged battery and report charging state,
adapter losses and background/display load. On battery use battery energy
telemetry or a suitable external measurement; percentage drops over short runs
are too coarse. If only SoC-domain power is usable, label that limit in every
energy claim. Do not project hours of battery life from one short kernel run.

Capture `vm_stat`, `sysctl vm.swapusage`, `/props` active/cache memory and process
physical footprint during the workload, not only after it. Retain counter deltas
for compression, swap and faults. Capacity stress is a separate experiment from
a no-swap kernel comparison. Do not disable memory preflight or the OS reserve.

## Workload ladder

| Tier | Checkpoint / workload | Coverage |
| --- | --- | --- |
| Required small hybrid | Available Qwen3.5-family 0.8–4B, fixed 4-bit pack | Launch cost, GDN, CPU pipeline, short prefill; fits ordinary laptops |
| Required small attention | Available Gemma 4 E2B/E4B pack, fixed precision | Attention/PLE/sliding behavior distinct from GDN |
| If memory permits | Qwen3.8-27B 4-bit, with its pinned MTP/DFlash sidecars | Dense weight bandwidth, large GEMM, tiled row-exact path and draft economics |
| If memory permits | Qwen3.6-35B-A3B 4-bit or another supported small MoE | Routed experts, batching, speculation with lower active weights |
| Extended only | Qwen3.8-Flash-Next, Bonsai ternary, Nemotron, MiMo or GLM | The specific PLE/QSA, qmv2, Mamba2, 192/128, or KDA kernel under test; no claim on absent models |

For each required model, start at approximately 512, 8192 and 32768 **actual
tokenized** prompt tokens where supported; include requested output in the
context/memory budget. Add 2048/4096/16384 boundary rungs for E4 and 64k only when
the checkpoint and hardware fit. Record the measured token count, not a word or
character estimate. For E1, the actual forwarded prefix/tail widths matter, not
just total prompt tokens.

Use three fixed prompt classes: novel prose/reasoning, code/structured output,
and copying/editing with a repeated prefix. Keep tool/thinking settings explicit.
Use 256 committed output tokens as the initial decode target where the model
does not end early; save EOS/stop reasons and actual lengths. Do not suppress EOS
or change requested content just to make a flattering denominator. Use separate
matched diagnostic token replays for forward attribution.

Cover these scenarios in the staged campaign:

- Resident model, prefix cold: no hot/disk prefix reuse; primed shader kernels.
- Same conversation extended: hot prefix reuse, restored SSM position, and tail
  prefill. Distinguish exact repeated prompt from a real appended turn.
- One stream decoding while another starts an 8k/32k cold prefill; then N=2/4
  steady streams. Measure per-stream latency and aggregate completion work.
- First-use loading/compile and warm steady state as separate energy budgets.
- Sustained 10–15 minute workload on both modes, allowing ordinary thermal
  evolution. Record it rather than discarding thermal effects that belong to use.

## Repeat order and controls

Use at least three counterbalanced blocks per shortlisted A/B comparison: ABBA,
BAAB, ABBA, with a saved prompt/seed order. Each cell still uses llmprobe's warmup
and median-of-3 protocol. Estimate uncertainty from paired **blocks**, not
individual tokens or correlated inner repetitions. Retest the selected result
in a second session; use a separate prompt set to confirm it.

For first E0 runs on the Max, A/B means High/Low with identical software (Auto
optional); on the base M5 it means Automatic/Low.
For optimization runs, A/B means baseline/candidate **inside the same power mode and source**;
repeat in the other mode. This prevents a mode change from being mistaken for a
software gain. Include a baseline return to detect drift. Let charge and idle
thermal state settle between cold-start blocks; use a predeclared charge band
and stop/recharge when leaving it. Do not cool only the preferred arm. Warm each
geometry before measuring; exclude JIT separately, but include controller learning
in a distinct first-request result.

Restart the server for controls cached at initialization, with persistence off
on both arms. E2's in-process mode transition is the deliberate exception. A
cached environment-variable value cannot be changed by editing another shell.
For intentionally content-varying repetitions use a fixed nonce per pair, reused
by both arms; do not compare different cache states or different prompts.

Apply the plan's gates to each scenario, not just a pooled average. Save paired
ratios and intervals for energy, TTFT and rate, plus p50/p95/max streaming gaps.
With too few repetitions to resolve the threshold, collect more independent
blocks or call the result inconclusive. No selection of only the best width,
prompt or boot without a held-out confirmation.

## Experiment cards

| ID | Arms / measurements | Proof, guard and stopping rule |
| --- | --- | --- |
| E0 | High/Low (Auto optionally) × AC/battery on the same binary; plain then shipping auto speculation; external collector off/on | Active-state samples agree across the block; label energy domain; quantify collector overhead. Resolve missing telemetry before energy claims. |
| E1 | Actual chunks 512/1024/2048/4096/8192 as allowed; `MLX_SERVE_PREFILL_DQ_GEMM=0` vs default, then the bounded DQ controls below. On 2-bit add 256/384/512 near the crossover. | `[prefill-trace]` shows actual widths/tails. `[prefill-dq]` proves the first experimental selection; use a separate kernel trace for per-layer reachability. Tiled weights require a separate lane-prefill experiment. |
| E2 | Plain; shipping auto; supported fixed MTP depths 1/2/4/6/8; DFlash blocks 2/4/5/8/16 up to config/hardware support; PLD alone for copying. Run fixed-width controls before adaptive comparisons. | `[spec-stats]`, attempted/accepted drafts, mode, width and serial fallback; exact acceptance only. Compare no-sidecar deployment cost separately from same-loaded-layout request-level drafter-off. Reject unengaged arms. |
| E2 transition | Warm auto in one mode, toggle to the other with model resident, return; compare against fresh server in each final mode | Measure first 32/128/512 committed-token windows, time to stabilize and total task energy. Test persistence off first. Propose epochs only if current adaptation loses materially. |
| E3 | N=1/2/4; default share 0 vs 0.25/0.5; safe chunk caps 512/1024/2048; interleave-off diagnostic control | Real streams and `[interleave]`/`[batched]` evidence. Require output/state correctness, useful per-stream rate and bounded gap; stop any unfair or memory-unsafe arm. |
| E4 | KV off/8, then 4; contexts around 2k/8k/16k; cold vs warm prefix; 0/1/4 RAM cache entries within budget; SSD tier only for a capacity-use case | Packed attention/fallback evidence, cached/forwarded tokens, restore time, physical memory, disk writes. Fixed request transcript and quality checks. No switching KV format inside a live cache. |
| E5 | Selected hotspot only: rows per simdgroup 1/2/4, legal SIMD-group/split counts around the shipped choice; matmul/attention crossover shapes; blocked vs pipelined GDN | Some geometry arms require future prototype code. Query per-pipeline limits; finite outputs, truth oracle, exact rows/states where promised, multi-seed tails. Stop if isolated improvement disappears in dependent full-forward and llmprobe runs. |
| E6 | Existing `MLX_SERVE_DECODE_ASYNC_LADDER` default/0/4/8; command-buffer op/MB caps around chip defaults, one dimension at a time | Only if trace shows exposed build/submit/gap cost. MLX defaults are 40/40 for base/Pro and 50/50 for Max/Ultra at the reviewed pin. Record actual buffers and traces; do not remove dependency barriers. |
| E7 | GPU-only vs `--ane-prefill`; supported model and sufficient RAM; warm shares e.g. 0.25/0.35/0.45, one compiled variant at a time | Actual MLP/GDN engagement, ready/total coverage and eval-failure counters, warm/cold energy and quality. Existing refusal stays authoritative; no forced M5 trial until a separate justified campaign. |
| E8 | Current policy vs proposed mode-specific timing/profile handling; mode/source changes during prefill, serial, spec and batch; unknown-state control | Future code. No mode-switch data corruption, false exactness claim, restart requirement or per-token notification overhead. Compare bounded relearning against existing EMA; retain current fallback. |

E5 subcases must name their exact target and keep all other kernels fixed:
`qmv2.planFor`, `lane_qmm.splitK`, `moe_affine4` ROWS/SGS,
`qkvMppWins`/`qkvMppSplits`, `qsa_decode` CH, `gdnBlockTFor`/pipelined route,
or the shape-specific 192/128 attention kernels. The existing
`MLX_SERVE_GDN_PIPELINED=0` and `MLX_SERVE_GDN_BLOCK_T` controls can test eligible
GDN alternatives; new variants require parity first. These are not a mandate to
change every kernel or copy M5 thresholds to an M1/M4.

### E1 prototype controls

| Environment setting | Meaning / valid values |
| --- | --- |
| `MLX_SERVE_PREFILL_DQ_MIN_ROWS` | Strict decimal 256..65536; overrides the ordinary affine DQ crossover for all supported bits. Unset keeps 2-bit at 384, other bits at 2048. |
| `MLX_SERVE_PREFILL_DQ_MAX_BYTES` | Strict positive decimal; maximum bytes for one expanded N×K bf16/fp16 weight. Overflow declines the route. Unset adds no cap. This is not a total-memory limit. |
| `MLX_SERVE_PREFILL_DQ_GEMM=0` | Existing master off switch; dominates the two prototype settings. |

After the default/off comparison, hold the actual chunk width fixed and sweep
minimum rows 512/1024/2048/4096, with a separate 256/384/512 sweep for 2-bit.
Then hold a promising crossover fixed and test weight caps around the model's
actual expanded projection sizes, e.g. 64/128/256 MiB. Size caps include the whole
projection, including fused gate/up where present. Do not run the full Cartesian
product before finding a reachable route.

For example, prefix the **server launch**, not the probe, with
`MLX_SERVE_PREFILL_DQ_MIN_ROWS=512 MLX_SERVE_PREFILL_DQ_MAX_BYTES=134217728`.
Unset both variables for the original policy. They are process-cached after the
first eligible call and apply to every loaded model; restart for each arm. Invalid
settings raise `InvalidPrefillDqMinRows` / `InvalidPrefillDqMaxBytes` on that call,
not at startup. Ineligible/tiled/GGUF routes and the master-off arm do not validate
unused settings. Save effective environment even for a declined arm.

The first selected experimental DQ route emits `[prefill-dq] experimental route
engaged` at info level with shape, bits, dtype and settings. That process-wide line
is not a per-layer count or proof of completed GPU work. No line can mean the cap
declined every weight, no ordinary affine call was reached, or logging was off;
resolve with a diagnostic trace before attributing results. Existing numerical
differences between DQ and stock remain; small-matrix parity does not establish
checkpoint-level quality at the newly selected widths.

## Existing commands and the limits of the harness

Use `tests/bench.sh --url` to measure a server launched with the exact arm. The
automatic model loop chooses fast speculative settings and separate sidecars;
it is not a plain-decode control. Example, after auditing per-model settings and
choosing a free port (run in bash, stop only this PID):

```bash
PORT=11490
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
# Continue only if this port has no listener.
MLX_SERVE_ROUND_COST_PERSIST=0 zig-out/bin/mlx-serve \
  --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --no-mtp --no-drafter --no-pld --kv-quant off \
  --prefix-cache-entries 0 --log-level info >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT
```

Wait for `/health`, confirm `/v1/models` and `/props`, then use the advertised
model ID (not an assumed path or alias). Set `MODEL_ID` from that response:

```bash
: "${MODEL_ID:?advertised model ID from /v1/models}"
curl --fail --silent "http://127.0.0.1:$PORT/props" >"$RUN_DIR/props.json"
LLMPROBE="$LLMPROBE" bash tests/bench.sh \
  --url "127.0.0.1:$PORT" -m "$MODEL_ID" --tag "$RUN_TAG"
kill "$SERVER_PID"
wait "$SERVER_PID" || true
trap - EXIT
```

The script can finish despite a failed probe; require an actual valid report with
usage and intended scenarios before accepting a cell. In URL mode its summary
does not automatically find `server.log`, so retain the log and annotate the
engaged spec mode yourself. Add `--full` only after the larger context ladder fits.
Audit all inherited `MLX_SERVE_*`/`MLX_*` values before a run; use explicit bash
argument arrays for multi-switch arms, not `env $CONFIG` under zsh.

For DFlash method correctness on row-exact models, load the same sidecar on both
arms and use request `enable_drafter:false`, `enable_mtp:false`, `enable_pld:false`
for the serial reference. A `--no-drafter` boot changes `rowExactDecode` and can
change weight layout. That different boot is still a useful **deployment**
comparison, but not the exact-row oracle. If llmprobe cannot express these body
fields, extend its scenario support first and pin the changed version.

Attribution helpers already present:

```bash
MLX_SERVE_DECODE_FWD_UBENCH_S=1 MLX_SERVE_DECODE_FWD_UBENCH_KV=8192 \
  bash tests/fwd_ubench.sh "$MODEL" 30 --no-mtp --no-drafter --no-pld
MLX_SERVE_KVQ_UBENCH=1 zig build test -Doptimize=ReleaseFast \
  -Dtest-filter="qkv packed-attention surface"
MLX_SERVE_VQMM_UBENCH=1 zig build test -Doptimize=ReleaseFast \
  -Dtest-filter="verifyQmm µbench"
```

`fwd_ubench.sh` kills matching processes on its chosen port: reserve a dedicated
unused `PORT` before using it. Use `MLX_SERVE_PREFILL_TRACE=1` and
`MLX_SERVE_STEP_TRACE=1` only in diagnostic passes. Follow
[metal-tracing.md](metal-tracing.md) for short phase-windowed captures. Verify
that kernel-name tables actually exist on this Xcode/OS; server attach recipes
have mixed historical results. Sampled shader times identify candidates, not
precise per-kernel wall-time totals. Hardware occupancy counters are usable only
if the counter names/units can be resolved on the target.

`tests/bench_concurrency_ladder.sh` is a useful diagnostic, but its client wall
rates include prefill and its log is temporary. It is not a drop-in source of
decode-only rates, p95 gaps or energy windows. E3 needs a tested llmprobe streaming
scenario/extension with per-request timestamps, counts and durable logs. Likewise,
phase-aligned energy collection and mode-transition orchestration are explicit
E0/E2 harness work, not existing bench.sh features.

## Tests required before adopting a candidate

Select by touched path, then run the full gates from the plan:

| Change | Existing starting points plus new regression coverage |
| --- | --- |
| Chunk/schedule | `effectivePrefillChunk`, `adaptivePrefillWidth`, `nextChunkEnd`, `interleaveTicksFor` unit cases; [test_prefill_interleave.sh](../tests/test_prefill_interleave.sh); test explicit override/admission agreement and a mode change at a boundary |
| Matmul geometry | `verifyQmm`/`lane_qmm`/`qmv2` tests, every adopted bits/group/dtype/width; exact-row references and unsupported-device decline |
| GDN/KDA | Blocked/pipelined f64-truth and chunk continuity tests; tree/sequence replay and partial acceptance; keep KDA's own recurrence oracle |
| KV/attention | `qkv` parity, ragged pages, masks, strided views, split merges and 32 KiB decline; [test_kv_quant_fused_equivalence.sh](../tests/test_kv_quant_fused_equivalence.sh) |
| Spec/batch | `round_cost`, `mtp_replay_test`, group planner tests; [test_batched_equivalence.sh](../tests/test_batched_equivalence.sh), [test_mtp_batched.sh](../tests/test_mtp_batched.sh); real N=2/4 and rollback/cancel/EOS |
| Prefix/state | [test_hybrid_reuse_equivalence.sh](../tests/test_hybrid_reuse_equivalence.sh), [test_prefix_cache_disk.sh](../tests/test_prefix_cache_disk.sh); restored-tail, eviction and memory-growth coverage |
| ANE | `anePrefillAllowed`, share/gate tests; [test_ane_prefill.sh](../tests/test_ane_prefill.sh); numerical/task quality plus failure-to-GPU fallback |
| Power telemetry/policy | New injected-state pure-helper tests: unavailable, transitions, duplicate notifications, straddled samples, finite bounded prices, explicit overrides and no MLX work on callback thread |

For the provided prototypes, run:

```bash
zig test src/prefill_dq_policy.zig -O ReleaseFast
zig build test -Doptimize=ReleaseFast -Dtest-filter="qmatmulBits: experimental prefill"
bash tests/test_low_power_state.sh
```

The GPU test checks actual engagement, fp32 truth-relative error and finite
outputs for bf16/fp16 and affine bits 2/3/4/5/6/8, exact stock output when the cap
declines, and an eight-row decode/verify guard. Default crossovers have independent
boundary tests. The sampler tests argument bounds, mode attribution, unknown
states, JSON round-trip, real output count and monotonic timestamps. Future E8
notification/epoch tests in the table do not apply to this external polling tool.

A unit test must run, not merely compile or skip because a model/GPU environment
variable is missing. MLX env-gated tests may be cached by Zig; rerun the actual
test artifact when varying an environment-only arm, as described in the test
matrix. Warm JIT outside timings. Numerical tuning must not silently relax an
existing test's tolerance to admit the candidate.

## Host validation of the initial prototypes

Run on the M1 Ultra Studio described in the plan, using the pinned Zig and newly
staged pinned MLX/MLX-C. These are correctness/build results, not E0–E8 performance
or power measurements.

| Check | Result |
| --- | --- |
| Pure DQ policy, `zig test src/prefill_dq_policy.zig -O ReleaseFast` | 4/4 passed; defaults, crossover/cap boundaries, strict parsing and overflow |
| New GPU test, `-Dtest-filter="qmatmulBits: experimental prefill"` | Passed, 3/3 in the filtered root (including test infrastructure); its numerical test exercises all 12 dtype/bit combinations |
| `zig build test -Doptimize=ReleaseFast -Dslow-tests --summary all` | **Not green:** 3,028 passed, 284 skipped, 1 failed in `glm5 one-token hcPre chain matches collapse, rms_norm and expand` |
| GLM failure isolation on current and base source | Both filtered runs exit 255 in the same GLM test; isolated base uses `0f01d299`'s unchanged `transformer.zig` with identical staged libraries. This reproduces a failing baseline test, not the full-suite numerical symptom exactly. |
| `zig build -Doptimize=ReleaseFast` | Passed |
| `bash tests/test_mlx_staged_nax.sh` after the build | 9 passed, 0 failed; NAX kernels are staged, but this M1 host cannot validate NAX execution |
| `bash app/test.sh` | 3,777 tests, 29 skipped, 0 failures |
| `cd app && swift build` | Passed |
| `bash tests/test_low_power_state.sh` | Passed, including real CLI output and invalid-argument exit status |
| `bash tests/test_prefill_interleave.sh` | Skipped: required checkpoint absent |
| Documentation / diff | Local source links, bash command syntax and whitespace checks passed |

The full-suite GLM comparison reported `-0.6015625` versus `-0.45898438`, exceeding
its existing `0.024589844` absolute tolerance. Its one-token fused hyper-connection
path is untouched by this branch and cannot reach the ≥256-row DQ experiment.
Treat GLM E5 work as blocked on understanding/fixing that baseline correctness
failure; no tolerance was relaxed. This branch is an experiment handoff, not an
all-green adoption candidate.

The build applied the repository's existing mlx-c patch; no submodule revision
was changed. Full-model CLI/HTTP equivalence, fixture-gated oracles, M4/M5 execution,
mode transitions, telemetry overhead, sustained runs and energy remain laptop
work. No checkpoint was downloaded for these prototypes.

## Results to bring back

Keep raw reports, logs and telemetry under `~/claude-tmp/bench-<tag>/` per the
project benchmark skill. Commit the concise result table and provenance after
measurement; do not fill `benchmarks.md` with projections or unreleased columns.

Each result needs: E-ID, baseline/candidate commits and binary/library hashes,
chip/RAM/OS, source/mode/thermal/charge, model/sidecar hashes, effective flags and
settings, prompt/seed hashes, cache state, actual token counts and stop reason,
engaged routes, repeats/order, TTFT/prefill/decode/task time, stream gaps, energy
domain/gross/idle-adjusted joules, peak footprint/swap, quality verdict, paired
interval, and adopt/reject/inconclusive with its reason. Scrub credentials and
personal machine identifiers before committing logs or manifests.

Minimum useful handoff: E0 baselines, one E1 chunk/matmul decision, one E2
speculation decision, one E3 contention result, one E4 memory/cache result, and
a phase trace explaining the next kernel target. E5–E8 may correctly remain
deferred when the evidence gives them no reachable benefit.
