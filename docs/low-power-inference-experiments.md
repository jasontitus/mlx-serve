# Low-power inference: laptop experiment runbook

Use with the [review and optimization plan](low-power-inference-plan.md).
The branch’s E0–E8 candidate experiments remain **planned, not run**. The supplied
M5 Max session establishes the [engine-bench protocol and reference evidence](low-power-engine-bench.md).
That protocol governs primary timing/energy claims and supersedes earlier
standalone llmprobe, battery-run and generic power-integration suggestions here.
Use the laptop’s existing harness; this branch adds no competing timing loop.

## Hardware allocation and M5 follow-ups

| Machine | Role and power modes | Work allocation |
| --- | --- | --- |
| **M5 Max 128 GB MacBook Pro — primary** | Separate Low and High campaigns; Automatic optional | Qualify correctness, then fresh Low Power E0/E2 arms to investigate DFlash2’s long tail, followed by E1/E3/E4. Select E5/E6 from traces. Add a larger dense and a MoE checkpoint only when actual RAM admits them with reserve. |
| Base M5 MacBook — secondary | Separate Low and Automatic campaigns; record actual available modes | Repeat the small-model baseline and shortlisted candidates, including their reachable NAX/fallback paths. Check for regressions and changed crossovers; do not repeat every Max sweep. |
| **M2 Max MacBook Pro — older-generation check** | Separate Low and High campaigns if exposed; otherwise Low and Automatic | Repeat E0 and shortlisted E1–E4 controls on shared checkpoints. Validate non-NAX dispatch, prefill crossovers, speculation and contention; prioritize a memory-eligible ANE trial if E7 is pursued. |

Apple lists High Power support for recent Pro/Max MacBook Pro models and older
Max models; verify the settings on each specific chassis, OS and power source.
([Supported power modes](https://support.apple.com/en-us/101613))
Record exact chip/core count, RAM, chassis size and OS build. Use the same small
checkpoint revision and scenarios for the portability check, but judge each
candidate against that machine's own baseline in the same mode. Different memory
capacity, cooling and dispatch choices prevent pooling raw rates or energy. The
M5 Max’s 0.918 W harness-overhead calibration and energy-counter behavior must
be validated separately before using this protocol on the base M5 or M2 Max.

After qualifying the M5 Max, reproduce the supplied long-reply speculation
problem with a fresh baseline, then prefill, contention/cache and the shortlisted
follow-ups. Transfer the useful
candidates to the base M5 and M2 Max before considering general defaults.

The M2 Max adds a non-NAX control: verify that unsupported NAX routes decline
and stock/custom shader fallbacks remain correct, then remeasure the DQ crossover
and speculative widths rather than copying the M5 choices. Repeat the GLM hcPre
oracle there too to help localize the existing M1 failure. A difference between
M2 and M5 cannot isolate NAX's contribution: CPU, memory, cooling and other GPU
changes are confounded. NAX attribution still needs a same-M5 path comparison.
If a candidate only wins on M5, preserve the original M2 behavior and require
evidence before adding a hardware-specific policy. ANE tests on M2 still need
memory eligibility, lossy-quality checks, engagement and cold/warm energy records.

| M5 follow-up | Concrete test and decision | Adversarial guard |
| --- | --- | --- |
| Correctness qualification, before timing | Run the new DQ oracle, reachable NAX kernel oracles and the GLM hcPre test in Low and High on the Max; repeat relevant checks in Low/Auto on base M5 | A staged metallib is not execution proof. Preserve tolerances; passing on M5 does not resolve the M1 baseline failure. |
| NAX crossover map, E1/E2/E5 | Probe actual verification widths around 4/8/16 rows and prefill widths 256–8192, then narrow around observed dispatch boundaries | Compare one reachable NAX path with its valid fallback, preserving weight layout. Per-kernel switches do not disable all MLX NAX use. Record engagement; decline synthetic wins absent from full requests. |
| Interactive bursts, E0/E2/E6 | Separate latency-only diagnostic after 5/30/120 seconds idle | The battery-controller counters cannot resolve a short burst. Do not attach J/token to it; energy requires a validated long steady-load harness scenario. Keep shaders/cache/idle duration matched. |
| Bottleneck attribution, E5/E6 | Compare production-sized weight-streaming operations, prefill GEMMs and small dependent chains in each mode | Use working sets that do not fit entirely in cache. Diagnostic timings identify the limiting resource; valid engine-bench arms remain the primary performance gate. |
| Realistic memory headroom, E4 | Repeat a shortlisted workload with a recorded browser/IDE workload, then grow context/cache within admission limits | Background work is a separate diagnostic, never mixed into the primary quiet-Mac arms. Record compression/swap and repeat the background load; never bypass reserve or preflight. |
| Sleep/wake and charger transitions, E0/E2/E8 | With the model resident, resume and test a cached conversation and speculative decode against a fresh-server reference | Run outside the primary AC-only arms, which reject sleep/battery transitions. Separate source from mode changes and record settling. Check restored state and timing-controller recovery; do not count sleep time as request service time. |

ANE remains a separate optional E7 follow-up after the GPU baseline. A bounded
M5 Low Power trial may revisit its energy tradeoff despite prior throughput losses;
it requires explicit force-path engagement, sufficient memory, lossy-quality
validation and cold/warm accounting. It is not part of the initial default matrix.

## First session: establish what changes

1. Read the current laptop `LOW-POWER-TESTING.md` and the linked protocol. Respect
   the occupied GPU, kernel/Sushi locks, watchdog and quiet periods. The supplied
   Rapid-MLX chain’s estimated finish is not a lock-release signal. No GPU work
   until it completes; no interactive analysis during any arm’s idle readings.
2. Prepare ReleaseFast binaries, correctness checks and candidate version roots
   between arms. Record commits, binary mtimes, library hashes, harness revision,
   model/template/sidecar identity and effective settings. Keep AC, 80% non-charging
   battery, display/peripheral state and quiet-Mac lifecycle consistent.
3. Begin with the existing 27B Low Power scenarios and fresh baseline/candidate
   arms. Set AC Energy Mode 1 through the verified setup and settle two minutes.
   Do not compare a Low candidate against a High baseline. High/Auto campaigns
   are separate within-mode comparisons; primary arms never run on battery.
4. Preserve the client’s thinking-off, temperature-1, top_p-1 workload. Validate
   the 1-user prompt count and actual draft engagement for every short/long step.
   Prioritize reproducing the DFlash2 disable event before changing its gate.
5. Use the supplied 4-user, long-reply and prefill suite, then choose extra
   diagnostic scenarios from the critical path. Counterbalance at least two arms
   per side in one sitting. Read both counts and energy-validity verdicts.

## Build and freeze the inputs

Repository build/correctness preparation follows [building.md](building.md) and
[tests/CLAUDE.md](../tests/CLAUDE.md). Run it outside occupied/quiet GPU periods;
these are repo commands, not primary benchmark arms:

```bash
export PATH="$PWD/.zig-toolchain:$PATH"
zig build -Doptimize=ReleaseFast
bash tests/test_mlx_staged_nax.sh
git rev-parse HEAD
git submodule status
shasum -a 256 zig-out/bin/mlx-serve lib/mlx/lib/libmlx.dylib
```

Inspect `otool -L` when the stage layout changes. The candidate version root in
`engine-bench/versions/<engine>/<tag>` must follow the existing adapter’s layout;
record the actual launched executable’s commit, mtime and hashes, not just the
checkout used to build it. Retain dirty diffs and rebuilt-library provenance.

All benchmark launches, plans, summaries and comparisons run from
`~/experiments/engine-bench` using the commands in the linked protocol. Pin its
revision and client/scenario configuration; do not substitute a guessed llmprobe
invocation. This supplied campaign protocol overrides the repository’s default
benchmark recipe for these measurements. Supporting standalone diagnostics keep
separate methodology labels and cannot populate the campaign’s comparison cells.

Inspect per-model settings: overrides can outrank launch flags. Preserve effective
KV, MTP acceptance, drafter, int8-prefill and thinking settings in each arm. Disable
round-cost persistence equally for an explicitly fresh-learning experiment; do
not silently change it when reproducing an existing harness baseline. Record
cache state and reported cached tokens. Inspect adapter forwarding and route logs
before attributing a new environment control to a measured result.

## Measurement and telemetry

The [engine-bench protocol](low-power-engine-bench.md#energy-and-validity) defines
both the energy domain and run validity. Primary arms use the battery controller’s
system-load counters, settled before/after idle readings and the supplied 0.918 W
harness-overhead calibration. Pair added J/token with speed from the **same step**;
keep prefill J/1K prompt tokens, short replies and long replies separate.

| Run | Purpose | Boundary |
| --- | --- | --- |
| Standard engine-bench arm | Primary speed and added-energy result | Quiet Mac on AC, valid counts and energy verdicts; no added profiler/sampler |
| Kernel / phase trace | Routes, gaps, dispatches, memory and CPU attribution | Outside primary arms; never quote its rate or chip watts as the campaign result |
| Correctness / transition / memory-stress diagnostic | Numerical/state checks and behavior outside steady conditions | Not a valid primary arm when it sleeps, uses battery, incurs user input or violates thermal criteria |
| Idle-burst diagnostic | Request latency after a pause | No short-burst energy claim from minute-scale counters |

The controller needs at least 55 seconds of steady load; the harness repeats steps
shorter than 150 seconds. Its settled-idle and drift criteria decide whether energy
is computed. Do not interpolate a short burst into an energy estimate or replace
missing energy with `22 W / tok/s`. Keep raw counter/idle/overhead data. Chip-only
`powermetrics` or GPU telemetry can attribute a bottleneck, not supply the energy
number. Loaded-idle noise alone cannot establish a gain.

The branch’s native `scripts/low-power-state.swift` sampler remains a diagnostic
for declared mode, low-power flag, source and thermal state. It is not an energy
meter and must not be added to a calibrated primary arm without harness integration
and overhead validation. Compile/test it between arms. A false low-power flag
cannot distinguish High from Automatic; verify the actual AC policy as well.

Record physical footprint and compression/swap through existing harness telemetry
or a separate diagnostic. Do not add interactive polling to idle windows. Keep
capacity stress separate from the no-swap comparison and preserve admission,
OS reserve and allocator safety constraints.

## Workload ladder

The primary M5 Max campaign starts with the harness’s existing 27B and Flash-Next
workloads and standard four-step suite; the table below extends coverage after
those reproduce. Small models remain useful for portability and attribution,
not a replacement for the supplied comparison workloads.

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

For extended diagnostics, use fixed novel-prose, code/structured-output and
copy/edit prompts. Keep them separate from the standard harness requests. Preserve
the harness’s reply-length settings for primary comparisons; save actual output
counts and EOS/stop reasons. Never suppress EOS, add thinking or change content
to improve the denominator. Token replays are attribution diagnostics only.

Cover these scenarios in the staged campaign:

- Resident model, prefix cold: no hot/disk prefix reuse; primed shader kernels.
- Same conversation extended: hot prefix reuse, restored SSM position, and tail
  prefill. Distinguish exact repeated prompt from a real appended turn.
- One stream decoding while another starts an 8k/32k cold prefill; then N=2/4
  steady streams. Measure per-stream latency and aggregate completion work.
- First-use loading/compile and warm steady state as separate energy budgets.
- Sustained 10–15 minute diagnostics in each mode, recording thermal evolution.
  Primary arms that reach thermal state 2 or above do not count; retain those
  traces as stress observations instead of using them in the accepted comparison.

## Repeat order and controls

Follow the harness’s ABBA or mirrored order: **at least two arms per side in one
sitting within about three hours**. A claim requires a difference at least 3% and
larger than 2.5 standard errors of measured run-to-run noise, as computed by the
harness’s comparison/standings implementation. Otherwise call it a tie. Review
counts and energy validity per run before interpreting a grouped comparison.

A/B always means baseline/candidate **within one Energy Mode, source and request
configuration**. Low Power optimizations must be measured in Low Power. Separate
High/Automatic campaigns can test portability of a candidate, but their cells
cannot serve as its Low baseline. If independently repeating in another session,
rerun both sides there; never import the earlier baseline or pool drifting raw
runs. Do not mix modes when estimating the noise threshold.

Restart for process-cached controls, and make persistent timing/cache state equal
within each pair. Mode transitions, sleep/wake and unplugging are separate
correctness/attribution tests outside primary arms. Their samples cannot establish
a steady-state performance or energy gain. Let the chain handle settling, locks,
quiet-Mac setup and caffeinate. No frequency caps or finer power controls.

Apply the plan’s latency/memory/quality guardrails to every scenario. In particular,
inspect short and long DFlash2 behavior separately: a gate may disable drafting
mid-request, so an arm-level mode label alone is insufficient. No selecting only
an appealing width, prompt, engine or boot; confirm a selected candidate with a
fresh same-sitting baseline.

## Experiment cards

| ID | Arms / measurements | Proof, guard and stopping rule |
| --- | --- | --- |
| E0 | Reproduce valid engine-bench Low Power arms on AC, with fresh baseline/candidate; separate High/Auto campaigns later | Honor counts and settled-idle energy verdicts, same-step rates, quiet periods and the existing calibration. Do not add a collector to primary arms or compare across modes. |
| E1 | Actual chunks 512/1024/2048/4096/8192 as allowed; `MLX_SERVE_PREFILL_DQ_GEMM=0` vs default, then the bounded DQ controls below. On 2-bit add 256/384/512 near the crossover. | `[prefill-trace]` shows actual widths/tails. `[prefill-dq]` proves the first experimental selection; use a separate kernel trace for per-layer reachability. Tiled weights require a separate lane-prefill experiment. |
| E2 | First reproduce DFlash2 short/long gate behavior against plain/MTP controls in Low Power; then supported MTP depths 1/2/4/6/8, DFlash blocks 2/4/5/8/16 and PLD copy cases. | `[spec-stats]`, attempted/accepted drafts, mode, width and serial fallback; exact acceptance only. Compare no-sidecar deployment cost separately from same-loaded-layout request-level drafter-off. Reject unengaged arms. |
| E2 transition | Separate diagnostic outside primary arms: warm auto speculation in one mode, toggle with model resident, return; compare against fresh server in each final mode | Measure latency/acceptance over first 32/128/512 committed-token windows and time to stabilize. No energy inference across a transition or from short windows. Test persistence off first; propose epochs only for material adaptation loss. |
| E3 | N=1/2/4; default share 0 vs 0.25/0.5; safe chunk caps 512/1024/2048; interleave-off diagnostic control | Real streams and `[interleave]`/`[batched]` evidence. Require output/state correctness, useful per-stream rate and bounded gap; stop any unfair or memory-unsafe arm. |
| E4 | KV off/8, then 4; contexts around 2k/8k/16k; cold vs warm prefix; 0/1/4 RAM cache entries within budget; SSD tier only for a capacity-use case | Packed attention/fallback evidence, cached/forwarded tokens, restore time, physical memory, disk writes. Fixed request transcript and quality checks. No switching KV format inside a live cache. |
| E5 | Selected hotspot only: rows per simdgroup 1/2/4, legal SIMD-group/split counts around the shipped choice; matmul/attention crossover shapes; blocked vs pipelined GDN | Some geometry arms require future prototype code. Query per-pipeline limits; finite outputs, truth oracle, exact rows/states where promised, multi-seed tails. Stop if isolated improvement disappears in dependent full-forward and valid engine-bench arms. |
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

Primary launch, detached-chain, summary and comparison commands are in the
[engine-bench protocol](low-power-engine-bench.md#builds-and-arms). Run them from
that harness directory, preserving its locks, watchdog and quiet periods. Stage
the baseline and candidate version roots first; launch-setting text does not
prove the executable, method or DQ route actually used.

The new prefill controls belong on the server arm via its environment or plan
line. Audit forwarding. Tiled weights can bypass the ordinary DQ route entirely;
prove reachability in a separate diagnostic before spending four long arms on it.

For DFlash method correctness on row-exact models, load the same sidecar on both
arms and use request `enable_drafter:false`, `enable_mtp:false`, `enable_pld:false`
for the serial reference. A `--no-drafter` boot changes `rowExactDecode` and can
change weight layout. That is a useful deployment comparison, not the exact-row
oracle. Verify the harness can express these controls; extend its scenario adapter
with tests if necessary rather than inventing flags or changing request semantics.

The repository’s `tests/bench.sh`, `tests/fwd_ubench.sh`, kernel microbenchmarks
and phase traces remain supporting tools outside the primary arm schedule.
`fwd_ubench.sh` can kill matching processes on its port; never run it during an
arm. A standalone llmprobe result has a different methodology label and does not
replace an engine-bench speed/energy cell. Microbenchmarks need warm JIT and cannot
establish full-request or energy wins by themselves.

Follow [metal-tracing.md](metal-tracing.md) for separate short captures. Check
kernel tables/counter availability before attribution. Existing concurrency
helpers do not supply every per-stream gap or phase-energy marker we want; any
harness extension must preserve validity gates and revalidate measurement overhead.
Do not add extra shell activity or per-round logging to a calibrated arm without
accounting for its effect.

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

Keep primary raw reports, logs, power events and summaries in the engine-bench
`runs/<date>/<label>` directories. Supporting repo diagnostics retain their own
artifacts/methodology labels. Commit a concise provenance/result table after
measurement; do not fill `benchmarks.md` with projections or mixed-method cells.

Each result needs: E-ID, baseline/candidate run labels, commits, binary mtimes and
binary/library hashes, harness revision, counts and energy-validity verdicts,
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
