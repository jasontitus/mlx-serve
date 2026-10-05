# Low-power inference: review and optimization plan

Status: **review, opt-in prototypes and supplied laptop evidence; no branch power measurements**.
Reviewed 2026-10-04 at `0f01d299d3d7759a7222b2117567053b6d350b89`.
Companion: [laptop experiment runbook](low-power-inference-experiments.md).

Start with a fresh Low Power baseline in the supplied engine-bench harness, then
reproduce DFlash2's reported long-reply fallback and measure prefill dispatch
crossovers. Optimize the kernels that remain on the laptop's critical path.
There is no evidence yet for a universal low-power chunk size, draft depth,
threadgroup size, or ANE split. Smaller work units can increase total work and
energy; faster completion can reduce energy even at higher instantaneous power.
Use **Low Power and High Power**, with Automatic optional, as separate campaigns:
compare each candidate with its baseline inside the same mode. We will use these
OS modes, not finer-grained frequency/power controls.
The primary target is the **M5 Max 128 GB MacBook Pro**, on AC throughout primary
arms. Use the base M5 MacBook for portability within the generation and the
**M2 Max MacBook Pro** for older-GPU fallback coverage. Test candidates separately
in Low and High where available, otherwise Low and Automatic. Choose candidates
on the M5 Max, then check the shortlisted changes
on the other machines; do not pool results or assume dispatch thresholds match.

The supplied [engine-bench protocol and reference results](low-power-engine-bench.md)
supersede the earlier generic energy plan for this campaign. The harness files
are absent on the review host; the pasted protocol is the source of the machine
constraints and measurements, not an independently verified result of this branch.
No GPU work or power-state change was performed while incorporating it.

## Scope and evidence

This review follows native text inference from admission through prefill, cache,
decode, sampling, and MLX Metal submission. It inspects both embedded shader
strings and `.metal` files, including the quantized matmul, attention, recurrent,
and MoE paths. The pinned upstream MLX and mlx-c sources were fetched for reading:

| Component | Reviewed revision / boundary |
| --- | --- |
| mlx-serve | Base commit above; source references below name symbols at that revision |
| MLX | `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`, including Metal dispatch, quantized shaders, attention fallback, and eval/submission |
| mlx-c | `56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6`; binding boundary, not a separate compute engine |
| Current host | M1 Ultra Mac Studio, 128 GB, macOS 27.0.1; no laptop low-power experiment performed |
| Not covered in depth | Media denoisers, vision towers, embedded ds4/llama Metal implementations, Sushi EXL3, and the separate MLX-GGUF kernel repository |

The first laptop campaign is text-only safetensors on the native MLX path.
Engine routing must be recorded: findings about `qmatmulBits` do not establish
anything about an embedded GGUF engine. Large-model kernels are secondary targets
only if a laptop has enough memory for their real checkpoints. Synthetic geometry
tests establish kernel correctness, not model throughput.
The primary Max has 128 GB RAM; still record GPU core count, chassis, OS and actual
headroom before selecting checkpoints. Other machines' memory budgets are separate.
The runbook's [hardware allocation and M5 follow-ups](low-power-inference-experiments.md#hardware-allocation-and-m5-follow-ups)
cover NAX qualification, idle bursts, memory pressure and resume behavior.

Apple documents Low Power Mode as an energy-saving mode, separately configurable
on battery and AC. It does not specify one universal CPU/GPU/DRAM frequency ratio.
Measure the actual machine; do not assume memory bandwidth is unchanged or that
Low Power Mode is equivalent to thermal throttling.
([Apple power modes](https://support.apple.com/en-us/101613))

Use `ProcessInfo.isLowPowerModeEnabled`, power-state notifications, and
`thermalState` as distinct observations. The local Foundation SDK confirms the
low-power API is available on macOS 12+. A false low-power flag alone does not
distinguish Automatic from High Power Mode.
([Apple ProcessInfo](https://developer.apple.com/documentation/foundation/processinfo))

## What the code already does

| Stage | Source and contract | Implication for low power |
| --- | --- | --- |
| Request / admission | [server.zig](../src/server.zig): `requestSpecModes`, `chooseRequestPrefillChunk`, `adaptivePrefillWidth`; [scheduler.zig](../src/scheduler.zig): admission and `PromiseLedger` | Keep memory safety, explicit settings, and request semantics authoritative. Current adaptive chunk policy follows memory headroom, not measured time or power. |
| Prefill | [generate.zig](../src/generate.zig): `effectivePrefillChunk`, `nextChunkEnd`, `effectiveSsmCheckpointStride`, chunk eval and cache release | Default base is 8192; score-free arches have a 16384 base. Caps, model pins, checkpoint boundaries, and company can reduce the actual width. The environment override bypasses the normal width cap. |
| Concurrent prefill | [scheduler.zig](../src/scheduler.zig): `companyPrefillChunk`, `interleaveTicksFor`, `decodeShareAdmissionCap` | Default company cap is 2048; explicit decode-share mode caps at 1024. Decode gets time only at chunk boundaries. A time share cannot bound a long chunk's latency. |
| Matmul routing | [transformer.zig](../src/transformer.zig): `qmatmulBits`, `verifyQmm`, `prefillDqGemm`; [lane_qmm.zig](../src/lane_qmm.zig) | Tiled weights take the lane path first. Ordinary affine weights try verify, then dequant+GEMM, then MLX. Dense and non-affine weights have separate routes. |
| Prefill recurrence | [transformer.zig](../src/transformer.zig): `gdnBlockTFor`, `gdnPipelinedFor`, `prefillEvalCadenceFor`; [gdn_pipelined.metal](../src/kernels/gdn_pipelined.metal) | Block dtype, register use, checkpoint spacing, and memory-bounding evals interact. The pipelined route is already gated by model and NAX availability. |
| Serial decode | [generate.zig](../src/generate.zig): `Generator.next`, `lazyForward`, `advanceStep` | Already builds/submits the next forward before resolving a pending token. Penalties, grammar, and logprobs take different paths. More async submissions are not automatically more overlap. |
| Batched decode | [scheduler.zig](../src/scheduler.zig): `runDecodeTick`, `runBatchedDecodeTickInner`, `mtpSubGroupPlan`; [transformer.zig](../src/transformer.zig): `perSlotBatchedAttn` | Existing batching shares projection work. Long KV uses per-slot attention rather than copying padded histories. A real two-stream run is required to exercise batch-sensitive kernels. |
| Speculation | [generate.zig](../src/generate.zig): MTP/DFlash controllers; [round_cost.zig](../src/round_cost.zig): `Table`, `cacheKey`; [mtp_group_planner.zig](../src/mtp_group_planner.zig): `choose` | Learns round milliseconds and tokens by width/context; group planner also bounds latency. Neither objective measures joules. Persistence is opt-in; in-memory history exists regardless. |
| KV / prefixes | [transformer.zig](../src/transformer.zig): packed KV reads; [prefix_cache.zig](../src/prefix_cache.zig), [kv_disk_writer.zig](../src/kv_disk_writer.zig) | Avoiding prefill can beat any kernel improvement, but retained state, copy-on-write, SSM snapshots, and disk writes must fit a laptop's memory/energy budget. |
| MLX backend | [quantized.cpp](../lib/mlx-src/mlx/backend/metal/quantized.cpp): `get_qmv_batch_limit`, `QuantizedMatmul::eval_gpu`; [device.cpp](../lib/mlx-src/mlx/backend/metal/device.cpp): `maybeInsertBarrier`, `needs_commit`; [transforms.cpp](../lib/mlx-src/mlx/transforms.cpp) | Routes and command-buffer limits depend on geometry/chip, not a low-power signal. Eval can block on the ten-active-task limit or memory pressure; long submit time is not necessarily slow CPU graph building. |

No explicit low-power/thermal policy was found in the inspected native inference
and MLX dispatch paths. This is an optimization opportunity, not evidence that
the OS fails to manage hardware power.

## Kernel findings and hypotheses

These are source-backed exposure points. Whether changing them helps under low
power remains an experiment.

1. **Prefill's dequantization crossover is fixed.** `prefillDqGemmMinRows` selects
   384 rows for 2-bit and 2048 for other supported affine widths. It materializes
   a full weight in the activation dtype on every call. Extra memory traffic can
   lose when bandwidth or package power is limiting; a faster dense GEMM can
   still win when matrix throughput dominates. On in-place tiled weights this
   route is unreachable: `lane_qmm.tiledQmm` wins first. Sweep these as different
   dispatch families, not one global switch.
2. **Small-M matmul has several discontinuities.** `vqmmLaneForTile` covers
   split-K/msg rows 2..7, NAX rows up to 16, and the M4 shader tile up to 24;
   mixed-bit eligibility is narrower. MLX's own `get_qmv_batch_limit` varies
   with K, N, generation and chip size. `lane_qmm` uses fixed row tiles and a
   split-K target of 1024 column-tile/split combinations. Power mode can change
   the profitable width without changing which hardware instructions exist.
3. **Attention has context-dependent crossovers.** `qkvMppWins` switches packed
   hd-256 attention at KV 2048 for 4+ query rows, 8192 for 2..3, and 16384 for
   one. `qkvAttnMppKernel` separately rejects excess padded rows/threadgroup
   memory. KV8 can reduce reads but decoding, partial buffers, merge work and
   prefill dequantization still cost energy. A shape that falls back to dense
   attention needs its own accounting.
4. **One-token MoE geometry is already aggressively tuned.**
   [moe_affine4.zig](../src/moe_affine4.zig) uses one output row per simdgroup,
   two simdgroups per threadgroup, fused gate/up, and register accumulation of
   the down expert sum. [qmv2.zig](../src/qmv2.zig) has per-generation plans
   with explicit stock fallbacks. Wider row reuse saves activation loads but
   raises register pressure; neither more occupancy nor more reuse is a goal
   by itself. The full model matters more than a repeatedly cached matrix.
5. **Sparse attention still has dependent work.**
   [qsa_decode.zig](../src/qsa_decode.zig) already bypasses mask/gather/SDPA for
   eligible solo dense-KV decode using a split and merge over selected keys.
   Its 32-key split and one-head-per-simdgroup layout are candidates for
   measurement, not missing optimizations. Batched, verify, and quantized-KV
   paths differ. [qsa_nax.metal](../src/kernels/qsa_nax.metal) handles its
   eligible gather shape with cooperative matrix operations. Preserve indexer
   arithmetic: a changed score can change selected blocks.
6. **Recurrent kernels exchange dispatch cost for live registers.**
   [gdn_decode.zig](../src/gdn_decode.zig) recomputes some prework per split,
   keeps state fragments in registers, and has distinct sequence/tree/commit
   paths. The pipelined prefill shader prefetches 12 tokens with 16 value rows
   and 128 threads. [kda_decode.metal](../src/kernels/kda_decode.metal) stages
   multiple token/head arrays and maintains f32 recurrent state. These are
   different recurrences; a GDN tuning result does not authorize a KDA change.
7. **Launch and event costs are already partially hidden.** MLX uses concurrent
   encoders but inserts a buffer-wide barrier for tracked dependencies. Every
   array in an async batch can inherit the batch-end event. GPU PLE already
   splits out the sampled-token event and defers history settling. Existing
   row joins, residual/norm and activation-table fusions must be accounted for
   before proposing them again. More command-buffer packing can delay GPU
   start and destroy useful overlap.
8. **Power transitions can leave learned prices from another regime.**
   `round_cost.cacheKey` includes chip, model path, quant, OS and build, but no
   power mode. Its EMA and stale-cell reseeding provide adaptation, so permanent
   failure is not established. They do not explicitly separate a sudden
   High-to-Low transition; the same concern applies to in-memory planner
   timing. Measure transient regret and recovery before adding persistent state.

For geometry work, query the compiled pipeline's thread limit and validate
threadgroup memory rather than assuming a device-wide maximum. Apple notes that
the limit depends on the kernel's resources, and that high occupancy can also
cause cache contention.
([Threadgroup sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes),
[occupancy interpretation](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy))

## Objective and decision model

Primary target: **lower joules per completed task while retaining usable TTFT
and streaming latency in Low Power Mode**, with throughput and latency reported
alongside energy. Also retain a speed-focused result if it saves latency without
materially increasing energy. Do not silently trade model quality for speed.

For a phase, reason about `max(bytes / effective bandwidth, operations /
effective compute)` plus exposed dependent dispatches and synchronization. CPU
build and GPU execution overlap, so adding their entire durations double-counts
time. Use this only to reject impossible savings; the effective rates must come
from the same laptop, mode, and working set. For MoE, count selected experts and
shared weights, not total checkpoint bytes. Include scales, KV and state traffic.

For speculation, compare `round time / committed tokens` and `round energy /
committed tokens` with plain decode. Rejected drafts are work, not output. The
maximum end-to-end benefit of a kernel occupying fraction f is bounded by
`1 / ((1 - f) + f / kernel_speedup)` before secondary effects.

Acceptance gates for the supplied campaign:

- A speed or energy difference counts only at **at least 3% and greater than
  2.5 standard errors** of measured run-to-run noise, using the harness's own
  comparison implementation. Otherwise report a tie. This replaces the earlier
  provisional percentage thresholds; it is not a guarantee against overfitting.
- Run at least two arms per side, ABBA or mirrored in one sitting within about
  three hours. Every run must count; an energy claim additionally requires valid
  settled-idle energy on both sides. Never use a baseline from another mode or
  earlier session. Independent confirmation reruns both arms in its own sitting.
- Pair speed and added J/token from the same step. Require unchanged correctness
  and quality, no more than 5% worse TTFT/p95 stream gap, and no more than 3%
  greater task energy for a speed-focused candidate. Keep short/long reply and
  prefill results separate. Capture missing latency metrics through validated
  harness extensions, not interactive polling inside an arm.
- No unexplained swap growth, memory-cap bypass, starvation, High Power Mode
  regression above 3%, or unsupported-shape regression. A useful tradeoff that
  misses these gates stays explicitly optional; inconclusive means do not ship.

This campaign uses battery-controller whole-machine system-load counters and
reports added energy after settled idle and calibrated harness overhead. Preserve
raw power and both subtractions. It is not a wall-outlet meter or a battery-life
projection. The reported roughly 21–23 W added during Low Power generation makes
throughput useful for ranking hypotheses, not a substitute for measured J/token.
Never impute missing energy with 22 W, GPU/chip power or a short burst. The coarse
counters require steady long steps; the existing harness supplies them.

## Prioritized plan, with adversarial review of every proposal

| ID / priority | Proposal and concrete experiment | Strongest objection / falsifier | Decision and implementation gate |
| --- | --- | --- | --- |
| E0 / first | Reproduce valid Low Power engine-bench arms on the M5 Max, AC only, then fresh candidate/baseline pairs. High/Auto are separate within-mode campaigns. | Extra tools/input disturb idle; mode/source drift and thinking preambles invalidate comparisons; the energy counters cannot resolve short bursts. | Honor locks, quiet periods, counts and energy-validity verdicts. Use the existing calibration and same-step energy; never replace a missing value with an estimate. |
| E1 / high | Sweep existing chunk sizes and dequant+GEMM on/off jointly, separately for tiled/NAX and ordinary affine trunks. | Small chunks repeat weight reads/graph work and may fall below fast matmul thresholds; large chunks worsen live-stream gaps or memory. | Use actual chunk/route evidence. Retain a Pareto winner within admission limits; only then change a shape-specific crossover or chunk policy. |
| E2 / first after E0 | Reproduce DFlash2's reported short/long divergence and disable event against plain/MTP controls, then sweep supported draft widths. Mode transitions are separate diagnostics. | Lowering the gate can retain losing drafts; longer warmup can hurt short replies; request thresholds, cost-table exceptions and tiled layouts complicate a blanket gate=2 explanation. | Pin executable/settings, prove engagement and disable timing, pair long speed with long energy. Test any controller change first; preserve acceptance and valid assistant context. |
| E3 / high | At N=2/4, combine a long prefill with an existing stream; sweep chunk cap and decode share. | More batching improves aggregate rate while making each user wait; tiny chunks lose GEMM efficiency. | Require per-stream TTFT, p95/max gap, fairness and completion energy. Add a time-budget cap only if the current policy misses a predeclared latency target. |
| E4 / high | Sweep KV off/8 first, then 4 as a separate quality tradeoff, across packed-attention crossover contexts and warm-prefix workloads. | Quantization/restore overhead dominates short context; hybrid state is not compressed by KV quantization; int4 can move output. | Measure active/cache/physical memory, actual packed/fallback route, useful prefix tokens and task quality. No unconditional low-power KV4 default. |
| E5 / medium | For a traced hotspot, sweep existing matmul/MoE/QSA/GDN tile and split choices, then prototype a narrow new geometry. | A microbenchmark fits in cache; more live accumulators spill; a new reduction order breaks exact-row verification or recurrent state. | Use production shapes and working sets, isolated plus dependent-chain timings, fp32/fp64 truth and existing bit-exact contracts; require full-forward and request wins. |
| E6 / conditional | If CPU submission or dependent idle gaps grow under Low Power Mode, test existing async-ladder and command-buffer limits, then a proven dependent fusion or bounded config-cache reuse. | Submission time can be GPU backpressure. Bigger buffers and more async evals have already lost; caching mutable/shape-incomplete configs is unsafe. | Keep only a measured request-level win; account for CPU and GPU energy. No speculative GPU-threading rewrite or one-kernel-per-layer project. |
| E7 / conditional | On a memory-eligible M1–M4 laptop, compare GPU-only prefill with warmed ANE and a small split sweep. | ANE uses lossy int8/fp16, extra memory, CPU copies and compile energy; same package power budget may erase overlap. M5 GPU was already faster in prior non-LPM tests. | Check `anePrefillAllowed` and actual memory gate; include cold amortization and quality. M5 force is a later explicit experiment, never a new default inferred from Low Power Mode. |
| E8 / last | If E1–E7 find stable mode-specific winners or E2 finds harmful transition lag, add a small power-aware policy and timing-table epoch. | Profiles overfit one laptop; mode notifications flap; clearing all learning makes short requests slower; thermal state adds excessive table cardinality. | Prefer one universal winner. Otherwise isolate timing by mode, preserve valid model state, bound relearning, and keep unknown/unsupported systems on current behavior. |

### Implementation seams after the measurements

E0 uses the existing external engine-bench measurement. Do not add the branch's
diagnostic sampler to calibrated arms without overhead validation. A later bridge
can read Foundation's power/thermal state and publish a small atomic snapshot.
Foundation is already linked by [build.zig](../build.zig). Notification callbacks must never
call MLX, free arrays, resize caches, or mutate active model state. The inference
thread consumes a changed epoch at a chunk/round boundary. Unknown is a real
state, not a synonym for Low or Automatic.

For E1/E3, separate the performance preference from the memory constraint:
`chosen width <= memory-safe width`. Preserve operator flags, checkpoint/backoff
semantics and admission/execution agreement. A time-budget controller needs a
measured chunk duration, hysteresis and a floor; it cannot promise to interrupt a
running Metal dispatch. First test fixed caps before introducing a controller.

For E2/E8, invalidate timing samples that straddle a mode transition. Test an
epoch-local serial baseline and cost estimates against the current EMA before
persisting anything new. Audit group timing, DFlash yield state and cold-start
priors too; changing only `cacheKey` misses running servers. Keep acceptance
history separate where it remains meaningful. Do not multiply every cost by one
frequency ratio. Keep persistence off in the first campaign; if enabled later,
version the format and reject missing/unknown mode identity rather than mixing it.

For E5, experiment branches should expose a finite geometry choice through pure
planning helpers, with an original-path control. Only turn that into a permanent
knob if both arms represent a useful tradeoff. Include dtype, full dimensions,
quant bits/group size, strides, split count and relevant layout in config keys.
Never send in-place lane-tiled weights to stock MLX. Retain per-kernel capability
and resource-limit gates independently of a power-mode preference.

For E4/E7, quality-changing options remain separate experiments. Neither power
mode nor thermal pressure authorizes changing trunk weights, KV precision,
acceptance semantics, thinking budget, model choice, or requested output length.
Memory reserve and allocator-release cadence remain safety constraints.

## Rejected or deferred directions

| Tempting plan | Adversarial disposition |
| --- | --- |
| Halve every threadgroup / prefill chunk in low power | Reject. Low power is not a smaller logical GPU; this can increase launches, rereads and partial-buffer traffic. E1/E5 measure geometry. |
| Disable speculation because the GPU is slower | Reject. Reusing a trunk's weights across accepted tokens may be more valuable. E2 prices the complete round. |
| Always increase speculation because decode is bandwidth-bound | Reject. Verify compute, draft reads, low acceptance and wider-kernel cliffs can dominate. |
| Use NAX or ANE whenever power mode is Low | Reject. NAX remains a capability/shape decision; ANE has a separate lossy seam and memory bill. |
| Remove prefill evals/cache clears to eliminate synchronization | Reject. They bound lazy graphs and allocator growth. Any cadence experiment must retain equivalent peak-memory bounds. |
| Redo QKV, residual/norm, SiLU/GELU or PLE fusions wholesale | Defer. Many already ship; other variants lost. Require a currently exposed dependency chain and route evidence. |
| Compress ternary weights more / keep all weights dequantized | Defer. Existing base-3 compression lost despite fewer bytes; permanent dequantization can exceed laptop RAM. No footprint-only performance claim. |
| Force CPU efficiency cores, sleep between tokens, or add GPU worker threads | Defer. May starve submission, extend total task energy, violate sole-MLX-caller ownership, or damage streaming. Only a separately requested paced-throughput mode could justify intentional delay. |
| Enable whole-table GPU PLE or SSD prefix caching everywhere | Reject. Residency and SSD traffic can outweigh saved work on laptops. Capacity-gated, workload-specific E4 subcases only. |

The prior failed experiments are documented in [engine gotchas](gotchas/engine-mlx.md),
especially “Small dense models spent a third of a token on dependent launches,”
“The intra-step async-eval ladder is a null lever for us,” “Per-kernel profile of
the Flash Next step,” and “Rows per simdgroup is an occupancy decision.” Their
measurements are prior evidence on named hardware, not laptop Low Power results.
Source takes precedence where historical notes describe superseded routes.

## Correctness and delivery sequence

1. Read the laptop's current engine-bench protocol. Respect the active chain,
   locks, watchdog and idle quiet periods. Run fresh E0 arms and save raw manifests,
   counts and energy verdicts. Invalid energy is a rerun, never an estimate.
2. Start E2 with the reported DFlash2 long-tail fallback, then E1/E3/E4 with
   existing controls. These distinguish policy problems from kernel problems.
   Keep all comparisons inside one mode and sitting. Report ties and nulls too.
3. Choose E5/E6 from the phase trace and Amdahl bound. E7 is a separate optional
   lossy campaign. Implement one measured change per PR, UI and engine together
   only where the feature needs both.
4. Before changing code, add the failing behavior/characterization test required
   by [AGENTS.md](../AGENTS.md). Kernel changes assert finiteness, truth-relative
   error, and any existing exactness contract; sequence kernels also test state,
   tails, chunk boundaries, partial acceptance and rollback. No source-scan tests.
5. For exact row/tiled paths require row and committed-output equality with a
   layout-matched reference. For general floating-point reorderings use the
   existing per-family oracle tolerances and logit margins, not invented cosine
   thresholds or unconditional long greedy byte equality. Run multiple seeds,
   every affected quant width, ragged shapes and actual N>1 batches.
6. Run full `zig build test -Doptimize=ReleaseFast -Dslow-tests`, the app test/build
   gates and relevant integration scripts from [tests/CLAUDE.md](../tests/CLAUDE.md).
   Rebuild the ReleaseFast executable before live A/B. Mark skipped hardware and
   fixture gates explicitly; a compiled env-gated test is not a passing oracle.
7. Validate with separate within-mode campaigns and a fresh pair in any second
   session; do not reuse its earlier baseline. Add E8 only if
   different modes demonstrably need different choices. Keep the original path
   recoverable until the hardware/shape coverage supports a default change.

## Initial implementations and adversarial review

Two prototypes make E0/E1 executable while preserving default inference choices:

* [Native state sampler](../scripts/low-power-state.swift): JSONL records the
  declared Low/High/Auto mode, observed low-power flag, thermal state, power
  source, wall time and boot-relative monotonic time. It observes; the operator
  changes modes. It is separate from the inference process and measures no energy.
* [Prefill DQ policy](../src/prefill_dq_policy.zig), wired into
  `transformer.prefillDqGemm`: optional minimum rows and a per-expanded-weight
  byte cap. Unset retains the 384-row 2-bit / 2048-row other-bit defaults.
  Existing `MLX_SERVE_PREFILL_DQ_GEMM=0` remains authoritative. No Metal arithmetic,
  decoder policy, power-aware defaults, or tiled-weight dispatch was changed.

| Adversarial concern | Guard / disposition | Remaining limit |
| --- | --- | --- |
| A typo silently benchmarks the default arm | Strict decimal parsing, bounded row threshold, positive byte cap, named parse errors; malformed CLI arguments exit 2 | DQ settings are validated lazily on the first eligible affine call; save environment and errors, restart between arms |
| Byte cap is mistaken for admission safety | Overflow-checked N×K×2; one-byte-below cap must take stock and exactly match its output | Does not include activations, other live weights, scratch or allocator cache; existing memory admission remains necessary |
| New threshold captures decode/verify or unsupported layouts | Minimum 256 rows, existing affine/bit/dtype gates, rank-2 weight check, lane route precedes DQ | Large row-axis batches can reach this shared route; validate a target workload's actual shapes |
| A numerically plausible fallback is mistaken for engagement | Counter assertions; first experimental engagement log includes M/N/K/bits/dtype/effective settings | Log proves one selected graph route, not every layer, successful evaluation or a faster kernel; retain separate diagnostic traces |
| DQ rounding or NaNs hide behind aggregate error | fp32 reference, explicit finite outputs, unchanged truth-relative tolerance; bf16/fp16 × 2/3/4/5/6/8-bit coverage; capped output equals stock | Small synthetic matrices are not full-model quality, NAX validation or proof for all shapes/group sizes |
| High and Automatic are mislabelled by one boolean | Declared mode and observed low flag are separate; false is labelled `low_flag_clear_only`, mismatch explicit | Verify selected High/Auto in OS settings for each source; no public API claim to distinguish them |
| Sampling contaminates inference or misses transitions | External process, bounded interval/sample arguments, actual timestamps and unknown-state labels | Polling can miss short transitions; invalidate settling/straddled windows, compare collector overhead, no phase-energy inference |

The policy and GPU route tests were first run against implementations that could
not honor the new settings and failed at the expected behavior, then implemented.
The sampler tests similarly failed before argument parsing was implemented.
Final validation and hardware limitations are recorded in the runbook. No energy
measurement or laptop speedup is claimed by this branch.

The full host suite also exposed a GLM one-token hyper-connection parity failure.
That test fails in isolation on unchanged base sources too; the new DQ oracle
passes. Resolve this baseline correctness issue before tuning GLM's fused path.
The [validation record](low-power-inference-experiments.md#host-validation-of-the-initial-prototypes)
distinguishes the numerical full-suite failure from isolated exit-255 failures.
