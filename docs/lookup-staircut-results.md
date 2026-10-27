# Lookup / StairCut: independent-replacement results

## Latest status and evidence boundary

The latest v30 live-history cohorts show substantial observed repeated-workload gains: combined writable features versus native improve **MTP +28.31% / DFlash2 +6.13%** overall; holding adaptive settings fixed, writable versus empty read-only history improves **+32.24% / +7.84%**. Each contrast completed 136 serial-exact responses. First-use regressions remain, and every cohort fails unchanged-system-swap qualification. These are different contrasts, not additive gains or universal/default-promotion claims. See [native-relative live history](#live-completed-response-reuse-mtp-and-dflash2), [isolated MTP history](#isolated-net-live-history-value-mtp), [isolated DFlash2 history](#isolated-net-live-history-value-dflash2), and the [executed regression diagnostic](#executed-mtp-trajectories-admission-delay-not-a-proven-missing-incumbent). Earlier failed and diagnostic cohorts remain historical evidence. No current-build standard llmprobe performance claim is established.

- **v30 measured binary:** `81911662b8213edbbc624d196615c4f188b2607e7c4cda1ba1c431cc4f34dcb1`; other sections identify their own binaries.
- Target: `ddalcu/Qwen3.8-27B-MLX-Serve-4bit`; assistant: `incoai/Qwen3.8-27B-DFlash2`.
- Every native/feature arm uses explicit **row-exact numerical mode**. These are numerical-mode-controlled references, not unchanged fastest defaults.
- Each section identifies its measured binary and protocol, not subsequent edits. Full receipt-enabled MTP diagnostics can additionally settle native GPU work before successor resolution; these are not instrumentation-free throughput measurements. Native-depth traces themselves perform no GPU evaluation/readback, but their CPU/I/O cost can enter later wall-clock prices.
- Initial replacement authors were source-excluded; the integrating parent previously saw the retired implementation. Neither clean-room nor legal certification is claimed. See [provenance, attribution and implementation contracts](lookup-staircut-port.md).

Paths below are relative to `~/.mlx-serve/bench/lookup-staircut/`. These are private artifacts, not bundled public datasets. Raw logs may contain credentials and reversible token data; do not publish them without redaction. Procedures and gates are in [the offline guide](lookup-staircut-offline.md).

## Historical v6: four complete workload windows

Binary: `f4121e1f222d5e86413e86e4d52b138becff0a1ad52888961ce67b0bf2672de5`. This revision preserved substantial MTP gains and improved DFlash related-configuration decode, but MTP first-exposure prose remained slower and DFlash short-code request latency was mixed. These regressions and failed promotion gates describe v6, not the latest ABBA cohorts.

Each window completed **24 exact-valid requests: 96 total, zero invalid samples**. Each has two paired boot blocks, both native families plus the persistent feature, two workloads, and first exposure plus a repeat. The boot block—not every request—is the independent unit.

| Evidence directory | Observed outer command wall time (s) | Additional exclusions beyond undeclared GPU isolation |
|---|---:|---|
| `runs/independent-v6-mtp-related-20261025` | 137.26 | Changed swap-ins |
| `runs/independent-v6-mtp-controls-20261025` | 173.62 | None listed |
| `runs/independent-v6-dflash-related-20261025` | 143.58 | Changed swap-ins |
| `runs/independent-v6-dflash-controls-20261025` | 179.51 | Changed swap usage and swap-outs |

All are **diagnostic, not confirmation**: two pairs, GPU isolation undeclared, confirmation/promotion gates false. Memory activity further confounds the indicated windows. There is no completed qualified current-binary canonical result. Power was AC; that does not establish stable GPU frequency or exclusive inference. Unprivileged GPU telemetry exposes utilization/memory but not the needed frequency trace; `powermetrics` requires superuser access, which was not obtained.

The frozen seed contains four completed training prompt/output documents, not evaluation answers. It was constructed and format-validated from published native outputs rather than through the production append path. First exposure is an unseen request against existing history; repeat can benefit from learned state and is not another independent held-out request. First exposure is not model-load latency. Exact publication agreement covers the exercised requests, not all inputs or full internal-state identity.

### Timed repeats: request latency and decode are different quantities

Values are medians of two observations. Latency reduction is `100 × (1 − feature seconds / native seconds)`; decode gain is `100 × (feature tok/s / native tok/s − 1)`. Positive means improvement. Decode is the server-reported decode interval, not whole-request throughput or a canonical benchmark rate; its source values are rounded to one decimal. Request latency includes prefill and other request work. Ratios of medians differ from medians of within-pair ratios.

| Family | Workload | Native (s) | Feature (s) | Latency reduction | Native decode (tok/s) | Feature decode (tok/s) | Decode gain |
|---|---|---:|---:|---:|---:|---:|---:|
| MTP | Related configuration | 7.44746 | 4.39108 | 41.04% | 24.10 | 50.95 | 111.41% |
| MTP | Related code | 4.40596 | 3.37735 | 23.35% | 22.50 | 33.20 | 47.56% |
| MTP | Conflicting configuration | 7.47024 | 4.88404 | 34.62% | 24.15 | 42.80 | 77.23% |
| MTP | Unrelated prose | 6.63915 | 6.70144 | -0.94% | 20.95 | 20.65 | -1.43% |
| DFlash | Related configuration | 4.81145 | 3.21329 | 33.22% | 50.80 | 82.20 | 61.81% |
| DFlash | Related code | 2.90600 | 2.85628 | 1.71% | 52.15 | 58.55 | 12.27% |
| DFlash | Conflicting configuration | 4.52234 | 4.10120 | 9.31% | 55.40 | 59.95 | 8.21% |
| DFlash | Unrelated prose | 6.33570 | 6.30422 | 0.50% | 23.30 | 23.50 | 0.86% |

Within-pair repeat latency reductions are MTP configuration **39.96% / 42.07%**, code **21.30% / 25.22%**, conflict **31.32% / 37.68%**, prose **−4.58% / 2.43%**. DFlash configuration is **22.86% / 41.67%**; code is **−10.50% / 11.44%**. These variations matter more than extra displayed decimal places.

DFlash code illustrates why the metrics must remain separate: both repeat decodes improve (55.5→58.6 and 48.8→58.5 tok/s), but prefill differs substantially (82.8→66.0 and 60.6→65.4 tok/s). One whole request therefore regresses despite lower summed verification time. A decode-component benefit does not waive the end-to-end gate.

The DFlash runtime did not receive a new optimization between v5 and v6; v5's related-configuration repeat latency reduction was 9.91%, versus 33.22% here. Adaptive width choices, environmental variation and sampling differ. The larger number must not be attributed to the MTP recovery fix or promised as a stable speedup.

### First exposure

Same latency aggregation and sign convention, with individual paired effects retained:

| Family | Workload | Native (s) | Feature (s) | Latency reduction | Pair 0 / pair 1 reductions |
|---|---|---:|---:|---:|---:|
| MTP | Related configuration | 6.81301 | 3.66207 | 46.25% | 43.88% / 48.18% |
| MTP | Related code | 4.36639 | 3.37453 | 22.72% | 18.81% / 26.26% |
| MTP | Conflicting configuration | 6.89731 | 4.17424 | 39.48% | 33.17% / 44.44% |
| MTP | Unrelated prose | 6.69624 | 6.95926 | -3.93% | -9.25% / 0.86% |
| DFlash | Related configuration | 3.85223 | 3.37653 | 12.35% | 15.78% / 8.34% |
| DFlash | Related code | 3.05005 | 2.75623 | 9.63% | -8.66% / 23.19% |
| DFlash | Conflicting configuration | 4.20507 | 3.04065 | 27.69% | 40.20% / 6.63% |
| DFlash | Unrelated prose | 6.44817 | 6.40990 | 0.59% | 2.47% / -1.38% |

MTP's first prose exposure violates the predeclared 5% regression allowance in one pair. Excluding a first execution from learned prices never excludes its user-visible latency. Removing bad lookup admissions does not guarantee free native-controller adaptation.

### Same-family gain is not fastest-reference gain

Native DFlash remains faster than MTP plus lookup on all four current repeat workload medians in the MTP windows. MTP plus lookup is 4.3911 versus native DFlash 4.3264 seconds for configuration, 3.3773 versus 2.5935 for code, 4.8840 versus 4.0488 for conflict, and 6.7014 versus 5.5725 for prose. Large gains over native MTP do not establish the fastest available decoder. These are within-window contrasts, not comparisons between unrelated windows.

### Useful work and remaining native-depth behavior

Slash-separated values are pair 0 / pair 1, for repeat requests:

| DFlash workload / arm | Rounds | Verified rows | Committed tokens | Summed verify/commit (ms) |
|---|---:|---:|---:|---:|
| Related configuration / native | 18 / 18 | 144 / 144 | 142 / 142 | 2427.2 / 2639.1 |
| Related configuration / feature | 11 / 11 | 144 / 144 | 142 / 142 | 1640.5 / 1549.7 |
| Related code / native | 9 / 9 | 72 / 72 | 70 / 70 | 1140.1 / 1302.7 |
| Related code / feature | 7 / 7 | 80 / 80 | 70 / 70 | 1098.4 / 1088.8 |
| Unrelated prose / native | 34 / 34 | 272 / 272 | 124 / 124 | 4993.8 / 4660.2 |
| Unrelated prose / feature | 34 / 34 | 272 / 272 | 124 / 124 | 4889.3 / 4676.3 |

Configuration reduces target rounds without increasing verified rows. Code trades more rows for fewer rounds; whether the trade pays depends on actual costs. Prose now performs native-equivalent verification work and no lookup-only rounds on all four feature exposures. V4 added rows and rounds for the same prose output. These are executed-work observations, not causal cross-version timing estimates.

Native-depth tracing exposed a concrete accounting defect in the preceding binary `929b60a1ef18ec326bc76e652d51b3a44c99b74a7a6016309730e7bf14da66db`, in `runs/independent-v5trace-mtp-controls-20261025`: first conflicting requests executed 7 / 8 native rounds, but final planning counters contained only 3 / 4 acceptance observations. Timing-ineligible head recovery discarded real native acceptance and then another clean timing interval.

V6 preserves that acceptance, still excludes recovery duration, and starts the next exact-depth clock at completed solo recovery. Deterministic tests prove that a 1,000 ms recovery cannot become the following 40 ms price, and that contended recovery cannot seed a solo price. Runtime traces show the previously dropped recovery evidence retained. Lookup outcomes are never invented as MTP-head successes.

This is not a complete cold-performance cure: v6 first prose exposures use **53 / 51 native rounds**, versus **48 / 51** in their baselines; repeats use **48 / 48** in both arms. No lookup-only prose rounds occur. Sparse native width prices remain visible—for example, one boot carries a 147.3 ms two-draft price into the next request. Altered native-depth behavior coincides with the remaining slowdown; its causal contribution is not isolated from environmental drift. Do not tune a fixture-specific width or prior to conceal it.

### Real target-price bootstrap

In both v5 MTP related boots, the existing two calibration opportunities executed two-row native chains. Head-settlement barriers took 2.86–6.39 ms. First target executions (113.40–122.70 ms) were excluded as cold; second observations (71.43–71.50 ms) learned the actual two-row target price. This is v5 mechanism evidence, not a v6 timing claim. The v6 control log also exercises target-price learning. Native remaining-duration timing includes the barrier; no additional model forward is introduced. Different width sequences can leave bootstrap incomplete, and another process's contention remains undetected.

## Current same-state verification experiment

Evidence: `runs/independent-v6-cf-serial-20261025`, `runs/independent-v6-cf-{assistant8,assistant16,lookup8,lookup16}-20261025`, and `runs/independent-v6-cf-comparison-20261025.json`. Each of five independently bounded windows completed two requests; outer wall times were 55.29, 69.23, 68.29, 67.15 and 68.29 seconds respectively.

Both serial requests produced the same 142-token answer. Every action's warmup and measurement matched complete serial publication. All four actions passed common prefix/KV/SSM/assistant/learner identity, including request-local trust. Each same-preparation width pair also matched complete offered candidates/probabilities and prepared state. Strict authentication and the exact third-request refusal were exercised. These are ten correctness/diagnostic requests, separate from the 96 workload requests above.

| First-offer action | Rows | Committed tokens | Verify + commit (ms) | ms / committed token |
|---|---:|---:|---:|---:|
| assistant8 | 8 | 6 | 98.833 | 16.472 |
| assistant16 | 16 | 10 | 136.353 | 13.635 |
| lookup8 | 8 | 7 | 97.060 | 13.866 |
| lookup16 | 16 | 7 | 153.723 | 21.960 |

At this predeclared first offer, widening assistant+lookup reduces verification/commit ms per committed token **17.22%**. Widening lookup-only increases it **58.38%**, because no additional tokens commit—even with the linear-GDN specialization. Nominal width and fewer calls are not sufficient optimization targets.

These are **readback-perturbed verification/commit measurements**, one measured offer per action, not replicated throughput evidence. Readback took 119.9–125.6 ms; subtraction cannot reconstruct unperturbed execution. Full-round timing is null. Assistant/lookup actions deliberately use different candidate sets; only within-preparation width contrasts hold the offer identical. First-offer evidence also does not establish optimal routing after request-local trust has learned later in a request.

## Why full-workload gates still fail

CPU aggregates are `runs/independent-v6-{mtp,dflash}-full-20261025/aggregate-summary.json`. All four cases are covered, and individual windows completed with zero invalid outputs. Aggregate `complete=false` reflects its fail-closed requirement for speed-comparison-valid windows, **not missing generated requests**. The recorded reason is `incomplete_or_invalid_window` because the timing windows are unqualified.

- Every window lacks declared exclusive GPU inference; several also change swap counters. Two pairs are diagnostic, not the four qualified balanced pairs required for confirmation.
- MTP cold prose includes a **9.25% regression**, exceeding the per-pair 5% allowance.
- DFlash short code includes **8.66% cold and 10.50% repeat regressions** in one pair, despite decode-component gains. It does not meet the per-pair related-workload improvement requirement.
- Native DFlash remains a stronger reference than MTP on these workloads; passing a same-family comparison would not automatically pass fastest-native comparison.
- No complete qualified standard canonical protocol exists. The v2 attempt below timed out; a successful subset cannot replace it.

No gate was weakened, invalid sample removed, or diagnostic result relabeled as confirmation. Keep the feature opt-in.

## Corpus and production-store correctness

`runs/independent-v6-{mtp,dflash}-corpus-oracle-20261025` each completed **eight exact-valid requests** against a current-binary serial oracle, including both native families and the corpus feature. Outer wall times were 79.43 and 69.24 seconds. Corpus nodes were actually selected and accepted: the conservative accepted-source lower bounds were 119 / 111 tokens for MTP first/repeat and 111 / 111 for DFlash. These are correctness-only windows, not performance evidence.

`runs/independent-v6-production-store-20261025` completed three requests in 18.70 seconds. An ordinary MTP request wrote a new private store through the production publication path. Its exact file payload matched the published prompt/output IDs and document separator, with the correct tokenizer header. Separate read-only MTP and DFlash boots then retrieved a unique known continuation: each published the expected eight tokens and accepted seven persistent-source drafts. The store hash stayed unchanged and its lock was released after every boot. Retrieval deliberately forced maximum selection to exercise the mechanism; it is not unseen-request generalization or a routing/performance claim.

Together with the workload and same-state windows, these are **125 current-binary runtime requests passing their scoped correctness checks**. The small production-store smoke does not replace the separate malformed-store, locking, trimming and rollback unit coverage or establish every storage failure path.

## Checks and adversarial corrections

Current recovery revision: **106 integrated lookup passes with four skips**, **ReleaseFast build 8/8**, and the model windows above. Earlier v5 checks of unchanged policy/helper components: **63 standalone adapter passes**, **10 integrated counterfactual passes**, **121 Python passes**. These tests do not certify statistical performance or unexercised architectures.

Review-driven corrections include zero support before a reached request-local hit; preserved conditional continuation evidence behind a neural parent; no redistribution of excluded sibling mass; owned native/lookup offers; no duplicate selected/off-policy learning; retained selected MTP edges omitted by the bounded offer; required request-trust diagnostic identity; and recovery acceptance separated from timing eligibility. Verified outcomes beyond publication EOS remain conditional evidence, not realized generation reward. Unsupported Qwen4 head state is excluded from target-price bootstrap. Bounded exploration can still leave a width unpriced after invalid/unmatched opportunities.

## Earlier independent revisions

Historical observations are binary-specific and are not causal code-change estimates:

| Revision | Evidence and result | Lesson |
|---|---|---|
| v2: `e8e18837a5e041c163c60bf1f16584a0af620393a88875f350207f2734d2c57d` | Four `runs/independent-v2-{mtp,dflash}-{related,controls}-20261025` windows; 96 exact-valid. Related repeat median-ratio reductions: MTP configuration/code 45.72%/37.25%, DFlash 7.60%/3.20%. Prose repeat median-paired reductions: MTP −12.45%, DFlash −12.82%. | Reuse gains did not excuse miss regressions. |
| v3: `d6912bcb51285098da008a5b8e4dce4c5ae0005c988e12d6b8e8d3089578eb92` | Three `runs/independent-v3-{mtp-controls,dflash-controls,mtp-related}-20261025` windows; 72 exact-valid. MTP related configuration/code repeat reductions 0.28%/−0.58%; prose repeat MTP −2.46%, DFlash −9.17%. | Exact-order conditioning exposed wide-price starvation. |
| v4: `e4a680da38b1817c81f58cb465861126efb60db5d04220fe6316c6516829f37e` | Four matching v4 windows; 96 exact-valid. Related repeat reductions: MTP 48.82%/34.01%, DFlash 11.17%/0.81%. Prose first/repeat: MTP −8.07%/−0.89%, DFlash −18.13%/−10.82%. | Wider exploration restored reuse, but pooled confidence admitted harmful prose bypasses. |
| v5: `339a44f073be6af1b008f79bfce0788361e9cbf98598b935dff217ba8b2d0867` | Four matching v5 windows; 96 exact-valid. Related repeat reductions: MTP 41.24%/24.75%, DFlash 9.91%/−0.78%. Prose first/repeat: MTP −6.49%/0.67%, DFlash −1.02%/4.06%. | Request-local trust removed exercised prose bypasses; native recovery accounting still discarded useful evidence. |

The older v1 same-state experiment (`runs/independent-cf-*`, binary `51110940f7e9e6dddbf669f489df428f963a441dcc2c48495e71ad25f3c8c668`) remains historical. Its receipts lack the now-required request-trust identity and are not current-protocol evidence. They were not upgraded or relabeled.

## Bounded confirmation runner smoke

The new confirmation wrapper freezes four pair slots before execution; each separately launched slot keeps all four workloads within each native/feature boot. Runtime code and the v6 binary are unchanged.

`runs/independent-v6-mtp-cohort-20261025/pair-0` completed **24 exact-valid requests in 137.81 seconds**, with zero invalid samples. It adds to the previous 125 scoped runtime checks: **149 current-binary requests checked in total**. The full Python helper suite passed 137 tests, including 16 new schedule/declaration/aggregation regressions. Adversarial review found no blocking wrapper issue.

An attempted second launch of slot 0 was refused before inference. CPU aggregation recomputed its sealed evidence without changing artifact hashes and correctly failed with missing slots `[1, 2, 3]`; no four-pair confirmation is claimed. Isolation was undeclared. The remaining slots were not automatically launched.

This new within-boot workload sequence also exposes a remaining regression: MTP first/repeat prose latency was **10.37% / 5.94% worse** than native MTP. First exposure used 55 native rounds versus 49; repeats used 48 each. Related configuration first/repeat latency improved 37.45% / 41.31%, and related code 19.87% / 23.97%. These are single-pair descriptive effects, not replacements for the two-pair tables or proof that the regressions are resolved.

## Storage portability follow-up

Remote macOS Zig/Swift and Linux CUDA x64/arm64 checks passed on `fbc59b0f`. Windows compilation found POSIX-only permission conversions in the history store. The follow-up uses the host's inherited-ACL convention on Windows, preserves Windows attributes, and leaves POSIX `0700`/`0600` enforcement intact. Static adversarial review found no new blocking issue; Windows requires a protected parent ACL and is not certified by a compile check.

After the fix, the history module and its tests cross-compiled for `x86_64-windows-gnu`; all **20 native history tests passed**. A standalone macOS program created a real store, appended a document, reopened it read-only and verified exact persisted IDs plus directory/payload/lock modes **0700/0600/0600**. The temporary program was removed. This storage-only follow-up has no new model-throughput measurement; the v6 performance tables retain their original binary identity.

## Separate v2 canonical timeout

`runs/independent-v2-canonical-mtp-0` failed after 252.78 seconds when llmprobe reached its **240-second subprocess timeout**, not the outer 290-second bound. The full 34-request protocol remained incomplete: zero accepted summary samples, one invalid sample. Logs reached three decode runs, prefill warmup and the first of three prefill runs. Partial rates are not promoted. The protocol was not silently shortened to obtain a pass. This is not a current-binary canonical measurement.

## Where a standard benchmark can expose the benefit

| Workload characteristic | Expected mechanism or limit |
|---|---|
| Unseen configuration/code requests sharing frozen fragments | History proposes reusable continuations; target verification still checks the accepted path. Held-out requests, not exact answer replay, exercise cross-request transfer. |
| MTP with long accurately predicted continuations | Batching useful target verification and avoiding sequential head work can substantially beat native MTP. It need not beat a stronger DFlash assistant. |
| DFlash already producing nearly full accepted blocks | Extra committed tokens must outweigh wider verification/preparation. Short outputs leave little room to amortize learning, and prefill can dominate total request latency. |
| Unrelated prose or conflicting successors | Local evidence restricts unsupported lookup. Fallback still has querying, setup and controller-adaptation costs; zero overhead is not guaranteed. |
| Prefill-heavy, long-context or concurrent protocols | This feature targets speculative decode, not prompt prefill. Contexts need their own prices; scheduler-known contention excludes price learning. Short solo reuse gains do not establish these rates. |

A standard suite needs an unchanged, completed decode protocol containing genuinely reusable or locally repetitive continuations to expose this mechanism. Training must be independent of evaluation answers. Prefill/batch aggregates can dilute a decode gain; omitting those stages and still calling the result the full benchmark is invalid.

## Promotion boundary

Further performance qualification needs exclusive-inference measurement, stable machine evidence, broader held-out/conflict/miss coverage, and at least four qualified balanced boots under the existing gates. Investigate sparse/drifting native width prices with actual measurements rather than a fixture-specific prior or forced width. Same-family benefits must survive fastest-native comparison before making that stronger claim. Canonical throughput needs its own complete qualified run.

Warm history is not unseen-request generalization; fewer rounds do not imply proportional savings; selected-action price error is not regret over unmeasured alternatives. No measured Flash-Next improvement or cross-architecture transfer follows from these dense-27B results.

## Variant-correct pricing and component validation (v7)

Runtime SHA-256: `de3fd40f31143f0da899a8f8e5e616787ec91a36962ee400243063690e3d14d0`. ReleaseFast build succeeded. Focused lookup integration passed **109 tests with four skipped**; PLD-focused checks passed **eight**. The price model no longer pools generic-tree and linear-GDN costs. Schema-three profiles, actual-prefix classification and variant-sensitive assistant routing have focused regressions.

`runs/independent-v7-pld-smoke` completed **30 serial/PLD-off/PLD-on requests with identical published token IDs, finish reasons and HTTP output/usage**. It covered the four held-out workloads and output caps 1, 2, 3, 5, 9 and 17. Actual backoff windows 1, 2, 4 and 8 were observed. Review found and fixed skipped-cold-step pricing against a stale context class. This sequential correctness smoke is **not a performance confirmation**; some PLD-on requests were slower, so backoff is not presumed to improve complete workloads.

### Real Flash-Next learned-table costs

The staged TensorFold Flash-Next pack uses affine `ngram_embedding.shard_N` tensors. The loader now recognizes that spelling without conversion or changed dequantization. The CPU-only component benchmark passed **9,216 exact serial/pooled vector comparisons and 1,024 exact production-path checks per run**; two whole runs completed. Five focused loader/benchmark tests passed. Benchmark executable SHA-256: `d38e10455139e9ed728b3413f13e8da135d13a7d063728b40ccfa57e68722b65`.

First run, mean hash-plus-gather time in milliseconds:

| Token batch | First-access serial | First-access pooled | Pooled latency reduction | Warm serial | Warm pooled | Warm pooled slowdown |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4.1533 | 0.4123 | 90.07% | 0.00476 | 0.21462 | 45.06× |
| 2 | 5.7922 | 0.5419 | 90.64% | 0.00991 | 0.24280 | 24.51× |
| 4 | 6.3869 | 0.5335 | 91.65% | 0.01919 | 0.25112 | 13.08× |
| 8 | 7.3659 | 0.9113 | 87.63% | 0.03863 | 0.49273 | 12.76× |

These compare existing implementations, **not a newly accelerated target model**. First access has uncontrolled OS cache, not proven disk-cold storage. Warm pairs repeat the same context; confidence intervals resample whole context blocks. The separate repeat run retained the same direction in every batch. Production used pooled reads for batches 1–4 in this component setup and serial reads for batch 8; forced pooling at batch 8 is diagnostic. Startup full-model calibration and memory pressure were absent.

The result exposes a real tradeoff: pooled reads help first-access faults but waste hundreds of microseconds on already-hot rows. Do not force either arm globally from this microbenchmark. Full-target weights can evict table pages, so a “warm table” assumption needs validation under the actual model's memory footprint. No full Flash-Next inference speedup, neural-drafter speedup, or additive contribution to suffix-lookup gains is established here. Artifacts: `runs/independent-v7-ple-component.json` and `runs/independent-v7-ple-component-reuse.json`.

### Full MTP component cohort: useful signal, failed qualification

`runs/independent-v7-mtp-lookup` completed all **96 requests**, with no invalid samples, under declared exclusive inference. Four balanced blocks compared native MTP, lookup-chain cut without adaptive native fallback, and adaptive lookup/native routing. First and repeated requests both count toward complete-workload latency.

| Comparison against native MTP | Geometric mean speed ratio | Speed change | 95% block-bootstrap interval |
|---|---:|---:|---:|
| Adaptive lookup/native routing, full workload | 1.321× | +32.1% | 1.056–1.739× |
| Forced lookup-chain cut control, full workload | 0.622× | −37.8% | 0.553–0.714× |
| Adaptive routing, unrelated prose first exposure | 0.957× | −4.3% | Not a qualified performance claim |
| Adaptive routing, unrelated prose repeat | 0.957× | −4.3% | Not a qualified performance claim |

These are **descriptive observations, not confirmed speedups**. Swap usage, swap-ins and swap-outs changed across the cohort; the speed-comparison validity gate failed. Some individual workload/block regressions also exceeded the allowance. Native prose decode varied from approximately 21 to 11 to 14 to 19 tokens/s across blocks despite similar accepted-token counts. Prefill slowed alongside decode, so controller behavior alone does not explain the drift. Boot-boundary machine snapshots cannot attribute paging specifically to measured requests rather than loading.

The forced cut control is not the production adaptive router: on prose it executed root-only verification rather than retaining native MTP. Its loss establishes the importance of the native fallback, not a neural-only MTP StairCut result. Separately, one adaptive prose block used 66 native attempts versus 48 for its baseline; that work-count difference needs investigation even if machine drift is resolved. No samples were discarded or old gates relaxed.

### DFlash neural-width cohort: the intended contrast did not activate

`runs/independent-v7-dflash-neural-width` completed **128 exact-valid requests**, zero invalid. Native, tiered fixed-eight, maximum and dynamic-cut arms each executed **632 rounds at eight rows**, accepting **3,192 drafts** and publishing **3,824 tokens**. Cut never reduced a tree. Thus this cohort does not test the advertised adaptive-width benefit, even though the requests and comparisons completed correctly.

| Full-workload contrast | Descriptive speed ratio | Speed change | 95% block-bootstrap interval |
|---|---:|---:|---:|
| Cut versus native DFlash | 1.004× | +0.4% | 0.937–1.058× |
| Cut versus tiered fixed-eight | 1.006× | +0.6% | 0.980–1.032× |
| Cut versus tiered maximum | 1.015× | +1.5% | 0.991–1.043× |

All contrasts are inconclusive; swap guards also failed. On this non-NAX machine, the inherited neural proposal heuristic limited the candidate pool to seven drafts before cost selection. Missing smaller-width prices further favored the already-measured eight-row execution. The next revision separates the bounded candidate pool from the native execution-width heuristic and requires actual training-only width prices. Candidate expansion is not itself a speedup: native versus new tiered fixed-eight will include a proposal/geometry change, not only width selection.

A passive `macmon` capture began partway through this cohort and continued into idle time: `passive-gpu-tail.jsonl`, 300 samples at approximately two-second intervals. Among its 159 samples with at least 90% GPU activity, clock frequency ranged **599–1,378 MHz** (median **843 MHz**) and GPU temperature ranged **76.6–86.2°C**. This establishes substantial operating-condition variation, not a thermal diagnosis or complete per-request coverage. It cannot retrospectively qualify this cohort or explain the earlier MTP cohort.

## Request-local timing and wider neural candidates (v9 smoke)

Runtime SHA-256: `7f2d50387995c6968e0427c1e257343d042c938f5ea56ad900047f73e5fb26cc`. ReleaseFast build completed **8/8 steps**. The applied MTP timing/empty-offer and DFlash candidate fixes passed static adversarial review. Before the latest protocol corrections, the complete Python helper suite passed **237 tests**; the subsequent focused counterfactual/telemetry/component suites passed **32/15/34** tests.

`runs/independent-v9-runtime-smoke` completed **32 requests**: serial, native MTP, adaptive MTP lookup and neural-only maximum-width DFlash, each exercising the four full held-out cases with first/repeat exposure. All published token IDs, text, reasoning, usage and finish reasons matched current serial output. **148 DFlash rounds actually verified more than eight rows**, all with the expected generic branching verifier. Owned passive telemetry captured all request intervals and passed its endpoint/coverage checks; mean pre/post endpoint collection cost was **86 ms per request**, outside recorded HTTP latency.

| Workload | Native MTP mean HTTP seconds | Adaptive MTP mean HTTP seconds | Descriptive speed gain |
|---|---:|---:|---:|
| Related configuration | 7.735 | 5.538 | +39.68% |
| Conflicting configuration | 8.212 | 5.697 | +44.13% |
| Related code | 4.962 | 3.911 | +26.87% |
| Unrelated prose | 8.471 | 7.539 | +12.37% |

Total measured HTTP time was **58.760 s native versus 45.370 s adaptive**, a descriptive **1.295×** ratio. This is a **single sequential smoke**, not balanced confirmation, a confidence interval, or proof that the code caused every observed gain. Instrumentation gaps are excluded from the summed HTTP durations, so this is not uninterrupted offered-load throughput. DFlash had no matched native arm in this smoke and receives no speedup claim. The standalone PLD cold-start exploration fix was not present in this binary.

## Standalone persistent suffix activation (v10 smoke)

Runtime SHA-256: `1f334b2673c428c2fbc216884146a7d0cb2cd20902521f730e1ce9eb0db85e1b`. All **30 policy tests** passed, including bounded standalone width learning; ReleaseFast build completed **8/8 steps**. Static adversarial review found no actionable issue in that patch.

`runs/independent-v10-pld-smoke` completed **24 exact requests**: current serial, frozen persistent-history PLD with backoff off, then on; four full held-out cases with first/repeat exposure. Telemetry coverage and endpoint qualification passed. Off executed **190 wider-than-root rounds and accepted 565 drafts**; on executed **151 and accepted 548**. Each PLD boot reserved the same four-row learning shape twice in context class zero, with no further reservation. This proves real activation and bounded reservations, not a performance confirmation.

| Workload | Serial mean HTTP seconds | PLD backoff off | PLD backoff on | Off speed change vs serial | On speed change vs serial |
|---|---:|---:|---:|---:|---:|
| Related configuration | 10.890 | 4.697 | 6.049 | +131.8% | +80.0% |
| Conflicting configuration | 11.690 | 5.520 | 6.924 | +111.8% | +68.8% |
| Related code | 6.458 | 3.387 | 4.071 | +90.7% | +58.6% |
| Unrelated prose | 9.990 | 12.429 | 12.416 | −19.6% | −19.5% |

Speed change is baseline time / treatment time − 1. Sequential arm order prevents causal attribution. For example, the first related-configuration request executed 27 rounds with 115 accepted drafts in both PLD arms, yet summed verification/commit time was 2.961 s off versus 4.166 s on. Machine operating conditions or other timing effects must be separated before blaming backoff for the wall-time difference.

Prose remains a real performance concern: backoff reduced lookup-verification rounds from 118/121 to 20/20, but full-request latency remained above serial. Reducing queries alone is insufficient. The earlier v7 PLD smoke used **local-only history**, whereas this run used the frozen persistent seed; their acceptance difference cannot be attributed solely to the width-learning patch. Neither this smoke nor its comparison with v7 establishes a balanced speedup, a native-MTP/DFlash win, or an isolated contribution from bounded exploration.

The separate `runs/independent-v10-pld-local-smoke` then exercised **24 exact requests without external history**. Off accepted **221 drafts in 277 wider rounds**; on accepted **143 in 140**. Telemetry qualification passed. This directly establishes that local-only lookup no longer stays root-only, while the sequential timings still do not establish a performance win. Static review identified an opportunity to enter the native pipeline earlier on an absent proposal; that optimization was not present in either v10 smoke.

### Calibration and larger-model checks

The current-binary training serial oracle and `lookup16` diagnostic completed in `runs/independent-v10-cf-training-{serial,lookup16}`. The measured receipt correctly named **tree / linear-GDN / 16 rows**, committed 16 tokens with 15 accepted drafts, and recorded **173.393 ms verification/commit** separately from **138.061 ms diagnostic readback**. Full-round time remained unavailable. The actual exporter CLI successfully produced one training price cell; subsequent review found missing table-to-sidecar binding and consumer-contract checks, so that historical export is not sufficient for qualified component calibration.

A stock-serial Flash-Next launch with context 512 was attempted with normal memory guards intact: `runs/independent-v10-flashnext-load`. The preflight refused loading at **48.04 GB estimated weights, about 55.0 GB required free including headroom, and 42.14 GB available**. It exited before inference. This establishes a current available-memory blocker, not architectural impossibility or a learned-table/full-model performance result.

## Six-permutation MTP cohort (v10b)

`runs/independent-v10b-mtp-lookup` completed **144 exact-valid requests, zero invalid**, using the v10 binary above. All six permutations of native MTP, forced lookup cut and adaptive lookup/MTP ran the complete four-case first/repeat workload. The production comparison is adaptive versus native; forced cut remains a separate control.

| Adaptive versus native MTP | First-request speed change | Repeat-request speed change |
|---|---:|---:|
| Related configuration | +88.5% | +72.6% |
| Conflicting configuration | +57.1% | +53.0% |
| Related code | +28.4% | +26.3% |
| Unrelated prose | −3.7% | −0.7% |

The full-workload geometric mean speed ratio was **1.343×**, with descriptive unadjusted block-bootstrap 95% interval **1.292–1.383×**. Every block's aggregate favored adaptive routing. Frozen-history accepted-output witnesses appeared in every required related cell. Forced cut remained much slower (**0.585× native**) and never accepted persistent candidates; it is not the production router.

Completed target-verification work, averaged over each case's 12 requests per arm:

| Workload | Native rounds/request | Adaptive rounds/request | Round reduction | Native verified rows/request | Adaptive verified rows/request |
|---|---:|---:|---:|---:|---:|
| Related configuration | 49.58 | 19.83 | 60.0% | 144.17 | 165.08 |
| Conflicting configuration | 48.00 | 23.08 | 51.9% | 143.42 | 186.75 |
| Related code | 25.42 | 16.58 | 34.8% | 75.42 | 86.92 |
| Unrelated prose | 48.58 | 49.08 | −1.0% | 144.58 | 144.25 |

Published outputs were identical within each case. This establishes fewer completed verification rounds on reusable workloads, **not proportionally less GPU work**: wider blocks verified more total rows there. It explains why hardware-specific width costs and full-request timing remain necessary.

**Full qualification failed.** Swap usage/counters changed, and some individual prose/block ratios violated the five-percent regression guard. Request-aligned evidence locates paging changes in three requests: block 3 forced-cut conflicting-config first exposure (four swap-ins), block 5 native related-config first exposure (4,804 swap-outs and approximately 78.7 MB increased swap usage), and block 5 native conflicting-config repeat (eight swap-ins). No adaptive request interval changed those counters. That attribution does not erase loading effects, certify other intervals, or justify excluding the affected baseline requests.

Most native/adaptive prose requests used 48–49 verification rounds. Block 5 adaptive first prose instead used **56 rounds**—21 at two rows and 35 at three—versus its native reference's 48 rounds. Its inherited alternate-depth cost remained unrefreshed early in the request. Block 1 also had a larger wall-time loss without a comparable work-count change. Controller adaptation and operating-condition variation therefore remain separate investigations. Neither failed evidence nor its thresholds were rewritten; these observations are not a qualified blanket speedup or canonical benchmark result.

## Native-pipeline miss entry (v11 correctness smoke)

Runtime SHA-256: `b15f7f708db73ded79daf41ab8586b2ec04bfc85910fc0f569dc5989b3eab6f5`. The integrated Python helper suite passed **257 tests** and ReleaseFast build completed **8/8 steps**. Static adversarial review found two receipt-contract mismatches in the first native-entry implementation; both were corrected before this build by retaining the configured selection label and removing a redundant backoff event.

`runs/independent-v11-pld-smoke` completed **42 exact-current-serial requests** across serial, persistent PLD off/on, the full four-case first/repeat workload, and output caps **1, 2, 3, 5, 9 and 17**. It observed **36 native-entry rounds**. Lookup-round publication plus skipped native/drain/cold emissions reconciled exactly to published output, and all backoff state transitions validated. Native-entry receipts remain explicitly **unsettled and unpriced**, with no synthetic verification duration or extra settle barrier.

This is correctness/activation evidence, not a measured speedup. Passive telemetry did not qualify the entire smoke because some subsecond boundary requests contained no one-second sample. Prose latency was still variable; the sequential arm order cannot isolate the optimization's contribution. The later bounded MTP alternate-refresh change is not present in this binary.

## Bounded MTP price refresh (v12 diagnostic)

Runtime SHA-256: `08f34039f98ff502affa2f0d5892d2255e6d6e1c0a37032450d798a14ef3a5cd`. ReleaseFast build completed **8/8 steps** after review corrected the refresh eligibility budget to account for an unpublished lazy predecessor.

`runs/independent-v12-mtp-refresh-diagnostic` completed **72 exact-current-serial requests**: eight serial references followed by two reverse-order blocks of old/new native and old/new adaptive MTP. Every boot retained the six configuration/code requests before prose. The new controller issued **23 refresh plans**, never more than one per request. This establishes bounded activation and output correctness, not improved economics.

| Block | Old native HTTP seconds | New native | Old adaptive | New adaptive |
|---|---:|---:|---:|---:|
| Forward arm order | 59.057 | 66.285 | 47.632 | 53.663 |
| Reverse arm order | 77.043 | 82.522 | 59.167 | 57.263 |

The new adaptive controller executed **437 completed verification rounds versus 419** for the old controller. Across both blocks, code increased from 60 to 73 rounds and prose from 193 to 204. Exact outputs matched, so this is a real work-count difference, although differing online timing histories also change the selected widths. The elapsed-time comparison is inconclusive and the telemetry gate failed on changed swap usage, swap-ins and swap-outs. The refresh is **not a demonstrated performance fix**; it must be investigated or removed rather than promoted on correctness alone.

### Combined DFlash activation on the same v12 binary

`runs/independent-v12-dflash-smoke` completed **16 exact-current-serial requests**, covering native DFlash then combined adaptive DFlash on all four held-out cases with first/repeat exposure. Native executed **158 eight-row generic-tree rounds**; adaptive executed **111 eight-row generic-tree rounds and 25 sixteen-row linear-GDN rounds**. Configuration requests dropped from 18 to 12 rounds each. Assistant bypass did not activate.

Total HTTP time was **40.895 s native versus 41.240 s adaptive**: no observed aggregate speedup in this sequential smoke, despite fewer verification rounds. Adaptive verified 1,288 rows versus native's 1,264. The telemetry gate also failed on changed swap usage/counters. This proves wider useful lookup execution and exact output, not DFlash acceleration; preparation costs, width prices and operating conditions still require matched measurement.

### Production-horizon neural width calibration

The v12 `production-neural` serial oracle and separate `assistant7`, `assistant8` and `assistant16` windows completed in `runs/independent-v12-cf-neural-*`. Each action used horizon seven, no suffix candidates, the same first-offer context length 158, and identical prepared target KV, recurrent state, assistant state and neural offer. The comparison validator accepted the same-preparation width contrast and exact serial publication.

| Verified rows | Actual variant | Committed tokens | Settled verify/commit ms | ms per committed token |
|---|---|---:|---:|---:|
| 7 | generic | 7 | 91.721 | 13.103 |
| 8 | generic | 7 | 97.224 | 13.889 |
| 16 | generic | 8 | 146.686 | 18.336 |

The larger tree recovered one extra token, but paid substantially more per committed token at this training prefix. These are single first-offer observations, **not pure GPU time or complete-round throughput**. Diagnostic readback duration is excluded; its effects on execution are not erased. No staircase plateau or universal best width follows from three points.

The actual exporter produced three bound schema-three cells, and the actual component declaration consumed them after checking exact price bytes, runtime identity, training contract and full target/assistant file hashes. Artifacts: `independent-v12-neural-training-prices{,.source}.json` and `independent-v12-neural-calibration-consumer.json`. This validates the calibration handoff, not production-equivalent timing or a full component performance gate. These prices are tied to the v12 binary and must not be relabeled for a later build.

Static analysis also confirmed a useful limit: a horizon-seven neural offer cannot commit more than eight tokens, regardless of a fifteen-node tree budget. Extra nodes buy alternate branches, not a longer continuation. On saturated configuration/code, wider neural verification alone therefore cannot exceed native DFlash's existing eight-token yield. A longer-horizon experiment must explicitly include its extra assistant and candidate-preparation costs.

## Stable native-cycle pricing and route economics (v13 smoke)

Runtime SHA-256: `3a53a365dbec6ab79d8f28890a5cdfc9cab96caa692f3891a0951c8cc7958429`. The Python helper suite passed **258 tests**; integrated lookup checks completed successfully and ReleaseFast build completed **8/8 steps**. Review identified mixed-depth preparation in inter-round timing. The fix restricts request-local price overrides to consecutive homogeneous native cycles, preserves acceptance/clock advancement on excluded transitions, and removes the unproven once-per-request refresh. Shared legacy estimates remain approximate fallback evidence.

`runs/independent-v13-family-smoke` completed **40 exact-current-serial requests** across serial, native/adaptive MTP and native/adaptive DFlash, with all four full cases and first/repeat exposure. Passive telemetry coverage and machine endpoint qualification passed. **135 route receipts** exposed the new pre-assistant lookup economics with valid measured/missing-value semantics.

| Family | Native summed HTTP seconds | Adaptive summed HTTP seconds | Descriptive speed change |
|---|---:|---:|---:|
| MTP | 63.942 | 43.182 | +48.08% |
| DFlash | 46.538 | 46.507 | +0.07% |

Sequential arm order still prevents a causal speedup claim or isolation of the timing correction. DFlash remains effectively unchanged in aggregate. The new receipts identify a specific gap: related-configuration anchors 222–270 executed successful sixteen-row linear-GDN trees, but their best priced **pre-assistant lookup** alternatives were eight-row generic trees. Their predicted yields rose from 5.03 to 6.03 tokens; the assistant remained cheaper under those predictions. This is not simply a pessimistic prediction for the same sixteen-row tree. Candidate geometry and exact-key price coverage must be checked before changing trust or assistant admission.

## Ranked lookup alternatives and production-horizon merging (v14)

Runtime SHA-256: `c83e0cd6f4af08ad6fedd26beb040b91c66199280c48bb5a4ff955f300bb7350`. The Python helper suite passed **260 tests**. Zig compilation initially caught a test call missing the new diagnostic anchor argument; after correction, integrated lookup checks completed and ReleaseFast build completed **8/8 steps**. An additional configuration-environment serialization fix passed all **57 offline helper tests** and an actual model-free dry-run with synthetic inherited credential sentinels: none appeared in emitted plan output. No inference subprocess environment was changed.

`runs/independent-v14-dflash-smoke` completed **24 exact-current-serial requests**, covering serial and native/adaptive DFlash on every full case with first/repeat exposure. Machine endpoints and passive telemetry qualified. All **1,214 ranked-prefix price receipts** passed checks for actual variant, null unpriced costs and query-inclusive cost/reward arithmetic.

Native summed HTTP time was **33.812 s**, adaptive **34.918 s**, a descriptive **−3.17% speed change**. This sequential, price-traced diagnostic is not an instrumentation-free comparison or a speedup claim.

The losing alternatives now resolve the v13 ambiguity. On first-exposure related configuration at anchor 254, lookup eight was generic, priced at 144.245 ms with expected yield 6.032; lookup sixteen was linear-GDN, priced at 264.196 ms with expected yield 10.158. Their query-inclusive costs were **23.915 versus 26.010 ms per predicted token**, so eight correctly won the lookup comparison. The assistant's estimate was lower still, at 19.588 ms per token. Repeat exposure later selected sixteen as the best lookup alternative, but it still lost to the assistant. Earlier sixteen-row alternatives were explicitly unpriced, not mislabeled as generic.

Thus, this evidence does not justify a variant-routing fix or arbitrary trust relaxation. Historical verifier prices and expected yield both matter; successful merged-tree execution does not establish the counterfactual yield of assistant-free lookup. Tracing adds host/I/O work and can affect learned wall-clock prices.

### Production-combined same-state calibration

The v14 serial oracle plus `assistant8`, `assistant16` and `lookup16` windows each completed two exact requests on the predeclared `train-config-a` first offer. The comparison validator accepted common baseline/prefix state and, for assistant-width comparisons, identical preparation and full offers. Assistant actions used the production horizon seven with suffix candidates enabled; lookup used no assistant preparation. Request trust remained cold.

| Action | Actual variant | Committed tokens | Settled verify/commit ms | ms per committed token |
|---|---|---:|---:|---:|
| assistant8 | generic | 7 | 106.444 | 15.206 |
| assistant16 | generic | 8 | 155.220 | 19.402 |
| lookup16 | linear_gdn | 16 | 152.267 | 9.517 |

At this independently declared training prefix, assistant-free lookup used **37.4% less verification/commit time per committed token** than assistant8. This establishes a useful local opportunity, not a complete-round or workload speedup: candidate sources differ, the readback perturbs execution, and these are individual first-offer observations. In contrast, widening the assistant tree bought only one extra committed token.

Artifacts are `runs/independent-v14-cf-combined-*`. The actual exporter created three exact-variant cells in `independent-v14-combined-training-prices{,.source}.json`; the actual component declaration accepted that bound profile in `independent-v14-combined-calibration-consumer.json` after full target/assistant content validation. That declaration has not itself established a performance result. Do not relabel these diagnostic prices as pure verifier or production-equivalent measurements.

### Actual server-path Metal attribution

`runs/independent-v14-metal-server-{dflash-native,dflash-combined-adaptive}` contain matched owned-server captures for configuration and unrelated prose, first/repeat exposure. All eight requests matched the current serial oracle. Native executed 18/18 configuration and 34/34 prose rounds; adaptive executed 12/13 and 35/33. The adaptive repeated configuration included three assistant-free lookup rounds with 40 published tokens.

These traces attach to the ready owned process and wait for an actual Darwin tracing-start notification before issuing requests. The earlier one-shot CLI capture did not activate DFlash despite its argument banner; it was rejected as mechanism evidence and its adaptive counterpart was cancelled.

| Later instrumented sample window | Native compute occupancy | Adaptive compute occupancy |
|---|---:|---:|
| Configuration repeat | 95.69% | 94.48% |
| Prose first | 96.38% | 96.18% |
| Prose repeat | 96.43% | 95.86% |

The denominator surrounds `record_sample()`, including setup, ownership checks, HTTP, receipt settling and persistence—not decode alone or precisely the HTTP interval. Owned compute intervals are clipped and unioned. Warm sampled shader duration is matrix-kernel dominated; sampled shares are not percentages of request wall time or absolute GPU cost.

Adversarial review caught an idle-gap calculation that omitted sample-window boundaries. Corrected `analysis-corrected.json` retains the originals, uses exact structured PIDs and includes leading/trailing gaps. First-window gaps are **348.093 ms native / 359.447 ms adaptive**; later-window largest gaps are **87–115 ms**, not the originally reported 3–8 ms inter-encoder gaps. Trace-envelope containment does not imply complete shader attribution: dominant target-kernel samples are missing from the native first configuration window. No first-window kernel-mixture or throughput claim is made.

Raw traces and exported TOCs can include inherited environment credentials and remain private. These profiles support investigating useful committed yield versus matrix-verification cost; they do not justify arbitrary trust-estimator relaxation or establish a performance win.

## Adjacent MTP ABBA/BAAB confirmation (v14)

`runs/independent-v14-mtp-abba` completed **six quartets, 24 fresh boots and 192 full requests** in 1,243.23 seconds. There were zero invalid responses; an independent post-run comparison also matched all 192 complete publications and semantic response fields to the already completed current-binary serial oracle. The prior six-permutation v14 run was interrupted when ABBA was requested and remains unqualified; none of its samples entered this cohort.

| Workload | First-exposure observed speed change | Repeat-exposure observed speed change |
|---|---:|---:|
| Related configuration | +95.58% | +73.93% |
| Conflicting configuration | +57.95% | +57.02% |
| Related code | +37.98% | +33.55% |
| Unrelated prose | −0.18% | −0.15% |

The geometric mean of the six complete-quartet native/adaptive latency ratios was **1.38096×: +38.10% reciprocal-latency speed, or 27.59% lower latency on that paired geometric basis**, with descriptive bootstrap interval **1.36646–1.39596×**. Individual quartet ratios ranged from **1.35673 to 1.40884×**. Every quartet's case/exposure paired totals passed the five-percent regression guard, and persistent-source activation passed for every required replica cell. Physical-boot timings remain in the summary; the interval uses six observations, not 192.

**The strict unchanged-swap qualification gate still failed:** reported swap use increased by approximately 1.3 MB, swap-ins by 12 and swap-outs by 80. Usage bytes derive from rounded OS output. Request endpoints bracket the usage/out changes in one adaptive configuration window, and the swap-ins in two native windows and one adaptive prose window. These are system-wide changes, not proven server-owned paging, and they were not exclusively during startup. Performance and promotion verdicts remain false; no sample was dropped and no frozen gate was relaxed.

Supported claim: **38.1% observed full-workload improvement in this ABBA run, consistently faster configuration/code reuse and essentially unchanged prose.** This is not a fully isolated hardware claim, A/A equivalence result, universal speedup or canonical-benchmark result. Fresh boots and ABBA do not prove independent hardware conditions across quartets. The earlier v10b prose failures remain historical evidence, not retrospectively corrected results.

## Adjacent DFlash ABBA/BAAB confirmation (v14)

`runs/independent-v14-dflash-abba` completed **six quartets, 24 fresh boots and 192 full requests** in 907.09 seconds. All 192 publications and semantic response fields matched the current-binary serial oracle. There were no invalid samples or execution/validation errors; measurement qualification failed on the unchanged-swap checks described below. Persistent-source activation passed for every required replica cell.

| Workload | First-exposure observed speed change | Repeat-exposure observed speed change |
|---|---:|---:|
| Related configuration | +13.16% | +16.02% |
| Conflicting configuration | +11.01% | +13.39% |
| Related code | +4.42% | +1.55% |
| Unrelated prose | +0.84% | +1.95% |

The geometric mean of complete-quartet native/adaptive latency ratios was **1.07208×: +7.21% reciprocal-latency speed**, with descriptive bootstrap interval **1.05796–1.08485×**. All six quartet ratios exceeded one, ranging from **1.04200 to 1.09565×**; one fell below the five-percent practical improvement threshold. Every quartet's case/exposure paired totals passed the five-percent regression guard. Configuration gains cleared the practical threshold; code and prose did not establish a practical five-percent gain.

**The unchanged-swap qualification gate failed here too:** reported swap use increased by approximately 0.34 MB, swap-ins by 152 and swap-outs by 20. Two request windows bracketed 148 swap-ins; usage and swap-out changes were outside the request endpoint windows. These counters are system-wide, not server attribution. Performance and promotion verdicts remain false; no samples or gates were changed.

Together, these cohorts support observed native-to-adaptive improvements for both drafter families on the declared reuse workload, not a direct MTP-versus-DFlash ranking. They do not isolate StairCut, lookup backoff or learned embedding-table contributions. DFlash used fewer rows than the legal offer in **1,294 of 1,608 published adaptive rounds**. The historical summary's `cut_reduced_rounds = 0` counts only the explicitly forced `cut` selection, not adaptive selection; interpreting it as no pruning was incorrect. Activation does not establish pruning's individual performance contribution. Canonical benchmark and individual-mechanism claims remain unestablished.

An adversarial reporting review found that the old `*_activated` flags also represented satisfied non-applicable requirements. Corrected mechanism analysis now separates observed counters from requirement-pass fields, explicitly filters published rounds, and records backoff as unobserved in these hybrid cohorts. The CPU-only `mechanisms-corrected.json` files retain the original summaries' hashes; no execution evidence, timing result or qualification verdict was overwritten.

Descriptive DFlash round attribution also narrows the optimization target: related configuration needed **288 rather than 432 verification rounds** for the same 3,408 published tokens across 24 requests, with no assistant-free rounds in either arm. Adaptive execution split evenly between eight-row generic and sixteen-row linear-GDN verification. Conflicting configuration used 322 rather than 432 rounds for the same token count. These observations support useful merged suffix continuations, not a claim that skipping the neural drafter caused the configuration gains. Only four code rounds bypassed the assistant; prose retained eight-row generic verification throughout.

Prefill remains a substantial fixed share of these short requests. Reconstructing approximate prefill duration from the server's rounded rate reports yields about 35 of 76 aggregate seconds for adaptive related configuration and 30 of 57 seconds for adaptive code. These are approximate, not direct phase timers. The whole-request gain therefore remains smaller than a decode-only gain; optimizing lookup admission does not remove prompt prefill.

### Additional same-state width-boundary diagnostics

Two additional training-only `production-combined` windows exercised lookup9 and lookup12 against the same current-binary serial oracle. Both completed two exact requests; actual comparison validated identical common state, prepared target state and lookup offer across lookup9/12/16.

| Lookup width | Actual dispatch | Committed tokens | Verify/commit time | Verify/commit time per token |
|---|---|---:|---:|---:|
| 9 | linear GDN | 9 | 140.192 ms | 15.577 ms |
| 12 | linear GDN | 12 | 140.932 ms | 11.744 ms |
| 16 | linear GDN | 16 | 152.267 ms | 9.517 ms |

These individual first-offer observations show useful nonlinear width economics at this prefix: nine to twelve rows added little measured cost, while sixteen rows gave the lowest observed cost per committed token. They are not a repeated, drift-controlled hardware plateau determination. Diagnostic readback effects remain, the sixteen-row window was earlier, and no complete-round or throughput claim follows. The actual exporter produced the separately named five-cell `independent-v14-combined-boundary-training-prices.json` profile; canonical and completed ABBA cohorts do not consume it.

## Request-local native economics and targeted ABBA (v15)

The first three physical arms of `runs/independent-v14-canonical-abba` completed the pinned workload, but the fourth was stopped after finding a benchmark coverage defect and a real MTP routing regression. The client issued **34 benchmark-body requests plus three reasoning-setup requests**, not 34 total. The corrected contract validates the source-defined two-to-four setup branches independently, actual metric memberships and every replay capture. All three saved probes pass that coverage check; this does not complete or retrospectively qualify the interrupted cohort.

The routing defect was a stale model-shared native comparator: a single post-warmup probe recorded **95.81 ms for one token** and remained the alternative for later requests. Native decode in the first arm actually averaged **44.86 ms per published token**, while lookup in the next arm averaged **69.61 ms/token**. Predicted two-row lookup verification cost was close to observed cost; relaxing confidence was not the fix.

V15 uses paired duration/yield from consecutive homogeneous native cycles in the current request, context class and selected depth. It subtracts matching interval query preparation, excludes prepared-native alternatives and terminal/truncated intervals, and passes the same fresh comparator to bounded width exploration. These are query-adjusted wall estimates, not isolated GPU timings. No extra GPU barrier or periodic refresh probe was introduced.

Runtime SHA-256: `b47cd4a09aad6bf7c04e05c826e895a23c6f76b9025b805326cb494d8367d838`. Focused Zig lookup tests and the ReleaseFast build succeeded. The targeted run `runs/independent-v15-native-comparator-smoke-2` executed native/adaptive/adaptive/native fresh boots followed by a separate serial boot, using three unchanged full requests per boot. All **15 outputs matched exact published IDs, HTTP text/usage and finish reasons**.

| Targeted workload | Native HTTP seconds, geometric mean | Adaptive HTTP seconds, geometric mean | Reciprocal-latency speed change |
|---|---:|---:|---:|
| Canonical decode request | 9.625 | 9.916 | −2.94% |
| Canonical novel request | 12.620 | 12.569 | +0.41% |
| Held-out related configuration | 7.946 | 4.454 | +78.41% |

This is one targeted ABBA quartet, not a completed full-client cohort or confidence interval. The general decode residual regression remains visible. Each native decode request executed 73 rounds; adaptive executed 76, including only six/four lookup rounds rather than the earlier extensive weak-lookups regime. New traces show current-request cost/yield updates instead of the fixed one-token shared estimate. This verifies the corrected decision path and retained reuse benefit; it does not causally compare the two binary versions under matched conditions.

### Incremental memory and storage observations

Both MTP arms use the same target and MTP head. No neural weights were added by enabling lookup/routing.

| Whole-process measurement | Native mean | Adaptive mean | Adaptive minus native |
|---|---:|---:|---:|
| Loaded physical footprint | 17,309.65 MiB | 17,309.84 MiB | +0.20 MiB |
| End-of-workload physical footprint | 17,619.04 MiB | 17,942.59 MiB | +323.55 MiB |
| Lifetime maximum physical footprint | 19,229.30 MiB | 19,229.19 MiB | −0.12 MiB |
| Persistent lookup files, logical content | 0 bytes | 3,920 bytes | +3,920 bytes |

The lifetime maximum includes loading and cannot establish zero additional inference peak. End footprints vary across the two boots per arm and include allocator/KV/verification-buffer retention, not just lookup allocations. The cache occupied 4 KiB of allocated file blocks; this small frozen 970-token fixture is not a production-capacity test.

A native arm64 type-layout probe measured the local ring at **1,048,576 bytes per armed request**, inline drafter state at **36,440 bytes**, inline policy at **8,608 bytes**, and the history-cache wrapper at **544 bytes**. Persistent token/position arrays scale at eight bytes per retained entry up to **2 MiB**; the payload cap is **1 MiB + 40 bytes**. Optional corpus arrays can reach **128 MiB** at their separate cap. Index replacement may temporarily retain old and new allocations. These implementation sizes exclude other tree scratch, dynamic price storage and GPU buffers; the offline guide documents the measurement contract.

The memory helper's Darwin ABI was checked against compiled SDK offsets and exercised on a live process and nonexistent PID. Adversarial review found a symlink-root storage-accounting defect; the corrected helper rejects it. All **289 Python lookup-helper tests passed** after the fix. Resource observations above came from real directory roots and remain unchanged.

### Upstream scope

The current working tree is not yet a minimally invasive submission: relative to upstream `191da71d`, excluding the user's dirty `lib` tree, the inspected surface comprised 49 files: 18 source files with 8,969 added lines and 175 removed lines (including inline Zig tests), 26 test/benchmark scripts with 13,488 added lines, four documentation/changelog files, and build wiring. These are an assessment snapshot before subsequent component work, not production-only line counts. Keep learned embeddings independently reviewable, isolate reusable verification prerequisites, and justify lookup/routing and dynamic width selection with their own measured contrasts before deciding the final PR boundaries.

## Learned-embedding decoded-row cache: first measured implementation (v16)

The first implementation adds a bounded, table-owned 256-row direct-mapped cache of exact decoded f32 vectors. Cache misses retain the existing pooled reads; widths above 64 embedding rows bypass the cache. This is an embedding-gather optimization, not an additional speculative drafter, and requires no new model weights.

Two real-pack runs, `runs/independent-v16-ple-reuse-{1,2}.json`, used `TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP`, seeds 1347175775 and 1347175776, 256 sampled contexts and eight immediate repeats. The `ple-bench` executable SHA-256 was `cd92a8cd2145ad91631d995555718c48c289b0e4c5a251e1b1d6a630d42ffc60`. Each run checked 9,216 forced-control pairs, 9,216 production microbenchmark comparisons and 24,576 sequential production gathers; decoded vectors matched exactly. The PLE tests and ReleaseFast benchmark build passed before execution.

| Hash-and-gather workload | Run 1 speed change | Run 2 speed change |
|---|---:|---:|
| Immediate repeated context, one token / 16 rows | +165.08% | +135.74% |
| Immediate repeated context, two tokens / 32 rows | +22.59% | +25.87% |
| Immediate repeated context, four tokens / 64 rows | +21.07% | +21.41% |
| Immediate repeated context, eight tokens / 128 rows, cache bypassed | +0.09% | −0.03% |
| Initial 32-token sequential request | −0.19% | +1.42% |
| Complete 32-token sequential repeat | +1.58% | +1.55% |
| Similar request: shared 24-token prefix, changed eight-token suffix | +1.06% | +0.52% |

Speed change is `(baseline time / treatment time − 1) × 100`, against the old production gather policy, not a forced serial control. Microbenchmark bootstrap intervals exclude parity for widths one, two and four; sequential results are descriptive ratios, not confidence intervals. Whole-policy order alternates; decoded-cache state resets before each three-request sequence, not between its requests. OS page-cache state is not reset, so these are not cold-disk measurements.

The sequential results expose the limitation hidden by immediate repeats: a 32-token stream can touch 512 rows, exceeding the 256-row direct-mapped cache. Complete-repeat row hits were only about 15.5%, so most gathers still woke the read pool. A bounded four-way cache experiment is justified by this measured churn; its gains remain unmeasured here.

Incremental cache payload and metadata were **165,920 bytes (162.03 KiB)** at this pack's dimension of 160, plus 1,024 bytes of narrow-call stack scratch. There are no additional worker threads or persistent files. These are implementation allocation sizes, not process-footprint deltas. No full-model Flash-Next inference speedup follows from the gather timings.

## Coding diagnosis: useful wide lookup, weaker mixed narrow rounds

The completed v14 DFlash2 cohort's `related-code` case contains 24 requests and 1,680 generated tokens per arm. Direct retained server timings total **30.546 s prefill / 27.879 s decode** for native and **30.397 s / 26.348 s** for adaptive. Aggregate HTTP durations were 58.471 s and 56.789 s: **+2.96% reciprocal-latency speed**, with **+5.81% decode-phase speed**. These aggregate phase diagnostics are not replacements for the declared quartet estimator or its machine-state gates. Prefill accounts for about 52% of native HTTP time on this short 70-token answer.

| Published round population | Rounds | Published tokens | Tokens per round | Verify/commit milliseconds per token |
|---|---:|---:|---:|---:|
| Native DFlash2, generic eight rows | 216 | 1,680 | 7.778 | 14.962 |
| Adaptive mixed offer, generic eight rows | 146 | 948 | 6.493 | 16.848 |
| Adaptive mixed offer, linear sixteen rows | 48 | 732 | 15.250 | 10.784 |

The wide lookup rounds are productive; narrow mixed rounds give back part of the gain. These populations occur at different prefixes and are not a causal topology comparison. In particular, native DFlash2 uses best-first admissions, not a guaranteed greedy spine; truncating the final preorder does not recover its original eight-row tree. The diagnostic follow-up retains exact native admissions from the same expanded lattice and compares ancestry-aware membership against the selected tree. Production preference must not be changed merely by comparing differently normalized neural and merged probability sums.

The next coding workload is separate from the original frozen transfer contract: fresh algorithm generation, an unseen related helper with changed record semantics, a full-module semantic edit, and a missing-suffix completion. Its four training helpers share one JSON-Lines template; that is a deliberate reuse case, not four independent coding domains. Report cases separately, preserve first-exposure costs and exact serial output, and do not treat Python syntax checks as functional-correctness evidence.

## Isolated DFlash2 fixed-eight versus StairCut (v16)

`runs/independent-v16-df2-combined-fixed8-cut` completed four adjacent ABBA/BAAB primary quartets, with native references and serial correctness outside them: **28 boots, 224 requests, zero invalid samples**. Runtime SHA-256 was `1ced2f1697ef2b160cca8eadd38ea67c4b96a2f7bd202e883a0fa34d64f14655`. Both primary arms use the same DFlash2 assistant, combined candidate sources, frozen history and explicit row-exact numerical mode. No offline prices were supplied. Fresh-controller learning, preparation and exploration remain charged to HTTP duration.

| Primary fixed-eight → cut contrast | First-exposure speed change | Repeat speed change |
|---|---:|---:|
| Related configuration | +10.75% | +19.61% |
| Conflicting configuration | +11.09% | +10.70% |
| Related code | +9.42% | +8.23% |
| Unrelated prose | −3.37% | −0.43% |

The complete-workload geometric mean of quartet summed-duration ratios is **1.067228× (+6.72%)**, with a descriptive quartet-bootstrap interval of **1.053496–1.083995× (+5.35% to +8.40%)**. Individual quartet ratios were 1.068511, 1.048538, 1.060338 and 1.091998. There are four statistical observations, not 224 independent samples.

This is an observed isolated **combined-offer width-policy** benefit, including its learning overhead. It is not an isolated kernel gain or neural-only StairCut result. Cut executed 16 bounded width-learning trials and reduced 868 of 1,075 published rounds; fixed-width selection used 1,264 published rounds. Both candidate-source requirements passed.

**Strict performance qualification remains false:** swap usage, swap-ins and swap-outs changed, and one first-prose quartet had a ratio of 0.899515, failing the five-percent per-cell regression guard. The average prose effect must not hide that result. All first-prose primary rounds actually executed generic eight-row verification; cut used 33–35 rounds per request versus fixed's 33. The regression cannot simply be attributed to executing wider prose kernels.

Native DFlash2 references are outside the primary quartet and remain descriptive: cut/native full-workload ratio was 1.054782 (+5.48%); related-code ratios were 1.030706 first and 1.005947 repeat. Fixed/native code ratios were 0.941993 and 0.929423. Width adaptation recovers mixed-offer costs, but these references do not establish a large coding improvement over native DFlash2.

### Resource observations in the completed width experiment

All 56 loaded/final allocator snapshots were complete, exercising the owned authenticated `/props` path. The following are means across eight physical boots per arm; serial boots are excluded.

| End-of-workload measurement | Native DFlash2 | Combined fixed-eight | Combined cut |
|---|---:|---:|---:|
| MLX active allocations, MiB | 18,239.324 | 18,239.324 | 18,239.324 |
| MLX reclaimable cache, MiB | 116.265 | 96.000 | 96.000 |
| MLX lifetime peak, MiB | 18,511.482 | 18,511.490 | 18,511.490 |
| Process physical footprint, MiB | 18,598.390 | 18,593.671 | 18,588.569 |

Reported model-weight bytes were identical across arms. These endpoint and lifetime measurements do not isolate live request peaks or imply that lookup itself saves memory; reclaimable-cache retention differs. The persistent fixture remained 3,920 logical bytes / 4,096 allocated bytes. It is write-disabled, so this run does not measure live cache growth or append costs. Adversarial review found no blocker in the frozen contrast or resource-accounting contract.

## Four-way learned-embedding cache (v17)

The bounded cache now holds **1,024 rows in 256 four-way sets**, with exact row-ID comparisons and deterministic round-robin replacement. Hits do not mutate replacement order. Duplicate insertion updates its existing slot; all gathered hits are copied before any miss insertion. Pooled misses, wide-gather dispatch and complete-output fallback are unchanged.

`ple-bench-test` and `ple-bench-build` passed. Four real-pack CPU runs then completed without concurrent inference or compilation, using executable SHA-256 `c3c35a8843e43bf4bc62f3ed35f81db6adef9ddd0f585dcfd38054be9c58118e`. Artifacts are `runs/independent-v17-ple-reuse-{1,2}.json` and `runs/independent-v17-ple-coding-{1,2}.json`. Seeds remain 1347175775/1347175776. Generated-text runs use 256 contexts; coding runs use 128. Each generated-text run checked 9,216 forced-control pairs, 9,216 micro production comparisons and 122,880 sequential production gathers; each coding run checked 4,608, 4,608 and 61,440 respectively. **All decoded vectors matched exactly.**

The coding corpus is a frozen concatenation of four independently authored Python benchmark helpers, 254,821 bytes, SHA-256 `0a458a06885a0e9b45dda083384dbe3f49b3e58cdfbf3e9d2eb5e015d1ebc789`. It contains no generated evaluation answers. Both 32- and 128-token cohorts use disjoint corpus windows and compare the new cache with the old production path without a decoded cache.

| Sequential hash/gather workload | Generated text, run 1 | Generated text, run 2 | Python source, run 1 | Python source, run 2 |
|---|---:|---:|---:|---:|
| Initial 32-token request | −0.30% | −0.30% | +4.56% | +1.93% |
| Complete 32-token repeat | +21.31% | +25.58% | +25.45% | +28.24% |
| Shared 24-token prefix, changed eight-token suffix | +17.93% | +18.24% | +19.12% | +21.23% |
| Initial 128-token request | +2.12% | +0.13% | +1.90% | +5.20% |
| Complete 128-token repeat | +2.40% | +0.87% | +4.41% | +4.43% |
| Shared 96-token prefix, changed 32-token suffix | +2.01% | +1.68% | +3.93% | +4.33% |

These are descriptive reciprocal-latency changes for the hash/gather component, not full-model generation gains or sequential bootstrap intervals. The complete 32-token coding repeat took 6.639 → 5.292 ms in run one and 7.389 → 5.762 ms in run two. Its row-hit rate was 88.71%; the 128-token repeat hit only 21.77%, exposing capacity/collision limits. Partial-hit gathers can still wake the pool, so row-hit percentage is not the latency reduction.

The new payload plus cache metadata is **663,840 bytes (648.28 KiB)** at dimension 160, **497,920 bytes (486.25 KiB)** more than the first cache. Narrow-call scratch remains 1,024 bytes; no persistent storage, worker threads or per-call heap allocations were added. Allocation rounding is excluded. This experiment changes capacity and associativity together; it does not independently attribute gains to either. The longer cohort exceeds total cache capacity, and the initial generated-text case preserves the small measured cold-path regression rather than hiding it.

Adversarial source review found no blocker in ownership, collision handling, fallback, memory accounting or sequential benchmark separation. Full-model Flash-Next inference remains a separate memory-gated validation; CPU gather savings cannot be applied directly as an end-to-end speed percentage.

## Composition: which gains can stack?

Suffix lookup and dynamic width selection cooperate with the configured neural drafter. The native-versus-adaptive MTP/DFlash2 results already measure their combined effect; an isolated fixed-width-versus-cut percentage must not be added to those combined gains. MTP and DFlash2 are alternative neural paths in this implementation, not simultaneous speed multipliers.

For example, the v16 descriptive native→combined-fixed ratio is 0.988338 and the isolated combined-fixed→cut ratio is 1.067228. Their product is the descriptive native→cut ratio, 1.054782. This is accounting across measured arms, not proof that component effects are independent; the native references remain outside the primary quartet.

Learned-embedding caching acts on a different operation and could compose on a model with the required table and supported speculative route. Its hash/gather speedup is not an inference multiplier: if the operation occupies fraction `f` of baseline runtime and speeds up by `s`, with everything else unchanged, whole-runtime speed is `1 / ((1 - f) + f / s)`. Speculation can change `f`, gather width and reuse, so a combined full-model experiment remains necessary. The Qwen 27B speculative cohorts do not exercise the Flash-Next learned table.

Miss backoff can reduce wasted suffix attempts but is relevant to misses, not another multiplier on successful lookup. Keep its standalone on/off comparison separate. PR claims should report measured combinations and controlled ablations, preserve interactions and regressions, and never sum headline percentages.

## Native candidate displacement diagnostic (v17)

Focused lookup/native-admission tests and the ReleaseFast build passed after correcting a test fixture whose near-certain child did not actually produce the intended sibling-first admission order. The expected ancestry behavior was retained, not weakened. Runtime SHA-256: `671581fa3e3561f49bb021c071614134591b6baaa7dc8725c046d43575fa36a7`.

`runs/independent-v17-native-candidate-diagnostic` completed eight exact-output requests across serial, native MTP, native DFlash2 and combined fixed-eight DFlash2, using the original related-code case. The trace preserves the original native first-admission tree without changing production selection. A reviewed offline analyzer joined 20 witnesses to their actual published rounds and complete serial output; 18 comparisons had equal width/variant and uncensored continuation.

At those same selected-run prefixes, the native tree matched **144 rows**, versus **132** for the selected mixed tree. Fourteen comparisons tied; native retained three more matching rows twice, four more once and two more once. Thus the diagnosed narrow-tree displacement can lose useful tokens even at identical prefixes—not merely in aggregate populations at different positions.

This is a token-path counterfactual, not measured alternative latency or independently sampled rounds. Geometry trusts the producer's explicitly enabled request log. The analyzer requires row-exact greedy output, validates ancestry-sensitive membership, rejects duplicate joins and excludes terminal continuation from the comparable subset. Production selection remains unchanged pending a causal throughput comparison.

## Broader coding workload, before the neural-calibration correction (v18 contract)

The first coding contract failed on the native baseline's indentation when asked to continue inside an unfinished Python `for` suite. `runs/independent-v17-coding-dflash-abba` remains a failed seven-request run; its answer was not whitespace-normalized or requalified. A separately frozen second contract completes a new top-level function instead. Four independent training responses were regenerated under the new train/evaluation identity declaration; no evaluation answer seeded history.

`runs/independent-v18-coding-{dflash,mtp}-abba` each completed **four ABBA/BAAB quartets plus serial reference: 17 boots, 136 exact-output requests**. Both used the pre-correction v17 runtime `671581fa3e3561f49bb021c071614134591b6baaa7dc8725c046d43575fa36a7`. The neural residual-calibration edit was not in that executable. Every case includes measured first and repeat exposure, with history writes disabled and prefix caching disabled. Syntax, declared function presence and unchanged-helper checks passed; generated code was not executed, so this is not functional task-correctness certification.

| Coding case | DFlash2 first | DFlash2 repeat | MTP first | MTP repeat |
|---|---:|---:|---:|---:|
| Fresh interval-merging function | −1.35% | −3.08% | −5.85% | −5.97% |
| Related JSON-Lines helper with changed key/retention semantics | −4.25% | −1.03% | +6.54% | +10.08% |
| Semantic edit of supplied module | +0.53% | −1.35% | −6.15% | −6.18% |
| Top-level function completion using supplied helper | −4.18% | −0.64% | −9.74% | −5.62% |

Combined native→adaptive workload ratios were **0.981114 (−1.89%) for DFlash2** and **0.971618 (−2.84%) for MTP**. Descriptive quartet-bootstrap intervals were 0.937829–1.036505 and 0.924014–1.020987 respectively. Neither establishes an overall improvement. Both machine-state qualifications failed due changed swap observations; retain the raw point estimates and uncertainty rather than claiming a clean causal regression magnitude.

The DFlash2 adaptive cohort executed wider sixteen-row rounds, but **all were generic**, not the cheaper linear-GDN variant: 55 on fresh code, 27 on the related helper, five on edits and two on completion. Fewer rounds therefore did not imply less verification time. The MTP fresh case increased from 1,280 to 1,337 rounds across its 16 requests, with 102 lookup rounds; completion increased from 664 to 709, with 450 lookup rounds. Native baselines already used their existing local-copy mechanisms on some cases. These observations bound the next optimization work: preserve useful neural ranking, account for executable shape rather than width alone, and investigate weak MTP admission/recovery instead of assuming more selected lookup tokens are beneficial.

The staged incoai Qwen3.8-27B DFlash2 configuration has **block size eight**: one anchor plus seven neural draft positions. Expanding its candidate tree adds alternatives, not neural depth. A neural-only chain from that forward cannot reach the greater-than-eight-row linear-GDN path. Longer suffix continuations can reach it, but that opportunity depends on actual reusable context. Do not assume the wider generic coding rounds could simply have been replaced with sixteen correct neural tokens, or raise the trained horizon without a separate supported-model experiment.

## Neural probability calibration experiment (v19)

The old neural estimator shrank every confidence in the same bucket toward its empirical mean. Its within-bucket slope was `4 / (4 + samples)`: once populated, distinct neural confidences became nearly indistinguishable before multiplying path reach. A six-edge high-confidence continuation could consequently lose its final node to a less promising root alternative.

The neural-only correction retains each raw probability and learns the bucket's mean residual: `clamp(raw + (hits - sum_predictions) / (4 + samples), 0, 1)`. Lookup entry/continuation calibration is unchanged. Eight prediction sums add 32 inline bytes, with no heap allocation, transcendental operation or GPU work. The sums decay with the existing hit/sample counters and participate in the policy fingerprint and value-copy snapshot. This does not guarantee globally monotone calibration across buckets or preservation of every native branch.

A focused adapter regression failed with the old estimator (expected continuation token 7, selected alternative 88) and passed with the correction. Constant-input equivalence through decay, source isolation and snapshot/fingerprint tests passed, followed by the lookup-filtered suite and ReleaseFast build. The corrected executable is `9729431a0e67fe19b8eb7bed0210b393bca1e9703aa59935fcb194a7487554ee`; the pre-correction executable is preserved separately.

`runs/independent-v19-native-candidate-diagnostic` completed eight exact responses. Its short-case witness still matched 144 native rows versus 132 selected rows across 18 comparable rounds—the same displacement as before. **The correction has not fixed that diagnostic loss, and no throughput benefit is established by these tests.**

The separate coding before/after experiment keeps combined-adaptive controls and frozen training history identical across two executables; it is not a native-baseline comparison. Adversarial runner review caught an arm-name fallback that would have given the baseline an empty history. The copied baseline definition now explicitly selects the persistent tier, invoking the existing seed-copy and full-tree-hash checks. Both executable/linkage identities were frozen and checked throughout the run; all sixteen measured boot stores had identical hashes.

`runs/independent-v19-coding-calibration-abba` completed four ABBA/BAAB quartets plus serial: **136 exact responses**. Whole-workload before/after ratio was **0.994197 (−0.58%)**, with descriptive quartet-bootstrap interval **0.981993–1.007441**. Swap bytes, swapins and swapouts changed, so machine qualification failed. This change has **not demonstrated a throughput improvement**.

| Coding case | First exposure | Repeat |
|---|---:|---:|
| Fresh interval merge | −4.09% | +1.02% |
| Related JSON-Lines helper | +0.88% | +0.08% |
| Semantic module edit | −0.47% | −0.13% |
| Function completion | −1.06% | −1.17% |

The correction reduced sixteen-row generic executions from 50 to 43 on fresh code, 35 to 15 on the related helper, 16 to six on edits and 11 to seven on completion. Total verification rounds were nearly unchanged: 540→539, 462→464, 418→418 and 277→280 respectively. Better mathematical preservation of neural confidence is therefore not sufficient evidence of better generation performance.

## MTP coding admission diagnosis

In the v18 completion case, native executed 646 depth-two and 18 depth-one rounds. Adaptive executed 251 native depth-two rounds, eight native depth-one rounds, 305 two-row lookup rounds and 145 fifteen-row lookup rounds. Review of its admission traces found 445 of 450 lookup admissions still using a native yield of exactly one. Fourteen of sixteen requests retained an unchanged paired comparator despite subsequent native executions.

The timing exclusion was intentional: recovery after lookup and mixed outgoing preparation are not homogeneous native-cycle durations. The error was using that timing eligibility to restrict all available native-yield evidence. A single poor eligible native sample could therefore remain the lookup comparator while later genuine native acceptances were excluded from its yield estimate.

The next implementation keeps the paired cost/yield record unchanged and separately tracks uncensored native output by request, context class and executed depth. Admission combines the existing homogeneous cost with this broader yield estimate. Recovery and contention may update yield, never timing; missing timing still forbids a measured comparator. Lookup, errors, terminal/budget truncation and context-crossing outcomes are excluded. No forced GPU refresh episode, depth pinning or synchronization was added.

This is a **factorized prediction from differently eligible populations**, not a freshly measured paired rate. Acceptance can affect duration, old cost can remain stale, and no estimator learns a new native outcome when lookup completely suppresses native execution. The edit case also has a separate problem: most lookup admissions already used native yield three, while some local-copy proposals greatly overpredicted acceptance. Do not attribute all coding regressions to the stale one-token sample.

Adversarial source review found no blocker in the new completion hook, censoring, depth/context isolation or retained timing gates. The lookup-filtered suite and ReleaseFast build passed; runtime SHA-256 is `b79deedbe02b409c669f42968a1ffadfb730d723c3664761fe0cb2fc421f91da`. `runs/independent-v20-mtp-yield-smoke-2` completed eight exact 70-token responses across serial, native MTP, native DFlash2 and adaptive MTP. Actual logs contain 24 native-yield observations and 18 admission decisions using `request_cost_current_yield`. The first smoke command was rejected before inference because price tracing is restricted to DFlash oracle diagnostics; it was not a model failure.

`runs/independent-v20-coding-mtp-abba` completed four native/adaptive ABBA/BAAB quartets plus serial: **136 exact responses**. Whole-workload ratio was **1.000483 (+0.05%)**, with descriptive quartet-bootstrap interval **0.975603–1.017532**. Swap bytes and swapins changed, so machine qualification failed. This is overall parity in the observed cohort, not a demonstrated whole-workload speedup or equivalence.

| Coding case | First exposure | Repeat |
|---|---:|---:|
| Fresh interval merge | −4.79% | −4.56% |
| Related JSON-Lines helper | +7.13% | +8.44% |
| Semantic module edit | −4.29% | −3.42% |
| Function completion | +3.23% | +1.43% |

None of the 450 logged lookup admissions used a native-yield estimate of one; all used the new factorized scope. Completion executed **605 adaptive rounds versus 664 native**, including 67 fifteen-row lookup rounds and no two-row lookup rounds. The older v18 adaptive cohort executed 709 rounds, including 305 two-row and 145 fifteen-row lookup rounds. These are different cohorts, so the before/after timing difference is not an isolated estimate of the fix's effect.

Fresh generation still executed 1,316 adaptive rounds versus 1,270 native. Its 14 lookup rounds committed only 20 tokens, with ten zero-draft-accept outcomes. Editing executed 625 versus 624 rounds; 16 of its 19 zero-accept lookup rounds matched only six suffix tokens, two matched three and one matched eight. Stronger native yield evidence addresses the diagnosed stale-comparator feedback but does not solve short-match prediction or transition costs. The completed v19 DFlash2 comparison does not contain this MTP change.

## Raw-neural scoring ablation (v21)

The residual-calibration experiment did not improve coding throughput or eliminate native candidate displacement. This ablation removes online neural calibration entirely, retaining the host drafter's bounded raw conditional probabilities. Lookup calibration, request trust, reached-path feedback and price selection remain active. Removing the neural hit/sample population and prediction sums saves 96 inline bytes relative to v20 before padding; it adds no GPU work.

Focused tests cover pure-neural ranking after lookup learning and lookup feedback through pure-neural ancestors. All 75 adapter/policy/history tests passed, followed by the lookup-filtered suite and ReleaseFast build. Runtime SHA-256 is `064dbb50f0394d61c9fc472d13f9e95b34096adf785b7c7f2463cbcdb51240a6`.

`runs/independent-v21-native-candidate-diagnostic` completed eight exact responses with no invalid samples. Across 18 equal-width, equal-variant, uncensored witnesses, native geometry matched 144 serial-path rows versus 131 selected rows. Four witnesses lost native matches: three lost three rows and one lost four. This **does not repair native candidate displacement**. Unselected lookup siblings can affect mixed-source mass projection even when every selected edge is neural; raw neural inputs alone do not preserve native admission order. These are same-prefix geometry observations, not counterfactual wall-time measurements.

`runs/independent-v21-coding-raw-neural-abba` completed four ABBA/BAAB quartets plus serial, **136 exact responses**, using identical combined-adaptive controls and training history with v20 versus v21 executables. Whole-workload ratio was **0.988231 (−1.18%)**, with descriptive quartet-bootstrap interval **0.960926–1.002963**. Individual quartet ratios were 0.947786, 1.001449, 1.001320 and 1.003511. Swap bytes, swapins and swapouts changed, so measurement qualification failed. The ablation **did not demonstrate a throughput benefit**; the point estimate is not a clean causal regression claim either.

| Coding case | Raw versus residual: first exposure | Repeat |
|---|---:|---:|
| Fresh interval merge | −1.95% | −0.94% |
| Related JSON-Lines helper | −1.70% | −0.82% |
| Semantic module edit | −1.10% | −0.81% |
| Function completion | −1.63% | −0.56% |

Published round counts changed from 534→540, 465→460, 418→422 and 281→287 respectively. Sixteen-row generic executions changed from 67→57, 25→35, nine→five and five→six. Neither arm executed linear-GDN rounds. Removing neural calibration simplifies the estimator and its state; it is not one of the measured speedups and does not resolve mixed-source candidate ranking.

## CPU lookup profiling at coding prefixes

`query-replay-v23-baseline-summary.json` preserves a standalone ReleaseFast CPU replay of the four frozen coding outputs against the 1,060-token, order-eight training history. It primes the local index, advances only published history, and calls lookup at **every output-token prefix**, with 32 deterministic replays per case and 22,848 measured queries. No inference or compilation ran concurrently. Timing surrounds `Drafter.look` only; output/node digests are computed outside that interval. The private harness, fixture, executable and production source hashes are retained.

| Coding case | Mean lookup query | Median sum of queries for one output replay |
|---|---:|---:|
| Fresh interval merge | 6.32 µs | 1.374 ms |
| Related JSON-Lines helper | 3.47 µs | 0.635 ms |
| Semantic module edit | 3.11 µs | 0.608 ms |
| Function completion | 3.56 µs | 0.410 ms |

This is not actual speculative query cadence, a whole-request overhead measurement, or an inference speedup. It excludes priming, policy selection, learner updates, logging, model work and pipeline transitions; CPU behavior under simultaneous GPU work may differ. Nevertheless, these isolated query costs do not support prioritizing a suffix-search rewrite to recover the much larger observed coding wall-time gaps. The next measurements should distinguish admission errors and native pipeline/depth effects from broader host/control overhead rather than assuming lookup search dominates.

## Exact-order request support (v22)

Pooled model calibration already separated suffix orders, but request-local admission support pooled them. A consumer-level regression warmed order-64 request support and order-six pooled calibration, then asked to admit a previously unsupported order-six offer. The old implementation returned support **0.9864865 instead of zero**. The same regression passes with order-specific request support.

The new request table separates orders 0–64, entry/continuation and eight confidence bins. Restriction reads each offered edge's stored order; reached feedback uses that child's order; nonoverlapping chain remainder uses the executed chain's order. Censoring, exactly-once shared-prefix handling and pure-neural ancestor traversal remain unchanged. The request fingerprint domain advances and covers every population. Inline state grows from **128 to 8,320 bytes (+8 KiB)**, with no heap/GPU allocation or disk payload.

Adversarial source review found no blocker. All 77 focused adapter/policy/history tests passed, followed by the lookup-filtered suite and ReleaseFast build. Runtime SHA-256 is `61cd5a3ce32a9f527ce2e047f2069f764bdf2fdd4ef402ae2945c642f1ce21cd`. `runs/independent-v22-mtp-order-trust-smoke` completed eight exact 70-token responses with no invalid samples; adaptive MTP executed 34 published rounds, including six selecting persistent lookup and one selecting local lookup.

`runs/independent-v22-native-candidate-diagnostic` also completed eight exact DFlash2 diagnostic responses without invalid samples. Its 18 eligible same-prefix witnesses matched **144 native rows versus 138 selected rows**, compared with 131 selected rows in v21; sixteen witnesses tied and two lost three rows each. Order-specific support reduces displacement in this diagnostic but does not eliminate it or establish faster inference.

This is not a hard minimum suffix length or proof that short matches are unsafe. Same-order content and continuation depths still share support; some previously useful new-order matches must wait for evidence. Timing comparisons must count both prevented misses and lost profitable admissions. The implementation has runtime correctness evidence, not yet a throughput claim.

## Standalone PLD miss-backoff isolation (v22)

Evidence: `runs/independent-v22-pld-backoff`; binary `61cd5a3ce32a9f527ce2e047f2069f764bdf2fdd4ef402ae2945c642f1ce21cd`. Four adjacent primary ABBA/BAAB quartets compare identical frozen-history/cut configurations with miss backoff off versus on. Serial references are outside the primary quartets. All 24 boots and **192 exact-valid responses** completed, with zero invalid samples. This isolates the backoff policy, not lookup versus native decoding or StairCut versus fixed width.

Full-workload speed ratio is **0.99856 (−0.14%)**, with quartet-bootstrap 95% interval **0.98282–1.00799**: no overall benefit demonstrated. Changed swap usage and swap-ins fail measurement qualification. Each quartet includes all four cases and first/repeat exposures; no cases or slow boots were dropped.

| Case | First exposure speed change | Repeat speed change |
|---|---:|---:|
| Related configuration | −9.98% | −3.24% |
| Conflicting configuration | −6.02% | −6.69% |
| Related code | −4.45% | −5.16% |
| Unrelated prose | +9.34% | +6.43% |

Percentages are `100 × (off/on full-request duration ratio − 1)`, using the geometric mean of complete-quartet ratios. Unrelated first-exposure prose has interval **+8.55% to +10.14%**, but conflicting repeat configuration regresses **−8.97% to −5.10%**. Hardware qualification still applies to both findings.

All 16 backoff-on prose exposures report the exercised 1/2/4/8 mechanism, ending with window 8 and seven plain steps remaining. Each records 18 misses, one reset and 103 plain emissions (17 drain plus 86 native), rather than inventing lookup work for skipped steps. Reduced lookup-receipt counts are not a reduction in total decoding rounds.

The tradeoff is real in this workload: avoiding repeated misses helps prose, while skipped opportunities hurt useful reuse. Do not extend this standalone policy to MTP/DFlash or claim universal gain on this evidence. Serial-reference comparisons remain descriptive; they do not inherit the isolated backoff contrast's statistical interpretation.

Adversarial follow-up found no safe branch substitution that removes the tradeoff while preserving exact 1/2/4/8 skip semantics. The terminal drain restores the uncached next anchor required by PLD; ordinary `next()` would leave a forwarded successor pending. A cheap probe of the last emitted suffix is not aligned with that future anchor, while resolving the pending anchor adds synchronization. Interrupting a skip on a probe would be a different backoff policy, not an implementation-equivalent optimization. Block-zero related/conflicting requests also lose accepted drafts with backoff enabled; their one or two cold skip emissions do not establish repeated drains as the sole latency cause. Retain opt-in behavior rather than adding an unmeasured presence index or claiming the full-workload regression is fixed.

### Which improvements stack?

Lookup supplies candidates, the width policy chooses verification work, and one selected neural family supplies neural drafts. These mechanisms can coexist; they are not three independently measured multipliers. The v16 within-cohort accounting is `fixed/native 0.98834 × cut/fixed 1.06723 = cut/native 1.05478`; only cut/fixed is the primary isolated comparison. Do not add v14 combined gains to v16 width gains from a different cohort. Learned-table caching affects another component and requires its own end-to-end contrast; component hash/gather gains are not whole-model multipliers.

## Publication-only measurement mode (v23)

Binary `59bf17e23613a2eb9d8793698d6aca2c41066279b98b676f22057bac4fd29918`; correctness evidence `runs/independent-v23-publication-smoke`. Full MTP round diagnostics can force settlement before successor resolution even on a native round. `MLX_SERVE_LOOKUP_RECEIPTS=commit` preserves complete published IDs and persistent-store identity without synthesizing those round receipts. Actual lookup execution, production pricing and bounded bootstrap probes retain their existing barriers; this is not a synchronization-free decoder.

The rebuilt runtime completed **eight exact 70-token requests**: serial, native MTP with publication-only receipts, adaptive MTP with full receipts, and adaptive MTP with publication-only receipts, each first/repeat. All publications matched serial. Full adaptive mode emitted 34 round receipts; publication-only modes emitted none. Both adaptive modes retained two per-request store receipts, and native/feature MTP engagement checks passed. Trace was explicitly disabled in this correctness smoke.

The selected-chain regression covers off, publication-only and full modes, preserving delayed per-edge lookup learning. Cache-control protocol tests ran with cache disabled and enabled; lookup-filtered runtime tests and ReleaseFast build completed. All **298 Python helper tests passed**. Adversarial review found no actionable receipt-mode, cache-control or full-model-driver defects after the serial-oracle fix.

The separate `runs/independent-v23-mtp-receipts-abba` comparison completed **136 exact responses / 24,276 generated tokens**, with four adjacent full-versus-publication-only ABBA/BAAB quartets and a serial oracle outside them. Both arms use the same binary, adaptive policy, frozen history and lookup tracing. The full-request ratio is **0.98586 (−1.41%)**, with 95% quartet-bootstrap interval **0.96260–1.00003**. Changed swap usage and swap-ins fail measurement qualification. No receipt-overhead speedup was demonstrated.

| Coding case | Publication-only versus full: first | Publication-only versus full: repeat |
|---|---:|---:|
| Fresh function | −3.29% | −1.47% |
| Related helper | −1.37% | −1.01% |
| Related edit | +0.11% | −0.52% |
| Related completion | −1.68% | −2.37% |

Non-diagnostic MTP statistics remain available in both arms: full mode records 3,754 native attempts plus 175 lookup rounds; publication-only records 3,758 plus 187. The full arm emits 3,929 round receipts and publication-only emits none. Missing receipt detail is not zero lookup activity. Because the controller consumes measured costs, removing a barrier can change later decisions; this contrast measures the complete mode change, not just the duration of an isolated synchronization call.

Publication-only mode remains useful for production-like timing, but it does not explain away the coding regression. Earlier full-receipt benchmarks remain historical instrumented results; do not silently reinterpret them as production-timing measurements.

## Flash-Next full-model server admission (v23)

The frozen cache-only serial declaration in `runs/independent-v23-ple-serial-abba` specifies 16 boots / 96 full requests, alternating off/on ABBA/BAAB, followed by a separate MTP declaration only after a completed serial oracle exists. The first server boot failed normal admission: **55.0 GB required versus 46.89 GB available**, with estimated weights of 48.04 GB. Zero benchmark requests ran. The failed declaration and startup log are retained; there is no cache speed estimate, measured-workload hit rate or completed serial oracle from this attempt.

No memory-preflight bypass was used. The earlier eight-token standalone CLI smoke remains execution evidence only: the CLI has a different single-prompt lifecycle and does not establish that the server/repeated-request configuration fits. The component hash/gather results remain valid at their stated scope, not as a substitute full-model claim.

A separate bounded CLI integration check ran the same eight-token prompt once with the cache off and once on, on the v23 binary. Complete published IDs and finish reason matched exactly. Both normal shutdowns reported 48 workers and 36 pool rounds. Off allocated zero decoded payload and recorded zero cache operations; on allocated 655,360 bytes and recorded **144 misses, zero hits**. This proves actual cache lookup/insertion-path engagement, not successful reuse or acceleration on that cold prompt.

Evidence is summarized in `independent-v23-ple-cli-analysis.json`. Both wrapper failures remain visible: the off wrapper initially looked for a serving-only log file instead of CLI stderr; the on wrapper incorrectly required a cache hit from a single cold prompt. The saved completed runtime outputs were analyzed without rerunning either request. This is post-run correctness/lifecycle evidence, not a salvaged performance cohort or a completed server benchmark.

## Publication-only native-versus-adaptive MTP coding (v23)

`runs/independent-v23-mtp-publication-coding-abba` completed **136 exact responses**, four primary adjacent ABBA/BAAB quartets plus an external serial oracle, using the same v23 binary and frozen coding history. Both primary arms use publication-only receipts; lookup tracing remains identical and enabled. The full-workload ratio is **0.93118 (−6.88%)**, with interval **0.92147–0.94836**: a regression, not a speedup. Swap usage and swap-ins changed, so hardware qualification also fails; that does not excuse the unfavorable result.

| Coding case | Adaptive versus native: first | Adaptive versus native: repeat |
|---|---:|---:|
| Fresh function | −3.23% | −2.25% |
| Related helper | +4.91% | +5.86% |
| Related edit | −25.74% | −25.45% |
| Related completion | +1.89% | −2.56% |

The native MTP reference includes its existing prompt-lookup behavior. Across the 16 related-edit requests it performs **476 native attempts plus 160 legacy lookup rounds**, accepting 1,650 of 1,904 lookup drafts. The tiered feature performs **975 native attempts plus 43 lookup rounds**, accepting 249 of 359 lookup drafts. The feature is losing useful incumbent reuse, not merely spending too much CPU searching suffixes. Publication-only evidence does not expose per-round candidate/order decisions.

The exact-order trust change is a candidate cause of lost support, not an isolated timing conclusion from comparisons across revisions. Two independently reviewed corrections are under validation: rate-aware MTP width exploration and preservation of native DFlash2 geometry when lookup adds no selected path. Neither is part of this measured v23 binary; neither may inherit a performance claim from earlier cohorts.

## Rate-screened exploration and native geometry retention (v24)

Binary: `b84d481e02925c4422d31b7a2423f5f1a693ae645a03c2ed9272d2e7d444bffa`. MTP upper-width exploration now compares `predicted wider tokens / (paid query + measured narrow verification)` against `predicted native tokens / (paid query + native remaining duration)`. This is a zero-added-verification-latency screening hypothesis, not a measured wider price or a mathematical latency bound. Losing hypotheses do not consume the bounded reservation. The actual wider action remains unpriced until its own eligible execution.

DFlash2 now retains recorded native admission geometry at native width when every selected draft edge already has neural support. Lookup corroboration alone cannot rerank the native incumbent. Genuinely new lookup paths and other widths still compete using their actual selected geometry and kernel variant. Lookup feedback retains its immutable offered tree independently of the selected native tree.

Validation caught and repaired two depth-contract defects before the final build: output-token budgets constrain draft depth, not sibling row capacity; `prepareTree` must not subtract the anchor twice. The regression exercises actual history queries at draft depths zero, one and fifteen plus two siblings at depth one. A fixture's unintended longer suffix was corrected without changing production semantics. The 83 focused adapter/policy/history tests passed; the corrected lookup-filtered integrated suite and ReleaseFast build then completed. Adversarial reviewers found no remaining depth-contract or rate-screening blocker.

`runs/independent-v24-native-candidate-diagnostic` executed eight complete, serial-identical responses with zero invalid samples. All **16 equal-width, equal-variant, uncensored witnesses tie: 128 native versus 128 selected serial-path rows**. Eighteen witnesses joined actual published rounds; terminal-censored witnesses are excluded from that comparison. This confirms the exercised native-path retention, not a throughput improvement.

The diagnostic's overall reuse gate **fails**: no persistent-origin nodes are selected after retaining the neural incumbent at fixed width eight. It is not a completed genuine-reuse benchmark, and the failed declaration is preserved. Exact publication agreement and same-prefix geometry are reported separately rather than weakening the activation gate or claiming copied neural paths as persistent-cache benefit. Adaptive wider lookup requires its own measurement.

`runs/independent-v24-publication-smoke` additionally completed eight exact 70-token responses across serial, native MTP and both adaptive receipt modes. Full adaptive mode emitted 37 round receipts; publication-only adaptive mode emitted none, while both retained per-request persistent-store identity. This exercises the rebuilt MTP path without treating the smoke as a performance comparison.

The next declared coding contrast uses preserved **v23 exact-order support versus v21 pooled request support**, not v24. It includes all four coding cases and first/repeat exposures, four adjacent ABBA/BAAB quartets and a serial oracle. This tests the complete binary-policy tradeoff before introducing continuation-state machinery. Pooled support can revive useful growing-copy matches but can also transfer long-match confidence to unrelated shorter matches; exact verification remains the output-correctness boundary in either case.

## Request-support isolation: MTP coding (v24 experiment, preserved binaries)

`runs/independent-v24-mtp-request-support-abba` compares **v23 exact-order request support** (`59bf17e2…`) with **v21 pooled-order request support** (`064dbb50…`). Both use the same combined-adaptive controls, frozen history, row-exact arithmetic and full diagnostic receipts. It does not use the v24 runtime or compare against native MTP. All **136 responses matched serial exactly** across four adjacent ABBA/BAAB quartets and a serial reference outside them.

The pooled/exact-order full-workload speed ratio is **1.09379 (+9.38%)**, with descriptive quartet-bootstrap interval **1.07228–1.12017**. All four complete-workload quartet ratios favor pooled support: 1.06443, 1.13292, 1.08277 and 1.09619. Swap usage and swap-ins changed, so hardware qualification fails. The fresh-code first-exposure cell in the first quartet regresses 8.11%; it is retained, not hidden by the aggregate.

| Coding case | Pooled versus exact-order: first | Repeat |
|---|---:|---:|
| Fresh function | −0.48% | +0.24% |
| Related helper | +2.38% | +2.96% |
| Related edit | +31.73% | +32.16% |
| Related completion | +4.46% | +6.41% |

Editing's realized published rounds fall **1,022→623**, while lookup rounds rise 26→180 and lookup-committed tokens rise **218→1,861**. Accepted lookup drafts rise 192→1,681 out of 364→2,520 offered drafts. Fresh-code rounds are 1,289→1,281, helper rounds 962→903 and completion rounds 662→604. This supports recovered productive reuse rather than merely more lookup attempts. These are complete treatment trajectories, not independently timed counterfactual rounds.

Adversarial inspection also found a same-prefix witness in the first edit exposure: after five identical three-token rounds, pooled support accepts a local order-17 offer and commits 15 tokens, while exact-order support executes native MTP and commits three. Pooled support then fully accepts local offers at orders 32, 47 and 62; exact-order support declines the growing orders until 64. This directly demonstrates profitable copy opportunities lost through order fragmentation, without claiming counterfactual timing. The pooled sample also includes an order-six 14-draft miss, so pooling does not make every admission profitable.

The evidence favors evaluating the simpler request-support rollback before adding continuation cursors or provenance-state machinery. It does not isolate the 8 KiB table footprint from trust-key behavior, eliminate long-to-short confidence contamination, establish publication-only timing, or prove the shared change is beneficial to DFlash2. A corresponding preserved-binary DFlash2 comparison is declared separately. Exact-order model calibration and exact target verification remain in both policies; the tradeoff concerns admission performance, not permission to publish incorrect drafts.

## Request-support isolation: DFlash2 coding (v24 experiment, preserved binaries)

The corresponding `runs/independent-v24-dflash-request-support-abba` contrast uses the same v23 exact-order and v21 pooled-order executables, frozen coding history and combined-adaptive full-receipt controls. All **136 responses matched serial exactly**. The full-workload pool/exact ratio is **0.99174 (−0.83%)**, with interval **0.97736–1.00632**; changed swap usage and swap-ins fail qualification. This is inconclusive overall, not proof of equivalence.

| Coding case | Pooled versus exact-order: first | Repeat |
|---|---:|---:|
| Fresh function | −0.34% | −1.03% |
| Related helper | +1.80% | −0.30% |
| Related edit | −1.44% | −2.51% |
| Related completion | −1.30% | −1.33% |

All four edit-repeat quartet ratios favor exact-order support: approximately 0.99093, 0.94573, 0.96835 and 0.99526. Their aggregate interval is **0.95683–0.99309**. Fresh first exposure also has a quartet at 0.93478, below the existing regression guard. These are meaningful cautions despite failed hardware qualification; neither parity nor no-tradeoff is established.

The smaller pooled-support implementation is selected as a candidate because of its MTP reuse recovery and reduced complexity, not because it dominates both families. The final combination with native DFlash geometry retention and rate-screened MTP exploration still needs its own two-family comparison. A large MTP gain cannot average away a DFlash regression, and effects from separate cohorts must not be multiplied.

## Combined correction candidate (v25)

Binary: `63aa73d85a8fdc526763cfb8105530761055a2ec609ffe4d77dbe772bc34c90f`. This combines rate-screened MTP exploration, native DFlash2 geometry retention and pooled request support. Model calibration remains exact-order; request support retains separate entry/continuation and confidence bins, reached-only learning, censoring and reset. Removing the larger request table saves **8 KiB/request**. The new growing-copy regression exercises actual price-based admission across orders 8→11→14→17 and rejects a separately miscalibrated short order despite pooled request support. All **83 focused tests passed**, followed by the lookup-filtered integrated suite and ReleaseFast build. Adversarial review found no remaining rollback defect after correcting stale test API calls.

`runs/independent-v25-publication-smoke` completed eight exact 70-token requests. Full adaptive mode emitted 31 round receipts; publication-only adaptive mode emitted none, while both retained store identity. Seven full-mode preparation records exercised carried query-to-handoff plus resumed-preverify accounting; all totals equaled those disjoint parts. No preparation records appeared in serial or publication-only modes. Their summed 8.81 ms is diagnostic scope on these seven rounds—not an isolated penalty or speedup.

`runs/independent-v25-preparation-fixed16` additionally completed eight exact-valid requests with zero invalid samples. Its 20 preparation records cover **two cold and 18 carried** intervals; every complete total matched its declared partition and cold records had no resumed charge. This forced-width smoke is not a production-adaptive throughput result.

Preparation records are immutable flight-owned evidence, not next-query scratch. They add no GPU work or synchronization and do not yet feed routing or prices. Their optional inline field occupies **96 bytes** even when logging is disabled. The interval includes shared history work, policy/logging, materialization and existing settlement; `incremental_penalty_ns` remains null. Missing query/handoff boundaries and shadow requeries must not be treated as free preparation.

### Completed native-versus-combined coding comparisons

`runs/independent-v25-mtp-coding-abba` and `runs/independent-v25-dflash-coding-abba` each completed **136 exact responses**: four adjacent ABBA/BAAB quartets (128 primary requests) and eight external serial-oracle requests. Both arms within each family use the same v25 runtime and frozen history. MTP uses publication-only receipts; DFlash2 uses full diagnostic receipts. These are within-family comparisons, not a direct MTP-versus-DFlash2 race.

| Family | Combined/native speed ratio | Change | 95% quartet-bootstrap interval |
|---|---:|---:|---:|
| MTP | 1.011923 | +1.19% | 0.989748–1.027778 |
| DFlash2 | 0.989293 | −1.07% | 0.969592–1.007108 |

Neither interval establishes a speedup or equivalence. Both cohorts fail hardware qualification because swap usage and swap-ins changed. The unfavorable cells remain part of the result; paging does not excuse them.

| Coding case | MTP first | MTP repeat | DFlash2 first | DFlash2 repeat |
|---|---:|---:|---:|---:|
| Fresh function | −7.92% | −1.52% | −2.82% | −5.70% |
| Related helper | +9.20% | +10.92% | −2.43% | +0.11% |
| Related edit | −3.34% | −1.58% | −0.73% | +1.11% |
| Related completion | +3.91% | +2.85% | +2.03% | +1.47% |

MTP helper reuse improves while fresh-code first exposure loses. First fresh requests spend 73.958→80.992 seconds in decode, against only 8.436→8.473 seconds in prefill. Every feature boot has two 14-draft fresh lookup misses at orders four and one. Their gross verification duration is not an isolated incremental penalty. Related-edit lookup accepts 820 versus 824 drafts on first exposures while offering 1,260 versus 952: more offered tokens do not automatically mean more useful tokens. Intermediate-width chain prices are missing; native bootstrap and upper-width observations do not price the intervening actions.

For DFlash2 fresh code, total rounds fall 560→544 but verified rows rise 4,480→4,848 and measured verification/commit time rises 67.987→71.757 seconds, with identical 3,488 committed tokens. Of 154 width-16 selections across cases, 138 are ordinary policy selections, not forced upper trials. Suppressing exploration alone therefore cannot explain or repair this result. Existing native-tree witnesses do not reconstruct a mixed, actually priced native-width alternative; that diagnostic must retain the alternative selected during the original decision.

These results measure the combined correction, not independent contributions from pooling, geometry retention, width screening or lookup. Earlier component and workload gains cannot be multiplied into a combined speedup.

## Flash-Next successful cache reuse smoke (v25)

`runs/independent-v25-ple-reuse-cli-smoke` completed one cache-off and one cache-on CLI request. Both stopped with **59 identical published tokens**, producing the requested twelve repeated lines. Both used 48 workers. Cache-on recorded **840 hits and 152 misses** and reduced pool rounds from 91 to 39. This exercises real full-model reuse, rather than only insertion.

The decoded row payload is **655,360 bytes (640 KiB)** when enabled and zero when disabled. Existing metadata is 8,480 bytes in either arm; full enabled cache accounting is 663,840 bytes (648.28 KiB). There is no additional on-disk table. Printed resident-model allocator peak is 45.914 GB in both runs, too coarse to resolve that payload.

Printed decode rates are 35.314 versus 35.602 tokens/second. This single sequential pair is not a performance claim. Load-inclusive outer durations (59.335 versus 43.051 seconds) include materially different prefill/loading behavior and must not be presented as a cache speedup. A balanced repeated CLI comparison must report decode and load-inclusive time separately; it remains distinct from the server configuration that failed normal memory admission.

The subsequent 32-request declaration `runs/independent-v25-ple-cli-abba` failed on its second request, before any cache-on arm. The model completed 407 tokens, but the coding output included explanation and Markdown and failed the unchanged Python-source gate. The CLI mapping had omitted the contract's system instruction requiring source-only output. Both completed observations and the failed cohort remain intact; no performance estimate is available.

A separately declared schema-two workload explicitly flattens system content, two newlines and user content into the CLI's single user message. This is documented input adaptation, not HTTP role parity. It does not strip model output, weaken the source gate, change token caps, or resume the failed schedule.

`runs/independent-v25-ple-cli-instructions-abba` also stopped at its second off-arm request. The adapted prompt completed 165 tokens, but the model still returned a Markdown-fenced answer and failed the unchanged source gate. There were no cache-on observations and no performance estimate. This is a remaining model/workload-quality limitation, not evidence for or against the cache's speed. Do not strip the fences or reinterpret either failed declaration as a completed comparison.

A new normal-admission server attempt, `runs/independent-v25-ple-server-headroom-abba`, used the preserved v25 runtime and the unchanged 96-request server declaration. Despite a low-pressure system reading, actual preflight reported **51.38 GiB available versus approximately 55.0 GiB required**. It stopped before any request. Further inspection found a stronger constraint: `scheduler.loadAvail` takes the minimum of host availability and Metal's recommended working-set limit. The actual device query returned **55,662,788,608 bytes (51.84 GiB)** with `iogpu.wired_limit_mb=0`. Closing apps alone cannot satisfy this pack's approximately 55-GiB admission requirement under the current ceiling. No guard or global GPU-memory override was used. The server runner's process-creation cancellation fix was independently reviewed and exercised with a real owned child before this attempt.

## Actual competing-width geometry (v26 diagnostic)

Binary: `218774a5a0d2292b361cd8105e558ad357e2041315f577c06b9d95ac61737f15`. The diagnostic retains the selection evaluation's actual native-width prefix mask and incumbent-retention decision. It reconstructs that alternative without rerunning selection, rebuilding the neural lattice or executing another target forward. A mixed alternative is not silently replaced by the original native tree. Emission requires full receipts and price tracing; this is not timing-neutral.

The build passed **83 focused Zig tests and 310 Python helper tests**, followed by the lookup-filtered integrated suite and ReleaseFast build. The eight-request `runs/independent-v26-competing-candidate-diagnostic` completed with zero invalid samples and exact serial publication. Its 14 joined native witnesses include eight uncensored equal-width/equal-variant ties (64 versus 64 serial-path rows).

Four additional witnesses compare the actual priced eight-row alternative with the chosen sixteen-row action. All four are uncensored and cover **eight versus sixteen serial-path rows**. Two are explicitly unpriced upper trials; two are ordinary priced selections. For the latter, the decision's predicted verifier costs are 86.364 ms (eight-row generic) and 124.421 ms (sixteen-row linear-GDN). These are historical decision inputs, not same-state executed alternate timings. This small repeated-code example demonstrates useful wider path coverage, not a throughput claim or an explanation of the fresh-coding regression.

Adversarial review found no remaining diagnostic or analyzer blocker. Analysis also rejected malformed widths/rate arithmetic and censored terminal output in CPU boundary checks. Earlier equal-width evidence still reduces to its original 16 ties / 128 versus 128 rows with the extended analyzer.

The subsequent `runs/independent-v26-coding-geometry-diagnostic` covers all four coding cases and first/repeat exposures: **16 exact responses across serial and adaptive boots**, with no invalid samples. It deliberately produces no throughput estimate. Its 193 native witnesses include 170 uncensored equal-width comparisons: 167 ties, two one-row losses and one three-row loss (1,137 native versus 1,132 selected path rows). Genuinely new lookup paths can still displace native paths; retaining corroborated neural geometry does not prohibit that competition.

| Coding case | Wider choices | Forced choices | Choices with no extra path rows | Predicted extra rows, summed | Actual extra serial-path rows, summed |
|---|---:|---:|---:|---:|---:|
| Fresh function | 5 | 2 | 4 | 8.422 | 4 |
| Related helper | 6 | 0 | 2 | 11.404 | 16 |
| Related edit | 3 | 0 | 0 | 9.826 | 12 |
| Related completion | 1 | 0 | 0 | 2.950 | 2 |

All fifteen competing-width witnesses are uncensored; two editing witnesses exercise **mixed alternatives**, not retained native geometry. The contrast exposes fresh-code marginal-yield overprediction in this trajectory while related helper/edit paths benefit from extra coverage. It does not time the unexecuted smaller actions, and these diagnostic counts cannot be converted into a speedup percentage. The same choice can reduce rounds yet lose wall time when its extra verification cost exceeds its extra useful coverage.

## Production chain-price acquisition: complete requests, failed qualification

`runs/independent-v25-mtp-chain-training` executed all **232 declared requests**, with exact serial publication and zero request-validation errors. The acquisition took 1,325.15 seconds overall; summed request time was 1,120.76 seconds. All fourteen supported context-zero chain/generic widths (2–15) had post-cold observations from both opposite-order sweeps. Receipt membership includes 28 first-actual-execution exclusions, 5,324 otherwise eligible observations and 152 inadmissible executions.

The cohort failed its declared hardware gate because swap usage and swap-ins changed. Endpoint evidence shows swap-ins during **174 of 232 requests**, totaling 29,615 system page-in counter increments; there were no request-aligned swap-outs. This is not merely a boot-only qualification failure, and the counters do not identify which process paged. No price profile was exported; profile-off/on evaluation is therefore not yet available.

The retained diagnostic population has median verification/commit durations around 117–122 ms at widths three through eight and 169–172 ms at widths nine through fifteen (width two: 98.97 ms). These are different-prefix, different-acceptance production populations under unqualified machine conditions, not a measured same-state hardware staircase. They justify investigating intermediate widths but do not establish their causal performance benefit or authorize filling missing production prices.

The CPU declaration and twelve profile-contract tests passed, including the reviewed relative-output-path fix. Adversarial review found no remaining acquisition/export or provenance blocker. The outstanding prerequisite for qualified export is a stable measured acquisition environment, not a configured profile filename or another unmodified retry.

## Actual native action remains eligible (v27)

Binary: `53b9c4fd3935de90b43cdc693612cff67724951ab45f3fc5dca629e29954cfbe`. The earlier incumbent-retention rule protected native only when the mixed prefix had no lookup-only edge. Otherwise, the mixed action replaced native at its width—even if native had a known kernel price and the mixed kernel was unknown. Bootstrap could consequently execute an unpriced mixed action under the native label.

The correction evaluates native and mixed candidates using their respective rewards and actual kernel prices, choosing the better eligible representative at that width and retaining native on exact ties. Bootstrap now prepares the actual native action. It adds no new price estimator, confidence shrinkage, near-tie margin, GPU operation or heap allocation. Explicit fixed-width controls keep their row count; upper exploration keeps its bounded reservation rules.

Both added regression tests failed before the fix. They cover cold/native-only/dominated/faster mixed kernel populations, native exact ties and legitimately better lookup candidates. All **85 focused tests passed**, followed by the lookup-filtered integrated suite and ReleaseFast build. Adversarial review found no remaining runtime-consumer, bound, forced-width or diagnostic reconstruction defect.

`runs/independent-v27-native-competition-diagnostic` completed **16 exact coding responses**, first/repeat across all four cases, with zero invalid samples. All **183 uncensored equal-width/equal-variant native comparisons tie**, totaling 1,244 versus 1,244 serial-path rows. Thirteen wider comparisons remain, including six with no extra coverage. This validates the exercised geometry and complete publications, not a speedup. In particular, native already competed in the earlier fresh-code near-tie; this structural correction does not claim to fix that marginal-yield prediction.

The isolated `runs/independent-v27-native-competition-abba` comparison uses v26 versus v27 with identical combined-adaptive controls and frozen history, not a native baseline. All **136 responses matched serial exactly**. The full-workload ratio is **1.00603 (+0.60%)**, with quartet-bootstrap interval **1.00049–1.01160**. Quartet ratios are 1.00101, 1.01372, 0.99998 and 1.00949. The small observed gain does not meet the declared 5% practical margin, and changed swap usage/swap-ins fail hardware qualification.

| Coding case | v27 versus v26: first | Repeat |
|---|---:|---:|
| Fresh function | −2.87% | +0.85% |
| Related helper | +0.27% | +1.38% |
| Related edit | +0.95% | +2.59% |
| Related completion | +1.57% | −0.25% |

Fresh first exposure regresses in all four quartets; edit repeat improves in all four. Neither pattern is erased by the aggregate. The structural correction remains justified by native-action eligibility and the reproduced dominated-action case, not by a universal performance claim. A direct native-versus-combined comparison is separate.

Adversarial attribution review found different work-count evidence for those cases. Across eight edit-repeat requests, rounds fall 208→200 and verified rows fall 1,696→1,616. That is a real reduction, although it cannot account for the entire measured timing difference. Fresh-first instead changes 271→272 rounds and 2,440→2,432 rows; seven of eight earlier requests already have the same width schedule and accepted-token sequence as all eight corrected requests. Large settled-duration differences also occur on matching offered geometry, dispatch and acceptance. These records do not isolate the fresh regression to increased work, nor establish hardware variability as its sole cause. No fitted near-tie margin or reward rescaling was added.

The direct native-versus-v27 comparison, `runs/independent-v27-dflash-coding-abba`, subsequently completed all **136 responses exactly**. Its combined full-workload ratio is **1.00479 (+0.48%)**, interval **0.99110–1.02281**; the four quartet ratios are 1.00420, 1.03335, 0.99035 and 0.99184. Swap usage and swap-ins again fail hardware qualification. This is an observed near-neutral aggregate, not established acceleration over native DFlash2.

| Coding case | Combined v27 versus native DFlash2: first | Repeat |
|---|---:|---:|
| Fresh function | −4.60% | −1.68% |
| Related helper | +0.10% | +2.15% |
| Related edit | +2.62% | +3.50% |
| Related completion | +0.51% | +0.97% |

Fresh-first loses in every quartet; edit-repeat improves in every quartet, with interval 1.00958–1.07212 before hardware qualification. These opposing workload effects are why isolated component gains must not be added together or advertised as a general stacked speedup.

The direct native comparison makes the workload tradeoff concrete. Across eight fresh-first requests, rounds fall 280→272 while verifier rows rise 2,240→2,448 (**+9.29%**). At anchors 130, 138 and 229, every feature request verifies sixteen versus eight rows yet commits the same eight tokens as native: 192 extra verified rows without extra coverage at those decisions. The first two anchors are bounded learning actions; the third is ordinary priced selection. Widening elsewhere does add coverage and reduces rounds, so suppressing every wider action is not justified.

Edit-repeat instead falls 216→200 rounds and 1,728→1,640 verifier rows. Assistant-bearing rounds rise 144→198: native already has a useful local-copy bypass, which the combined policy often trades away. The observed edit gain therefore is not evidence that the feature primarily saves assistant calls. Adversarial review found no reporting defect; these actual work differences explain mechanisms without assigning an isolated millisecond cost to unexecuted alternatives.

## MTP post-query preparation correction (v28)

Binary: `6a5750780959964f46414c5d1b98afcdde9dc47a618248ded2ad4f536feb1a14`. Row-exact MTP native-cycle admission now adds each lookup width's known post-query preparation before selecting the best width. Native cost remains unchanged. Cold/carried estimates are separate; missing preparation preserves the prior hypothesis without inventing an observation or reopening verifier exploration. This corrects omitted empirical lookup work, not exact same-state total-action pricing.

Verification passed **92 focused Zig tests and 310 Python helper tests**, the lookup-filtered integrated suite and ReleaseFast build. Adversarial review found no remaining implementation defect. Review of the new smoke caught a missing executed-width check; the corrected gate rejects a requested fifteen-row action executing eight rows while allowing legitimate capped widths.

`runs/independent-v28-preparation-accounting-smoke` completed **eight exact 70-token responses** across serial, native MTP, forced-width lookup with full receipts, and the same lookup control with publication-only receipts. The full arm produced twenty eligible preparation records: two cold records with one learned sample, and eighteen carried records with fifteen learned samples after per-cell cold exclusions. Both feature arms exposed twelve admissions with positive learned carried preparation. Publication-only emitted no full round/preparation records. This proves exercised acquisition and complete publication, not adaptive performance.

Cold preparation measured 0.050–0.108 ms and carried preparation 0.672–2.643 ms in this smoke; these are active wall intervals, not pure GPU or incremental switching costs. The measured estimator storage is **9,720 bytes per model policy**, with no added disk file. The optional flight snapshot is **112 bytes**, up 16 bytes from v25. Separate v27/v28 publication-only ABBA evaluates whether this correction improves real coding latency.

The initial isolated comparison, `runs/independent-v28-mtp-preparation-abba`, stopped after **96 request-valid observations** and before its serial-reference boot. The runtime guard detected a hostname change, `Mac.localdomain`→`Jason-M2-MacBook.local`; comparison of both frozen executable identities found no other changed field. The cohort remains incomplete and has no throughput estimate or complete serial-exactness claim.

Runtime identity schema two excludes the network hostname while retaining OS/kernel build, architecture, dependency resolution, executable/library hashes and Metal assets. The hostname regression failed before this correction and the eight runtime-identity tests passed afterward. Actual installed-runtime inspection covered seven images and thirteen Metal candidates and remained identical under a simulated hostname change. This fixes a false interruption, not a hardware qualification failure; the old cohort is not resumed or reclassified.

## Native DFlash2 backed-copy preservation (v29)

Binary: `e357c1c866a329f7c7aaa8a87416aaf1bda95d1ccbb5681331cbf6e0a55a7031`. The earlier tiered route disabled the host's backed-copy shortcut and could invoke the assistant for an already available seven-token copy. At the retained v27 edit-repeat prefix 285, the local offer contained that continuation, but its discounted predicted yield lost to the assistant comparator. This was loss of an existing action's identity/economics, not absence of its tokens.

The correction retains the actual native copy as a distinct candidate, including its backing length and silence eligibility. Unknown economics preserve that action; known copy competes using paired actual remaining duration/yield, without a perfect-copy prior or fitted margin. It executes preselected geometry rather than appending copies to a selected tree. Copy and assistant remaining clocks now share the post-query, pre-policy boundary. Therefore the isolated before/after contrast includes baseline-action preservation **and this required clock alignment**, not an independently toggled copy-only effect.

Adversarial review caught the clock mismatch before integration. Parent integration also reproduced and fixed a receipt-price mismatch: a selected copy had inherited the competing lookup/plain price. Unknown selected-copy prices now remain null; known selected-copy prices match that action. Losing-copy economics remain separately visible. Two regression assertions failed before the reporting fix. Final verification passed **101 focused Zig tests, 316 Python tests, all five targeted native-copy runtime regressions and ReleaseFast compilation**; the lookup-filtered integrated suite also passed during integration.

`runs/independent-v29-native-copy-diagnostic` completed **sixteen exact responses** across the four coding cases and both exposures, with zero validation errors. Editing offered nine native copies in each exposure. First exposure selected six copies—two with unknown economics and four with known economics—committing 42 tokens with 36 accepted drafts. Repeat selected one known copy, committing two tokens with one accepted draft. No native-copy opportunities appeared in the other three cases. The diagnostic proves unknown/known execution and complete output, not a latency gain; its price tracing perturbs the measured populations.

Measured state growth is **960 bytes per existing context-map value**, excluding spare map capacity; query scratch is **256 KiB of bounded stack space**. No new disk file or second estimator map is added. Native-copy selected geometry has its own identity; the repository component reducer excludes it from lookup-offer pruning counts. The subsequent four-quartet v28/v29 comparison uses full receipts without price tracing.

The isolated `runs/independent-v29-native-copy-abba` comparison completed **136 exact responses**. Its full-workload ratio is **1.00506 (+0.51%)**, quartet-bootstrap interval **0.99533–1.01366**; quartet ratios are 0.98990, 1.00321, 1.01181 and 1.01552. Swap usage, swap-ins and swap-outs changed, failing hardware qualification. The interval also includes no improvement. This does not establish acceleration from the correction.

| Coding case | v29 versus v28: first | Repeat |
|---|---:|---:|
| Fresh function | −0.48% | +1.24% |
| Related helper | +2.01% | +1.43% |
| Related edit | −1.88% | +1.37% |
| Related completion | +0.36% | +0.02% |

Editing first exposure regresses in all four quartets. Across eight such requests, the correction saves assistant-bearing rounds (172→143), but increases total rounds (204→211) and verifier rows (1,640→1,704). Its 64 selected copies commit 445 tokens, including 381 accepted drafts. Summed assistant draft duration falls 2,212.64→1,906.45 ms while verification/commit duration rises 24,808.77→26,268.97 ms. Preserving a cheap draft source is therefore not sufficient: its realized coverage and verifier work can outweigh the draft saving. These are executed-work observations, not same-state alternative timings.

Edit repeat instead changes 200→199 rounds, 1,632→1,624 verifier rows and 199→191 assistant-bearing rounds. Five selected copies accept all 35 draft tokens. Copy opportunities remain absent from the other three cases; their timing changes cannot be attributed to copy execution. The comparison also contains the shared remaining-clock correction. Raw `published_width_reductions` from this private driver are not isolated StairCut attribution; selected native copies must be excluded before interpreting such counts. Direct native-versus-v29 evaluation is a separate cohort.

Independent adversarial review found no remaining clock, action-identity, low-yield retention or bounded-state defect in the copy integration. Prefix-level work evidence retains the unfavorable outcomes: all eight first-edit copies at anchor 341 accept one draft, while the earlier arm executes four assistant actions accepting seven and four lookup actions accepting one. Later copy selections at anchors 387 and 415 accept three and two drafts respectively. Historical paired economics do not guarantee future copy coverage; the review does not turn this measured regression into a speedup.

The separate direct-native cohort, `runs/independent-v29-dflash-coding-abba`, completed **136 serial-exact responses**. Combined v29 versus native DFlash2 measures **+0.66%** overall (ratio 1.00662), with quartet-bootstrap interval **0.99522–1.01726** and quartet ratios 1.02178, 1.00383, 0.98996 and 1.01120. Changed swap usage and swap-ins fail isolated-hardware qualification; the aggregate interval also includes no improvement.

| Coding case | Combined v29 versus native DFlash2: first | Repeat |
|---|---:|---:|
| Fresh function | −2.67% | +0.42% |
| Related helper | +1.12% | +2.95% |
| Related edit | +1.54% | +1.56% |
| Related completion | +0.91% | −1.70% |

Related-helper repeat has interval 1.00802–1.04738, while completion repeat has interval 0.96938–0.99686, before hardware qualification. The aggregate is near-neutral and the workload effects are mixed; neither exact output nor a majority of positive point estimates establishes general acceleration. All contrasts retain the same explicit row-exact arithmetic and incoai DFlash2 drafter, not a fastest-stock-arithmetic comparison.

## Prospective PLE CLI response-format protocol

The two earlier failed CLI cohorts remain failed and are not reclassified. A new private schema-three declaration, `runs/independent-v25-ple-cli-assistant-python-abba`, retains the same two schema-two workloads, prompts, frozen v25 binary, 512-token limits, four quartets and 32-process schedule. Its explicit prospective quality contract permits raw Python or one complete Python Markdown fence, rejecting outside prose, multiple/malformed fences and non-stop publications. Extracted source is used only for the unchanged syntactic/structural checks; complete raw text, prompt/generated token IDs, usage and finish reason must still match across arms.

This changes response-format acceptance, not the model invocation or task. It does not prove compliance with the source-only instruction or functional correctness of generated Python. CLI invocation already uses the model's chat formatter; flattened system words are not a true system-role message. The new declaration binds both failed predecessor plan hashes and exact schema-two cases, rejecting historical plans/foreign observations rather than pooling them.

Independent adversarial review found no remaining parser, identity, reducer or provenance defect. Seventeen throwaway parser/quality/schema/provenance checks passed, followed by a complete synthetic reducer case and eight corruptions that all withheld performance estimates (raw-format/whitespace/ID mismatches, missing/foreign protocol stamps, non-stop output and incomplete cohort). The actual declaration command completed without loading a model. No schema-three model execution or new PLE inference performance result is claimed.

## Optimization checkpoint: preparation result and overall-process review

The freshly declared `runs/independent-v28-mtp-preparation-hostname-fixed-abba` completed **136 exact responses**, comparing v27/v28 with identical adaptive MTP controls and publication-only receipts. The isolated preparation-accounting effect is **+0.39%**, ratio 1.00387, interval **0.97037–1.03510**; quartet ratios are 1.00393, 1.04571, 0.95736 and 1.01046. Swap usage and swap-ins fail qualification. This is not an established preparation speedup and is not a new native-relative comparison.

| Coding case | v28 versus v27: first | Repeat |
|---|---:|---:|
| Fresh function | +1.58% | +0.70% |
| Related helper | −0.79% | −1.40% |
| Related edit | +1.21% | +0.41% |
| Related completion | −2.00% | +3.55% |

Across eight fresh-first requests per arm, the same sixteen upper-width learning actions each draft fourteen tokens and accept none. Ordinary one-draft lookup executions fall 13→10, but native-head rounds rise 631→635, leaving total rounds 660→661. All 27 executed fresh v28 lookup actions, including the one repeat action, have unknown preparation cost at selection. Positive preparation prices occur on related workloads, but the fresh improvement cannot be attributed to directly pricing away these bad fresh selections.

The earlier v25 fresh-first ledger likewise contains sixteen learning misses plus ten ordinary one-draft selections, accepting only three lookup drafts. Their gross logged lookup duration is 444.73 ms/request versus the approximately 879.25-ms observed decode gap. These durations replace native actions; neither their sum nor an assumed recovery charge is an incremental saving. The source applies and settles existing head history on lookup entry, and two existing transition observations show no consistent large recovery spike. A new recovery surcharge would risk double counting.

The hypothesized long-order-to-short-order trust transfer is not established as the fresh-first cause: no preceding offer exceeds order four before the first learning miss in any of the eight requests. Long-hit/short-miss witnesses exist elsewhere, but useful intervening short matches also exist. Do not restore exact-order request fragmentation or fit a short-match threshold to these fixtures without an attainable-latency ledger and an isolated full-request contrast.

### Adversarial process findings and resulting priorities

- **P1: production opportunity cost remains unmeasured.** Current exact-mode comparisons do not establish improvement over the fastest supported production-native configuration. Keep exact-mode feature value separate from the numerical-mode/production gap; never disable exactness to manufacture a win. A numerical-mode diagnostic must retain output differences instead of treating them as equal-work acceleration.
- **P1: stop scheduling broad performance cohorts under an unchanged failed environment.** In the latest DFlash native comparison, 55/136 request intervals have swap-ins, totaling 6,508 increments; the preparation comparison has 54/136, totaling 5,585. Neither has request-aligned swap-outs, and every request reports normal pressure on AC. This does not establish model-induced memory pressure. A subsequent sixty-second diagnostic with no model loaded still observes 182 swap-in increments, with unchanged swap usage/outs. Background paging persists independently of inference; its owning process is not identified.
- **P2: optimize attainable latency, not counters.** The stable learning misses justify investigation, not credit for removing their entire gross duration. Reconcile preparation, native alternatives, actual coverage and useful admissions before adding another policy. Copy-call savings alone already failed to establish a gain.
- **P2: close the PLE end-to-end question before expanding its cache.** The validated thirty-two-process CLI cohort remains the next independent performance question when hardware prerequisites permit it. A null result closes that workload's current speed hypothesis; it is not permission to enlarge the cache automatically.

The proposed 576-request broad component package and 64 canonical measurement/replay windows are deferred, not cancelled or replaced by narrow claims. First establish comparator relevance and measurement feasibility; then run the direct native/adaptive contrast that answers a deployment decision. Further DFlash copy-policy changes rank behind these gaps. Sushi work is deferred at the user's request.

Progress checkpoints retain the hypothesis, observed work changes, isolated timing, native-relative timing, incremental resources and decision to keep/change/defer. Repeat the overall-process adversarial review after each material frozen-runtime change and before a broad cohort, not merely after individual fixes.

### Mechanism-reporting correction after v29

The private coding reducer now excludes selected native copies from its raw width-reduction count and reports selected-copy rounds separately. A throwaway execution of the actual reducer expressions distinguishes copy, ordinary reduction, equal-width and zero-commit cases; the completed isolated v29 candidate records contain 69 selected-copy rounds. Historical summaries and qualifications were not rewritten. This corrects attribution, not latency, and an observed reduction still does not isolate a causal StairCut speedup.

### Native arithmetic opportunity diagnostic

A separate prospective diagnostic runs the frozen v29 executable with native DFlash2 and native MTP, each in stock / row-exact / row-exact / stock order. Each fresh server runs the same four complete coding cases, first and repeat: 64 requests total. Existing native lookup/copy defaults remain enabled. Both modes use publication-only receipts with lookup tracing disabled. The harness retains token identities, output lengths, finish state, syntax observations and latency separately; divergent outputs are not equal-work speedups. One quartet per family is descriptive evidence, not a statistical promotion gate or cross-family race.

Publication-only validation now accepts absent DFlash round diagnostics only when the arm explicitly requests `commit` receipts and native DFlash statistics establish actual engagement. Wrong-family or zero-attempt statistics remain failures, as do absent publication receipts; full-diagnostic mode still requires its rounds. The missing-round rejection reproduced before the fix, and all 57 benchmark-helper tests passed afterward. This reporting change does not alter inference code or certify mechanism attribution.

The diagnostic completed **64 valid publications**, with no syntax/structure failures. All eight DFlash case/exposure cells and seven of eight MTP cells have identical publications across their four observations. MTP fresh-code repeat differs within stock mode: one output has 219 tokens, the other 218; exact mode produces 218 in both boots. That cell is excluded from matched-work timing, not silently repaired.

| Native decoder | Matched cells / all cells | Stock total request seconds | Row-exact total request seconds | Row-exact latency change |
|---|---:|---:|---:|---:|
| DFlash2 | 8 / 8 | 89.035 | 92.069 | +3.41% |
| MTP, matched subgroup only | 7 / 8 | 122.833 | 118.789 | −3.29% |

These are descriptive sums of the two observations per mode in matched cells, not confidence intervals or qualified performance claims. Swap usage and swap-ins fail hardware qualification. DFlash executes 440 native attempts and accepts 2,416 drafts in each mode across its sixteen requests. MTP changes its native work substantially: across all sixteen requests per mode, stock/exact executes 567/871 native attempts, accepts 1,852/1,520 drafts, and executes 45/44 native lookup rounds. Its mode contrast therefore is not a pure arithmetic-kernel cost measurement. Related-edit exact latency is approximately 10.6% lower in both exposures, while fresh-first is 5.3% higher.

**Decision:** no evidence here supports a large universal exact-mode penalty hiding an otherwise strong adaptive-policy win. DFlash's observed 3.41% cost increases the production hurdle; MTP's work-policy changes and one stock output divergence forbid treating mode removal as a free optimization. Preserve correctness controls and pursue actual useful work per verification rather than weakening the oracle. The frozen plan's top-level family/numerical-mode fields are inherited source metadata; the mixed diagnostic is defined by its schedule and actual per-boot/per-request activation records. Native boots use empty stores and no corpus despite retained seed provenance metadata.

Adversarial review found no lifecycle, ownership, receipt-validation or hidden-treatment defect. This runtime exercise also validates the publication-only validator fix on real native DFlash execution. Evidence root: `runs/independent-v29-native-arithmetic-diagnostic`.

### Next optimization decision: live history versus fresh-miss tuning

For the completed v28 fresh-first population, gross lookup time is 429.86 ms/request. Applying each selected action's logged native milliseconds/token estimate to its actual committed-token count gives a **136.25-ms/request native-equivalent estimate**, leaving approximately **293.62 ms/request** of modeled difference across an 11.22-second mean request. The two upper learning misses contribute 211.02 ms and ordinary lookups 82.60 ms to that difference. This is a historical-rate estimate, **not an executed same-state counterfactual, strict savings bound or measured speedup**; transition costs and subsequent trajectories can change. It does not justify assigning the whole gross lookup interval to removable overhead or adding a second recovery surcharge.

The current coding cohorts freeze per-boot history and disable writes. Their “repeat” exposure therefore does **not** append the preceding answer to the persistent suffix store. Those cohorts remain valid frozen-history transfer tests, but do not answer the live completed-response reuse question. A separate prospective writable-history protocol is being implemented with fresh stores per boot, exact publication-derived append checks, cold/repeat reporting and storage growth. Native versus writable combined features and read-only versus writable combined features must remain distinct contrasts; neither isolates StairCut alone. Prefix/KV cache settings and append/service-completion timing must be explicit.

The independently reviewed PLE CLI protocol has now been freshly declared against the current shared helper source and launched as `runs/independent-v25-ple-cli-assistant-python-current-abba`. The earlier unexecuted declaration is not silently updated after helper changes. This first complete-format cohort is intended to close the cache-only end-to-end question; it does not relax hardware qualification or promise a speedup.

## Completed PLE CLI cache-only cohort

`runs/independent-v25-ple-cli-assistant-python-current-abba` completed **32 exact publications**, four alternating ABBA/BAAB quartets, with no fatal errors. Both complete tasks passed the prospective response-format/structure gate: 59 generated tokens for repeated lines and 165 for the related edit. The primary metric is tokens divided by the CLI's printed decode rate, **not HTTP latency or load-inclusive process wall time**.

| Workload | Cache-on decode throughput change | 95% quartet interval |
|---|---:|---:|
| Both complete tasks pooled | +0.28% | −0.34% to +0.98% |
| Repeated lines | +1.31% | +0.25% to +2.36% |
| Related coding edit | −0.09% | −0.56% to +0.48% |

Swap usage, swap-ins and swap-outs fail hardware qualification. The pooled interval includes no improvement; neither case meets the declared 5% practical margin. This does **not** establish a full-inference cache speedup. It closes this cohort's end-to-end hypothesis without changing its gates or enlarging the cache.

The cache is genuinely active: every enabled repeated-lines process reports 840 hits / 152 misses; every enabled related-edit process reports 353 hits / 2,335 misses. Those are **table-lifetime counters, including startup/calibration**, not measured-decode hit rates. Related-edit pool rounds fall by eight in each corresponding lifecycle regime (252→244 or 188→180); this is substantially less avoided work than the repeated-lines reductions of 91→39 or 82→30. The earlier +19–28% narrow-gather component results cannot be substituted for these full-decode observations.

Incremental decoded payload remains **655,360 bytes (640 KiB)**, with **8,480 bytes of metadata in both arms**; no extra disk table or workers. Printed MLX allocator peaks are approximately 45.914 GiB for repeated lines and 46.177 GiB for the edit; their three-decimal rounding and scope cannot resolve or attribute the cache's sub-MiB host allocation.

## Terminal persistent-history append optimization

Writable production history previously constructed a complete new sorted index as part of its final append, before the first disk write, then immediately destroyed it during generator teardown. The next request reopens the file and constructs its own index. The terminal `appendFinalTurn()` operation removes that unused rebuild. Ordinary append writes existing input directly, with no combined-history or positions allocation; eviction prepares only replacement token bytes. Success retires the view while keeping borrowed storage and the exclusive lock alive until `deinit()`. Validation failures and successful rollback retain a usable old snapshot; uncertain durable state retires it.

The obsolete nonterminal append API and its production caller are migrated, not retained as a shim. Tests that model successive requests now reopen between successful turns. The generator publication test reads durable history after reopening and continues to exclude unpublished tails.

**22 token-history tests passed**, including borrowed inputs, no-eviction append under allocation failure, invalid/empty final input, rollback, whole-document trimming and allocation failure before trim. A CPU/filesystem experiment compared the old implementation at `e465fa37` with the terminal implementation using complete published documents from the native arithmetic diagnostic. Each size has four alternating ABBA/BAAB quartets, eight observations per arm; all **64 durable byte sequences match exactly**.

| Existing history token slots | Old append mean | Terminal append mean | Append latency reduction | Open/append/close latency reduction |
|---|---:|---:|---:|---:|
| 0 | 0.070 ms | 0.041 ms | 40.75% | 16.50% |
| 7,979 | 1.043 ms | 0.057 ms | 94.57% | 47.74% |
| 65,458 | 7.757 ms | 0.055 ms | 99.29% | 50.01% |
| 261,920, requiring eviction | 34.609 ms | 0.895 ms | 97.41% | 50.16% |

Each appended document contains 457 tokens plus a separator. These are **host persistence-component measurements with initialized filesystem caches**, not GPU decoding or request-throughput gains. No inference or build ran concurrently with this experiment. At 65,458 existing slots, append's measured cumulative allocation falls **527,328→0 bytes**; the eviction case falls **2,096,344→1,048,172 bytes**. These are allocated bytes during the operation, not peak RSS. Retained store format, capacity and disk payload are unchanged. Private numeric evidence: `history-append-observed-v30.json`.

The opt-in publication diagnostics now also emit `[lookup-finalize]` after append and `LookupRound.deinit()`. Its timer measures **production lookup finalization**, including the document buffer and history destruction; it excludes earlier publication logging, scheduling delay and the rest of generator/slot cleanup. HTTP/SSE completion can precede this work. The writable benchmark reports HTTP latency, this production component time and observer-fence timing separately—not their sum as an invented full-request duration. The rebuilt executable exercised this receipt and exact durable publication in both completed live-history cohorts below.

## Live completed-response reuse: MTP and DFlash2

The v30 executable (`81911662b8213edbbc624d196615c4f188b2607e7c4cda1ba1c431cc4f34dcb1`) completed separate native-versus-adaptive-writable cohorts for MTP and **incoai DFlash2**. Private roots are `runs/independent-v30-mtp-writable-history-abba` and `runs/independent-v30-dflash-writable-history-abba`. Each contains four alternating ABBA/BAAB quartets and a serial reference: **136 exact responses per backend**, 128 timed and eight reference responses, with no fatal or publication errors. Each boot starts with an empty history; the four complete coding cases run first/repeat in the original order, with no discarded warmup requests. Both arms use the same frozen executable, row-exact arithmetic and disabled prefix/KV caching.

Unlike the earlier frozen-history cohorts, each completed adaptive request publishes its actual prompt and authoritative output to the production store. Its immediate repeat can reuse that response. This is a favorable, intentional repeated-workload scenario, not an unseen-task generalization test. Only the first fresh-code request is globally empty-history; later cases' first exposures see earlier documents.

| Backend | Exposure | HTTP/SSE throughput change | 95% quartet interval |
|---|---|---:|---:|
| MTP | All requests | **+28.31%** | +25.60% to +30.53% |
| MTP | First exposures | −3.06% | −5.85% to −0.86% |
| MTP | Immediate repeats | **+88.62%** | +86.93% to +90.31% |
| DFlash2 | All requests | **+6.13%** | +5.30% to +6.76% |
| DFlash2 | First exposures | −1.61% | −2.35% to −0.86% |
| DFlash2 | Immediate repeats | **+14.80%** | +13.03% to +16.33% |

These are geometric means of quartet throughput ratios, not percentages of latency saved. Every overall quartet favors the writable feature in both cohorts. **Swap usage and swap-ins changed in both runs**, so these are observed, hardware-unqualified workload gains; the qualification and promotion gates remain false. Neither result isolates StairCut, persistent storage alone or the fastest stock configuration.

Hardware is not the only promotion concern. Applying the existing five-percent latency-regression allowance to each case/exposure/quartet would fail **five of 32 MTP cells** (fresh-first quartet 0, edit-first 0/2/3, completion-first 0) and **two of 32 DFlash2 cells** (fresh-first 0/1). Their latency increases range 5.50–10.21% for MTP and 9.11–9.53% for DFlash2. These diagnostic cohorts do not promote themselves; a positive aggregate cannot erase those first-use losses.

| Coding case | MTP first | MTP repeat | DFlash2 first | DFlash2 repeat |
|---|---:|---:|---:|---:|
| Fresh generation | −3.06% | +135.78% | −5.05% | +24.12% |
| Related helper | −0.35% | +127.36% | −3.28% | +23.93% |
| Related edit | −6.32% | +33.60% | +1.45% | +3.67% |
| Related completion | −2.82% | +75.39% | +0.70% | +10.00% |

Executed work supports the reuse benefit without identifying selected source origins: MTP fresh-repeat lookup acceptance is **1,461/1,461 drafts**, with head rounds falling 624→62 across eight requests. DFlash2 repeat verification attempts fall 280→223 for fresh code, 240→185 for the helper, 216→199 for the edit and 144→128 for completion, with identical publications. Commit-only diagnostics do not prove which rounds selected persistent rather than local history, or price unexecuted alternatives.

The first-edit MTP regression is also real executed work: native versus feature totals are **312→331 rounds**, **952→1,316 lookup drafts**, and **824→783 accepted lookup drafts**, for the same 1,568 published tokens. All four first-edit quartet ratios regress. Static review finds that tiered MTP currently bypasses the native prompt-copy gate and compares against head economics; exact intermediate-width coverage is another limitation. Neither aggregate counts nor the fourteen-draft mean establish which alternative would win at the same prefix. The next diagnostic must preserve native action identity, proposal membership, preparation and exact-width price provenance before changing admission. A blanket cap reduction would risk the productive repeat copies.

### Persistence timing and incremental resources

HTTP/SSE totals across 64 requests per arm are 560.772→437.045 seconds for MTP and 343.799→323.939 seconds for DFlash2. The separately observed request-through-store-snapshot envelope gives **+28.14%** and **+5.88%** respectively using quartet-geometric ratios; it includes observer overhead and is not pure production slot latency. Scoped production lookup finalization totals **12.612 ms for MTP** and **11.688 ms for DFlash2**, approximately **0.197 and 0.183 ms/request**. Native finalization is unmeasured (`null`), not zero.

Every feature boot ends with **eight documents / 2,750 token slots**, including document separators: **11,040 logical file bytes**, **12,288 allocated file bytes**, versus no native store. The feature's initial header occupies 40 logical / 4,096 allocated bytes, so within-feature growth is 11,000 logical / 8,192 allocated bytes. Allocated APFS bytes exclude directories and are not exclusive physical extents or device-write traffic. These sizes describe this small workload, not the configured capacity.

| End-of-boot mean across eight processes | MTP native | MTP writable | DFlash2 native | DFlash2 writable |
|---|---:|---:|---:|---:|
| Process physical footprint, MiB | 17,422.45 | 17,892.07 | 18,632.90 | 18,611.93 |
| MLX active, MiB | 17,005.42 | 17,005.42 | 18,239.32 | 18,239.32 |
| MLX reclaimable cache, MiB | 109.14 | 256.17 | 141.01 | 124.29 |
| MLX lifetime peak, MiB | 18,304.12 | 18,503.44 | 18,638.03 | 18,637.43 |

These are matched post-fence endpoints, **not allocator-quiescent samples or intrinsic lookup-memory bounds**. MTP's observed +469.61-MiB footprint includes retention and process variability; DFlash2's −20.97-MiB difference is not evidence that adding lookup intrinsically saves memory. MLX counters overlap process footprint and must not be added to it. Writable requests rebuild their changed-history index at opening; terminal append avoids only the redundant teardown rebuild.

To isolate the incremental value and cost of live history, the next declared comparison holds adaptive features fixed and compares empty read-only history with writable history. Its result must remain separate from these native-versus-combined contrasts and the older frozen-history transfer experiments.

## Isolated net live-history value: MTP

`runs/independent-v30-mtp-history-increment-abba` holds the v30 adaptive controls fixed and compares an initially empty **read-only** store with an initially empty **writable** store. It completed **136 serial-exact responses**, four alternating quartets plus serial references, with no observation errors. This isolates the net live-history configuration—including changed-history index reopening and publication versus cached read-only lifecycle—not just a retrieval kernel or disk-write operation.

| Exposure | Writable versus read-only throughput change | 95% quartet interval |
|---|---:|---:|
| All requests | **+32.24%** | +30.77% to +33.82% |
| First exposures | −0.50% | −1.65% to +0.68% |
| Immediate repeats | **+95.64%** | +93.93% to +97.74% |

| Coding case | First exposure | Immediate repeat |
|---|---:|---:|
| Fresh generation | +0.25% | +144.64% |
| Related helper | +1.54% | +138.59% |
| Related edit | −3.15% | +38.44% |
| Related completion | −1.25% | +79.23% |

Read-only/writable HTTP totals are **588.516/444.832 seconds**, with separate request-through-snapshot totals **590.578/446.956 seconds**. Lookup finalization averages **0.014/0.200 ms/request**. Those components are not additive estimates of full-slot latency. Swap usage and swap-ins again fail qualification; the positive complete-workload result remains descriptive. Do not multiply this ratio by the separate native-relative cohort to synthesize another comparison.

This strengthens the attribution to **net live completed-response reuse under fixed adaptive settings**. It does not identify selected source origins, separate policy-state interactions from retrieval, or establish an isolated StairCut gain. The related-edit first exposure still regresses despite the fixed adaptive configuration; preceding-history availability and resulting decisions matter, not only the feature-versus-native dispatch boundary. DFlash2's corresponding history-increment comparison is separate.

## Bounded MTP proposal-coverage diagnostic

A private CPU diagnostic directly imported the production `mtp_lookup.Index` and `token_history.Drafter`, replaying exact published prefixes with the original chronological history. Its single filtered diagnostic passed. It examined 218 fresh-repeat prefixes with one prior document and 196 first-edit prefixes with four prior documents. Each query includes its anchor in context but indexes only earlier tokens; future output is used only to score proposals after construction.

At **first-edit output position 110**, native latest-match backing is eight tokens. Its seven-token cap candidate matches all seven oracle tokens; the tiered modal fourteen-token offer matches zero. The native continuation is present in the raw vote tree, but only off the modal chain that MTP currently makes executable. Across 115 suffix-eligible edit prefixes, this is the sole native-cap coverage advantage; the tier modal offer accepts more at 23 prefixes. No accepted native prefix is absent from the raw tree. Fresh repeat has only one native suffix-eligible prefix, where the tier offer accepts more.

This is an **executable-candidate gap witness**, not proof that the runtime visited that prefix, that the native gate admitted the copy, or that it would be faster. Native gate EMA/streak/cost state and actual selected widths are unavailable in this offline replay; terminal lengths are clipped to published output. There were no equal-coverage/fewer-row witnesses under these cap comparisons. The next bounded full-receipt run must join the witness to actual executed anchors before changing selection or acquiring new prices. Private evidence: `mtp-membership-observed-v30.json`; raw fixture tokens remain private.

## Isolated net live-history value: DFlash2

`runs/independent-v30-dflash-history-increment-abba` completed the corresponding **136 serial-exact responses** with fixed adaptive DFlash2 controls, initially empty read-only/writable stores, four alternating quartets and no observation errors.

| Exposure | Writable versus read-only throughput change | 95% quartet interval |
|---|---:|---:|
| All requests | **+7.84%** | +6.55% to +9.52% |
| First exposures | +0.77% | −1.38% to +4.45% |
| Immediate repeats | **+15.67%** | +14.35% to +17.52% |

| Coding case | First exposure | Immediate repeat |
|---|---:|---:|
| Fresh generation | +6.28% | +26.00% |
| Related helper | −1.12% | +28.19% |
| Related edit | −0.94% | +3.02% |
| Related completion | −0.99% | +8.03% |

The fresh-first point estimate must **not** be credited to retrieval: both histories are empty at that point, while verifier attempts change only 272→270 across eight requests. Its wide interval and timing-sensitive adaptive state are reasons not to attribute that observed difference to a particular mechanism. Repeat work reductions are clearer: attempts fall **276→224** for fresh code, **234→177** for the helper, **210→201** for the edit and **141→128** for completion, with unchanged publications.

Read-only/writable HTTP totals are **387.007/358.867 seconds**; observer-inclusive request-through-snapshot totals are **389.122/361.111 seconds**. Scoped lookup finalization averages **0.011/0.213 ms/request**. Read-only history remains 40 logical / 4,096 allocated bytes; writable ends at 11,040 / 12,288 bytes. Mean endpoint process footprint differs by approximately **+2.46 MiB**, not an intrinsic allocation or peak-memory bound.

Swap usage, swap-ins **and swap-outs** fail qualification. This is descriptive net live-history benefit, including differing storage/cache lifecycles and their policy interactions—not a clean-machine, isolated retrieval, StairCut or fastest-stock claim. Together with the MTP control it establishes useful live-history behavior in both exercised neural families; their percentages cannot be added to other contrasts.

## Executed MTP trajectories: admission delay, not a proven missing incumbent

`runs/independent-v30-mtp-full-trajectory` completed **12 exact, syntax-valid publications** with no validation errors: fresh generation, related helper and related edit, first/repeat, in one empty-read-only boot and one writable boot. Full diagnostics were enabled in both; original request bodies, chronological publication, finalizers, lock fences and store bytes were retained. Five offline boundary checks covered unique-anchor joins, differing acceptance, mismatched prefixes, duplicate anchors and invalid publication identities.

**Neither executed edit trajectory visited output position 110.** The offline off-modal candidate witness therefore does not justify a native-incumbent production change. The earliest actual divergence is at **absolute anchor 273 / output position 12**, after four identical three-token native-head rounds:

- Both arms query a local order-fourteen, confidence-one offer. Persistent indexes support order twelve, so they cannot directly alter this order-fourteen proposal through voting.
- Read-only takes fourteen lookup drafts, accepts all fourteen and commits fifteen tokens.
- Writable declines with `mtp_rate`, takes two head drafts, then admits and fully accepts fourteen lookup drafts at output fifteen.

Readonly/writable edit-first execution totals are **39/42 rounds**, with **154/168 lookup drafts** and **104/96 accepted lookup drafts**. At output 109 the read-only arm again starts a successful copy three tokens earlier. Writable fresh repeat still accepts **182/182 lookup drafts** and uses eight head rounds versus read-only's 79.

A separate interruption at writable output sixty is the existing **64-plan alternate native-depth challenge**: depth one is selected as an explicit trial. Its request-local depth-one cost is unknown, so admission conservatively retains that head instead of borrowing depth-two pricing. Copy rounds advance the model-wide planning counter even when they replace the head. This is a policy interaction, not evidence of an incorrect depth index or fabricated timing.

The losing lookup alternative's economics are not retained by existing receipts: a rejected choice reports the fallback rate, not the rejected yield/price/preparation inputs. Selected records at different prefixes cannot distinguish calibration, request support and price-history effects at anchor 273. **Stop the unproved native-incumbent fix.** The next diagnostic captures bounded host admission inputs and replays the original decision before exchanging support and price blocks on disposable state. Unknown prices must stay unknown; a replayed decision flip is not evidence of a faster GPU action.

Full tracing perturbs cost learning and trajectory. These observations describe the diagnostic execution, not a reconstruction of the earlier commit-only benchmark or a new throughput measurement.

## Terminal-append final verification

After the completed v30 cohorts, the obsolete, unused `PersistentStore.max_order` field was removed; persistent index construction retains its existing default order. **All 22 token-history tests passed again**. The lookup-filtered integrated test command and ReleaseFast build exited successfully. The resulting v31 executable is `ad4ad30e082c401420f9c3fd8db4d2ec2119081f8fba3900dc993eb73ca8e91f`.

`runs/independent-v31-terminal-history-smoke` exercised the rebuilt executable with three full original coding requests: serial, writable first, writable repeat. **All three publications match exactly**, with no errors, successful finalizers, released production locks and exact durable history growth **0→1→2 documents**. This is an untimed runtime smoke, not a shortened benchmark or a new performance estimate. Adversarial review corrected the private smoke's source-plan provenance and collector cancellation ownership before execution; those findings were confirmed resolved.

## Standard mlx-serve benchmark comparison: frozen v31 protocol

This comparison uses the repository's standard `tests/bench.sh` client protocol: pinned **llmprobe 0.6.17**, `HOST -m MODEL --bench-only --save REPORT`, without scenario overrides. The private declaration is `runs/standard-v31-default-abba/plan.json`. All eight MTP boots completed; the campaign stopped on its own validation gate after DFlash feature boot 09, as detailed below. It was not cancelled for the refactor port.

- **Sixteen fresh server boots:** MTP and incoai DFlash2 each run native/feature/feature/native, then feature/native/native/feature. The same frozen v31 binary runs every arm. Native means feature-disabled stock numerical mode in this PR binary, **not an untouched upstream build**. Feature includes the required row-exact mode, adaptive lookup/width selection and initially empty writable history.
- Preserve default prefix caching, automatic context capacity, server concurrency and client medium reasoning. Standard greedy scenarios include decode, prefill, predictable normalization, novel prose, prefix-cache cold/warm, a four-stream batching probe, the **512/4K/8K/16K context ladder**, and sustained-load decode. There are three measured samples per headline scenario and one coding/counting pair per context rung. Startup detection requests are additional.
- The sole client-source change is the previously reviewed cache-bust initializer, shared within each quartet. Prompts, token limits, scenario order, timing code and all other client bytes remain unchanged. Each server has fresh configuration/cache directories and history; no fixture seeds or cross-boot training. Standard warmups and earlier requests can train history, so this measures **benchmark-warmed behavior**, not held-out generalization.
- Direct client connections avoid a relay. Per-round tracing and publication receipts are off in both arms. Frozen identities cover the executable, linked runtime/Metal libraries, client/dependencies, interpreter and model metadata; safetensors payloads are not rehashed. Passive telemetry, memory snapshots and settled storage are separate from client timing.
- Completion requires all three primary samples, all four context rungs, native-family engagement, and successful tokenizer-bound history activation/publication for feature boots. Failed slots are retained and stop the campaign; no retry or replacement. Predictable-task results filtered by the standard client remain unavailable, not zero or a substitute task.
- Effects use within-quartet arm means and the geometric mean of the two feature/native ratios. Report both quartet effects and their range, not a confidence interval from two units. Throughput increases and latency reductions are labeled separately. Resource snapshots are not isolated lookup allocation; finalization is not added to HTTP timing.

Adversarial review required linked-runtime identity checks, proof of writable-history activation, balanced-quartet reduction, and measured-sample completeness before launch. Offline checks exercised missing-sample/rung rejection, unavailable speculative metrics, client-stage errors, durable-history structure, and real owned-child termination. Those checks started no inference. This Qwen-27B comparison evaluates the combined opt-in lookup/width/routing configuration; it does not evaluate Flash-Next's learned-embedding cache.

### Interim MTP quartet: repeated decode wins, cold/short-context and batching losses

The first complete MTP quartet, boots **00 native / 01 feature / 02 feature / 03 native**, has no request-validation or client-stage errors. These are means of two boot-level report medians per arm; the completed eight-boot MTP estimate follows below.

| Standard-suite metric | Native MTP | Feature | Observed change |
|---|---:|---:|---:|
| Headline decode | 20.80 tok/s | 71.45 tok/s | +243.51% |
| Predictable decode | 42.15 tok/s | 71.80 tok/s | +70.34% |
| Novel-prose decode | 15.30 tok/s | 50.65 tok/s | +231.05% |
| 512-context decode | 19.90 tok/s | 17.60 tok/s | −11.56% |
| 4K-context decode | 16.65 tok/s | 19.40 tok/s | +16.52% |
| 8K-context decode | 18.20 tok/s | 21.20 tok/s | +16.48% |
| 16K-context decode | 17.80 tok/s | 19.90 tok/s | +11.80% |
| Headline TTFT | 1,103.5 ms | 1,220.5 ms | 10.60% longer |
| Prefill throughput | 131.25 tok/s | 128.10 tok/s | −2.40% |
| Four-stream aggregate | 24.65 tok/s | 22.85 tok/s | −7.30% |
| Whole-client observer envelope | 1,061.260 s | 1,003.002 s | 5.49% shorter |

Every boot processes the same **105,970 input / 5,137 output tokens over 36 benchmark requests**, plus identical aggregate setup usage. Actual headline, predictable, novel and context generations reach 192 tokens; batch streams reach 96. Equal lengths do not prove equal token identities or quality. Only headline/drift requests force length through `ignore_eos` and `min_tokens`.

Normal runtime counters support the repeated-decode mechanism: late feature headline repeats accept **168 lookup drafts** and use **four head rounds**, versus approximately forty head rounds and five to seven accepted lookup drafts in native execution. No prefix-cache reuse is logged for these initial requests. Feature novel samples progress from **17.0→51.7→55.2** and **16.4→49.6→52.5 tok/s**: the nominally novel task itself repeats and trains history. These counters establish accepted lookup work replacing head work, not which local/persistent source supplied it.

The losses have concrete reached-path evidence:

- The first feature length probe uses **71/72 head rounds** with only six/five accepted lookup tokens, versus 42 head rounds in the first native boot. It is not already fast before warming.
- At the actual 513-token context, feature uses **76/77 head rounds** versus 57 in both native boots. Higher per-draft acceptance does not compensate for lower yield per round; lookup accepts only fifteen/thirteen drafts. Individual feature head rounds are not uniformly slower.
- Four-stream cohorts have **zero lookup rounds** and identical 384-token output totals. All arms engage batched trunk verification/decode. Native engages the shared batched MTP-head lane; row-exact feature mode disables rerank and reaches per-chain full-logit head construction instead. This is a dispatch-cost hypothesis for a future component diagnostic, not proof that it accounts for the full aggregate loss. The client's “serialized” efficiency label must not be read as absence of actual batching.

Qualification remains failed because system swap-ins change. Default automatic resource sizing also yields different effective contexts (**105,472 / 115,712 / 117,760 / 116,736**) and prefix-cache budgets across boots. Actual context requests still match, and every prefix-cache probe reports **1,510/1,541 cached tokens**; those resource differences do not explain the observed short-prompt lookup transition, but prevent an isolated identical-resource claim. All context-coding quality notes report that the answer never used the planted constant; medium reasoning and the short generation cap do not establish useful long-context coding quality.

Feature stores finish at **444,564/445,152 logical bytes**, or **487,424 allocated filesystem bytes** each, containing 36/37 documents. This is roughly 42% of the existing token capacity, with no enlarged limit. Process footprint and GPU cache retention include multi-gigabyte prefix-cache allocations and are not intrinsic lookup memory costs.

Two independent adversarial reviews confirmed the arithmetic, matching token counts, reached work-count evidence and limits. No new campaign-continuation blocker was found. Preserve the frozen run; after completion, target cold row-exact head yield and reached batched-head dispatch rather than changing lookup admission based on an unreached candidate.

### Completed MTP subset and DFlash publication failure

All eight declared MTP boots completed, covering both ABBA and BAAB. The balanced estimator is the geometric mean of the two within-quartet feature/native mean ratios; two quartets do not support a confidence interval.

| Standard-suite metric | Balanced change | First quartet | Second quartet |
|---|---:|---:|---:|
| Headline decode throughput | +221.71% | +243.51% | +201.29% |
| Predictable decode throughput | +58.37% | +70.34% | +47.23% |
| Repeated novel-prose throughput | +273.23% | +231.05% | +320.79% |
| 512-context decode throughput | −10.31% | −11.56% | −9.05% |
| 4K-context decode throughput | +6.14% | +16.52% | −3.32% |
| 8K-context decode throughput | +10.08% | +16.48% | +4.02% |
| 16K-context decode throughput | +11.84% | +11.80% | +11.89% |
| Headline TTFT | 5.54% longer | 10.60% longer | 0.71% longer |
| Prefill throughput | −2.06% | −2.40% | −1.73% |
| Four-stream aggregate throughput | −4.64% | −7.30% | −1.89% |
| Whole-client observer envelope | 6.63% shorter | 5.49% shorter | 7.76% shorter |

These are observed results on the old main-based v31 implementation, not qualification of a new refactor-based port. The warmed-history, automatic-resource-sizing, swap and output-quality limitations above remain. The whole-client envelope includes the complete standard workload and client overhead; the 3.22× headline rate is not a whole-workload speedup.

DFlash native boot 08 completed. Feature boot 09 finished its client successfully with no client-stage errors and produced a valid 35-document, 110,435-token history. However, one overlapping request logged **`[lookup] persistent store unavailable: StoreBusy`**. The predeclared history-availability gate therefore marked that boot failed and stopped the campaign; boots 10–15 were not run. No balanced DFlash standard-suite claim is available. Retain this failed cohort without replacing, resuming or relabeling it.

Adversarial port review traced two incompatible overlap assumptions in the old implementation: a writable request retains an exclusive store lock throughout generation, and cached leases allow only one active borrower. The refactor port must use shared immutable request snapshots and short terminal append transactions against the current durable log. Appending an old request snapshot would lose another request's completed document; simply suppressing `StoreBusy` would hide missing history. The new baseline also permits preserving stock MTP numerical/head routing while adding external candidate sources, rather than carrying the global row-exact/cap-2 configuration responsible for additional cold and batching costs.

## Refactor history: maximum-order query correction and full-suite recovery

The focused history branch is now `8f2278b2`. In `Request.match`, Zig 0.17 inferred `@min(committed.len, 63)` as `u6`; adding the pending token wrapped the query length from 64 to zero in ReleaseFast. Runtime diagnostics showed 20–52 match calls but only one actual query per request, with no matches. A direct compiler reproduction printed `n_type=u6 count_type=u6 n=63 count=0`; the new production-method regression failed before the fix. Explicit `usize` arithmetic fixes the query. The independently ported DFlash assembly receives the same correction.

The unchanged eight-case coding diagnostic then passed its source-quality and engagement checks, using stock native MTP and eight prefix-cache entries. No future answers were seeded; each request could learn only from earlier requests:

| Request | Output tokens | External rounds | Accepted external drafts |
|---|---:|---:|---:|
| First catalog helper | 144 | 0 | 0 |
| Similar billing helper | 144 | 10 | 123 |
| Unrelated interval function | 218 | 0 | 0 |
| Related inventory helper | 182 | 5 | 22 |
| Related semantic edit | 196 | 1 | 1 |
| Related completion | 116 | 4 | 7 |
| Repeated catalog helper | 144 | 10 | 134 |
| Repeated inventory helper | 182 | 13 | 169 |

This is mechanism evidence, **not a speedup estimate**. The diagnostic had no native timing arm, and compilation overlapped the investigation. Syntax, requested-function presence and preserved helpers were checked; task semantics were not executed. The two repeated prompts also reused 89/120 and 95/126 prefix tokens, respectively: history engagement was additional to enabled prefix caching.

The permanent `tests/test_token_history.py` HTTP regression passed on the corrected binary. It verifies a seed document plus overlapping requests finishing B before A, exact prompt prefixes, emitted output counts and detokenized stream equality, with no `StoreBusy` or publication warnings. Artifacts: `runs/refactor-history-coding-diagnostic-v4` and `runs/refactor-history-http-v2` under the private benchmark directory. Failed diagnostics v1–v3 remain unchanged.

The history validation worktree's complete `zig build test -Doptimize=ReleaseFast --summary all` reported **33/33 steps succeeded, 4,516 tests passed, 448 skipped, zero failed**. This tree also includes separate baseline fixture repairs not included in `8f2278b2`: EXL3 hardware-eligibility gating, dropping recurrent group references before freeing their fixture lanes, and replacing const-cast mutation of a GGUF tokenizer fixture with mutable storage. The last issue caused the previously missing shard: ordered protocol execution reached test 420 and timed out in `appendDerivedMerges`; a process sample captured that stack. The full suite completes after the fixture repair. Do not claim the published feature commit alone has that full-suite result.

A bounded CPU setup probe at the full 1,048,616-byte history log measured 291.796 microseconds for initialization plus cached acquisition versus 38.180 microseconds for cached acquisition alone, with 400 calls per arm in ABBA/BAAB order. The redundant validation costs approximately 0.254 milliseconds per request on this run; it is not the main cause of the missing speedup and has not been removed. No GPU throughput claim follows from this microbenchmark.

## Refactor history: completed coding ABBA/BAAB comparison

`runs/refactor-history-coding-abba-v1` completed all eight boots and 64 requests on Qwen3.8-27B. Both arms use the same history-port binary, stock native MTP, and eight prefix-cache entries. A disables external history; every B boot starts with an empty writable store. No StairCut controller, DFlash2 changes or decoded-PLE cache participates in this comparison. Client request time includes terminal publication but excludes model boot.

| Workload group | Request-time reduction | Inverse-latency gain |
|---|---:|---:|
| Entire eight-request sequence | 8.41% | 9.19% |
| First two chronological requests | 22.79% | 29.52% |
| Four unseen/related evaluation requests | −13.72% | −12.07% |
| Two repeated requests | 50.15% | 100.61% |

Each estimate is the geometric mean of two quartet mean(B)/mean(A) ratios; no confidence interval or significance claim is made. Both quartets improved overall (7.33% and 9.48% shorter), and both regressed on the four evaluation requests (16.50% and 11.01% longer). The first-two-request group is **not wholly cache-cold**: the second request can reuse the first turn. The evaluation group is unseen-at-entry but online learning continues; it is not a frozen-history transfer experiment.

Across four boots per arm, A took 244.344 client-request seconds for 5,311 output tokens; B took 223.636 seconds for 5,314 tokens. B executed 172 external rounds and accepted 1,759 external draft tokens. Related inventory, edit and completion requests accepted 79, 4 and 28 external drafts respectively yet regressed; accepted-token counts alone do not establish a worthwhile candidate. The first catalog request and unrelated interval request accepted none and showed 2.48% and 4.68% inverse-latency regressions.

All responses pass the declared source checks. A separate static review of every distinct text found the interval function omitted requested type annotations in **both** arms (4/4 each); no differing functional defect was found by inspection. This is neither an executed semantic pass rate nor proof of complete task adherence. The original source-only quality result is retained, not reclassified.

**Qualification:** the campaign was explicitly declared contended/unknown-isolation. Telemetry also rejected whole-campaign changes in swap usage and swap-ins; individual request telemetry reported no errors. These are observed same-session performance differences, not an isolated causal or universal speedup claim. Do not discard the regressing cases or retroactively relax the gates.

**Incremental costs:** B stored 9,956–9,960 logical bytes and 12,288 allocated bytes per boot. Mean post-request sampled RSS maxima were 8,468.254 MiB for A and 8,470.906 MiB for B, an observed difference of 2.652 MiB. This is not peak unified-memory measurement or an exact attribution of allocator overhead. No extra model weights were needed.

Adversarial benchmark review corrected future runners to check effective server configuration before timing, explicitly gate performance qualification on declared GPU isolation, and allowlist recorded environment values. Four earlier private diagnostic launch files had inherited environment fields removed with a redaction annotation; request measurements were unchanged.
