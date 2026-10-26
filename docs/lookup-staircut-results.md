# Lookup / StairCut: independent-replacement results

## Status and evidence boundary

**Useful decode gains on reusable requests; not a general performance win.** The current revision preserves substantial MTP gains and improves DFlash related-configuration decode. Unrelated-prose requests no longer execute harmful lookup-only rounds in these windows. Nevertheless, MTP first-exposure prose remains slower, DFlash short-code request latency is mixed, and the full-workload promotion gates fail. The accounting fixes are implemented; not every performance regression is resolved.

- **Current measured v6 binary:** `f4121e1f222d5e86413e86e4d52b138becff0a1ad52888961ce67b0bf2672de5`.
- Target: `ddalcu/Qwen3.8-27B-MLX-Serve-4bit`; assistant: `incoai/Qwen3.8-27B-DFlash2`.
- Every native/feature arm uses explicit **row-exact numerical mode**. These are numerical-mode-controlled references, not unchanged fastest defaults.
- Results identify that binary, not subsequent edits. Receipt-enabled diagnostics include host logging overhead. Native-depth traces perform no GPU evaluation/readback, but their CPU/I/O cost can enter later wall-clock prices; these are not instrumentation-free throughput measurements.
- Initial replacement authors were source-excluded; the integrating parent previously saw the retired implementation. Neither clean-room nor legal certification is claimed. See [provenance, attribution and implementation contracts](lookup-staircut-port.md).

Paths below are relative to `~/.mlx-serve/bench/lookup-staircut/`. These are private artifacts, not bundled public datasets. Raw logs may contain credentials and reversible token data; do not publish them without redaction. Procedures and gates are in [the offline guide](lookup-staircut-offline.md).

## Current revision: four complete workload windows

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
