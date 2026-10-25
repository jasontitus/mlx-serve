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
