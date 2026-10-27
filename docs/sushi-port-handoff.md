# Sushi port: goals, sources and acceptance gates

## Goal

Independently port useful inference optimizations into Sushi, preserving its native decoder strengths and target-model semantics. Measure incremental generation speed and memory/storage costs. These changes aim to generate code faster, not improve the generated code or make its Metal kernels faster.

Start on a fresh branch from current Sushi HEAD. Do not merge the large mlx-serve research branch or transplant its generator. Reinspect current Sushi: the previous compatibility assessment used `beamivalice/sushi` main at `124cb7d042788478414961f1772192e857b7bc5b`, not a promise about today's HEAD.

## Authoritative source snapshots

Repository: <https://github.com/jasontitus/mlx-serve>.

| Source | Revision / branch | Use |
|---|---|---|
| Focused cross-request history | `8f2278b2`, `feature/token-history-refactor` | Preferred storage/ownership/publication design; includes the maximum-order query fix and permanent HTTP overlap regression. |
| Focused decoded learned-embedding cache | `ac7861b0`, `feature/decoded-ple-refactor` | Preferred narrow host-gather cache and TensorFold affine shard-name compatibility. |
| Refactored economics and DFlash integration | `12cd42e5`, `feature/speculation-economics-refactor` | Preferred current-engine integration of independent StairCut policy, candidate adapters, native-incumbent preservation and matched-scope pricing. Includes separate baseline fixture repair commit `0b497db2`. |
| Historical independent economics research | `27e70fbc6cffab3cc753bf3754dcfe8edc9d1ddb`, `feature/independent-speculative-lookup` | Older engine layout and richer experiment history; not a mechanical patch. |

The focused branches descend from upstream `refactor/big-bang` at `9d5cfc73b5c014e8e16b8a523082c8516079a179`. They are published source snapshots, not upstream-approved or fully release-qualified PRs. The economics integration includes the history implementation but is not already part of the focused history branch; the decoded-PLE cache remains separate.

Existing **fork draft PR**, against the older `main` layout: <https://github.com/jasontitus/mlx-serve/pull/9>. It is a research/review entry point, not the patch to apply to Sushi. No focused refactor PR has been submitted upstream yet.

Upstream has since merged the refactor into `main` at `d0a51ca8b1deedb65e71daef56f692bc246bddaa`; `refactor/big-bang` is no longer advertised. Local current-main ports are prepared as `feature/token-history-main` (`12016e26`), `feature/speculation-economics-main` (`8a13b088`) and `feature/decoded-ple-main` (`cbaee97b`). These are **not yet built, runtime-verified or published**. The running benchmark still uses the frozen refactor binaries above; its measurements must not be relabeled as current-main results. Use the published, verified source snapshots until the current-main ports pass their own checks.

Read [the architecture and porting write-up](lookup-staircut-port.md), especially “Sushi HEAD compatibility,” “Adapter-boundary correctness” and “Upstream refactor submission boundary.” [The results ledger](lookup-staircut-results.md) retains favorable results, regressions and qualification failures.

## Mechanisms and preferred order

### 1. Cross-request suffix lookup: strongest reuse-workload opportunity

Retain bounded token documents from completed engine output, match a suffix of the current context, and propose its continuation. The active target verifies every proposal. This is exact token-pattern matching, not semantic retrieval or cached-answer substitution.

On the focused branch inspect:
- `src/engines/mlx/spec/token_history.zig`: immutable corpus/persistent indexes, reference-counted leases, short current-log append transactions and storage tests.
- `src/engines/mlx/spec/history.zig` and `history_publication.zig`: model/request ownership and emitted-prefix accounting.
- `src/engines/mlx/spec/mtp/chain.zig`, `mtp/round_finish.zig`: native-incumbent admission and verified flight attribution.
- `src/engines/mlx/slot.zig`, `history_slot_test.zig`, `plugin/proof_mtp.zig`: real publication boundaries and verification/rollback tests.
- `src/sdk/tokenizer/{tokenizer,load}.zig`: identity from the exact consumed tokenizer bytes.

Preserve Sushi's native local lookup and its verifier. Prefer native when its proposal remains the admitted incumbent; external history is not permission to replace a better existing action. Integrate at Sushi's own authoritative output boundary, not by copying mlx-serve's Slot structure.

Default persistent capacity is 262,144 token/separator words: at most 1,048,616 logical log bytes, roughly 2 MiB per full token/index snapshot, plus scratch and overlapping retained snapshots. Imported corpus capacity can reach 128 MiB of token/index arrays. Treat stored tokens as reversible sensitive content; explicitly opt in and isolate trust boundaries. Private permissions are not encryption or secure deletion.

### 2. Decoded learned-embedding cache: smallest independent port

Inspect focused `src/plugins/qwen4_exp/qwen4_exp.zig` and `ple_table.zig`. Cache exact decoded f32 rows on eligible narrow host-pooled gathers, leaving architecture-required embeddings enabled. Preserve worker count, wide/calibration dispatch, raw-BF16 bypass, allocation/read fallback and table lifetime. Copy all hits before inserting misses so collisions cannot corrupt a batch.

At dimension 160: 655,360 payload bytes plus 8,480 metadata bytes per eligible pool; no extra model download or persistent cache file. At 16 embedding heads, the <=64-row cached gather covers at most four tokens per call, not arbitrary prefill/verify widths. GPU-table dispatch is not accelerated by this host cache, although its payload can remain allocated.

This is not another next-token drafter: Qwen PLE and DeepSeek Engram produce model-internal embedding values. Turning them off changes the target model and is not a valid cache ablation.

### 3. Own StairCut / matched-cost routing

Inspect refactored `src/engines/mlx/spec/speculation_policy.zig`, `lookup_adapter.zig`, `mtp/economics.zig` and `dflash/economics.zig`. The historical research branch additionally retains `lookup_counterfactual.zig` experiments. Choose ancestor-closed candidate subsets using expected useful tokens per measured completed-work cost. Respect discrete kernel-width costs; never infer a hardware staircase from token count alone.

Keep native neural and native copied-continuation actions explicitly eligible. Duplicate neural/lookup support is correlated evidence, not additive probability. Calibrate only reached decisions; clipped or unreachable descendants are censored. Keep flight-owned proposal evidence stable when the next query replaces scratch.

Compare costs from the same decision boundary. Distinguish cold preparation from already-carried work, avoid charging sunk work twice, and settle GPU work before recording latency. Do not add synchronization to every native round just to instrument it. Exclude scheduler-contended/incomplete observations from wall-price learning; accepted-token evidence has different eligibility. Explicitly report other-process contention, which a scheduler-local solo flag cannot detect.

Sushi already has adaptive policies. Demonstrate an increment over its existing policy, not over an artificially fixed or weakened baseline. Do not import dense-Qwen price files as measurements for Flash-Next, another quantization, streamed residency or another GPU. The research schema-three price loader validates schema/cells, not complete model/hardware identity; the benchmark harness supplies additional identity checks. A port must not claim stronger runtime validation than it implements.

### 4. DFlash2 integration

Port through Sushi's actual assistant/verifier lifecycle. Preserve native candidate admission order and backed-copy actions when lookup adds no selected path. Mixed candidates must compete against the actual native action, not a same-width mixed substitute. Budget limits constrain accepted path depth; sibling count and selector top-k are different quantities.

The inspected Sushi had a Flash-Next chain-width policy and separate GLM tree/copy behavior. Do not transplant dense-Qwen tree geometry, backing thresholds or GDN specialization assumptions into either. Exact topology, rollback, cache state, assistant recovery and executed kernel-variant pricing need architecture-specific proofs. Keep MTP and DFlash2 comparison results separate.

## Correctness and adapter boundaries

- Record only the engine-emitted prefix, never an unseen speculative tail. Handle early-first emission, duplicate suppression, cancellation inside a verified block, EOS/length limits, failed attempts, retries and repeated terminal calls. Engine emission is not proof of network delivery.
- Requests borrow immutable snapshots without holding a writer lock throughout generation. Terminal append locks briefly and reloads the current durable log. Test overlapping requests finishing B before A without losing B's turn or invalidating A's borrowed snapshot.
- Drain outstanding work before changing effective weights. Reset or namespace KV/GDN/QSA/convolution state, pending logits/chains, assistant state and prefix snapshots by effective model/adapter generation, including disk snapshots. Reset or partition learned acceptance/cost state.
- Old token documents may remain proposals after an adapter change only with matching tokenizer identity and current-target verification. Decoded embedding rows may survive only if the effective table, scales, layout and decode semantics are demonstrably unchanged. A path, mmap or equal file size does not prove immutability.
- Prove fixed-geometry verification/commit correctness through zero/partial/full acceptance. Do not assert universal serial byte-equivalence for stock variable-width floating-point execution. If a supported row-exact contract exists, test it explicitly rather than silently forcing a slower numerical mode.

## Validation and performance reporting

1. Build and smoke-test the actual Sushi target/model path. Identify which mechanisms are supported; the previously inspected Sushi did not serve the dense Qwen3.8-27B architecture used for the main mlx-serve coding measurements.
2. Test cold/unrelated requests, unseen related requests, exact repeats, and chronological multi-request coding. Frozen training-only transfer is a separate condition from online history learning. Never seed future outputs or reference answers.
3. Enable the same prefix/KV cache configuration in both arms. Suffix lookup can help after prompt divergence; prefix caching saves leading-context prefill. Neither substitutes for measuring their interaction.
4. Use independent identical history/policy starting states per arm and balanced ABBA/BAAB complete-workload repetitions. The second B must not inherit the first B's output. Requests within a pass are not independent timing repetitions.
5. Report whole-request/workload time, decode throughput, TTFT, output counts and quality, actual source engagement, target/assistant work counts, preparation/query/append costs, incremental RAM/disk and telemetry. Keep model loading and measurement overhead separately scoped. Measure generated-kernel runtime separately from inference.
6. Isolate suffix history, width policy and decoded-row cache before testing combinations. Do not add their percentages. Preserve source/output failures and regressions; two quartets do not establish a tight confidence interval.

## Evidence available now — not Sushi claims

- Focused history: the corrected long-context coding diagnostic executes 43 external rounds with 456 accepted drafts across eight source-valid responses. The permanent HTTP regression preserves all three documents when overlapping requests finish B before A. These establish operation, not matched timing or task-semantic correctness.
- Focused history matched coding: eight ABBA/BAAB boots, 64 source-valid responses, observed inverse-latency gains of **9.19% overall**, **100.61% for repeats**, and **−12.07% for unseen/related evaluation requests**. This isolates history on stock MTP, not StairCut or DFlash2. The second chronological request also benefits from the first turn; it is not wholly cold. Contention was declared and swap qualification failed. Both arms omitted requested interval-function annotations; static review found no differing functional defect, but generated code was not executed. See the ledger for exact scope and costs.
- Focused PLE: embedded suite 67 passed / one skipped; real Flash-Next off/on smoke emitted identical text and 59-token counts, with cache hits observed. This is operation/parity evidence, not a performance comparison or token-ID equivalence proof.
- Older combined coding cohort: MTP +28.31%, DFlash2 +6.13% throughput overall; favorable immediate repeats, first-exposure regressions, and failed swap qualification. These are neither isolated history gains nor Sushi predictions.
- Older PLE full-decode comparison: +0.28%, interval -0.34% to +0.98%; no established end-to-end win. StairCut has promising isolated measurements but no qualified universal or additive gain.
- The history validation worktree reports 4,516 passed / 448 skipped / zero failed from the full `zig build test`. It includes separate, still-uncommitted baseline fixture repairs: EXL3 hardware eligibility, recurrent-state teardown lifetime, and a GGUF tokenizer test mutating a constant object. Do not attribute this full-suite result to the published history commit alone.
- Use the updated history revision: at `72419820`, Zig 0.17 inferred the capped query-tail length as `u6`, so adding the pending token wrapped 63 + 1 to zero in ReleaseFast. The fix explicitly widens the arithmetic and tests queries below, at and beyond the 64-token boundary. The same correction is applied to the in-progress DFlash query assembly.
- Refactored economics integration: ReleaseFast build and full suite passed (4,625 passed / 448 skipped / zero failed). Real MTP and DFlash2 diagnostics each completed eight source-valid responses, with external lookup engagement and actual adaptive settings verified. DFlash2 exercised both generic and linear-GDN verification. This establishes operation, not speed; the unchanged standard-suite ABBA/BAAB comparison against a separate frozen upstream binary is running. See the ledger for artifacts and attribution limits.

## Attribution and source boundary

Credit **Khaled Bakeer / [TandemLLM](https://github.com/0xBakeer/TandemLLM)** for StairCut and multi-tier lookup/drafting inspiration, mlx-serve/Sushi contributors for native machinery, and checkpoint authors for their models/drafters. The intended port sources above are independently authored replacements and existing host implementation, not TandemLLM source.

Do not copy TandemLLM implementation or retired local source-derived versions. Preserve applicable notices for the actual code reused, keep clean feature-only history and describe concept attribution in docs/PRs. Independent authoring is not legal certification; the detailed provenance caveat is in the architecture write-up. Use a separate Sushi checkout and do not modify the running service or the mlx-serve worktrees while the other agents operate.
