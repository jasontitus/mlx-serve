# Three-tier lookup and verification-width selection

## Scope and attribution

The concepts of three-tier token-history drafting and hardware-cost-aware StairCut selection are credited to [TandemLLM](https://github.com/0xBakeer/TandemLLM), by Khaled Bakeer. This feature belongs in **mlx-serve**: its host owns candidate preparation, verification, KV/SSM commit, request publication and speculative scheduling. mlx-stream's opaque DeepSeek `DraftLane` would require a separate interface change; its Engram table is not a continuation cache.

**Authoring provenance:** earlier local implementations were acknowledged source-derived ports. They have been removed, not renamed or retained as compatibility shims. The replacements in `token_history.zig`, `speculation_policy.zig` and `lookup_adapter.zig` were newly authored from behavioral/interoperability contracts and this project's performance observations. Initial replacement authors were excluded from the identified TandemLLM implementations and retired local port modules. The parent subsequently integrated and revised those components using host tests, new measurements and adversarial findings; the parent had previously seen the old implementation. Integration uses the existing host-native verifier and scheduler. This is not a clean-room certification or a claim that every contributor was source-unexposed.

Declared read sets and retired local artifacts are retained privately for audit. TandemLLM identifies its code as AGPL-3.0-only and documentation as CC BY 4.0; mlx-serve's own code is MIT, with separate dependency notices. Concept attribution is not a permission grant. Independent replacement and clean feature history are **not legal certification**, nor do they resolve obligations from historical distribution or third-party code.

The feature branch starts at upstream `191da71de32a65f69ebceeb6cb2e91036ab50b30`; intermediate local ports and experimental merge history are not part of the feature patch. This does not assert that prior work never existed. The existing fork can carry the upstream-based branch; no force-push or replacement fork is necessary.

## Architecture

| Component | Responsibility |
|---|---|
| `src/token_history.zig` | Bounded incremental request history; immutable corpus/persistent indexes; source-tagged proposals; validated storage and request-scoped leases |
| `src/speculation_policy.zig` | Calibrated proposal reach; ancestor-closed width selection; settled prices; paired remaining native duration/yield; bounded recovery probes |
| `src/lookup_adapter.zig` | Host candidate conversion, duplicate evidence merging, context classes and pre-assistant routing |
| `src/generate.zig` | MTP/PLD chain selection, supported DFlash tree integration, actual settled observations and publication-aware history |
| `src/transformer.zig` | Model-owned policies/index cache; native verification and opt-in dispatch instrumentation |
| `src/lookup_counterfactual.zig` | Diagnostic state/offer fingerprints, learner reset, exact first-offer publication and warmup qualification |

The three history tiers are request-local prompt plus committed output, a static tokenizer-matched corpus, and completed cross-request history. Matching respects document boundaries and uses the longest usable suffix. Conflicting successors remain alternatives. The local ring retains at most 65,536 tokens; persistent storage is bounded at 262,144 tokens; corpus loading is bounded at 16,777,216 tokens. Candidate storage and query depth are bounded. MTP/DFlash use bounded suffix copying; PLD retains the host's per-step full prompt/generated-history construction.

Source confidence is calibrated from reached verifier outcomes, separately by exact matched suffix order (up to 64), lookup entry/continuation and raw-confidence bucket. A successful long template must not inflate the estimated reliability of a one-token suffix match. Chain selection uses each offered edge's actual conditional evidence, not repeated root confidence. Scheduling-clipped and rejected descendants are censored. Owned in-flight evidence preserves the selected order/confidence even when the next query replaces scratch. Duplicate neural/lookup proposals are correlated evidence, not independent probabilities to add together. Sibling projection preserves conditional descendant reach; pruning does not redistribute excluded mass.

In greedy row-exact MTP and DFlash, the already verified, anchor-excluded target path also calibrates the retained full bounded lookup offer. This uses accepted selected tokens plus at most one reached correction/bonus decision; rows beyond native rejection are not actual-prefix observations. MTP additionally learns reached selected-chain edges missing from the bounded offer, without double-counting overlapping edges. Selected neural and offered lookup outcomes are learned separately, with no extra target evaluation/readback for this feedback. Verified outcomes beyond publication EOS may train conditional probabilities, but are not realized generation yield; EOS-censored native cost/yield samples remain excluded. Other modes retain their existing selected-candidate learning.

These two greedy row-exact paths also require **request-local evidence** before lookup admission. Entry/continuation confidence bins pool suffix orders locally, while model-pooled calibration retains exact-order separation. No reached hit means zero local support; otherwise the Beta(1,1) predictive mean caps the pooled estimate. This is an admission heuristic, not a confidence bound. The full bounded offer remains available for learning when its execution copy is restricted. Conditional edge evidence is preserved independently of parent reach, so a supported lookup continuation can follow an accepted neural parent without reviving an unsupported lookup entry. Request-local trust resets between requests and participates in diagnostic state identity.

Selection maximizes expected committed tokens divided by already-paid preparation plus verification/commit cost. Ancestors precede descendants. Every admitted width is evaluated rather than stopping at the first locally unprofitable addition. Unknown prices are estimates, not measurements: cold DFlash selection retains the caller's executable native width, and unknown native MTP economics retain the native incumbent.

This is **not** a universal router between loaded MTP, DFlash, DSpark or other model families. It selects history/neural proposals and verification work within the configured supported mode. Another drafter needs a compatible proposal/confidence adapter and exact verify/commit semantics. An opaque plugin needs an interface change.

## Timing and learned routing

- Verification prices measure settled target verification **and committed state**, not lazy graph construction. Context labels are `0` below 2,048, then lower power-of-two bounds `2048`, `4096`, `8192`, etc., capped at `262144`.
- Warmup is keyed by execution kind, context, actual row count and actual generic/linear-GDN kernel variant. The first valid execution of a shape is excluded; invalid observations do not consume warmup. Exclusion from learning does not erase user-visible cold latency.
- Both families can reserve one upper-width shape per kind/context, with at most one retry after a real cold execution; a naturally cold shape has only its remaining retry. MTP requires fresh measured native economics, an actually priced executable chain reference, and greater current calibrated yield than both the reference prefix and native expected yield. Its floor, available evidence and output budget bound the action; declining yield blocks the retry too. DFlash compares the upper shape against its known native width. These are bounded learning actions, not proof that widening is faster. Reservations are consumed before execution, including cancellation or invalid timing; they never fabricate prices. Forced controls and same-state diagnostics do not explore widths.
- Native fallback samples preserve a **pair: remaining round milliseconds and expected committed tokens**. Compare `(current query cost + remaining round cost) / expected tokens`. Adding query milliseconds to an already normalized milliseconds/token value is dimensionally wrong.
- Adaptive DFlash queries once before assistant admission. Its remaining-work clock begins after that query and ends after settled commit. Lookup-only and assistant-first forced rounds cannot overwrite this native cost history. Bypassing the assistant uses the exact priced alternative, not a second speculative reselection.
- Adaptive MTP obtains at most two clean calibration opportunities per missing context/request: first valid sample excluded, second learned. Only these opportunities suppress overlapping successor preparation. The timer ends before successor predrafting. Native baseline requests, forced lookup controls and already-priced contexts retain their normal pipeline. Budget/EOS-censored yields are ineligible. Calibration gaps remain part of cold end-to-end latency.
- For the proven Qwen MTP head only, those same two opportunities also settle the existing draft/head graphs and KV cache before starting a separate target verification/commit timer. This is an explicit synchronization cost, not an extra model forward. The independent native remaining-duration timer includes it. Target prices retain normal per-shape cold exclusion; different widths or invalid observations may leave bootstrap incomplete rather than inventing a price or adding unlimited probes. Prepared/grouped/lookup flights and Qwen4 heads are excluded. Receipts expose head-settlement time, target duration and price eligibility/learning.
- Native MTP's exact-depth controller retains verified acceptance from the first head-recovery round after lookup, while excluding that recovery duration. A completed solo recovery starts the next clean depth-price interval; lookup, contention and intervening prefill cannot become native head prices. This avoids discarding real head evidence merely because its timing is ineligible. Sparse measurements and cross-request hardware/workload drift can still leave native width prices stale.
- Model policies and immutable read-only indexes survive requests; file locks do not. `[lookup-setup]` separates local/corpus/persistent setup and cache hits. `[lookup-native-cost]` reports eligible samples and pipeline gaps. `[lookup-round]` receipts distinguish selected/accepted work and decision inputs.
- `[lookup-native-depth]` logs the native controller's host-side acceptance and depth-price inputs when receipts are enabled. It describes a planning call before caller depth bounds, not necessarily an executed round: a successor may be prepared or lookup may override the plan. Logging performs no GPU evaluation/readback, but its CPU/I/O cost can affect wall-clock prices. Treat traced binaries as diagnostic evidence, not instrumentation-free throughput.
- Carried MTP chains may be shortened by group admission. Learning uses the actual verified width, while receipts reconcile executed rows, expected yield, width price and source counts. Original offer hashes/counts and requested widths remain unchanged; a clipped forced action cannot claim that its requested width executed.
- Scheduler-known co-tenancy excludes wall-clock price learning while acceptance remains useful. It cannot detect another process using the GPU. Contended timings are descriptive, not causal speed claims.

Optional price files require `schema: 2` and `context_bucket: "floor-power-of-two-2048"`. Numeric context keys contain `plain`, `chain` or `tree` width-to-millisecond tables. Legacy/malformed optional files are not silently relabeled. Priors must match the actual model, device, numerical mode and measurement scope; file acceptance alone is not calibration proof.

## Persistence and privacy

Only published prompt/output documents are persisted, never rejected drafts. The host `MLXLOOK1` format contains a tokenizer SHA-256 and little-endian token IDs separated by `1 << 30`. Missing/wrong-tokenizer/malformed stores are refused. Corpus input is the public NumPy format (`tokens.npy`, `sa.npy`) plus validated metadata; no external source builder is required.

Stable advisory locking protects append/trim. Trimming atomically replaces complete documents and syncs the directory. Partial append rolls back; failed rollback retires the store. Unsafe symlinks/hard links are rejected before permission changes. A reserved interrupted replacement is recovered under the stable lock. Read-only leases validate file identity and never chmod or rewrite the frozen fixture. Cooperative locking is required; metadata checks do not defend against an arbitrary malicious writer that ignores the protocol.

POSIX writers enforce directory mode `0700` and file mode `0600`. Windows follows the host's inherited-ACL convention and does not tighten or validate ACLs: place the configurable store beneath a directory with an appropriately restrictive **inheritable ACL**. New replacement files inherit the directory ACL, not necessarily an old payload's explicit ACL. Windows compilation is not evidence of private ACLs or runtime locking/durability correctness.

Token IDs are reversible data, **not anonymized text**. Enabling persistent storage retains prompts and answers across requests. Use private directories separated by tokenizer and trust boundary. Raw diagnostic logs can contain text, token IDs and temporary authentication credentials; do not publish them without redaction.

## Model and numerical boundaries

| Feature | Local Qwen3.8-27B pack | Qwen3.8-Flash-Next |
|---|---|---|
| Local/corpus/persistent lookup | Supported host chain/tree mechanisms | Host chain mechanism; runtime not established by 27B results |
| Native MTP | In-checkpoint head available | Host `qwen4_exp` path exists; not exercised by this feature's 27B checks |
| DFlash2 tree integration | Compatible `incoai/Qwen3.8-27B-DFlash2`; dense row-exact Qwen verifier | `qwen4_exp` excluded by tree support; do not bypass architecture checks |
| Memory | Target plus assistant previously fit the 64-GiB machine at small context; admission still enforced | Downloaded payload about 65.9 GiB including 17.9 GiB PLE; text/MTP excluding PLE/vision about 47.2 GiB before overhead; runtime fit unproved |

`MLX_SERVE_ROW_EXACT=1` is an explicit numerical reference, not a promise of stock MTP output bytes. Unsupported/uncovered execution fails closed, and the prompt-cache namespace is isolated. Compare every arm in the same numerical mode. Ordinary non-row-exact chain reductions can differ from serial; this feature does not repair the preexisting PLD raw-argmax versus reserved-token suppression difference.

Remaining output tokens bound tree **depth**, not competing sibling count. The native pre-scaled 4-bit projection covers up to 16 rows. The retained host-native GDN linear specialization is used only above eight rows and only for proven chains: earlier microbenchmarks found a width-eight regression. Native acceptance/KV/SSM commit algorithms remain authoritative.

## Exercise and validate

```sh
M="$HOME/.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit"
D="$HOME/.mlx-serve/drafters/incoai/Qwen3.8-27B-DFlash2"
MLX_SERVE_LOOKUP=1 MLX_SERVE_ROW_EXACT=1 \
  ./zig-out/bin/mlx-serve serve --model "$M" --drafter "$D" \
  --host 127.0.0.1 --port 18097 --ctx-size 2048 --no-vision
```

Set `MLX_SERVE_LOOKUP_CORPUS` and/or `MLX_SERVE_LOOKUP_STORE` to opt into external tiers. `MLX_SERVE_LOOKUP_STORE_WRITE=0` freezes persistent history. Omit `--drafter` for the target's native MTP configuration. `MLX_SERVE_LOOKUP=0` retains existing native behavior. Never bypass memory admission.

Correctness checks cover publication identity, tree compaction and continued KV/SSM state, clipping/censoring, store boundaries, and actual execution. Performance checks must compare **both native MTP and native DFlash**, separately report cold first exposure and timed repeats, and include conflicting-history and unrelated misses. Exact replay demonstrates a cache-hit mechanism; frozen-history unseen related requests demonstrate transfer. Neither alone proves standard-benchmark improvement.

Same-state diagnostic windows restore the full learner before two identical requests, including request-local trust, fingerprint actual target/assistant state and dispatch geometry, and match complete published tokens to a current-binary serial oracle. The validator requires the request-trust identity field; old receipts cannot be upgraded into current evidence. These windows report verify/commit time only; full-round duration is deliberately unavailable. GPU readback perturbs execution, so they are not throughput benchmarks. Strict-authenticated single-slot diagnostics admit exactly two requests and reject a third with the explicit request-limit error.

Diagnostic offer hashes cover the full merged selection offer. Lookup rows excluded by its merge cap can still be off-policy learning inputs; that hash alone does not enumerate those rows. Common-state fingerprints cover the deterministic frozen history, prefix and learner inputs. Verification-only width comparisons must not be presented as a measurement of all unselected alternatives.

Performance qualification requires declared GPU isolation plus complete stable power/thermal/memory-pressure/swap observations. Two balanced pairs are diagnostic; four qualified pairs are required for transfer confirmation. Family-improvement and fastest-native gates are separate. Historical schema-two summaries cannot become schema-three confirmation evidence. A full canonical protocol that times out is incomplete, never a passing shortened benchmark.

See [results and evidence boundaries](lookup-staircut-results.md) and [offline operation](lookup-staircut-offline.md). Measurements are tied to exact binary identities; earlier revisions do not certify later changes.
