# PR999: private, user-operated M5 replay

This is a setup and replay handoff, **not verified M5 evidence or a performance comparison**. Nothing here connects to an M5, downloads a model, or starts a benchmark campaign automatically. `prepare.sh` downloads build/tooling dependencies and builds; only an explicit `run.py` invocation without `--plan` starts inference.

## Checkout and prerequisites

On the intended M5, use native arm64 macOS **26.2+**, Xcode **26.2+** selected with `xcode-select`, its macOS SDK and **Metal Toolchain**, Homebrew, Git, Python **3.11+**, and Node **24.19.0 exactly** with npm. Install that exact Node release using your normal version manager; do not substitute the latest Node. If Metal is missing, install it with `xcodebuild -downloadComponent MetalToolchain`. See [the repository build instructions](../../docs/building.md) for Xcode/Homebrew setup. Allow space for two source/build trees, dependency caches, private outputs, and the approximately **18.2 GB** model weights (plus assets); sufficient runtime memory is also required.

The M5 name alone does not establish sufficient RAM: those weights precede KV-cache and runtime allocations. Record installed memory (`sysctl hw.memsize`) and current memory pressure in the private admission/results; setup records RAM, VM, and swap snapshots, but its post-build snapshot is not runtime admission. Observe Activity Monitor's memory-pressure state before each block. Do not silently reduce the 32K context, replace the quantization, or alter workload settings to fit. Preserve OOM/failure results; any explicitly changed protocol is a different experiment, not a comparable score.

```bash
git clone --branch test/pr999-m5-handoff https://github.com/jasontitus/mlx-serve.git mlx-serve-pr999-m5-handoff
cd mlx-serve-pr999-m5-handoff
brew bundle install --file=Brewfile
node --version                         # must be v24.19.0
TOOLING="$PWD/tests/pr999_replay"
WORK="$HOME/pr999-m5-work"             # must NOT exist; outside this checkout
bash "$TOOLING/prepare.sh" --help        # safe: no setup, downloads, or builds
bash "$TOOLING/prepare.sh" "$WORK"
```

Setup resolves `WORK` to an absolute path, refuses existing paths (including symlinks), uses mode 0700/umask 077, and creates independent `WORK/main` and `WORK/pr` clones. It uses local repository objects first and fetches the exact commit from upstream/fork only if missing. Each detached tree runs the documented server-only sequence: `git submodule update --init`, `scripts/fetch-zig.sh`, tree-local `.zig-toolchain` on `PATH`, `scripts/fetch-llama.sh`, `scripts/build-mlx.sh`, then `zig build -Doptimize=ReleaseFast`. It does not run either binary. Setup failures preserve logs and partial files; do not reuse that directory or hide a failure by retrying in place.

Engine code is pinned independently of the handoff/tooling branch:

| Purpose | Exact commit |
| --- | --- |
| A: upstream main baseline | `49977fe79744dbe622403bb0a21ffa831122d516` |
| B/C: PR999 engine | `3fb027ddbd5a3d7fbd7e06b5d15f95f28831d320` |
| Handoff scripts/docs | Your checked-out `test/pr999-m5-handoff` commit, recorded separately |

`WORK/setup-receipt.json` records actual engine/tooling commits, helper-file hashes, binary and Zig compiler SHA-256/version, submodule state, OS/hardware, Xcode/SDK/compiler versions, and timing artifacts. Preserve it with private results. The helper hashes distinguish uncommitted tooling changes from the recorded tooling commit. Build logs are `WORK/build-main.log` and `WORK/build-pr.log`.

Real runs verify the receipt's helper-file hashes. Keep this tooling checkout (including this README) unchanged after setup; changed tooling requires a deliberately fresh setup/receipt, not editing receipt hashes to bypass provenance checks.

Setup installs **llmprobe 0.6.17**, not `latest`, into `WORK/timing`, records its npm lock hash, verifies the pinned SHA-256 of `node_modules/llmprobe/bin/dist/llmprobe.mjs`, and derives `WORK/timing/llmprobe-timing-client.mjs` using `prepare-timing-client.mjs`. The adapter exports the existing timing client without launching llmprobe's CLI. This is normalized recorded replay using llmprobe's `streamTimed`, **not the stock llmprobe workload**.

## Private data and local model

Keep data, model files, transfer archives, logs, generated responses, and results **outside every Git checkout**. Never commit or publicly paste them. Obtain the private archive through the separate handoff; raw source transcripts/projects are not required or included. The public checkout does not supply episode IDs or private cross-project sequencing.

Example transfer, run manually with your actual private source host/path:

```bash
umask 077
PRIVATE="$HOME/pr999-private"           # choose a new directory outside the checkout
mkdir -m 700 "$PRIVATE"
scp SOURCE_HOST:/absolute/private/path/pr999-private-data.tar.gz "$PRIVATE/"
shasum -a 256 "$PRIVATE/pr999-private-data.tar.gz"
# Require ec729b0823d88801559f71c7d8119097e11d302b16c7a1286478534e7b06d453.
# Stop on any mismatch; extract only the trusted handoff archive after comparison.
tar -xzf "$PRIVATE/pr999-private-data.tar.gz" -C "$PRIVATE"
DATA="$PRIVATE/pr999-private-data"
MODEL="/absolute/path/to/your/local/pinned-model"
python3 "$TOOLING/validate.py" "$DATA" --model "$MODEL"
```

The model is user-supplied: [`ddalcu/Qwen3.8-27B-MLX-Serve-4bit` at revision `901aa73a1ff5456752c02e55582609e0040ca470`](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit/tree/901aa73a1ff5456752c02e55582609e0040ca470). `pins.json` is the authoritative revision and per-file SHA-256/byte-size inventory. Supply the complete snapshot, not a similarly named model or a mutable `main` revision. No helper downloads it. Full model validation reads all pinned weight files; allow time for that I/O before thermal admission.

`DATA` must contain `requests.jsonl`, `manifest.json`, and `package.json`. Validation checks the frozen data/package hashes and request-body hashes, optionally all model-file hashes; it performs no inference or private-source rerendering. It prints the available **episode IDs and cross-project control sequence**. Use those private IDs/ordering exactly; do not put them into public docs. Validation without `--model` checks data only and does not establish model identity.

## Plan, then explicitly admit one block

A block is one fresh server/store/HOME and one selected arm. Normal episode blocks select one episode; a cross-project control block repeats `--episode` in the exact sequence printed by validation/private handoff, preserving that sequence within the single block. Never reuse the normal episode store across blocks. Benchmark settings are serial concurrency **1**, context **32768**, native MTP, no drafter/PLD; history is off for B and on for C. A is the pinned main baseline. The runner owns only the process it launches, not any existing server.

Choose a private episode ID, a **new** output directory, an unused port, and an honest admission statement describing AC power, thermal readiness, and exclusive GPU access or actual FIFO ownership. Do not launch while other inference, compilation, indexing, or GPU work is active. Use the same admission policy across arms; preserve failures rather than silently retrying/replacing blocks.

```bash
read -r -p 'Private episode ID from validate.py: ' EPISODE
read -r -p 'Actual AC/thermal/exclusive-GPU or FIFO ownership statement: ' ADMISSION
OUT="$PRIVATE/preliminary-A"             # must not exist
python3 "$TOOLING/run.py" --work "$WORK" --data "$DATA" --model "$MODEL" \
  --arm main --episode "$EPISODE" --out "$OUT" --port 18999 \
  --admission "$ADMISSION" --plan
```

Inspect the printed command/assignment. `--plan` runs local provenance checks but launches no server or replay. To deliberately execute this one preliminary block, repeat the command **without `--plan`**. That single block is a plumbing/admission check, **not benchmark evidence**, and must not be relabeled as part of a completed comparison. A later measured block needs a fresh output directory and fresh server/store.

Planning requires the exact clean tracked engine checkouts under `WORK/main` and `WORK/pr` plus validated frozen data. It can report a plan before binaries, the setup receipt, Node, or model verification are ready; its explicit `skipped_checks` are **not** a successful run admission. Real execution requires those checks. `--admission` remains required for planning.

All arms use `--prefix-cache-entries 32 --prefix-cache-disk 0`; the memory-budget flag is omitted to preserve the model-dependent production default. Inspect the captured `/props` effective settings: mismatched effective prefix-cache settings invalidate a causal comparison. Runtime receipts also capture RAM, VM/swap state, and `memory_pressure -Q` before, between episodes, and after execution.

For the declared two-episode control sequence, use the same CLI with `--episode "$FIRST" --episode "$SECOND"`, with both distinct IDs obtained privately. Do not invent a sequence, repeat an episode within a block, or use a different sequence for different arms.

**First-run gate:** runtime `prompt_tokens` must equal the frozen `input_tokens` for each request; any mismatch invalidates the block. Equality is necessary but **not sufficient** to establish exact chat-template/token parity. Frozen request hashes plus pinned tokenizer/template/model assets are additional provenance, not proof that a different runtime rendered identical token IDs. Do not make a parity or speed claim from token counts alone. Preserve mismatch/error logs and exclude invalid blocks; do not repair requests or adjust counts to make a block pass.

## Manual fixed ordering, not an automatic campaign

A = `--arm main`; B = `--arm history-off`; C = `--arm history-on`.

If proceeding beyond preliminary checks, manually execute all six fixed orders below for each prescribed episode/control block, following the private manifest/handoff's block ordering. Every letter is a separate `run.py` invocation with a **new** `--out` directory and fresh server/store/HOME. Within a cross-project control block only, keep its complete repeated-`--episode` sequence in that invocation. Preserve the same frozen input/settings for corresponding A/B/C blocks.

| Replicate | Fixed order |
| --- | --- |
| 1 | A → B → C (`ABC`) |
| 2 | B → C → A (`BCA`) |
| 3 | C → A → B (`CAB`) |
| 4 | C → B → A (`CBA`) |
| 5 | A → C → B (`ACB`) |
| 6 | B → A → C (`BAC`) |

Use explicit output names such as `r1-A`, `r1-B`, `r1-C` under a private episode-specific directory, then `r2-B`, etc. Review admission and the planned command before each block. There is intentionally no shell loop or campaign launcher here. No inference has been verified on M5 by this handoff, and completing only some orders does not establish the requested comparison. Retain original receipts/logs, invalid blocks, and the private execution order for later analysis; these helpers do not make statistical or performance claims.

## Separate overlapping functional check (optional, manual)

`tests/test_token_history.py` checks publication of emitted history when overlapping requests finish out of order. It is **not** the serial timing benchmark or a numerical-equivalence/speed test. It requires a separate dedicated PR server, private fresh lookup store and scratch HOME, native MTP, `--no-drafter --no-pld --max-concurrent 2 --ctx-size 4096`, and **not** `--lookup-read-only`. Do not reuse a benchmark store or the serial 32K server. Stop any benchmark first; start and stop this dedicated server yourself using the repository's normal server CLI. Then, with its actual port/store/log:

```bash
python3 tests/test_token_history.py "$FUNCTIONAL_PORT" --store "$FUNCTIONAL_STORE" \
  --tokenizer "$MODEL/tokenizer.json" --server-log "$FUNCTIONAL_LOG" \
  --out "$PRIVATE/functional-new-output"
```

The test does not manage that server's lifecycle and fails on unsupported scheduling/model behavior. Preserve its separate results privately; it cannot substitute for the fixed-order serial replay.

## Handoff verification

Locally exercised without inference: setup help and shell syntax; `run.py --plan` against both pinned source trees; extracted-archive validation of all 48 requests; SHA-256 validation of all 14 model files; and rejection of a corrupted transfer. The canonical timing adapter handled scripted HTTP success and 503 failure without retrying.

The serving-block lifecycle also handled scripted HTTP success, failure/unattempted accounting, token-count mismatch rejection, and owned-server cleanup. That smoke used a temporary HTTP server and synthetic inputs with injected provenance/admission fixtures—not either engine or actual thermal admission. These checks do not establish M5 build compatibility, serving-template parity, model correctness, or performance. The two fresh builds and first real M5 serving block remain user-operated checks.
