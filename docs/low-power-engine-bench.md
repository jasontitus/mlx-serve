# M5 Max Low Power campaign: engine-bench protocol and supplied evidence

Authority: the user supplied the laptop session's `LOW-POWER-TESTING.md` protocol
and 2026-10-03/04 results. The target is an **M5 Max 128 GB MacBook Pro**.
The full file is `~/experiments/engine-bench/LOW-POWER-TESTING.md` on that Mac.
It is not present on the review host; the harness implementation and raw runs
have not been independently inspected here. No laptop measurements were made by
this branch. Read the current file and harness scripts on the laptop before use.

This protocol supersedes this campaign's earlier generic energy-collection and
standalone llmprobe recipes. Use the existing **engine-bench harness**, not a new
timing loop. The repo's kernel/phase helpers remain separate diagnostic tools.
All harness commands below run from `~/experiments/engine-bench`.

## Machine access and quiet periods

- Only one GPU job at a time. Arms hold the kernel lock on
  `/tmp/engine-bench.arm.lock` and also take Sushi's `scripts/gpu-lock.sh`.
  `lsof /tmp/engine-bench.arm.lock` identifies a holder. Do not delete a lock file
  or treat its existence/absence as lock ownership; use the harness locking.
- The supplied session reports a Rapid-MLX chain occupying the GPU until about
  **05:00 on 2026-10-05**, with log `runs/2026-10-03/chain-rapid.log`.
  That is an estimate, not permission to overlap: confirm chain completion and
  lock release before any GPU build/test/profile/benchmark work on that Mac.
- The watchdog scans shell command text for engine launch signatures. Never put
  the unbroken signatures in shell commands during an arm, even a search. Safe
  regex spellings for searches are `[o]mlx serve`, `[m]tplx serve`,
  `[r]apid-mlx serve`, `[t]ensorfold.cli import main`, `[l]lama_cpp.server`,
  `[m]lx_lm.server`, and `/[t]ensorfold serve`.
- Build, inspect, edit and analyze between arms, immediately after `ARM OK` and
  before the next arm's quiet period. No interactive tool polling or busy terminal
  during idle readings. Input within 60 seconds of a step invalidates the run.
  The first idle reading waits for chip power at or below 150 mW; that chip
  reading is a settling signal, not the energy measurement.
- Run `tools/quiet-mac.sh quiet` before the session and
  `tools/quiet-mac.sh restore` after it. Follow the existing chain lifecycle;
  preserve a restore path if a session stops early.
- Stay on **AC**, with the battery held at **80%, not charging**, across arms.
  The reported charger limit is 100 W. `Using Batt` in `power-events.txt` rejects
  the run. Hold `caffeinate -i -s` throughout; the chain scripts do this. The Mac's
  one-minute battery sleep setting makes an accidental unplug especially costly.

For Low Power, the supplied passwordless command is:

```bash
sudo -n /usr/bin/pmset -c powermode 1
pmset -g custom
```

Use this only during setup under the existing chain's exclusion, never while an
arm is running. Verify **AC `powermode 1`**, then settle for **two minutes** before
measurement. `powermode 0` means **Automatic**, not High. Use the laptop's verified
High-mode setup when running a separate High campaign; do not invent a mapping.
Never edit sudoers. GPU caps from `m5gpu` do not emulate macOS Low Power and are
excluded from this campaign.

## Builds and arms

Prepare a ReleaseFast candidate under `versions/<engine>/<tag>`, matching an
existing version root's layout. Audit the adapter before staging it: a version
root is not necessarily just a binary. Preserve the executable, linked-library
and model provenance. Record **commit and binary modification time** alongside
each result, plus hashes where available. Build Sushi with
`zig build -Doptimize=ReleaseFast`; use the same optimization requirement for
mlx-serve. Perform build/correctness checks outside running arms.

For a single arm, after the locked, caffeinated session setup:

```bash
: "${ENGINE:?harness engine key, e.g. mlx-serve-27b}"
: "${VERSION_ROOT:?prepared candidate or baseline version root}"
: "${ARM_DIR:?unique runs/date/label path}"
scripts/run-arm.sh "$ENGINE" "$VERSION_ROOT" "$ARM_DIR"
```

Supported supplied engine keys are `sushi`, `mlx-serve`, `mlx-serve-27b`,
`tensorfold`, `omlx`, `omlx-27b`, `mtplx`, `llamacpp`, `ds4`, and `rapid-mlx-27b`.
Pass experiment settings as `KEY=VALUE` environment variables, and verify the
adapter forwards them and the server's actual route engages.

A plan has one line per arm:
`<label> <engine> <version-root> [KEY=VALUE ...]`, with no spaces inside a value.
Example structure, after replacing the version roots with staged builds:

```text
before-a mlx-serve-27b versions/mlx-serve-27b/BASE
after-b mlx-serve-27b versions/mlx-serve-27b/CANDIDATE
after-c mlx-serve-27b versions/mlx-serve-27b/CANDIDATE
before-d mlx-serve-27b versions/mlx-serve-27b/BASE
```

```bash
: "${PLAN:?prepared plan file}"
: "${DAY_DIR:?runs/date directory}"
MIN_BATTERY_PCT=50 scripts/run-plan.sh "$PLAN" "$DAY_DIR"
```

Use `runs/2026-10-03/chain-rapid.sh` as the existing lifecycle model: Energy Mode,
caffeinate, lock acquisition and plan execution. The admission floor of 50% does
not replace the 80%, non-charging comparison condition. For detached execution:

```bash
: "${CHAIN_SCRIPT:?prepared chain script}"
: "${CHAIN_LOG:?durable log path}"
python3 scripts/detach.py "$CHAIN_LOG" /bin/bash "$CHAIN_SCRIPT"
```

An arm takes settled idle, boots the engine, records loaded idle, then runs
4-user generation, 1-user long replies three times, 6K prefill and one long
single-prompt prefill. It finishes with another settled idle reading, stops the
engine and removes per-arm caches. Steps shorter than 150 seconds are repeated
to cover enough energy-counter updates. Preserve these scenarios for comparable
numbers; extend the harness with tests before adding a new measured scenario.

## Energy and validity

The supplied method reads `AccumulatedSystemLoad / SystemLoadAccumulatorCount`
from `ioreg`'s `AppleSmartBattery`: a whole-machine system-load counter estimate,
not a wall-outlet meter. Counters update about once a minute, requiring at least
55 seconds of steady load. Short bursts can supply latency results, not energy
claims through this method. Loaded-idle power alone is too noisy for a claim.

Reported added power is **step power − settled idle − 0.918 W harness overhead**.
The overhead comes from `tools/harness-overhead.sh` on this setup; do not transplant
it to the base M5/M2 Max or a changed instrumentation setup without validation.
Added joules per output token is that step's added watts divided by that same
step's output tokens/second. Keep prompt joules/1K tokens separate. Retain raw
step power, idle, overhead and derived values; never subtract overhead twice.

An idle is settled when two consecutive one-minute means differ by at most
100 mW; the arm waits up to ten minutes. Energy is valid only when both endpoint
idles are settled and differ by at most 0.5 W, or one is settled and they differ
by at most 0.15 W. If the summary says `energy: not computed`, rerun; do not fill
the gap using estimated 22 W, chip telemetry or another arm's idle.

Reject a run if it slept, used battery, reached thermal state 2 or above, had
keyboard/trackpad input within 60 seconds of a step, or shows a thinking preamble.
Treat the summary's `counts: NO` reason as authoritative for this protocol; a
run that counts for speed can still lack valid energy. Chip `powermetrics` and
filtered GPU draw from `m5gpu status` are diagnostics, not the reported energy.

The client requests thinking off (`chat_template_kwargs.enable_thinking=false`),
temperature 1 and top_p 1. The 1-user prompt should be about 71–72 tokens; above
about 90 is a preamble/comparability failure, not a longer equivalent workload.
Retain exact request/model/template/sidecar identity and inspect effective settings.

Prove speculation from `spec_engaged` in `scripts/ingest.py` and server logs.
For mlx-serve/Sushi, preserve `[spec-stats] mode=`, gate threshold, attempts,
accepts, `runtime_disabled` and disable events. Summarize each step separately:
short-reply speed cannot be paired with long-reply energy.

```bash
: "${ARM_DIR:?completed run directory}"
python3 tools/arm-summary.py "$ARM_DIR"
# Substitute the actual four completed run directories.
python3 scripts/compare.py before=runA,runD after=runB,runC
```

At least **two arms per side**, interleaved ABBA or mirrored in one sitting within
about three hours. A difference counts only when **at least 3%** and **greater
than 2.5 standard errors** of measured run-to-run noise, using the harness's
`compare_blocks` / standings rules in `site/build_site.py`; otherwise call it a
tie. Rerun the baseline in that sitting. Never mix Energy Modes or request settings.
An independent confirmation session must rerun both arms; old baselines cannot
be carried into it. Report every run label, commit, binary mtime, Energy Mode,
counts verdict and energy-validity verdict.

## Supplied observations: priorities, not replacement baselines

The user reports the following valid Low Power runs from 2026-10-03/04. Exact raw
arm labels/provenance are not included here and must be retrieved from the harness
before reusing a comparison. These are not results measured or verified by this
branch. Source table: `site/battery-data.json` on the laptop and the supplied
[battery results page](https://mac-inference-bench.web.app/battery).

| Model class / engine | Long reply tok/s | Added J/output token | Added J/1K prompt tokens | Qualification |
| --- | ---: | ---: | ---: | --- |
| 27B / TensorFold 0.6.3 DFlash2 | 25.7 | 0.86 | 91 | Supplied reference |
| 27B / oMLX 0.7.0 MTP | 18.0 | 1.22 | 76 | Supplied reference |
| 27B / mlx-serve 26.10.1 MTP | 18.5 | 1.28 | 92 | Fresh same-sitting baseline still required |
| 27B / MTPLX 2.12.2 | 16.6 | 1.40 | 81 | Supplied reference |
| 27B / mlx-serve DFlash2 | 13.7 | 1.64 | 110 | Short replies reported at 24.4 tok/s; do not pair that rate with long-reply energy |
| 27B / llama.cpp b11365 MTP | 10.4 | 2.40 | 113 | One run; insufficient replication for a win claim |
| Flash-Next / mlx-serve 26.10.1 MTP | 31.0 | 0.76 | 24 | One run; insufficient replication for a win claim |
| Flash-Next / Sushi e4be467 MTP | 27.9 | 0.86 | 28 | Supplied reference |
| Flash-Next / ds4 0aaea5a | 21.0 | 1.08 | 43 | Supplied reference |

The session reports High Power about 2.3–2.5× faster at about 1.6× the energy per
token, and generation in Low adding about 21–23 W across measured engines.
These are workload/host observations, not fixed chip laws or conversion factors.
Nearly constant added power makes speed a useful hypothesis-ranking signal;
only same-step measured J/token can establish an energy gain. Rejected drafts
reduce committed throughput and may also change power.

## First investigation: DFlash2's long-reply gate

The supplied session attributes the short/long gap to `gate_min=2.00` disabling
drafting after poor early acceptance. The reviewed source has an eight-round
default warmup (`Generator.dflashGateWarmup`), a request threshold derived by
`scheduler.dflashGateMinimum`, sticky request-local disable, and a
`roundBeatsSerial` cost-table exception. The reported number is not a universal
threshold for every width/model/request; pin the measured executable's commit.

| Proposal | Evidence to collect / adversarial test |
| --- | --- |
| Reproduce the long-tail fallback before tuning kernels | Fresh Low Power ABBA; keep the standard short/long steps separate. Record disable round, accepted drafts/round, effective threshold/width and disabled duration. `mode=dflash` alone does not prove drafting lasted through the long reply. |
| Compare shipping policy with plain decode and MTP | Use the same checkpoint/settings. A layout-matched drafter-off reference separates method economics from loaded-layout cost; a no-sidecar boot is a separate deployment comparison. Verify the harness can express the required request controls before running. |
| Diagnose premature gating or mispriced rounds | Inspect early versus late acceptance and the cost-table serial/round prices in separate diagnostic runs. Log only between arms or through prearranged capture; extra per-round logging can contaminate measured overhead. |
| Test a bounded warmup/gate/controller candidate only after diagnosis | A lowered gate can keep unprofitable drafts running; a longer warmup can worsen short replies. Require regression tests, unchanged acceptance semantics, and gains in same-step long-reply J/token without a short-reply regression. Re-enabling after serial requires valid assistant-context recovery; do not simply clear the sticky bit. |

Then run E1's DQ crossover/cap experiment on an actually reachable ordinary
affine path. A tiled DFlash layout can bypass it entirely. This keeps the measured
fallback problem ahead of speculative Metal tuning while preserving prefill work.
