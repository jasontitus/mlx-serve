#!/usr/bin/env bash
# User-operated setup only: downloads build dependencies and builds, never runs inference.
set -euo pipefail
umask 077

fail() { printf 'prepare: %s\n' "$*" >&2; exit 1; }
if [[ $# -eq 1 && "$1" == --help ]]; then
    printf '%s\n' \
        'Usage: bash tests/pr999_replay/prepare.sh NEW_WORK_DIR' \
        'Requires macOS arm64 26.2+, Xcode/Metal, Homebrew build dependencies,' \
        'Python 3.11+, and Node 24.19.0 with npm. WORK must be fresh and outside the checkout.' \
        'Builds exact pinned main/PR engines and installs llmprobe 0.6.17 in private WORK.' \
        'Preserves logs and writes setup-receipt.json. Never downloads a model or runs inference.' \
        '--help performs no prerequisite checks, downloads, builds, or filesystem writes.'
    exit 0
fi
[[ $# -eq 1 ]] || fail 'usage: bash tests/pr999_replay/prepare.sh NEW_WORK_DIR'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail 'requires native macOS arm64 (not Rosetta)'
for command in git python3 node npm cmake brew xcrun xcodebuild; do
    command -v "$command" >/dev/null || fail "missing prerequisite: $command"
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
PINS="$HERE/pins.json"
PIN_NODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["node_version"])' "$PINS")"
PIN_PROBE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["llmprobe_version"])' "$PINS")"
[[ "$(node --version)" == "v$PIN_NODE" ]] || fail "activate Node $PIN_NODE exactly before setup"
python3 - <<'PY'
import platform
version = tuple(int(x) for x in platform.mac_ver()[0].split('.')[:2])
if version < (26, 2):
    raise SystemExit('prepare: macOS 26.2 or later is required')
PY
xcrun -sdk macosx metal --version >/dev/null 2>&1 || fail 'install Xcode Metal Toolchain: xcodebuild -downloadComponent MetalToolchain'
# Resolve symlinks before enforcing the private work location; require its parent to exist.
WORK="$(python3 - "$1" "$REPO" <<'PY'
import os
from pathlib import Path
import sys
requested = Path(sys.argv[1]).expanduser()
if os.path.lexists(requested):
    raise SystemExit('prepare: WORK already exists; refusing to overwrite or resume')
work = requested.resolve()
repo = Path(sys.argv[2]).resolve()
if work == repo or repo in work.parents:
    raise SystemExit('prepare: WORK must be outside the tooling checkout')
if not work.parent.is_dir():
    raise SystemExit('prepare: create the parent of WORK first')
work.mkdir(mode=0o700, exist_ok=False)
print(work)
PY
)"
trap 'printf "prepare failed; preserved private files/logs in %s (no automatic retry or cleanup)\n" "$WORK" >&2' ERR
printf 'Preparing private work directory: %s\n' "$WORK"

for arm in main pr; do
    commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["engine_commits"][sys.argv[2]])' "$PINS" "$arm")"
    if [[ "$arm" == main ]]; then
        remote=https://github.com/ddalcu/mlx-serve.git
    else
        remote=https://github.com/jasontitus/mlx-serve.git
    fi
    printf 'Preparing and building %s at %s (log: %s/build-%s.log)\n' "$arm" "$commit" "$WORK" "$arm"
    (
        # Copy local objects without shared hardlinks or modifying the user's repository.
        git clone --no-checkout --no-local "$REPO" "$WORK/$arm"
        git -C "$WORK/$arm" remote set-url origin "$remote"
        if ! git -C "$WORK/$arm" cat-file -e "$commit^{commit}" 2>/dev/null; then
            git -C "$WORK/$arm" fetch --no-tags origin "$commit"
        fi
        git -C "$WORK/$arm" checkout --detach "$commit"
        [[ "$(git -C "$WORK/$arm" rev-parse HEAD)" == "$commit" ]] || fail 'engine commit mismatch'
        cd "$WORK/$arm"
        git submodule update --init
        # Do not allow caller overrides to replace either tree's pinned toolchain or MLX target.
        unset ZIG_VERSION MLX_DEPLOYMENT_TARGET ZIG
        bash scripts/fetch-zig.sh
        export PATH="$PWD/.zig-toolchain:$PATH"
        export ZIG="$PWD/.zig-toolchain/zig"
        bash scripts/fetch-llama.sh
        bash scripts/build-mlx.sh
        zig build -Doptimize=ReleaseFast
    ) >"$WORK/build-$arm.log" 2>&1
    [[ -x "$WORK/$arm/zig-out/bin/mlx-serve" ]] || fail "$arm build produced no server binary"
done

mkdir "$WORK/timing"
printf '{"private":true,"name":"pr999-private-timing","version":"1.0.0"}\n' >"$WORK/timing/package.json"
printf 'Installing llmprobe %s (log: %s/timing-install.log)\n' "$PIN_PROBE" "$WORK"
npm_config_cache="$WORK/npm-cache" npm install --prefix "$WORK/timing" --ignore-scripts --no-audit --no-fund --save-exact "llmprobe@$PIN_PROBE" >"$WORK/timing-install.log" 2>&1
BUNDLE="$WORK/timing/node_modules/llmprobe/bin/dist/llmprobe.mjs"
python3 - "$PINS" "$BUNDLE" "$WORK/timing/node_modules/llmprobe/package.json" <<'PY'
import hashlib, json, sys
pins = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(source.read()).hexdigest()
if digest != pins['llmprobe_bundle_sha256']:
    raise SystemExit('prepare: installed llmprobe bundle does not match frozen SHA-256')
if json.load(open(sys.argv[3]))['version'] != pins['llmprobe_version']:
    raise SystemExit('prepare: installed llmprobe version does not match pin')
PY
node "$HERE/prepare-timing-client.mjs" "$BUNDLE" "$WORK/timing/llmprobe-timing-client.mjs"
python3 - "$WORK" "$HERE" "$REPO" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

work, here, repo = map(Path, sys.argv[1:])
pins = json.loads((here / 'pins.json').read_text())
def command(*args):
    return subprocess.check_output(args, text=True).strip()
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def artifact(path):
    return {'path': str(path), 'sha256': sha(path)}

commits, binaries, compilers, submodules = {}, {}, {}, {}
for arm in ('main', 'pr'):
    tree = work / arm
    commits[arm] = command('git', '-C', str(tree), 'rev-parse', 'HEAD')
    if commits[arm] != pins['engine_commits'][arm]:
        raise SystemExit('prepare: engine commit mismatch while recording receipt')
    binaries[arm] = artifact(tree / 'zig-out/bin/mlx-serve')
    zig = tree / '.zig-toolchain/zig'
    compilers[arm] = {**artifact(zig), 'version': command(str(zig), 'version')}
    submodules[arm] = command('git', '-C', str(tree), 'submodule', 'status')
timing = work / 'timing/llmprobe-timing-client.mjs'
receipt = {
    'schema': 'pr999-m5-setup-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'work': str(work),
    'tooling_commit': command('git', '-C', str(repo), 'rev-parse', 'HEAD'),
    'tooling_files': {name: sha(here / name) for name in (
        'prepare.sh', 'README.md', 'pins.json', 'prepare-timing-client.mjs',
        'replay-timed.mjs', 'validate.py', 'run.py')},
    'engine_commits': commits,
    'binaries': binaries,
    'compilers': compilers,
    'submodules': submodules,
    'build_argv': ['zig', 'build', '-Doptimize=ReleaseFast'],
    'os': {'system': platform.system(), 'machine': platform.machine(),
           'version': platform.mac_ver()[0], 'build': command('sw_vers', '-buildVersion'),
           'hardware': command('sysctl', '-n', 'machdep.cpu.brand_string')},
    'memory': {'hw_memsize_bytes': int(command('sysctl', '-n', 'hw.memsize')),
               'vm_stat': command('vm_stat'),
               'swapusage': command('sysctl', '-n', 'vm.swapusage'),
               'note': 'Post-build snapshot; runtime memory pressure requires separate admission.'},
    'xcode': command('xcodebuild', '-version'),
    'sdk': command('xcrun', '--sdk', 'macosx', '--show-sdk-version'),
    'clang': command('xcrun', 'clang', '--version'),
    'metal': command('xcrun', '-sdk', 'macosx', 'metal', '--version'),
    'cmake': command('cmake', '--version'),
    'node_version': command('node', '--version').removeprefix('v'),
    'npm_version': command('npm', '--version'),
    'llmprobe_version': pins['llmprobe_version'],
    'llmprobe_bundle_sha256': sha(work / 'timing/node_modules/llmprobe/bin/dist/llmprobe.mjs'),
    'timing': {**artifact(timing), 'provenance_sha256': sha(Path(str(timing) + '.provenance.json'))},
    'npm_lock_sha256': sha(work / 'timing/package-lock.json'),
}
with (work / 'setup-receipt.json').open('x') as output:
    json.dump(receipt, output, indent=2)
    output.write('\n')
PY
printf 'Setup complete; receipt: %s/setup-receipt.json\nNo model downloaded and no server/inference started.\n' "$WORK"
