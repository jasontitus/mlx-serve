#!/usr/bin/env python3
"""Run one explicitly assigned private serving block; never a campaign or score."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from validate import validate

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parent.parent
LIMITS = (
    'Fixed-request serving replay, not task-completion or benchmark-win evidence. '
    'Prompt-token parity is necessary, not proof of byte-exact runtime rendering. '
    'Compare effective settings (including resolved prefix memory) across arms before '
    'causal comparison; unequal settings invalidate that comparison. '
    'Capped outputs are not successful task completion. No retries or scoring.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    # Every artifact is create-only: failed blocks are never reused or overwritten.
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def command(argv, *, env=None):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)
    require(result.returncode == 0, f'Command failed: {argv!r}: {result.stderr.strip()}')
    return result.stdout.strip()


def capture(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'argv': argv, 'error': str(error)}


def host_snapshot(pid=None):
    result = {'at': stamp(), 'platform': platform.platform(), 'machine': platform.machine(),
              'hostname': socket.gethostname(),
              'power': capture(['/usr/bin/pmset', '-g', 'batt']),
              'thermal': capture(['/usr/bin/pmset', '-g', 'therm']),
              'hardware': capture(['/usr/sbin/sysctl', 'hw.model', 'hw.memsize', 'machdep.cpu.brand_string']),
              'ram': capture(['/usr/bin/vm_stat']),
              'swap': capture(['/usr/sbin/sysctl', 'vm.swapusage']),
              'memory_pressure': capture(['/usr/bin/memory_pressure', '-Q'])}
    if pid is not None:
        result['server_memory'] = capture(['/bin/ps', '-p', str(pid), '-o', 'pid=,rss=,vsz=,etime='])
    return result


def require_ac(snapshot):
    power = snapshot['power']
    require(power.get('returncode') == 0 and 'AC Power' in power['stdout'],
            'AC power is required; pmset did not confirm AC Power')
    require(snapshot['thermal'].get('returncode') == 0, 'Cannot capture actual thermal state')


def external(path, label):
    require(not path.is_relative_to(CHECKOUT), f'{label} must be outside the public checkout')


def check_sources(work, pins):
    sources = {}
    # Git override variables could inspect a different index/worktree than requested.
    git_env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    for arm, expected in pins['engine_commits'].items():
        tree = work / arm
        require(tree.is_dir(), f'Missing source checkout: {tree}; run setup first')
        root = command(['git', '-C', str(tree), 'rev-parse', '--show-toplevel'], env=git_env)
        require(Path(root).resolve() == tree.resolve(), f'Not a separate source checkout: {tree}')
        commit = command(['git', '-C', str(tree), 'rev-parse', 'HEAD'], env=git_env)
        require(commit == expected, f'{arm}: engine commit differs from pin')
        dirty = command(['git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=no'], env=git_env)
        require(not dirty, f'{arm}: tracked source changes; refusing mixed provenance')
        sources[arm] = {'path': str(tree), 'commit': commit, 'tracked_clean': True}
    return sources


def check_setup(work, pins, plan):
    missing = []
    receipt_path = work / 'setup-receipt.json'
    if not receipt_path.is_file():
        require(plan, 'Missing setup-receipt.json; finish setup/build first')
        return None, ['Setup receipt, binary hashes, and timing adapter not checked: setup incomplete']
    setup = json.loads(receipt_path.read_text())
    require(setup['schema'] == 'pr999-m5-setup-v1', 'Unexpected setup receipt schema')
    require(Path(setup['work']).resolve() == work, 'Setup receipt belongs to a different work directory')
    for name in ('prepare.sh', 'README.md', 'pins.json', 'prepare-timing-client.mjs',
                 'replay-timed.mjs', 'validate.py', 'run.py'):
        require(sha(HERE / name) == setup['tooling_files'][name], f'{name}: tooling changed since setup')
    require(setup['engine_commits'] == pins['engine_commits'], 'Setup engine pins mismatch')
    require(setup['node_version'] == pins['node_version'], 'Setup Node pin mismatch')
    require(setup['llmprobe_version'] == pins['llmprobe_version'], 'Setup llmprobe version mismatch')
    require(setup['llmprobe_bundle_sha256'] == pins['llmprobe_bundle_sha256'], 'Setup bundle pin mismatch')
    for arm in ('main', 'pr'):
        path = work / arm / 'zig-out/bin/mlx-serve'
        item = setup['binaries'][arm]
        require(Path(item['path']).resolve() == path.resolve(), f'{arm}: unexpected binary path')
        if plan and not path.is_file():
            missing.append(f'{arm} binary hash skipped: not built')
        else:
            require(path.is_file() and os.access(path, os.X_OK), f'{arm}: missing executable')
            require(sha(path) == item['sha256'], f'{arm}: binary differs from setup receipt')
    client = work / 'timing/llmprobe-timing-client.mjs'
    provenance = Path(str(client) + '.provenance.json')
    require(Path(setup['timing']['path']).resolve() == client.resolve(), 'Unexpected timing adapter path')
    if plan and (not client.is_file() or not provenance.is_file()):
        missing.append('Timing adapter hashes skipped: not prepared')
    else:
        require(sha(client) == setup['timing']['sha256'], 'Timing adapter differs from setup')
        require(sha(provenance) == setup['timing']['provenance_sha256'], 'Timing provenance differs from setup')
        timing = json.loads(provenance.read_text())
        require(timing['source_sha256'] == pins['llmprobe_bundle_sha256'], 'Unpinned timing source')
        require(timing['adapted_sha256'] == sha(client), 'Timing adapter hash mismatch')
    require(setup.get('build_argv'), 'Setup is missing build command provenance')
    return setup, missing


def isolated_environment(out):
    # Allowlist rather than an MLX-only denylist: DYLD, Metal, Node options,
    # proxies, HF settings, and future tuning variables cannot silently leak in.
    allowed = {'PATH', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ'}
    environment = {k: v for k, v in os.environ.items() if k in allowed}
    environment.update({'HOME': str(out / 'home'), 'CFFIXED_USER_HOME': str(out / 'home'),
                        'XDG_CACHE_HOME': str(out / 'cache'), 'XDG_CONFIG_HOME': str(out / 'config'),
                        'XDG_DATA_HOME': str(out / 'data'), 'TMPDIR': str(out / 'tmp'),
                        'HF_HOME': str(out / 'hf'), 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'NO_PROXY': '127.0.0.1,localhost'})
    return environment, sorted(set(os.environ) - allowed)


def listener_owner(port, pid):
    result = subprocess.run(['/usr/sbin/lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
                            capture_output=True, text=True, timeout=10)
    owners = set(result.stdout.split())
    require(result.returncode == 0 and owners == {str(pid)},
            f'Port {port} is not exclusively owned by the launched server PID {pid}')


def get_json(port, route):
    # Explicitly disable environment proxies for local administrative requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}{route}', timeout=5) as response:
        return json.load(response)


def ready(server, port):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        require(server.poll() is None, f'Owned server exited during readiness: {server.returncode}')
        try:
            health = get_json(port, '/health')
        except (OSError, urllib.error.URLError, ValueError):
            time.sleep(0.5)
            continue
        listener_owner(port, server.pid)
        require(health.get('status') == 'ok', 'Unexpected health response')
        props = get_json(port, '/props')
        if 'settings' not in props:
            time.sleep(0.5)
            continue
        require(server.poll() is None, 'Owned server exited after readiness')
        return {'health': health, 'props': props, 'models': get_json(port, '/v1/models')}
    raise TimeoutError('Server was not ready within 600 seconds; no generation warmup was sent')


def check_effective(observed):
    props = observed['props']
    settings = props['settings']
    require(settings['engine'] == 'mlx', 'Expected native MLX engine')
    require(settings['mtp']['loaded'] and settings['mtp']['default_on'], 'Native MTP is not loaded/enabled')
    require(settings['drafter'] == 'none' and not settings['pld']['default_on'], 'Unexpected drafter or PLD')
    require(settings['max_concurrent'] == 1, 'Unexpected concurrency')
    require(settings['prefix_cache']['ram_enabled'] and settings['prefix_cache']['disk_bytes'] == 0,
            'Expected production RAM prefix retention and no SSD prefix tier')
    require(props['default_generation_settings']['n_ctx'] == 32768, 'Effective context differs from 32768')


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def account(path, assigned):
    starts, terminal, errors = set(), {}, []
    if path.exists():
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except ValueError:
                    errors.append(f'Incomplete/invalid JSONL record at line {number}')
                    continue
                if event.get('kind') == 'request-start':
                    starts.add(event['id'])
                elif event.get('kind') == 'request':
                    identity = event['id']
                    if identity in terminal:
                        errors.append(f'Duplicate terminal result: {identity}')
                    terminal[identity] = event
    expected_ids = {row['id'] for row in assigned}
    if (starts | set(terminal)) - expected_ids:
        errors.append('Replay emitted an unassigned request ID')
    requests = []
    for row in assigned:
        identity = row['id']
        event = terminal.get(identity)
        status = event.get('status') if event else ('interrupted' if identity in starts else 'unattempted')
        item = {'id': identity, 'status': status, 'expected_input_tokens': row['input_tokens']}
        if event:
            item.update({key: event[key] for key in ('usage', 'output_capped', 'problems', 'error',
                         'first_delta_ms', 'total_ms', 'finish_reason', 'comparison_sha256') if key in event})
        if status == 'completed':
            actual = event.get('usage', {}).get('prompt_tokens')
            item['input_tokens_match'] = actual == row['input_tokens']
            if not item['input_tokens_match']:
                item['status'] = 'invalid-input-token-parity'
                errors.append(f'{identity}: expected {row["input_tokens"]} input tokens, got {actual}')
        requests.append(item)
    counts = Counter(row['status'] for row in requests)
    return {'assigned': len(assigned), 'counts': dict(counts),
            'capped': sum(bool(row.get('output_capped')) for row in requests),
            'valid': not errors and counts.get('completed', 0) == len(assigned),
            'errors': errors, 'requests': requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--arm', choices=('main', 'history-off', 'history-on'), required=True)
    parser.add_argument('--episode', action='append', required=True,
                        help='Repeat only for an explicitly declared same-server control sequence')
    parser.add_argument('--out', type=Path, required=True, help='New private directory, outside checkouts')
    parser.add_argument('--port', type=int, default=18999)
    parser.add_argument('--admission', required=True,
                        help='Operator statement: AC, stable thermal state, exclusive GPU or owned FIFO turn')
    parser.add_argument('--plan', action='store_true', help='Validate and print; never launch server/replay')
    args = parser.parse_args()
    os.umask(0o077)
    for key in ('work', 'data', 'model', 'out'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
        external(getattr(args, key), key)
    require(args.work.is_dir() and args.data.is_dir(), 'Work and data directories must exist')
    require(args.out.parent.is_dir() and not args.out.exists(), 'Output must be new with an existing parent')
    require(1 <= args.port <= 65535, 'Invalid port')
    require(args.admission.strip(), 'An explicit operator admission statement is required')
    require(len(set(args.episode)) == len(args.episode), 'Do not repeat an episode inside a block')
    for root in (args.work / 'main', args.work / 'pr', args.data, args.model):
        require(not args.out.is_relative_to(root), 'Output must not be inside an input/source directory')
    for tree in (args.work / 'main', args.work / 'pr'):
        require(not args.data.is_relative_to(tree) and not args.model.is_relative_to(tree),
                'Private data/model must not be inside source checkouts')
    pins = json.loads((HERE / 'pins.json').read_text())
    validation = validate(args.data, None if args.plan else args.model)
    records = [json.loads(line) for line in (args.data / 'requests.jsonl').read_text().splitlines()]
    assignments = [[row for row in records if row['episode_id'] == episode] for episode in args.episode]
    require(all(assignments), 'Unknown/empty episode; discover private IDs with validate.py')
    sources = check_sources(args.work, pins)
    setup, skipped = check_setup(args.work, pins, args.plan)
    node = shutil.which('node')
    if node:
        version_env, _ = isolated_environment(args.out)
        version = command([node, '--version'], env=version_env)
        if args.plan and version != 'v' + pins['node_version']:
            skipped.append(f'Node version mismatch: {version}; real run requires v{pins["node_version"]}')
        else:
            require(version == 'v' + pins['node_version'], 'Node version differs from pin')
    else:
        require(args.plan, 'Node is not on PATH')
        skipped.append('Node unavailable; real run requires pinned Node')
    if args.plan:
        skipped.append('Model full hashes, AC/thermal admission, free/owned port and effective settings not checked')
    source_arm = 'main' if args.arm == 'main' else 'pr'
    binary = args.work / source_arm / 'zig-out/bin/mlx-serve'
    client = args.work / 'timing/llmprobe-timing-client.mjs'
    server_argv = [str(binary), '--model', str(args.model), '--serve', '--host', '127.0.0.1',
                   '--port', str(args.port), '--mtp', '--no-drafter', '--no-pld', '--no-vision',
                   '--no-plugins', '--max-concurrent', '1', '--ctx-size', '32768',
                   '--prefix-cache-entries', '32', '--prefix-cache-disk', '0']
    if args.arm == 'history-on':
        server_argv += ['--lookup-store', str(args.out / 'history')]
    replay_commands = [[node or 'node', str(HERE / 'replay-timed.mjs'), str(client),
                        str(args.data / 'requests.jsonl'), episode, f'http://127.0.0.1:{args.port}',
                        str(args.out / f'episode-{index:02d}.jsonl'), str(args.out / 'receipt.json')]
                       for index, episode in enumerate(args.episode)]
    environment, removed = isolated_environment(args.out)
    assignment = {'schema': 'pr999-serving-block-v1', 'created_at': stamp(), 'arm': args.arm,
                  'episodes': [{'episode': episode, 'ids': [row['id'] for row in rows]}
                               for episode, rows in zip(args.episode, assignments)],
                  'source_commit': pins['engine_commits'][source_arm], 'source_checkouts': sources,
                  'dataset_sha256': pins['dataset_sha256'], 'model_identity': pins['model'],
                  'server_argv': server_argv, 'replay_commands': replay_commands,
                  'admission': args.admission, 'environment': environment,
                  'removed_environment_keys': removed, 'validation': validation,
                  'prefix_policy': '32 RAM entries, production model-dependent default memory, no SSD tier',
                  'limits': LIMITS}
    if args.plan:
        print(json.dumps({'plan_only': True, 'skipped_checks': skipped, **assignment}, indent=2))
        return 0
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Real run requires Apple Silicon macOS')
    args.out.mkdir(mode=0o700)
    save(args.out / 'assignment.json', assignment)
    server = replay = reservation = None
    errors, episodes = [], []
    started = time.monotonic()
    interrupted = False

    def interrupt(signum, frame):
        raise InterruptedError(f'Operator interruption: signal {signum}')

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name in ('home', 'cache', 'config', 'data', 'tmp', 'hf', 'history'):
            (args.out / name).mkdir(mode=0o700)
        snapshot = host_snapshot()
        save(args.out / 'host-before.json', snapshot)
        require_ac(snapshot)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reservation.bind(('127.0.0.1', args.port))
        receipt = {**assignment, 'binary_sha256': sha(binary),
                   'setup_receipt_sha256': sha(args.work / 'setup-receipt.json'), 'setup': setup,
                   'tooling_sha256': {name: sha(HERE / name) for name in
                                     ('run.py', 'validate.py', 'pins.json', 'replay-timed.mjs')},
                   'node_path': node, 'node_sha256': sha(Path(node)), 'node_version': version,
                   'store_initial_state': {'empty': True, 'preseeded': False,
                                           'enabled': args.arm == 'history-on', 'path': str(args.out / 'history')},
                   'power': snapshot['power'], 'thermal': snapshot['thermal'],
                   'host': snapshot, 'fifo_owner': args.admission,
                   'ownership_note': 'Operator assertion, not an automatically acquired FIFO lock'}
        save(args.out / 'receipt.json', receipt)
        with (args.out / 'server.log').open('xb') as log:
            reservation.close()
            reservation = None
            server = subprocess.Popen(server_argv, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=args.out, env=environment, start_new_session=True)
            save(args.out / 'server-start.json', {'pid': server.pid, 'at': stamp(), 'argv': server_argv})
            observed = ready(server, args.port)
            save(args.out / 'effective-before.json', observed)
            check_effective(observed)
            for index, (episode, rows, argv) in enumerate(zip(args.episode, assignments, replay_commands)):
                require(server.poll() is None, 'Owned server exited before episode')
                listener_owner(args.port, server.pid)
                snapshot = host_snapshot(server.pid)
                save(args.out / f'host-episode-{index:02d}.json', snapshot)
                require_ac(snapshot)
                with (args.out / f'replay-{index:02d}.log').open('xb') as replay_log:
                    replay = subprocess.Popen(argv, stdout=replay_log, stderr=subprocess.STDOUT,
                                              cwd=args.out, env=environment, start_new_session=True)
                    save(args.out / f'replay-start-{index:02d}.json',
                         {'pid': replay.pid, 'episode': episode, 'at': stamp()})
                    code = replay.wait()
                report = account(args.out / f'episode-{index:02d}.jsonl', rows)
                require(code == 0 and report['valid'], f'Episode {index} failed or has invalid token parity')
                require(server.poll() is None, 'Owned server exited during episode')
            save(args.out / 'effective-after.json', {'props': get_json(args.port, '/props')})
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError))
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        # Repeated signals must not interrupt owned-PID cleanup or failure accounting.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if reservation is not None:
            reservation.close()
        for label, proc in (('replay', replay), ('server', server)):
            try:
                stop_owned(proc)
            except BaseException as error:
                errors.append(f'{label} cleanup failed: {error}')
        for index, (episode, rows) in enumerate(zip(args.episode, assignments)):
            report = account(args.out / f'episode-{index:02d}.jsonl', rows)
            episodes.append({'episode': episode, **report})
        after = host_snapshot()
        save(args.out / 'host-after.json', after)
        try:
            require_ac(after)
        except ValueError as error:
            errors.append(str(error))
        counts = Counter()
        for episode in episodes:
            counts.update(episode['counts'])
        history_bytes = sum(path.stat().st_size for path in (args.out / 'history').rglob('*') if path.is_file())
        history_stats = []
        if (args.out / 'server.log').exists():
            with (args.out / 'server.log').open(errors='replace') as log:
                for line in log:
                    if '[history] external_rounds=' in line:
                        history_stats.append({key: int(value) for key, value in
                                              re.findall(r'([a-z_]+)=(\d+)', line)})
        summary = {'schema': 'pr999-serving-block-summary-v1', 'finished_at': stamp(), 'arm': args.arm,
                   'valid_serving_block': not errors and all(e['valid'] for e in episodes),
                   'interrupted': interrupted, 'errors': errors, 'episodes': episodes,
                   'assigned': sum(e['assigned'] for e in episodes), 'counts': dict(counts),
                   'capped': sum(e['capped'] for e in episodes), 'history_bytes': history_bytes,
                   'history_counters_per_logged_request': history_stats,
                   'history_counter_note': 'Empty means no history counter records, not measured zero engagement',
                   'orchestration_elapsed_seconds': time.monotonic() - started,
                   'timing_note': 'Per-request serving timings are canonical llmprobe fields, not orchestration elapsed time',
                   'server_pid': server.pid if server else None,
                   'server_returncode_after_cleanup': server.returncode if server else None,
                   'assignment_sha256': sha(args.out / 'assignment.json'),
                   'receipt_sha256': sha(args.out / 'receipt.json') if (args.out / 'receipt.json').exists() else None,
                   'limits': LIMITS}
        save(args.out / 'summary.json', summary)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({'output': str(args.out), 'valid_serving_block': summary['valid_serving_block'],
                      'counts': summary['counts'], 'errors': errors}))
    return 0 if summary['valid_serving_block'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'Refused: {error}', file=sys.stderr)
        sys.exit(1)
