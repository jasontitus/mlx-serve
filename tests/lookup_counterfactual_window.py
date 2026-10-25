#!/usr/bin/env python3
"""One explicit owned <=290s diagnostic window, or CPU-only comparison.

Each run starts ONE fresh server and sends TWO identical related-config chat requests.
Run serial first with the current binary; each subsequent action requires that oracle.
No builds, automatic follow-up windows, throughput claims, or later-offer matching.

Examples (separate invocations, never chained):
  python tests/lookup_counterfactual_window.py run --action serial --binary zig-out/bin/mlx-serve --output /tmp/cf-serial
  python tests/lookup_counterfactual_window.py run --action assistant8 --binary zig-out/bin/mlx-serve --oracle /tmp/cf-serial --output /tmp/cf-a8
  python tests/lookup_counterfactual_window.py compare /tmp/cf-a8 /tmp/cf-a16 --oracle /tmp/cf-serial
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HELPERS = Path(__file__).resolve().parent
SEED = Path.home() / '.mlx-serve/bench/lookup-staircut/runs/20261025-transfer-training-literal'
MARKER = '[lookup-counterfactual] '
ACTIONS = ('serial', 'assistant8', 'assistant16', 'lookup8', 'lookup16')
IDENTITY_KEYS = ('prefix', 'target_kv', 'target_ssm', 'assistant', 'controls', 'training', 'learner', 'request_lookup')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_content(model):
    result = {}
    for role, key in (("target", "model_dir"), ("assistant", "drafter_dir")):
        root = Path(model[key])
        files = sorted(path for path in root.iterdir() if path.suffix in (".json", ".safetensors", ".bin"))
        require(any(path.suffix == ".safetensors" for path in files), "missing model weight files")
        result[role] = {path.name: sha(path) for path in files}
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostic_limit_refused(response):
    if response.get('http_status') != 409:
        return False
    try:
        payload = json.loads(response.get('error', ''))
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get('error'), dict) and (
        payload['error'].get('type') == 'diagnostic_request_limit' and payload['error'].get('code') == 409)


def digest(value):
    if isinstance(value, list):
        require(len(value) == 32 and all(type(x) is int and 0 <= x <= 255 for x in value), 'invalid digest bytes')
        return bytes(value).hex()
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid SHA256 digest')
    return value


def normalized(receipt):
    row = json.loads(json.dumps(receipt))
    for key in IDENTITY_KEYS:
        row['identity'][key] = digest(row['identity'][key])
    for key in ('target_kv', 'target_ssm', 'assistant'):
        row['prepared'][key] = digest(row['prepared'][key])
    for part in ('offer', 'selected'):
        for key in ('candidates', 'probabilities'):
            row[part][key] = digest(row[part][key])
    row['kernel']['dispatch'] = digest(row['kernel']['dispatch'])
    return row


def validate_receipt(receipt, action, phase):
    row = normalized(receipt)
    require(row['phase'] == phase and row['action'] == action and row['first_offer'] == 0, 'wrong predeclared first offer')
    width = 16 if action.endswith('16') else 8
    require(row['kernel']['rows'] == width, 'actual forced width differs')
    require(row['kernel']['variant'] in ('generic', 'linear_gdn') and row['kernel']['kind'] in ('tree', 'chain'), 'missing actual kernel classification')
    require(row['kernel']['dispatch'] != '0' * 64, 'missing actual dispatch proof')
    require(row['warmup_proven'] is (phase == 'measured'), 'warmup proof differs from phase')
    count = row['published_len']
    require(type(count) is int and 0 < count <= width and count == row['accepted_drafts'] + 1, 'incomplete native publication')
    require(len(row['published']) >= count and all(type(x) is int and x >= 0 for x in row['published'][:count]), 'invalid published token IDs')
    timing = row['timing']
    require(type(timing['verify_commit_ns']) is int and timing['verify_commit_ns'] > 0, 'invalid settled verify/commit time')
    require(type(timing['diagnostic_readback_ns']) is int and timing['diagnostic_readback_ns'] >= 0, 'missing readback accounting')
    require(timing['full_round_ns'] is None or (type(timing['full_round_ns']) is int and timing['full_round_ns'] >= timing['verify_commit_ns']), 'invalid complete round timing')
    return row


def validate_pair(receipts, action):
    require(len(receipts) == 2, 'exactly one warmup and one measurement required')
    warm, measured = [validate_receipt(row, action, phase) for row, phase in zip(receipts, ('warmup', 'measured'))]
    for key in ('identity', 'prepared', 'offer', 'selected', 'kernel'):
        require(warm[key] == measured[key], 'warmup/measurement mismatch: ' + key)
    require(warm['published'][:warm['published_len']] == measured['published'][:measured['published_len']], 'warmup output differs')
    return measured


def oracle_for(path, plan):
    root = Path(path)
    declaration = load(root / 'plan.json')
    summary = load(root / 'summary.json')
    require(summary.get('complete') is True and declaration['action'] == 'serial', 'incomplete/nonserial oracle')
    for key in ('binary_sha256', 'model', 'model_content', 'semantic_request_sha256', 'contract_sha256', 'seed_hashes'):
        require(declaration[key] == plan[key], 'serial oracle mismatch: ' + key)
    samples = load(root / 'samples.json')
    require(len(samples) == 2 and samples[0]['identity'] == samples[1]['identity'], 'serial repeats differ')
    return samples[1]['identity']


def helpers(path):
    sys.path.insert(0, str(Path(path).resolve()))
    return importlib.import_module('lookup_similar_requests')


def build_plan(args):
    h = helpers(args.helpers)
    d, r = h.d, h.r
    contract_path = h.CONTRACT
    contract = load(contract_path)
    training = [h.render(contract, item) for item in contract['training']]
    evaluation = [h.render(contract, item) for item in contract['evaluation']]
    case = next(item for item in evaluation if item['id'] == 'related-config')
    manifest = load(d.KIT / 'manifest.json')
    model = next(item for item in manifest['models'] if item['id'] == 'qwen27b')
    require(r.digest(Path(model['model_dir']) / 'tokenizer.json') == model['tokenizer_sha256'], 'tokenizer changed')
    binary = Path(args.binary).expanduser().resolve()
    plan = {'version': 1, 'action': args.action, 'binary': str(binary), 'binary_sha256': sha(binary),
            'helpers': str(Path(args.helpers).resolve()), 'model': model,
            'training_cases': training, 'request_identities': h.unique_requests(training, evaluation),
            'case': case, 'semantic_request_sha256': h.semantic_hash(case), 'contract_sha256': sha(contract_path),
            'server_args': {'port': args.port, 'ctx_size': 2048, 'ready_timeout': 35, 'request_timeout': 90},
            'first_predeclared_offer': 0, 'requests': 2, 'deadline_seconds': 290,
            'numerical_mode': 'row-exact', 'diagnostic_only': True, 'throughput_claim': False,
            'oracle': str(Path(args.oracle).resolve()) if args.oracle else None}
    seed = h.load_seed(Path(args.seed), plan)
    plan['reuse'] = seed
    plan['seed_hashes'] = seed['hashes']
    plan['model_content'] = model_content(model)
    plan['helper_hashes'] = {str(Path(module.__file__).resolve()): sha(module.__file__) for module in (h, d, r)}
    if args.action != 'serial':
        require(args.oracle is not None, 'feature window requires current-binary serial oracle')
        oracle_for(args.oracle, plan)
    return plan


def sample_identity(response, stats, body):
    require(not response.get('error') and response.get('http_status') == 200, 'HTTP/SSE failure')
    require(response.get('finish_reason') == 'stop' and not response.get('finish_details'), 'truncated/abnormal response')
    require(isinstance(response.get('text'), str) and bool(response['text']), 'empty output')
    commits = stats['commits']
    require(len(commits) == 1, 'exactly one complete native commit receipt required')
    commit = commits[0]
    for key in ('prompt_ids', 'generated_ids'):
        require(isinstance(commit.get(key), list) and bool(commit[key]) and all(type(t) is int and t >= 0 for t in commit[key]), 'missing exact ' + key)
    prompt, generated = commit['prompt_ids'], commit['generated_ids']
    require(len(generated) <= body['max_tokens'], 'token cap exceeded')
    require(commit.get('finish_reason') == 'stop', 'native commit incomplete')
    counts = {'prompt_tokens': len(prompt), 'completion_tokens': len(generated), 'total_tokens': len(prompt) + len(generated)}
    require(all(response.get('usage', {}).get(k) == v for k, v in counts.items()), 'usage/commit mismatch')
    return {'text': response['text'], 'usage': response['usage'], 'finish_reason': 'stop', 'prompt_ids': prompt, 'generated_ids': generated}


def worker(out):
    out = Path(out)
    plan = load(out / 'plan.json')
    h = helpers(plan['helpers'])
    d, r = h.d, h.r
    for path, expected in plan['helper_hashes'].items():
        require(sha(path) == expected, 'helper changed after declaration')
    require(sha(plan['binary']) == plan['binary_sha256'], 'binary changed after declaration')
    require(model_content(plan['model']) == plan['model_content'], 'model content changed before launch')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('bounded window stopped')))
    args = argparse.Namespace(**plan['server_args'])
    folder = out / 'owned-boot'
    folder.mkdir()
    store, corpus, before = d.prepare_boot_assets(folder, plan, 'persistent')
    serial = plan['action'] == 'serial'
    mode = 'serial' if serial else 'dflash-persistent'
    env = {'MLX_SERVE_ROW_EXACT': '1', 'MLX_SERVE_LOOKUP_STORE_WRITE': '0', 'MLX_SERVE_LOOKUP_RECEIPTS': '1'}
    if not serial:
        env.update(MLX_SERVE_LOOKUP_COUNTERFACTUAL='1', MLX_SERVE_LOOKUP_COUNTERFACTUAL_ACTION=plan['action'],
                   MLX_SERVE_LOOKUP_SELECTION='fixed:' + ('16' if plan['action'].endswith('16') else '8'))
    else:
        env['MLX_SERVE_LOOKUP_COUNTERFACTUAL'] = '0'
    server = r.Server(args, plan['binary'], plan['model'], mode, folder / 'server.log', corpus=corpus, store=store, experiment=env)
    save(folder / 'launch.json', {'command': server.command, 'environment': server.env})
    samples, receipts = [], []
    try:
        with server:
            save(folder / 'owned.json', {'pid': server.proc.pid, 'worker_pid': os.getpid()})
            require(server.numerical_mode == 'row-exact', 'row-exact activation not observed')
            if not serial:
                unauthenticated = argparse.Namespace(base=server.base)
                refusal = r.request(unauthenticated, r.request_body(plan['case'], mode, None), 5)
                save(folder / 'authentication-refusal.json', refusal)
                require(refusal.get('http_status') in (401, 403), 'strict authentication not enforced')
                server.assert_owned()
            for phase in ('warmup', 'measured'):
                sample_dir = out / phase
                sample_dir.mkdir()
                body = r.request_body(plan['case'], mode, None)
                require(body.get('temperature') == 0 and body.get('seed') == 0 and body.get('enable_thinking') is False, 'request controls changed')
                save(sample_dir / 'request.json', body)
                server.assert_owned()
                offset = server.log_path.stat().st_size
                original = r.HTTP
                with (sample_dir / 'response.sse').open('wb') as raw:
                    try:
                        r.HTTP = d.RecordingHTTP(original, raw, sample_dir / 'http.json')
                        response = r.request(server, body, args.request_timeout)
                    finally:
                        r.HTTP = original
                save(sample_dir / 'response.json', response)
                log = r.settled_log(server, offset, timeout=2)
                (sample_dir / 'receipts.log').write_text(log)
                stats = r.parse_stats(log, 'serial' if serial else 'dflash')
                identity = sample_identity(response, stats, body)
                quality_errors = h.quality(plan['case'], {'response': response, 'body': body})
                require(not quality_errors, 'semantic control failed: ' + repr(quality_errors))
                if serial:
                    require(not stats.get('rounds'), 'serial oracle unexpectedly speculated')
                else:
                    require('[lookup-counterfactual-invalid]' not in log, 'diagnostic invalidated during request')
                    found = [json.loads(line.split(MARKER, 1)[1]) for line in log.splitlines() if MARKER in line]
                    require(len(found) == 1, 'missing/duplicate first-offer diagnostic receipt')
                    receipt = validate_receipt(found[0], plan['action'], phase)
                    require(receipt['published'][:receipt['published_len']] == identity['generated_ids'][:receipt['published_len']], 'first offer is not first published output')
                    receipts.append(receipt)
                samples.append({'phase': phase, 'identity': identity})
                save(out / 'samples.json', samples)
                save(out / 'receipts.json', receipts)
            if not serial:
                refusal = r.request(server, body, 5)
                save(folder / 'third-request-refusal.json', refusal)
                require(diagnostic_limit_refused(refusal), 'third diagnostic request was not rejected by the limit gate')
                server.assert_owned()
        require(samples[0]['identity'] == samples[1]['identity'], 'identical requests produced different outputs')
        if not serial:
            validate_pair(receipts, plan['action'])
            reference = oracle_for(plan['oracle'], plan)
            require(all(sample['identity'] == reference for sample in samples), 'exact same-binary serial oracle mismatch')
        require(sha(plan['binary']) == plan['binary_sha256'], 'binary changed during run')
        require(model_content(plan['model']) == plan['model_content'], 'model content changed during run')
    finally:
        assets = d.finish_boot_assets(folder, plan, store, corpus, before)
        require(assets['unchanged'] and assets['seeds_unchanged'], 'frozen history changed')
    save(out / 'summary.json', {'complete': True, 'action': plan['action'], 'diagnostic_only': True, 'performance_pass': False,
                              'timing_scope': 'verify+commit; complete round only when full_round_ns is nonnull', 'requests': 2})
    return 0


def run(args):
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', build_plan(args))
    if args.plan_only:
        return 0
    # The owned worker and server share one killable process group. Stop at285s,
    # reserve5s for cleanup, hard kill at290s. Never start another window here.
    start = time.monotonic()
    child = None
    def stop_owned():
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    def hard_deadline(*_):
        stop_owned()
        os._exit(124)
    def interrupted(*_):
        raise KeyboardInterrupt('owned window interrupted')
    old_alarm = signal.signal(signal.SIGALRM, hard_deadline)
    old_term = signal.signal(signal.SIGTERM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 290)
    try:
        with (out / 'worker.log').open('wb') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker', str(out)],
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = child.wait(timeout=285)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=max(0.01, 290 - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise ValueError('bounded window expired; incomplete comparison rejected')
            require(code == 0 and load(out / 'summary.json').get('complete'), 'owned worker failed; inspect retained raw artifacts')
    finally:
        stop_owned()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
    return 0


def compare(paths, oracle):
    records = []
    reference_plan = None
    actions = set()
    for path in paths:
        root = Path(path)
        plan = load(root / 'plan.json')
        require(load(root / 'summary.json').get('complete') is True, 'incomplete diagnostic window')
        require(plan['action'] not in actions and plan['action'] != 'serial', 'duplicate/nonfeature action')
        actions.add(plan['action'])
        if reference_plan is not None:
            for key in ('binary_sha256', 'model', 'semantic_request_sha256', 'contract_sha256', 'seed_hashes'):
                require(plan[key] == reference_plan[key], 'window declaration mismatch: ' + key)
        reference_plan = plan
        reference = oracle_for(oracle, plan)
        samples = load(root / 'samples.json')
        require(len(samples) == 2 and all(row['identity'] == reference for row in samples), 'serial exact-output mismatch')
        records.append(validate_pair(load(root / 'receipts.json'), plan['action']))
    require(len(records) >= 2, 'at least two independent explicit windows required')
    for a in records:
        for b in records:
            require(a['identity'] == b['identity'], 'actual common state or learner mismatch')
            for key in ('target_kv', 'target_ssm'):
                require(a['prepared'][key] == b['prepared'][key], 'actual target before verify differs')
            same_preparation = a['action'].startswith('assistant') == b['action'].startswith('assistant')
            if same_preparation:
                require(a['offer'] == b['offer'] and a['prepared'] == b['prepared'], 'width contrast candidate/probability/prepared-state mismatch')
    report = {'valid_counterfactual': True, 'diagnostic_only': True, 'throughput_claim': False,
              'first_offer': 0, 'actions': sorted(actions), 'same_preparation_width_contrast': len({a.startswith('assistant') for a in actions}) == 1,
              'preparation_note': 'Assistant/lookup action contrasts deliberately use different candidate sets; width contrasts require identical offers.',
              'observations': []}
    for row in records:
        count = row['published_len']
        full = row['timing']['full_round_ns']
        report['observations'].append({'action': row['action'], 'actual_rows': row['kernel']['rows'], 'variant': row['kernel']['variant'],
            'committed_tokens': count, 'accepted_drafts': row['accepted_drafts'],
            'verify_commit_ms_per_committed_token': row['timing']['verify_commit_ns'] / 1e6 / count,
            'full_round_ms_per_committed_token': None if full is None else full / 1e6 / count,
            'readback_ms_excluded': row['timing']['diagnostic_readback_ns'] / 1e6})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    one = subs.add_parser('run')
    one.add_argument('--action', choices=ACTIONS, required=True)
    one.add_argument('--binary', type=Path, required=True)
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--oracle', type=Path)
    one.add_argument('--seed', type=Path, default=SEED)
    one.add_argument('--helpers', type=Path, default=HELPERS)
    one.add_argument('--port', type=int, default=11279)
    one.add_argument('--plan-only', action='store_true')
    cpu = subs.add_parser('compare')
    cpu.add_argument('windows', type=Path, nargs='+')
    cpu.add_argument('--oracle', type=Path, required=True)
    hidden = subs.add_parser('_worker', help=argparse.SUPPRESS)
    hidden.add_argument('output', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == 'run':
        return run(args)
    if args.command == 'compare':
        return compare(args.windows, args.oracle)
    return worker(args.output)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        raise SystemExit(1)
