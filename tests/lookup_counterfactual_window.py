#!/usr/bin/env python3
"""One explicit owned <=290s diagnostic window, or CPU-only comparison/price export.

Each run starts ONE fresh server and sends TWO identical predeclared chat requests.
Run serial first with the current binary; each subsequent action requires that oracle.
No builds, automatic follow-up windows, throughput claims, or later-offer matching.

Examples (separate invocations, never chained):
  python tests/lookup_counterfactual_window.py run --preparation production-neural --action serial --case train-config-a --binary zig-out/bin/mlx-serve --output /tmp/train-serial
  python tests/lookup_counterfactual_window.py run --preparation production-neural --action assistant8 --case train-config-a --binary zig-out/bin/mlx-serve --oracle /tmp/train-serial --output /tmp/train-a8
  python tests/lookup_counterfactual_window.py compare /tmp/train-a8 /tmp/train-a16 --oracle /tmp/train-serial
  python tests/lookup_counterfactual_window.py export-prices /tmp/train-a8 /tmp/train-a16 --oracle /tmp/train-serial --output /tmp/assistant-prices.json
"""
import argparse
import hashlib
import importlib
import json
import math
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
ACTIONS = ('serial',) + tuple(f'{source}{rows}' for source in ('assistant', 'lookup') for rows in range(1, 17))
PREPARATIONS = ('merged16', 'production-neural', 'production-combined')
IDENTITY_KEYS = ('prefix', 'target_kv', 'target_ssm', 'assistant', 'controls', 'training', 'learner', 'request_lookup')
DECLARATION_KEYS = ('binary_sha256', 'runtime_identity', 'model', 'model_content',
                    'semantic_request_sha256', 'contract_sha256', 'seed_hashes', 'preparation')


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


def action_width(action):
    require(action in ACTIONS and action != 'serial', 'invalid diagnostic action')
    return int(action.removeprefix('assistant').removeprefix('lookup'))


def validate_preparation(preparation, action):
    require(preparation in PREPARATIONS, 'missing/invalid predeclared preparation')
    require(action in ACTIONS, 'invalid diagnostic action')
    require(preparation != 'production-neural' or not action.startswith('lookup'),
            'production-neural preparation rejects lookup actions')


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


def validate_receipt(receipt, action, phase, preparation='merged16'):
    row = normalized(receipt)
    validate_preparation(preparation, action)
    prepared = row['prepared']
    require(prepared.get('preparation') == preparation, 'receipt preparation mismatch')
    require(prepared.get('suffix_candidates') is (preparation != 'production-neural'), 'candidate-source control mismatch')
    horizon = prepared.get('assistant_horizon')
    require(type(horizon) is int and (horizon > 0 if action.startswith('assistant') else horizon == 0),
            'missing actual assistant horizon')
    if preparation == 'merged16' and action.startswith('assistant'):
        require(horizon == 16, 'merged16 assistant horizon mismatch')
    require(type(prepared.get('context_length')) is int and prepared['context_length'] > 0,
            'missing actual context length')
    require(row['phase'] == phase and row['action'] == action and row['first_offer'] == 0, 'wrong predeclared first offer')
    width = action_width(action)
    require(type(row['kernel']['rows']) is int and row['kernel']['rows'] == width, 'actual forced width differs')
    require(row['kernel']['variant'] in ('generic', 'linear_gdn') and row['kernel']['kind'] == 'tree', 'missing actual DFlash tree-verifier classification')
    require(row['kernel']['dispatch'] != '0' * 64, 'missing actual dispatch proof')
    require(row['warmup_proven'] is (phase == 'measured'), 'warmup proof differs from phase')
    count = row['published_len']
    require(type(count) is int and 0 < count <= width and count == row['accepted_drafts'] + 1, 'incomplete native publication')
    require(len(row['published']) >= count and all(type(x) is int and x >= 0 for x in row['published'][:count]), 'invalid published token IDs')
    geometry = row['kernel']['geometry']
    parents, depths = geometry['selected_parents'], geometry['selected_depths']
    require(len(parents) >= width and len(depths) >= width and parents[0] == -1 and depths[0] == 0,
            'missing selected parent/depth geometry')
    for index in range(1, width):
        parent = parents[index]
        require(type(parent) is int and 0 <= parent < index and depths[index] == depths[parent] + 1,
                'invalid selected parent/depth geometry')
    if preparation == 'production-neural':
        require(max(depths[:width]) <= horizon and count <= horizon + 1,
                'neural-only geometry/publication exceeds actual assistant horizon')
    committed = geometry['committed_rows'][:count]
    require(geometry['committed_count'] == count and len(committed) == count and committed[0] == 0,
            'commit geometry count mismatch')
    require(all(type(index) is int and 0 <= index < width for index in committed) and
            all(parents[index] == prior for prior, index in zip(committed, committed[1:])),
            'invalid committed path')
    require(geometry['kv_compacted'] is any(index != position for position, index in enumerate(committed)),
            'compaction geometry mismatch')
    timing = row['timing']
    require(type(timing['verify_commit_ns']) is int and timing['verify_commit_ns'] > 0, 'invalid settled verify/commit time')
    require(type(timing['diagnostic_readback_ns']) is int and timing['diagnostic_readback_ns'] >= 0, 'missing readback accounting')
    require(timing['full_round_ns'] is None or (type(timing['full_round_ns']) is int and timing['full_round_ns'] >= timing['verify_commit_ns']), 'invalid complete round timing')
    return row


def validate_pair(receipts, action, preparation='merged16'):
    require(len(receipts) == 2, 'exactly one warmup and one measurement required')
    warm, measured = [validate_receipt(row, action, phase, preparation) for row, phase in zip(receipts, ('warmup', 'measured'))]
    for key in ('identity', 'prepared', 'offer', 'selected', 'kernel'):
        require(warm[key] == measured[key], 'warmup/measurement mismatch: ' + key)
    require(warm['published'][:warm['published_len']] == measured['published'][:measured['published_len']], 'warmup output differs')
    return measured


def validate_runtime_evidence(root, plan):
    expected = plan.get('runtime_identity')
    require(isinstance(expected, dict) and bool(expected), 'missing runtime identity; redeclare historical windows')
    evidence = load(Path(root) / 'runtime-identity.json')
    require(evidence.get('before') == expected and evidence.get('after') == expected,
            'runtime libraries or kernels changed during diagnostic')


def validate_samples(samples):
    require(len(samples) == 2 and [row.get('phase') for row in samples] == ['warmup', 'measured'],
            'exactly one warmup and one measured output required')
    require(samples[0]['identity'] == samples[1]['identity'], 'serial repeats differ')
    return samples[1]['identity']


def oracle_for(path, plan):
    root = Path(path)
    declaration = load(root / 'plan.json')
    summary = load(root / 'summary.json')
    require(summary.get('complete') is True and declaration['action'] == 'serial', 'incomplete/nonserial oracle')
    validate_preparation(declaration.get('preparation'), declaration['action'])
    validate_preparation(plan.get('preparation'), plan['action'])
    validate_runtime_evidence(root, declaration)
    for key in DECLARATION_KEYS:
        require(declaration[key] == plan[key], 'serial oracle mismatch: ' + key)
    samples = load(root / 'samples.json')
    return validate_samples(samples)


def helpers(path):
    sys.path.insert(0, str(Path(path).resolve()))
    return importlib.import_module('lookup_similar_requests')


def build_plan(args):
    validate_preparation(args.preparation, args.action)
    h = helpers(args.helpers)
    d, r = h.d, h.r
    contract_path = h.CONTRACT
    contract_json = Path(contract_path).read_bytes().decode('utf-8')
    contract = json.loads(contract_json)
    training = [h.render(contract, item) for item in contract['training']]
    evaluation = [h.render(contract, item) for item in contract['evaluation']]
    matching = [item for item in training + evaluation if item['id'] == args.case]
    require(len(matching) == 1, 'case must name one declared training or evaluation request')
    case = matching[0]
    manifest = load(d.KIT / 'manifest.json')
    model = next(item for item in manifest['models'] if item['id'] == 'qwen27b')
    require(r.digest(Path(model['model_dir']) / 'tokenizer.json') == model['tokenizer_sha256'], 'tokenizer changed')
    binary = Path(args.binary).expanduser().resolve()
    plan = {'version': 1, 'action': args.action, 'preparation': args.preparation, 'binary': str(binary), 'binary_sha256': sha(binary),
            'runtime_identity': r.runtime_identity(binary), 'helpers': str(Path(args.helpers).resolve()), 'model': model,
            'training_cases': training, 'request_identities': h.unique_requests(training, evaluation),
            'case': case, 'semantic_request_sha256': h.semantic_hash(case),
            'contract_sha256': hashlib.sha256(contract_json.encode('utf-8')).hexdigest(),
            'contract_json': contract_json,
            'case_phase': 'training' if any(item['id'] == case['id'] for item in training) else 'evaluation',
            'server_args': {'port': args.port, 'ctx_size': 2048, 'ready_timeout': 35, 'request_timeout': 90},
            'first_predeclared_offer': 0, 'requests': 2, 'deadline_seconds': 290,
            'numerical_mode': 'row-exact', 'diagnostic_only': True, 'throughput_claim': False,
            'oracle': str(Path(args.oracle).resolve()) if args.oracle else None}
    seed = h.load_seed(Path(args.seed), plan)
    plan['reuse'] = seed
    plan['seed_hashes'] = seed['hashes']
    plan['model_content'] = model_content(model)
    plan['helper_hashes'] = {str(Path(module.__file__).resolve()): sha(module.__file__) for module in (h, d, r)}
    plan['helper_hashes'][str(Path(__file__).resolve())] = sha(__file__)
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
    validate_preparation(plan.get('preparation'), plan['action'])
    h = helpers(plan['helpers'])
    d, r = h.d, h.r
    for path, expected in plan['helper_hashes'].items():
        require(sha(path) == expected, 'helper changed after declaration')
    require(sha(plan['binary']) == plan['binary_sha256'], 'binary changed after declaration')
    require(model_content(plan['model']) == plan['model_content'], 'model content changed before launch')
    runtime_before = r.runtime_identity(Path(plan['binary']))
    require(runtime_before == plan['runtime_identity'], 'runtime changed before diagnostic')
    save(out / 'runtime-identity.json', {'before': runtime_before})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('bounded window stopped')))
    args = argparse.Namespace(**plan['server_args'])
    folder = out / 'owned-boot'
    folder.mkdir()
    store, corpus, before = d.prepare_boot_assets(folder, plan, 'persistent')
    serial = plan['action'] == 'serial'
    mode = 'serial' if serial else 'dflash-persistent'
    env = {'MLX_SERVE_ROW_EXACT': '1', 'MLX_SERVE_LOOKUP_STORE_WRITE': '0', 'MLX_SERVE_LOOKUP_RECEIPTS': '1',
           'MLX_SERVE_LOOKUP_COUNTERFACTUAL_PREPARATION': plan['preparation'],
           'MLX_SERVE_LOOKUP_CANDIDATES': '0' if plan['preparation'] == 'production-neural' else '1'}
    if not serial:
        env.update(MLX_SERVE_LOOKUP_COUNTERFACTUAL='1', MLX_SERVE_LOOKUP_COUNTERFACTUAL_ACTION=plan['action'],
                   MLX_SERVE_LOOKUP_SELECTION='fixed:' + str(action_width(plan['action'])))
    else:
        env['MLX_SERVE_LOOKUP_COUNTERFACTUAL'] = '0'
    server = r.Server(args, plan['binary'], plan['model'], mode, folder / 'server.log', corpus=corpus, store=store, experiment=env)
    save(folder / 'launch.json', {'command': server.command, 'environment': r.evidence_environment(server.env)})
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
                    receipt = validate_receipt(found[0], plan['action'], phase, plan['preparation'])
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
            validate_pair(receipts, plan['action'], plan['preparation'])
            reference = oracle_for(plan['oracle'], plan)
            require(all(sample['identity'] == reference for sample in samples), 'exact same-binary serial oracle mismatch')
        require(sha(plan['binary']) == plan['binary_sha256'], 'binary changed during run')
        require(model_content(plan['model']) == plan['model_content'], 'model content changed during run')
        runtime_after = r.runtime_identity(Path(plan['binary']))
        save(out / 'runtime-identity.json', {'before': runtime_before, 'after': runtime_after})
        validate_runtime_evidence(out, plan)
    finally:
        assets = d.finish_boot_assets(folder, plan, store, corpus, before)
        require(assets['unchanged'] and assets['seeds_unchanged'], 'frozen history changed')
    save(out / 'summary.json', {'complete': True, 'action': plan['action'], 'diagnostic_only': True, 'performance_pass': False,
                              'preparation': plan['preparation'],
                              'timing_scope': 'verify+commit including host work and diagnostic readback effects; not pure verifier or full round', 'requests': 2})
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


def validated_windows(paths, oracle, *, unique_actions=True):
    """Shared completed-window, exact-output and same-state evidence gate."""
    records, plans, roots = [], [], []
    reference_plan = None
    actions = set()
    for path in paths:
        root = Path(path).expanduser().resolve()
        require(root not in roots, 'duplicate source window')
        plan = load(root / 'plan.json')
        validate_preparation(plan.get('preparation'), plan['action'])
        require(load(root / 'summary.json').get('complete') is True, 'incomplete diagnostic window')
        validate_runtime_evidence(root, plan)
        require(plan['action'] != 'serial' and (not unique_actions or plan['action'] not in actions),
                'duplicate/nonfeature action')
        actions.add(plan['action'])
        if reference_plan is not None:
            for key in DECLARATION_KEYS:
                require(plan[key] == reference_plan[key], 'window declaration mismatch: ' + key)
        reference_plan = plan
        reference = oracle_for(oracle, plan)
        samples = load(root / 'samples.json')
        require(validate_samples(samples) == reference, 'serial exact-output mismatch')
        row = validate_pair(load(root / 'receipts.json'), plan['action'], plan['preparation'])
        require(row['published'][:row['published_len']] == reference['generated_ids'][:row['published_len']],
                'first offer is not first published output')
        records.append(row)
        plans.append(plan)
        roots.append(root)
    require(bool(records), 'at least one completed explicit window required')
    for a in records:
        for b in records:
            require(a['identity'] == b['identity'], 'actual common state or learner mismatch')
            for key in ('target_kv', 'target_ssm'):
                require(a['prepared'][key] == b['prepared'][key], 'actual target before verify differs')
            same_preparation = a['action'].startswith('assistant') == b['action'].startswith('assistant')
            if same_preparation:
                require(a['offer'] == b['offer'] and a['prepared'] == b['prepared'], 'width contrast candidate/probability/prepared-state mismatch')
    return roots, plans, records


def compare(paths, oracle):
    _, plans, records = validated_windows(paths, oracle)
    require(len(records) >= 2, 'at least two independent explicit windows required')
    actions = {plan['action'] for plan in plans}
    report = {'valid_counterfactual': True, 'diagnostic_only': True, 'throughput_claim': False,
              'first_offer': 0, 'actions': sorted(actions), 'same_preparation_width_contrast': len({a.startswith('assistant') for a in actions}) == 1,
              'preparation': plans[0]['preparation'],
              'timing_scope': 'verify+commit including host work and diagnostic readback effects; not pure verifier',
              'preparation_note': 'Assistant/lookup action contrasts deliberately use different candidate sets; width contrasts require identical offers.',
              'observations': []}
    for row in records:
        count = row['published_len']
        full = row['timing']['full_round_ns']
        report['observations'].append({'action': row['action'], 'actual_rows': row['kernel']['rows'], 'variant': row['kernel']['variant'],
            'committed_tokens': count, 'accepted_drafts': row['accepted_drafts'],
            'prepared': row['prepared'], 'geometry': row['kernel']['geometry'],
            'verify_commit_ms_per_committed_token': row['timing']['verify_commit_ns'] / 1e6 / count,
            'full_round_ms_per_committed_token': None if full is None else full / 1e6 / count,
            'readback_ms_excluded': row['timing']['diagnostic_readback_ns'] / 1e6})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def require_training(plan):
    require(plan.get('case_phase') == 'training', 'price export rejects evaluation cases')
    case = plan['case']
    matches = [item for item in plan['training_cases'] if item.get('id') == case.get('id')]
    require(matches == [case], 'price case is not the exact declared training case')
    snapshot = plan.get('contract_json')
    require(isinstance(snapshot, str) and bool(snapshot),
            'missing frozen contract snapshot; historical windows cannot qualify for price export')
    require(hashlib.sha256(snapshot.encode('utf-8')).hexdigest() == plan['contract_sha256'],
            'frozen contract hash mismatch')
    contract = json.loads(snapshot)
    declared = [item for item in contract['training'] if item.get('id') == case.get('id')]
    require(len(declared) == 1 and
            not any(item.get('id') == case.get('id') for item in contract['evaluation']),
            'price case is not training in the frozen contract')
    require(all(case.get(key) == value for key, value in declared[0].items() if key != 'prompt'),
            'price case differs from frozen contract training declaration')
    require(all(case.get(key) == value for key, value in contract['request_contract'].items() if key != 'endpoint'),
            'price request controls differ from frozen contract')
    identities = plan.get('request_identities')
    require(isinstance(identities, list), 'missing frozen request identities')
    matching = [item for item in identities
                if item.get('case') == case['id'] or
                item.get('rendered_request_sha256') == plan['semantic_request_sha256']]
    require(matching == [{'phase': 'training', 'case': case['id'],
                          'rendered_request_sha256': plan['semantic_request_sha256']}],
            'price case/hash is not a unique frozen training request identity')


def save_new_prices(output, table, provenance):
    """Reserve both names exclusively before writing; never replace any artifact."""
    output = Path(output).expanduser().absolute()
    source = output.with_suffix('.source.json')
    require(output != source, 'price and provenance paths must differ')
    # Encode before reserving either name, including rejecting nonfinite JSON.
    price_bytes = (json.dumps(table, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    provenance = dict(provenance, prices_sha256=hashlib.sha256(price_bytes).hexdigest())
    source_bytes = (json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    with output.open('xb') as price_stream:
        try:
            source_stream = source.open('xb')
        except BaseException:
            output.unlink()
            raise
        with source_stream:
            price_stream.write(price_bytes)
            source_stream.write(source_bytes)
    return output, source


def export_prices(paths, oracle, output):
    """Export only measured first-offer verify/commit costs, with no interpolation."""
    roots, plans, records = validated_windows(paths, oracle, unique_actions=False)
    require_training(load(Path(oracle).expanduser() / 'plan.json'))
    table = {'schema': 3, 'context_bucket': 'floor-power-of-two-2048'}
    sources, occupied = [], {}
    for root, plan, row in zip(roots, plans, records):
        require_training(plan)
        kernel = row['kernel']
        cls, kind, variant, width = (kernel[key] for key in ('class', 'kind', 'variant', 'rows'))
        require(type(cls) is int and (cls == 0 or 2048 <= cls <= 262144 and cls & (cls - 1) == 0),
                'invalid measured context class')
        cell = (str(cls), kind, variant, str(width))
        require(cell not in occupied,
                'conflicting price cell; export source/action costs separately, not pooled: ' + repr(cell))
        occupied[cell] = root
        try:
            milliseconds = row['timing']['verify_commit_ns'] / 1e6
        except OverflowError as error:
            raise ValueError('nonfinite verify/commit price') from error
        require(math.isfinite(milliseconds) and milliseconds > 0, 'invalid verify/commit price')
        table.setdefault(str(cls), {}).setdefault(kind, {}).setdefault(variant, {})[str(width)] = milliseconds
        sources.append({'path': str(root), 'plan_sha256': sha(root / 'plan.json'),
                        'receipts_sha256': sha(root / 'receipts.json'),
                        'action': plan['action'], 'case': plan['case']['id'],
                        'acquisition_timings': [receipt['timing'] for receipt in load(root / 'receipts.json')],
                        'prepared': row['prepared'], 'geometry': kernel['geometry'],
                        'accepted_drafts': row['accepted_drafts'], 'committed_tokens': row['published_len']})
    plan = plans[0]
    oracle_root = Path(oracle).expanduser().resolve()
    provenance = {
        'version': 1, 'purpose': 'staircase_calibration', 'phase': 'training',
        'binary_sha256': plan['binary_sha256'], 'runtime_identity': plan['runtime_identity'],
        'model': plan['model'], 'model_content': plan['model_content'],
        'contract_sha256': plan['contract_sha256'], 'seed_hashes': plan['seed_hashes'],
        'source_windows': sources,
        'preparation': plan['preparation'],
        'timing_scope': 'verify+commit including host work and diagnostic readback effects; not pure verifier or full round',
        'production_equivalence': 'unestablished; requires repeated production-horizon neural training controls and instrumentation-off controls',
        'serial_oracle': {'path': str(oracle_root), 'plan_sha256': sha(oracle_root / 'plan.json'),
                          'samples_sha256': sha(oracle_root / 'samples.json')},
        'scope': 'same-state first-offer verify/commit only; no throughput claim',
        'throughput_claim': False, 'aggregation': 'none; duplicate cells rejected',
        'excluded_timings': ['warmup', 'diagnostic_readback_ns', 'full_round_ns'],
        'plain_evidence': 'none; forced one-row chain/tree is not relabeled as plain',
        'calibration_cost_scope': 'Both warmup and measured acquisition timings retained per source; '
                                  'boot, prefill, serial oracle and remaining request costs are not '
                                  'amortized into these verify/commit priors. Not a full-round price.',
    }
    price_path, source_path = save_new_prices(output, table, provenance)
    print(json.dumps({'prices': str(price_path), 'provenance': str(source_path),
                      'cells': len(occupied), 'throughput_claim': False}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    one = subs.add_parser('run')
    one.add_argument('--action', choices=ACTIONS, required=True)
    one.add_argument('--preparation', choices=PREPARATIONS, required=True,
                     help='Freeze on oracle and actions; production-* preserves the native horizon, with suffix candidates enabled only for production-combined')
    one.add_argument('--binary', type=Path, required=True)
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--oracle', type=Path)
    one.add_argument('--seed', type=Path, default=SEED)
    one.add_argument('--case', default='related-config',
                     help='Predeclared contract case; use train-* only for independent price calibration')
    one.add_argument('--helpers', type=Path, default=HELPERS)
    one.add_argument('--port', type=int, default=11279)
    one.add_argument('--plan-only', action='store_true')
    cpu = subs.add_parser('compare')
    cpu.add_argument('windows', type=Path, nargs='+')
    cpu.add_argument('--oracle', type=Path, required=True)
    export = subs.add_parser('export-prices', help='CPU-only training verify/commit schema3 export; no pooling')
    export.add_argument('windows', type=Path, nargs='+')
    export.add_argument('--oracle', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True,
                        help='New price JSON and adjacent .source.json; neither may already exist')
    hidden = subs.add_parser('_worker', help=argparse.SUPPRESS)
    hidden.add_argument('output', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == 'run':
        return run(args)
    if args.command == 'compare':
        return compare(args.windows, args.oracle)
    if args.command == 'export-prices':
        return export_prices(args.windows, args.oracle, args.output)
    return worker(args.output)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        raise SystemExit(1)
