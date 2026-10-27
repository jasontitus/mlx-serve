#!/usr/bin/env python3
"""Training-only production MTP chain prices, not a same-state staircase.

Declare (CPU only), acquire once without a workload deadline, then export:
  python3 tests/lookup_chain_profile.py declare --binary PATH --contract PATH --seed DIR --output RUN
  python3 tests/lookup_chain_profile.py acquire --run RUN
  python3 tests/lookup_chain_profile.py export --run RUN --output PRICES.json

Every width 2..15 gets two opposite-order fresh-process sweeps, all four frozen
training prompts twice per boot, and a same-runtime serial oracle. Width 16 is
unsupported, never clamped or synthesized. Failed cohorts cannot resume/export.
The median is over actual eligible training executions, not matched model states.
Private raw receipts/IDs remain in RUN; the .source.json sidecar is private too.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

import lookup_similar_requests as s
import lookup_gpu_telemetry as telemetry

d, r = s.d, s.r
CONTRACT = 'production-mtp-chain-training-population-v1'
WIDTHS = tuple(range(2, 16))
AGGREGATION = 'median of all eligible training executions; exclude first actual valid execution per process/context/kind/variant/width; require >=2 eligible observations and both sweeps; not same-state staircase'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def load(path):
    return decode(Path(path).read_bytes())


def exclusive(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def sources():
    return [Path(__file__).resolve(), Path(s.__file__), Path(d.__file__), Path(r.__file__),
            Path(telemetry.__file__), Path(s.evidence.__file__),
            Path(d.write_npy_i32.__code__.co_filename)]


def schedule():
    return [{'sweep': None, 'width': None, 'arm': 'serial'}] + [
        {'sweep': sweep, 'width': width, 'arm': 'width' + str(width)}
        for sweep, widths in enumerate((WIDTHS, tuple(reversed(WIDTHS)))) for width in widths]


def definitions():
    base = d.arm_definitions('reuse', 'mtp', ['persistent'])['persistent']
    arms = {'serial': {'mode': 'serial', 'family': 'serial', 'selection': 'adaptive',
                      'actual': 'legacy', 'meaning': 'same-runtime complete training oracle'}}
    for width in WIDTHS:
        arms['width' + str(width)] = dict(copy.deepcopy(base), tier='persistent',
            selection='fixed:' + str(width), actual='fixed',
            meaning='production fixed lookup chain, native fallback on miss; exact widths only')
    for spec in arms.values():
        spec['experiment'] = {'MLX_SERVE_LOOKUP_RECEIPTS': '1'}
    d.declare_numerical_mode(arms, True)
    return arms


def binding(plan):
    return {key: plan[key] for key in ('binary_sha256', 'runtime_identity', 'model',
        'model_inventory', 'numerical_mode', 'contract_sha256', 'request_identities', 'training_cases')} | {
        'seed_hashes': plan['reuse']['hashes'],
        'seed_declaration_sha256': d.content_hash(plan['reuse']),
        'request_controls': plan['contract']['request_contract'], 'ctx_size': plan['server_args']['ctx_size']}


def validate_training(plan):
    require(plan.get('profile_contract') == CONTRACT and plan.get('phase') == 'training',
            'not a production training profile declaration')
    require(sha_bytes(plan['contract_bytes'].encode()) == plan['contract_sha256'], 'contract bytes changed')
    contract = json.loads(plan['contract_bytes'])
    require(contract == plan['contract'], 'contract snapshot changed')
    training = [s.render(contract, item) for item in contract['training']]
    evaluation = [s.render(contract, item) for item in contract['evaluation']]
    require(plan['training_cases'] == training and plan['cases'] == training,
            'evaluation data or changed training prompts')
    require(plan['request_identities'] == s.unique_requests(training, evaluation), 'training identity mismatch')
    require(plan['repetitions'] == [0, 1] and plan['server_args']['ctx_size'] == 4096 and
            plan['deadline_seconds'] is None and plan['worker_stop_seconds'] is None,
            'changed request coverage or workload deadline')
    require(plan['schedule'] == schedule() and plan['arm_definitions'] == definitions(), 'changed acquisition controls/schedule')
    require(plan['numerical_mode'] == 'row-exact' and plan['aggregation'] == AGGREGATION,
            'changed numerical/aggregation contract')
    require(plan['unsupported_widths'] == {'16': 'unsupported/unmeasured: production maximum 14 drafts = 15 rows'},
            'unsupported width declaration missing')


def declare(args):
    out = args.output.expanduser().resolve()
    out.mkdir(mode=0o700)
    contract_bytes = args.contract.expanduser().read_bytes()
    contract = json.loads(contract_bytes)
    seed_root = args.seed.expanduser().resolve()
    source = load(seed_root / 'plan.json')
    binary = str(args.binary.expanduser().resolve())
    training = [s.render(contract, item) for item in contract['training']]
    evaluation = [s.render(contract, item) for item in contract['evaluation']]
    plan = {'profile_contract': CONTRACT, 'phase': 'training', 'family': 'mtp',
        'binary': binary, 'binary_sha256': r.digest(binary), 'runtime_identity': r.runtime_identity(binary),
        'model': source['model'], 'numerical_mode': 'row-exact', 'contract': contract,
        'contract_bytes': contract_bytes.decode(), 'contract_sha256': sha_bytes(contract_bytes),
        'training_cases': training, 'cases': training,
        'request_identities': s.unique_requests(training, evaluation), 'seed_root': str(seed_root),
        'schedule': schedule(), 'arm_definitions': definitions(), 'repetitions': [0, 1],
        'server_args': {'ctx_size': 4096, 'ready_timeout': 120, 'request_timeout': 300, 'port': args.port},
        'aggregation': AGGREGATION, 'measurement_isolation': 'declared',
        'unsupported_widths': {'16': 'unsupported/unmeasured: production maximum 14 drafts = 15 rows'},
        'deadline_seconds': None, 'worker_stop_seconds': None,
        'scope': 'production training population; full diagnostic receipts; no throughput claim or same-state attribution',
        'telemetry': telemetry.declaration('/opt/homebrew/bin/macmon', 1000, 3000),
        'created_unix': time.time()}
    require(r.digest(Path(plan['model']['model_dir']) / 'tokenizer.json') == plan['model']['tokenizer_sha256'],
            'current tokenizer mismatch')
    plan['model_inventory'] = r.inventory(Path(plan['model']['model_dir']))
    plan['reuse'] = s.load_seed(seed_root, plan)
    plan['source_hashes'] = {str(path): r.digest(path) for path in sources()}
    snapshots = out / 'sources'
    snapshots.mkdir(mode=0o700)
    for index, path in enumerate(sources()):
        exclusive(snapshots / (str(index) + '-' + path.name), path.read_bytes())
    validate_training(plan)
    exclusive(out / 'plan.json', encode(plan))
    exclusive(out / 'declaration.sha256', (r.digest(out / 'plan.json') + '\n').encode())
    (out / 'plan.json').chmod(0o400)
    (out / 'declaration.sha256').chmod(0o400)
    print(json.dumps({'run': str(out), 'requests': len(schedule()) * len(training) * 2,
                      'unsupported': plan['unsupported_widths'], 'acquired': False}))


def check_frozen(plan):
    require(all(r.digest(path) == digest for path, digest in plan['source_hashes'].items()), 'frozen helper source changed')
    require(r.digest(plan['binary']) == plan['binary_sha256'] and
            r.runtime_identity(plan['binary']) == plan['runtime_identity'], 'frozen runtime changed')
    require(r.inventory(Path(plan['model']['model_dir'])) == plan['model_inventory'], 'frozen target assets changed')
    require(s.load_seed(Path(plan['seed_root']), plan) == plan['reuse'], 'frozen seed changed')


def sample_errors(row, case):
    errors = d.check_sample(row)
    if row['response'].get('finish_reason') != 'stop':
        errors.append('training_request_incomplete')
    if row['body'] != r.request_body(case, row['arm_definition']['mode'], None):
        errors.append('request_not_declared_training')
    return errors


def acquire(root):
    root = root.expanduser().resolve()
    plan = load(root / 'plan.json')
    require(r.digest(root / 'plan.json') == (root / 'declaration.sha256').read_text().strip(), 'declaration changed')
    validate_training(plan)
    check_frozen(plan)
    start = time.time()
    exclusive(root / 'acquisition-started.json', encode({'time': start, 'plan_sha256': r.digest(root / 'plan.json')}))
    rows, oracle = [], {}
    options = argparse.Namespace(**plan['server_args'])
    try:
        with telemetry.Collector(root, plan['telemetry'], r.machine_state) as collector:
            for slot, item in enumerate(plan['schedule']):
                check_frozen(plan)
                arm = item['arm']
                folder = root / f'{slot}-{arm}'
                folder.mkdir(mode=0o700)
                spec = plan['arm_definitions'][arm]
                with d.owned_boot(root, folder, plan, options, slot, arm, telemetry=collector) as server:
                    for case in plan['training_cases']:
                        for rep in plan['repetitions']:
                            row = d.record_sample(root, folder, server, options, spec, slot, arm, case, rep, telemetry=collector)
                            row.update(sweep=item['sweep'], requested_width=item['width'], phase='training',
                                       process_instance=slot, process_pid=server.proc.pid)
                            row['errors'] = sorted(set(row['errors'] + sample_errors(row, case)))
                            identity = d.identity(row)
                            if arm == 'serial' and case['id'] not in oracle:
                                oracle[case['id']] = identity
                            if identity != oracle.get(case['id']):
                                row['errors'].append('serial_training_oracle_mismatch')
                            rows.append(row)
                            r.append(root / 'observations.jsonl', row)
                            require(not row['errors'], 'invalid request; cohort stopped without retry: ' + row['sample'])
                check_frozen(plan)
        boots, errors = d.read_rows(root / 'boots.jsonl')
        hardware = telemetry.qualify(root, [row['sample'] for row in rows],
            [(slot, item['arm']) for slot, item in enumerate(plan['schedule'])], plan['telemetry'],
            boot_states=[boot.get(key, {}) for boot in boots for key in ('machine_before', 'machine_after')])
        errors += s.evidence.transfer_measurement_errors(boots, plan['measurement_isolation'])
        errors += ['telemetry:' + value for value in hardware['errors']]
        result = {'complete': not errors and hardware['qualified'], 'errors': errors, 'hardware': hardware,
                  'started_unix': start, 'finished_unix': time.time(),
                  'request_seconds': sum(row['seconds'] for row in rows), 'requests': len(rows)}
        exclusive(root / 'acquisition.json', encode(result))
        require(result['complete'], 'acquisition hardware qualification failed; no export/retry')
        # Include all raw HTTP, receipt, ownership, launch and source bytes, not just selected prices.
        artifacts = d.tree_hashes(root)
        exclusive(root / 'completed.json', encode({'artifacts': artifacts, 'complete': True}))
        d.freeze(root)
    except BaseException as exc:
        exclusive(root / 'failed.json', encode({'error': type(exc).__name__ + ': ' + str(exc), 'time': time.time()}))
        raise


def valid_class(value):
    return type(value) is int and (value == 0 or 2048 <= value <= 262144 and value & (value - 1) == 0)


def eligible_receipt(receipt, width):
    if width not in WIDTHS:
        return 'unsupported_width'
    for key, value in [('family', 'mtp'), ('kind', 'chain'), ('variant', 'generic'),
                       ('assistant', False), ('selection', 'fixed'), ('admissible', True),
                       ('timing_scope', 'settled_round')]:
        if receipt.get(key) != value or isinstance(value, bool) and receipt.get(key) is not value:
            return 'ineligible_' + key
    if (type(receipt.get('requested_rows')) is not int or type(receipt.get('rows')) is not int or
            receipt['requested_rows'] != width or receipt['rows'] != width):
        return 'short_offer_or_clipped_width'
    if not valid_class(receipt.get('class')):
        return 'invalid_context_class'
    value = receipt.get('verify_commit_ms')
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        return 'invalid_settled_verify_commit_ms'
    return None


def population(rows):
    """Keep every receipt's membership, including cold/short/native/absent offers."""
    seen, cells, membership = set(), {}, []
    for row in rows:
        if row['arm'] == 'serial':
            continue
        receipts = row.get('rounds', [])
        if not receipts:
            membership.append({'sample': row['sample'], 'receipt': None, 'excluded': 'absent_proposal'})
        for index, receipt in enumerate(receipts):
            reason = 'failed_request' if row.get('errors') else eligible_receipt(receipt, row['requested_width'])
            member = {'sample': row['sample'], 'receipt': index, 'receipt_sha256': d.content_hash(receipt),
                      'process_instance': row['process_instance'], 'sweep': row['sweep'],
                      'requested_width': row['requested_width'], 'excluded': reason}
            if reason is None:
                key = (receipt['class'], 'chain', 'generic', receipt['rows'])
                cold = (row['process_instance'],) + key
                if cold not in seen:
                    seen.add(cold)
                    member['excluded'] = 'first_actual_valid_execution'
                elif receipt.get('priced') is not True:
                    member['excluded'] = 'unpriced_execution'
                else:
                    member.update(cell=list(key), milliseconds=receipt['verify_commit_ms'])
                    cells.setdefault(key, []).append(member)
            membership.append(member)
    table = {'schema': 3, 'context_bucket': 'floor-power-of-two-2048'}
    summaries = []
    for key, members in sorted(cells.items()):
        values = [member['milliseconds'] for member in members]
        sufficient = len(values) >= 2 and {member['sweep'] for member in members} == {0, 1}
        median = statistics.median(values)
        summaries.append({'cell': list(key), 'count': len(values), 'exported': sufficient,
            'sweeps': sorted({member['sweep'] for member in members}),
            'requests': len({member['sample'] for member in members}),
            'processes': len({member['process_instance'] for member in members}),
            'median_ms': median, 'min_ms': min(values), 'max_ms': max(values),
            'median_absolute_deviation_ms': statistics.median(abs(value - median) for value in values),
            'population_stdev_ms': statistics.pstdev(values)})
        if sufficient:
            table.setdefault(str(key[0]), {}).setdefault('chain', {}).setdefault('generic', {})[str(key[3])] = median
    return table, membership, summaries


def validate_table(table):
    require(type(table) is dict and table.get('schema') == 3 and
            table.get('context_bucket') == 'floor-power-of-two-2048', 'invalid schema3 price profile')
    count = 0
    for cls, kinds in table.items():
        if cls in ('schema', 'context_bucket'):
            continue
        require(cls.isdigit() and str(int(cls)) == cls and valid_class(int(cls)), 'invalid context bucket')
        require(type(kinds) is dict and set(kinds) == {'chain'} and
                type(kinds['chain']) is dict and set(kinds['chain']) == {'generic'}, 'exact chain/generic namespace required')
        widths = kinds['chain']['generic']
        require(type(widths) is dict and bool(widths), 'empty width population')
        for width, price in widths.items():
            require(width in {str(w) for w in WIDTHS}, 'unsupported/unmeasured width')
            require(type(price) in (int, float) and math.isfinite(price) and price > 0, 'invalid price')
            count += 1
    require(count > 0, 'no eligible measured prices')


def export_data(root):
    root = Path(root).expanduser().resolve()
    require(not (root / 'failed.json').exists(), 'failed cohort cannot export')
    completed = load(root / 'completed.json')
    require(completed.get('complete') is True, 'incomplete acquisition')
    artifacts = completed['artifacts']
    actual_artifacts = d.tree_hashes(root)
    actual_artifacts.pop('completed.json', None)
    require(actual_artifacts == artifacts, 'raw acquisition artifact changed or omitted')
    plan = load(root / 'plan.json')
    validate_training(plan)
    require(r.digest(root / 'plan.json') == (root / 'declaration.sha256').read_text().strip(), 'declaration hash mismatch')
    rows, errors = d.read_rows(root / 'observations.jsonl')
    expected = [(slot, item, case, rep) for slot, item in enumerate(plan['schedule'])
                for case in plan['training_cases'] for rep in (0, 1)]
    require(not errors and len(rows) == len(expected), 'incomplete request coverage')
    oracle = {}
    for row, (slot, item, case, rep) in zip(rows, expected):
        arm = item['arm']
        require((row['pair'], row['arm'], row['case'], row['rep'], row['phase'], row['process_instance'], row['sweep'], row['requested_width']) ==
                (slot, arm, case['id'], rep, 'training', slot, item['sweep'], item['width']), 'request membership differs from training schedule')
        require(row['arm_definition'] == plan['arm_definitions'][arm] and not sample_errors(row, case), 'invalid training response')
        sample = root / f'{slot}-{arm}' / (case['id'] + '-' + str(rep))
        require(load(sample / 'request.json') == row['body'] and load(sample / 'response.json') == row['response'], 'raw request/response mismatch')
        response = r.response_receipt((sample / 'response.sse').read_text(), True)
        require(response['complete'] and all(response.get(key) == row['response'].get(key)
                for key in ('text', 'usage', 'finish_reason', 'events')), 'incomplete or changed raw SSE publication')
        stats = r.parse_stats((sample / 'receipts.log').read_text(), row['family'])
        require(stats == row['stats'] and stats['commits'] == row['commits'] and stats['rounds'] == row['rounds'], 'raw receipt membership mismatch')
        identity = d.identity(row)
        if arm == 'serial' and case['id'] not in oracle:
            oracle[case['id']] = identity
        require(identity == oracle.get(case['id']), 'not exact versus same-runtime serial training oracle')
    boots, errors = d.read_rows(root / 'boots.jsonl')
    require(not errors and [(boot['pair'], boot['arm']) for boot in boots] ==
            [(slot, item['arm']) for slot, item in enumerate(plan['schedule'])], 'incomplete boot coverage')
    require(all(boot['completed'] and boot['assets_preserved'] and boot['numerical_mode'] == 'row-exact' for boot in boots), 'invalid boot')
    acquisition = load(root / 'acquisition.json')
    require(acquisition['complete'] and not acquisition['errors'] and acquisition['hardware']['qualified'], 'unqualified acquisition')
    hardware = telemetry.qualify(root, [row['sample'] for row in rows],
        [(slot, item['arm']) for slot, item in enumerate(plan['schedule'])], plan['telemetry'],
        boot_states=[boot.get(key, {}) for boot in boots for key in ('machine_before', 'machine_after')])
    require(hardware['qualified'] and not hardware['errors'] and
            not s.evidence.transfer_measurement_errors(boots, plan['measurement_isolation']),
            'raw hardware evidence does not qualify')
    table, membership, cells = population(rows)
    validate_table(table)
    provenance = {'profile_contract': CONTRACT, 'phase': 'training', 'bindings': binding(plan),
        'source_hashes': plan['source_hashes'], 'source_run': str(root),
        'completed_sha256': r.digest(root / 'completed.json'), 'artifacts': artifacts,
        'aggregation': AGGREGATION, 'membership': membership, 'cells': cells,
        'membership_sha256': d.content_hash(membership), 'declared_coverage': plan['schedule'],
        'unsupported_widths': plan['unsupported_widths'], 'acquisition_cost': acquisition,
        'scope': plan['scope'], 'throughput_claim': False}
    return table, provenance


def export(root, output):
    table, source = export_data(root)
    data = encode(table)
    source['prices_sha256'] = sha_bytes(data)
    output = output.expanduser().resolve()
    sidecar = output.with_suffix('.source.json')
    require(output != sidecar and not output.exists() and not sidecar.exists(), 'export paths must be new and distinct')
    exclusive(sidecar, encode(source))
    exclusive(output, data)
    sidecar.chmod(0o400)
    output.chmod(0o400)


def freeze_profile(path, plan, out):
    """Validate provenance and freeze the exact bytes used by selection, before evaluation."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    table = decode(raw)
    source_path = path.with_suffix('.source.json')
    source_raw = source_path.read_bytes()
    source = decode(source_raw)
    validate_table(table)
    require(source.get('profile_contract') == CONTRACT and source.get('phase') == 'training', 'evaluation/non-production profile rejected')
    require(source.get('prices_sha256') == sha_bytes(raw), 'tampered price bytes')
    require(source.get('bindings') == binding(plan), 'profile target/runtime/numerical/seed/training contract mismatch')
    require(source.get('source_hashes') == {str(p): r.digest(p) for p in sources()}, 'profile helper source changed')
    rebuilt, provenance = export_data(source['source_run'])
    require(rebuilt == table and dict(provenance, prices_sha256=sha_bytes(raw)) == source, 'profile source provenance mismatch')
    frozen = Path(out).expanduser().resolve() / 'training-chain-prices.json'
    frozen_source = frozen.with_suffix('.source.json')
    exclusive(frozen, raw)
    exclusive(frozen_source, source_raw)
    frozen.chmod(0o400)
    frozen_source.chmod(0o400)
    return {'path': str(frozen), 'sha256': sha_bytes(raw), 'source_path': str(frozen_source),
            'source_sha256': sha_bytes(source_raw), 'profile_contract': CONTRACT,
            'original_path': str(path), 'bindings': source['bindings']}


def assert_profile_frozen(identity):
    require(r.digest(identity['path']) == identity['sha256'] and
            r.digest(identity['source_path']) == identity['source_sha256'], 'frozen profile bytes changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    declaration = commands.add_parser('declare')
    for flag in ('binary', 'contract', 'seed', 'output'):
        declaration.add_argument('--' + flag, type=Path, required=True)
    declaration.add_argument('--port', type=int, default=11279)
    acquisition = commands.add_parser('acquire')
    acquisition.add_argument('--run', type=Path, required=True)
    exporter = commands.add_parser('export')
    exporter.add_argument('--run', type=Path, required=True)
    exporter.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    if args.command == 'declare':
        declare(args)
    elif args.command == 'acquire':
        acquire(args.run)
    else:
        export(args.run, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
