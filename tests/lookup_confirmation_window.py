#!/usr/bin/env python3
"""Predeclare four full-workload pairs; explicitly launch one <=290s window.

--declare ROOT --seed TRAINING freezes a CPU-only cohort. Independently invoke
--cohort ROOT --pair N for each N in 0..3; no command launches the next window.
--aggregate ROOT --output NEW recomputes all four windows on CPU, without
rewriting their evidence. Row-exact native references are not stock defaults.
Local hashes detect accidental mutation, not a hostile operator forging receipts.
"""
import argparse
import copy
import json
from pathlib import Path
import os
import sys
import tempfile

import lookup_similar_requests as similar

d = similar.d
r = similar.r
TOTAL_PAIRS = 4
SEALED_FILES = ('plan.json', 'summary.json', 'observations.jsonl', 'boots.jsonl',
                'worker-complete.json', 'bounded-run.json', 'errors.jsonl',
                'sample-starts.jsonl', 'deadline.jsonl')


def validate_cohort(config, root):
    """Validate declaration structure/content, without consulting runtime assets."""
    root = Path(root).expanduser().resolve()
    if config.get('hash') != d.content_hash({key: value for key, value in config.items() if key != 'hash'}):
        raise r.BenchError('cohort content hash mismatch')
    if config.get('version') != 1 or config.get('root') != str(root):
        raise r.BenchError('cohort version/root mismatch; cohorts cannot be relocated')
    plan = config['template']
    opposite = ('dflash' if plan['family'] == 'mtp' else 'mtp') + '-legacy'
    arms = ['legacy', opposite, 'persistent']
    orders = [arms[:] if pair % 2 == 0 else list(reversed(arms)) for pair in range(TOTAL_PAIRS)]
    expected = {'phase': 'evaluation', 'scenario': 'transfer', 'gate_schema': 3,
                'pairs': 1, 'arms': arms, 'feature_arms': ['persistent'],
                'native_baseline_arms': arms[:2], 'baseline_arm': 'legacy',
                'both_native_baselines': True, 'numerical_mode': 'row-exact',
                'feature_selection': 'adaptive', 'serial_oracle': False,
                'price_trace': False, 'cross_family_reference': None,
                'orders': orders[:1], 'total_requests': 24,
                'warmup_per_case_per_arm_per_boot': 1,
                'timed_per_case_per_arm_per_boot': 1,
                'deadline_seconds': 290, 'worker_stop_seconds': 285}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise r.BenchError('cohort protocol is not the fixed full-case one-pair protocol')
    if (plan['family'] not in ('mtp', 'dflash') or config.get('orders') != orders
            or plan.get('measurement_isolation') not in ('contended', 'declared')
            or plan['cases'] != plan['evaluation_cases'] or len(plan['cases']) != 4
            or len({case['id'] for case in plan['cases']}) != 4
            or set(similar.RELATED) - {case['id'] for case in plan['cases']}):
        raise r.BenchError('cohort orders/cases/isolation mismatch')
    if not isinstance(config.get('wrapper_sha256'), str) or len(config['wrapper_sha256']) != 64:
        raise r.BenchError('missing wrapper source identity')
    return config


def declare(args):
    """CPU-only declaration; an existing root is never reused."""
    root = Path(args.declare).expanduser().resolve()
    if not args.seed or args.family not in ('mtp', 'dflash'):
        raise r.BenchError('declaration requires --seed and a supported family')
    isolation = getattr(args, 'isolation', 'contended')
    port = getattr(args, 'port', 11279)
    if isolation not in ('contended', 'declared') or not 1 <= port <= 65535:
        raise r.BenchError('invalid isolation or port')
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    options = argparse.Namespace(seed=Path(args.seed), family=args.family, isolation=isolation,
        port=port, contract=similar.CONTRACT, prepare=False, pairs=1,
        arms='legacy,persistent', both_native_baselines=True, cases=None,
        row_exact=True, serial_oracle=False, price_trace=False,
        feature_selection='adaptive', cross_family_reference=None)
    plan = similar.build_plan(options, root)
    (root / 'executed-window.py').write_bytes(Path(__file__).read_bytes())
    config = {'version': 1, 'root': str(root), 'template': plan,
              'orders': [plan['arms'][:] if pair % 2 == 0 else list(reversed(plan['arms']))
                         for pair in range(TOTAL_PAIRS)],
              'wrapper_sha256': r.digest(Path(__file__)),
              'declaration_artifacts': {name: r.digest(root / name) for name in
                  ('plan.json', 'declared-cases.json', 'executed-similar.py',
                   'executed-diagnostic.py', 'executed-window.py')}}
    config['hash'] = d.content_hash(config)
    validate_cohort(config, root)
    d.save(root / 'cohort.json', config)
    for name in ['cohort.json', *config['declaration_artifacts']]:
        (root / name).chmod(0o444)
    return config


def load_cohort(root):
    root = Path(root).expanduser().resolve()
    config = validate_cohort(d.load_json(root / 'cohort.json'), root)
    expected = {'plan.json', 'declared-cases.json', 'executed-similar.py',
                'executed-diagnostic.py', 'executed-window.py'}
    if set(config.get('declaration_artifacts', {})) != expected:
        raise r.BenchError('missing declaration artifact receipts')
    for name, digest in config['declaration_artifacts'].items():
        if r.digest(root / name) != digest:
            raise r.BenchError('declaration artifact changed: ' + name)
    if d.load_json(root / 'plan.json') != config['template']:
        raise r.BenchError('declaration template changed')
    if r.digest(root / 'executed-window.py') != config['wrapper_sha256']:
        raise r.BenchError('declaration wrapper source mismatch')
    return config


def pair_plan(config, pair):
    validate_cohort(config, config['root'])
    if type(pair) is not int or not 0 <= pair < TOTAL_PAIRS:
        raise r.BenchError('pair must be an integer in 0..3')
    plan = copy.deepcopy(config['template'])
    plan['orders'] = [config['orders'][pair][:]]
    plan['cohort'] = {'root': config['root'], 'hash': config['hash'],
                      'slot': pair, 'total': TOTAL_PAIRS}
    plan['wrapper_sha256'] = config['wrapper_sha256']
    return plan


def _current_assets(config):
    plan = config['template']
    sources = {'source_script_sha256': similar.__file__, 'shared_driver_sha256': d.__file__,
               'request_helper_sha256': r.__file__,
               'evidence_helper_sha256': similar.evidence.__file__,
               'seed_helper_sha256': d.write_npy_i32.__code__.co_filename,
               'contract_sha256': similar.CONTRACT, 'manifest_sha256': d.KIT / 'manifest.json',
               'binary_sha256': plan['binary']}
    if r.digest(Path(__file__)) != config['wrapper_sha256']:
        raise r.BenchError('wrapper source changed after declaration')
    for key, path in sources.items():
        if r.digest(Path(path)) != plan.get(key):
            raise r.BenchError('current source/asset identity mismatch: ' + key)
    if r.digest(Path(plan['model']['model_dir']) / 'tokenizer.json') != plan['model']['tokenizer_sha256']:
        raise r.BenchError('current tokenizer identity mismatch')
    if similar.load_seed(Path(plan['seed_root']), plan) != plan['reuse']:
        raise r.BenchError('frozen seed declaration changed')


def validate_schedule(plan, rows, boots):
    """Pure strict sequence validation, including local pair and sample IDs."""
    expected = [(0, arm, case['id'], rep)
                for arm in plan['orders'][0] for case in plan['cases'] for rep in (-1, 0)]
    actual = [(row.get('pair'), row.get('arm'), row.get('case'), row.get('rep')) for row in rows]
    errors = []
    if actual != expected or any(type(row.get('pair')) is not int or type(row.get('rep')) is not int for row in rows):
        errors.append('request_schedule_mismatch')
    if any(row.get('sample') != f'{pair}-{arm}-{case}-{rep}'
           for row, (pair, arm, case, rep) in zip(rows, expected)):
        errors.append('sample_identity_mismatch')
    if ([(boot.get('pair'), boot.get('arm')) for boot in boots]
            != [(0, arm) for arm in plan['orders'][0]]
            or any(type(boot.get('pair')) is not int for boot in boots)):
        errors.append('boot_schedule_mismatch')
    return errors


def _validated_rows(out, plan, rows):
    validated = []
    cases = {case['id']: case for case in plan['cases']}
    for raw in rows:
        row = copy.deepcopy(raw)
        case = cases[row['case']]
        spec = plan['arm_definitions'][row['arm']]
        failures = d.check_sample(row) + similar.quality(case, row) + similar.controlled_receipt_errors(row)
        body = r.request_body(case, spec['mode'], None)
        expected = {'body': body, 'arm_definition': spec, 'family': spec['family'],
                    'mode': spec['mode'], 'phase': 'evaluation', 'numerical_mode': 'row-exact',
                    'warmup': row['rep'] < 0, 'request_sha256': d.content_hash(body),
                    'rendered_request_sha256': similar.semantic_hash(case),
                    'seconds': row['response'].get('total_seconds')}
        expected['request_identity_sha256'] = d.content_hash({
            'run': str(out), 'phase': 'evaluation', 'numerical_mode': 'row-exact',
            'pair': 0, 'arm': row['arm'], 'case': row['case'], 'rep': row['rep'],
            'body_sha256': expected['request_sha256']})
        failures.extend('observation_mismatch:' + key for key, value in expected.items() if row.get(key) != value)
        row['errors'] = sorted(set(failures))
        row['transfer'] = (similar.transfer_evidence(row, plan['reuse']) if not row['errors']
                           else {'evidence_unavailable': 'invalid sample'})
        validated.append(row)
    return validated


def _recompute(out, plan, final, supervisor):
    """Use the existing reducers, with regenerated validation/activation evidence."""
    rows, parse_errors = d.read_rows(out / 'observations.jsonl')
    boots, boot_errors = d.read_rows(out / 'boots.jsonl')
    rows = _validated_rows(out, plan, rows)
    schedule_errors = validate_schedule(plan, rows, boots)
    with tempfile.TemporaryDirectory(prefix='lookup-cohort-summary-') as temporary:
        scratch = Path(temporary)
        for row in rows:
            r.append(scratch / 'observations.jsonl', row)
        for boot in boots:
            r.append(scratch / 'boots.jsonl', boot)
        for name in ('errors.jsonl', 'worker-complete.json'):
            if (out / name).is_file():
                (scratch / name).write_bytes((out / name).read_bytes())
        result = similar.summarize(scratch, plan, final=final, supervisor=supervisor)
    errors = parse_errors + boot_errors + schedule_errors
    if (out / 'worker-complete.json').is_file() and d.load_json(out / 'worker-complete.json') != {'completed': True}:
        errors.append('invalid_worker_completion')
    if final and (not supervisor or supervisor.get('timed_out') is not False
                  or type(supervisor.get('exit_code')) is not int or supervisor['exit_code'] != 0
                  or not d.finite(supervisor.get('elapsed_seconds'))
                  or not 0 < supervisor['elapsed_seconds'] <= 290):
        errors.append('invalid_bounded_run_completion')
    result['errors'].extend(errors)
    result['schedule_errors'] = schedule_errors
    result['complete'] = bool(result['complete'] and not errors)
    result['speed_comparison_valid'] = bool(result['speed_comparison_valid'] and result['complete'])
    result['descriptive_timing_valid'] = bool(result['descriptive_timing_valid'] and result['complete'])
    if not result['speed_comparison_valid']:
        result['arm_family_improvement_pass'] = {arm: False for arm in plan['feature_arms']}
        result['fastest_native_baseline']['arm_threshold_pass'] = {arm: False for arm in plan['feature_arms']}
    result.update(similar.transfer_gate_decisions(
        result['arm_family_improvement_pass'], result['fastest_native_baseline']['arm_threshold_pass'],
        1, True, True, valid=result['speed_comparison_valid']))
    result.update(cohort=plan['cohort'], wrapper_sha256=plan['wrapper_sha256'],
                  canonical_speedup_claim=False, promotion_pass=False,
                  scope='One predeclared full-case pair; never confirms or promotes alone. No automatic next window.')
    return result, rows, boots


def _artifact_hashes(out):
    return {name: r.digest(out / name) if (out / name).is_file() else None for name in SEALED_FILES}


def summarize(out, plan, final=False, supervisor=None):
    out = Path(out).resolve()
    result, _, _ = _recompute(out, plan, final, supervisor)
    d.save(out / 'summary.json', result)
    if final and (out / 'bounded-run.json').is_file():
        if d.load_json(out / 'bounded-run.json') != supervisor:
            raise r.BenchError('final supervisor receipt differs from bounded-run.json')
        d.save(out / 'seal.json', {'version': 1, 'cohort': plan['cohort'],
               'root': str(out), 'artifacts': _artifact_hashes(out),
               'trust': 'Local accidental-mutation receipt, not hostile-operator authentication.'})
    return result


def validate_slot(out, config, pair):
    """Check fixed slot receipts and recompute without modifying historical files."""
    out = Path(out).resolve()
    if out != Path(config['root']) / ('pair-' + str(pair)):
        raise r.BenchError('slot root mismatch')
    expected = pair_plan(config, pair)
    if d.load_json(out / 'plan.json') != expected:
        raise r.BenchError('slot plan/global order mismatch')
    seal = d.load_json(out / 'seal.json')
    if (seal.get('version') != 1 or seal.get('root') != str(out)
            or seal.get('cohort') != expected['cohort'] or seal.get('artifacts') != _artifact_hashes(out)):
        raise r.BenchError('slot seal/artifact mismatch')
    if any(seal['artifacts'].get(name) is None for name in SEALED_FILES[:6]):
        raise r.BenchError('slot is missing final artifacts')
    saved = d.load_json(out / 'summary.json')
    if saved.get('final') is not True or saved.get('cohort') != expected['cohort']:
        raise r.BenchError('slot is not a final declared-window summary')
    result = _recompute(out, expected, True, d.load_json(out / 'bounded-run.json'))
    if d.content_hash(result[0]) != d.content_hash(saved):
        raise r.BenchError('sealed summary differs from recomputed evidence')
    return result


def aggregate(root, output):
    """CPU-only, fixed-slot aggregation; missing or invalid slots fail closed."""
    config = load_cohort(root)
    _current_assets(config)
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    summaries, all_boots, windows, errors, missing, identities = [], [], [], [], [], {}
    for pair in range(TOTAL_PAIRS):
        slot = Path(config['root']) / ('pair-' + str(pair))
        if not slot.is_dir():
            missing.append(pair)
            errors.append('missing_slot:' + str(pair))
            continue
        try:
            summary, rows, boots = validate_slot(slot, config, pair)
            summaries.append(summary)
            all_boots.extend(boots)
            windows.append({'slot': pair, 'root': str(slot), 'seal_sha256': r.digest(slot / 'seal.json'),
                            'complete': summary['complete'], 'summary': summary})
            if not summary['complete'] or summary['invalid_samples']:
                errors.append('incomplete_or_invalid_slot:' + str(pair))
            for row in rows:
                identity = d.identity(row)
                if row['case'] in identities and identities[row['case']] != identity:
                    errors.append('cross_window_exact_identity_mismatch:' + row['case'])
                identities.setdefault(row['case'], identity)
        except (r.BenchError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append('invalid_slot:' + str(pair) + ':' + str(exc))
    plan = config['template']
    qualification = similar.evidence.transfer_measurement_errors(all_boots, plan['measurement_isolation'])
    complete = not errors and len(summaries) == TOTAL_PAIRS
    valid = bool(complete and not qualification and all(item['speed_comparison_valid'] for item in summaries))
    family = {arm: bool(valid and all(item['arm_family_improvement_pass'].get(arm, False) for item in summaries))
              for arm in plan['feature_arms']}
    fastest = {arm: bool(valid and all(item['fastest_native_baseline']['arm_threshold_pass'].get(arm, False)
                                     for item in summaries)) for arm in plan['feature_arms']}
    gates = similar.transfer_gate_decisions(family, fastest, TOTAL_PAIRS, complete, True, valid=valid)
    result = {'version': 1, 'gate_schema': 3, 'cohort_hash': config['hash'], 'cohort_root': config['root'],
              'complete': complete, 'execution_complete': complete, 'missing_slots': missing,
              'errors': sorted(set(errors)), 'windows': windows, 'pairs': TOTAL_PAIRS,
              'measurement_qualified': not qualification, 'measurement_exclusions': qualification,
              'speed_comparison_valid': valid, 'full_workload_covered': complete,
              'covered_cases': sorted(identities), 'numerical_mode': plan['numerical_mode'],
              'native_reference_label': plan['native_defaults'], 'pass_rule': plan['pass_rule'],
              **gates, 'promotion_pass': False, 'canonical_speedup_claim': False,
              'scope': 'Four independently launched predeclared full-case pairs. All raw family and fastest-native gates ANDed; cross-window machine qualification required. No stock-default or canonical claim.',
              'receipt_trust': 'Frozen local hashes detect accidental mutation; no hostile-operator authentication.'}
    d.save(out / 'aggregate-summary.json', result)
    print(json.dumps({'output': str(out), 'complete': complete,
                      'performance_pass': result['performance_pass'], 'missing_slots': missing}), flush=True)
    return 0 if complete else 1


def worker(out):
    out = Path(out).expanduser().resolve()
    plan = d.load_json(out / 'plan.json')
    config = load_cohort(plan['cohort']['root'])
    pair = plan['cohort']['slot']
    if out != Path(config['root']) / ('pair-' + str(pair)) or plan != pair_plan(config, pair):
        raise r.BenchError('worker slot/plan mismatch')
    if (out / 'seal.json').exists() or (out / 'observations.jsonl').exists():
        raise r.BenchError('worker refuses an already-started slot')
    _current_assets(config)
    return similar.worker(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--declare', type=Path)
    mode.add_argument('--cohort', type=Path)
    mode.add_argument('--aggregate', type=Path)
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--family', choices=('mtp', 'dflash'), default='mtp')
    parser.add_argument('--isolation', choices=('contended', 'declared'), default='contended')
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--pair', type=int, choices=range(TOTAL_PAIRS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.declare and not args.seed:
        parser.error('--declare requires --seed')
    if args.cohort and args.pair is None:
        parser.error('--cohort requires exactly one --pair')
    if args.aggregate and args.output is None:
        parser.error('--aggregate requires a NEW --output directory')
    if args.pair is not None and not args.cohort:
        parser.error('--pair is only valid with --cohort')
    if args.output is not None and not args.aggregate:
        parser.error('--output is only valid with --aggregate; pair output slots are fixed')
    if args.seed is not None and not args.declare:
        parser.error('--seed is only valid with --declare')
    try:
        if args.worker:
            return worker(args.worker)
        if args.declare:
            config = declare(args)
            print(json.dumps({'cohort': config['root'], 'hash': config['hash'], 'slots': TOTAL_PAIRS}), flush=True)
            return 0
        if args.aggregate:
            return aggregate(args.aggregate, args.output)
        config = load_cohort(args.cohort)
        args.output = Path(config['root']) / ('pair-' + str(args.pair))
        args.plan_only = False
        def prepare(out):
            _current_assets(config)
            plan = pair_plan(config, args.pair)
            d.save(out / 'plan.json', plan)
            return plan
        return d.supervise(args, prepare, __file__, summarize)
    except (r.BenchError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print('confirmation window: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
