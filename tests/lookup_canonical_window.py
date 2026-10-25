#!/usr/bin/env python3
"""One complete pinned canonical arm OR its exact replay, never an automatic sweep.

Reuse the host canonical proxy/report/replay and transfer frozen-store/supervisor
contracts. Each invocation owns a 285s worker / 290s absolute process-group bound.
Only CPU aggregation joins explicitly completed windows; no truncated protocol is
accepted. Rebuild the independently implemented host before any GPU invocation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import lookup_similar_requests as s
d, r = s.d, s.r
from lookup_bench_evidence import stable_states, response_equal

ARMS = ('mtp-legacy', 'dflash-legacy', 'mtp-persistent', 'dflash-persistent')
TARGET = str(Path.home() / '.mlx-serve/models/ddalcu/Qwen3.8-27B-MLX-Serve-4bit')
DRAFTER = str(Path.home() / '.mlx-serve/drafters/incoai/Qwen3.8-27B-DFlash2')
PROTOCOL = {
    'client': 'llmprobe 0.6.17 full bench-only; all warmups, prefix, batch and drift retained',
    'numerical_mode': 'row-exact',
    'server_args': {'ctx_size': 16384, 'ready_timeout': 35, 'request_timeout': 60,
                    'probe_timeout': 240, 'probe_rungs': '512', 'repetitions': 3,
                    'max_tokens': None, 'row_exact_tree': True, 'row_exact': True},
    'deadline_seconds': 290, 'worker_stop_seconds': 285,
}


def source_hashes():
    paths = [Path(__file__), Path(d.__file__), Path(s.__file__), Path(r.__file__),
             Path(s.evidence.__file__), Path(d.write_npy_i32.__code__.co_filename), s.CONTRACT]
    return {path.name: r.digest(path) for path in paths}


def validate_cohort(config):
    if not isinstance(config, dict) or config.get('source_hashes') != source_hashes():
        raise r.BenchError('canonical cohort helper/contract identity missing or changed; redeclare, never upgrade history')
    if (d.content_hash(config.get('protocol')) != d.content_hash(PROTOCOL) or type(config.get('runs')) is not int
            or config['runs'] != 3 or config.get('rungs') != '512'
            or type(config.get('pairs')) is not int or config['pairs'] not in (2, 4)):
        raise r.BenchError('canonical cohort protocol changed or malformed')
    if config.get('arms') != list(ARMS) or not isinstance(config.get('cache_tag'), str) or not config['cache_tag'].isalnum():
        raise r.BenchError('declare both native and both persistent arms and an alphanumeric cache tag')
    orders = [list(ARMS) if pair % 2 == 0 else list(reversed(ARMS)) for pair in range(config['pairs'])]
    if config.get('orders') != orders:
        raise r.BenchError('canonical cohort boot orders changed or malformed')


def validate_plan(plan):
    config = plan.get('config')
    validate_cohort(config)
    if (plan.get('source_hashes') != config['source_hashes'] or d.content_hash(plan.get('protocol')) != d.content_hash(config['protocol'])
            or any(d.content_hash(plan.get(key)) != d.content_hash(PROTOCOL[key]) for key in ('numerical_mode', 'deadline_seconds', 'worker_stop_seconds'))):
        raise r.BenchError('canonical plan helper/contract/protocol identity mismatch')
    server_args = plan.get('server_args')
    if (not isinstance(server_args, dict) or type(server_args.get('port')) is not int
            or not 1 <= server_args['port'] <= 65535
            or d.content_hash(server_args) != d.content_hash(dict(PROTOCOL['server_args'], port=server_args['port']))):
        raise r.BenchError('canonical plan server protocol changed or malformed')
    if (plan.get('phase') not in ('measure', 'replay') or plan.get('mode') not in ARMS
            or type(plan.get('pair')) is not int or plan['pair'] not in range(config['pairs'])):
        raise r.BenchError('canonical plan phase/arm/pair changed or malformed')
    family, tier = plan['mode'].split('-')
    definitions = d.arm_definitions('reuse', family, [tier])
    d.declare_numerical_mode(definitions, True)
    if plan.get('tier') != tier or plan.get('arm_definitions') != definitions:
        raise r.BenchError('canonical plan arm controls changed or malformed')


def prepare(args, out):
    config = d.load_json(args.config)
    validate_cohort(config)
    manifest = d.load_json(d.KIT / 'manifest.json')
    model = next(m for m in manifest['models'] if m['id'] == 'qwen27b')
    if model['model_dir'] != TARGET or model['drafter_dir'] != DRAFTER:
        raise r.BenchError('target must be the declared Qwen27B with explicit incoai DFlash2')
    binary = r.absolute_asset(manifest['binary'])
    if r.digest(binary) != config['binary_sha256']:
        raise r.BenchError('cohort binary hash changed; redeclare after rebuilding, never mix binaries')
    if r.digest(Path(model['model_dir']) / 'tokenizer.json') != model['tokenizer_sha256']:
        raise r.BenchError('tokenizer changed')
    contract = d.load_json(s.CONTRACT)
    training = [s.render(contract, item) for item in contract['training']]
    evaluation = [s.render(contract, item) for item in contract['evaluation']]
    plan = {'model': model, 'training_cases': training,
            'request_identities': s.unique_requests(training, evaluation)}
    seed = s.load_seed(Path(config['seed']), plan)
    source = None
    if args.source:
        source = args.source.expanduser().resolve()
        previous = d.load_json(source / 'plan.json')
        validate_plan(previous)
        summary = d.load_json(source / 'summary.json')
        if not summary.get('complete') or previous.get('phase') != 'measure':
            raise r.BenchError('replay needs a completed full measurement, not a partial or replay run')
        if previous['config_sha256'] != r.digest(args.config) or previous['reuse']['hashes'] != seed['hashes']:
            raise r.BenchError('replay cohort or frozen history mismatch')
        mode, pair = previous['mode'], previous['pair']
    else:
        mode, pair = args.mode, args.pair
    if mode not in ARMS or pair not in range(config['pairs']):
        raise r.BenchError('explicit --mode and --pair must belong to declared cohort')
    family, tier = mode.split('-')
    definitions = d.arm_definitions('reuse', family, [tier])
    d.declare_numerical_mode(definitions, True)
    plan.update(version=1, phase='replay' if source else 'measure', binary=binary,
        binary_sha256=config['binary_sha256'], config=config, config_sha256=r.digest(args.config),
        source_hashes=config['source_hashes'],
        source=str(source) if source else None, mode=mode, pair=pair, tier=tier,
        numerical_mode='row-exact', arm_definitions=definitions, reuse=seed,
        probe_cmd=manifest['probe_cmd'], probe_assets={p: r.digest(p) for p in manifest['probe_cmd']},
        server_args=dict(PROTOCOL['server_args'], port=args.port),
        deadline_seconds=PROTOCOL['deadline_seconds'], worker_stop_seconds=PROTOCOL['worker_stop_seconds'],
        protocol=config['protocol'],
        training_policy='Only four predeclared held-out-transfer training requests; frozen clones, writes off; canonical answers never seed history',
        promotion_pass=False)
    validate_plan(plan)
    d.save(out / 'plan.json', plan)
    return plan


def summarize(out, plan, final=False, supervisor=None):
    errors, parse_errors = d.read_rows(out / 'errors.jsonl')
    boots, boot_errors = d.read_rows(out / 'boots.jsonl')
    complete = (out / 'worker-complete.json').is_file() and not errors and not parse_errors and not boot_errors
    if final:
        complete = complete and supervisor is not None and supervisor.get('exit_code') == 0 and not supervisor.get('timed_out')
    if plan['phase'] == 'measure':
        complete = complete and len(boots) == 1 and all(b.get('completed') and b.get('assets_preserved')
                                                        and b.get('numerical_mode') == 'row-exact' for b in boots)
    result = {'complete': bool(complete), 'final': final, 'invalid_samples': len(errors) + len(parse_errors) + len(boot_errors),
              'samples': 1 if complete else 0, 'performance_pass': False, 'promotion_pass': False,
              'phase': plan['phase'], 'errors': errors + parse_errors + boot_errors,
              'supervisor': supervisor, 'scope': 'One complete canonical arm or separate exact replay; no standalone speed claim'}
    d.save(out / 'summary.json', result)
    return result


def worker(out):
    plan = d.load_json(out / 'plan.json')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        validate_plan(plan)
        args = argparse.Namespace(**plan['server_args'])
        if r.digest(plan['binary']) != plan['binary_sha256'] or any(r.digest(p) != h for p, h in plan['probe_assets'].items()):
            raise r.BenchError('binary or pinned client changed after preparation')
        if plan['phase'] == 'measure':
            folder = out / 'arm'
            folder.mkdir()
            command = r.controlled_probe_command(plan['probe_cmd'], folder, plan['config']['cache_tag'] + str(plan['pair']))
            with d.owned_boot(out, folder, plan, args, plan['pair'], plan['tier']) as server:
                preflight = r.preflight_context(server, args, [], command)
                d.save(folder / 'context-preflight.json', preflight)
                probe = r.canonical_probe(server, args, plan['mode'], command, folder)
            d.save(out / 'probe.json', probe)
        else:
            previous = d.load_json(Path(plan['source']) / 'plan.json')
            validate_plan(previous)
            if (previous['phase'] != 'measure' or previous['config_sha256'] != plan['config_sha256']
                    or previous['mode'] != plan['mode'] or previous['pair'] != plan['pair']
                    or previous['reuse']['hashes'] != plan['reuse']['hashes']):
                raise r.BenchError('replay source cohort/protocol/slot/seed changed after preparation')
            probe = d.load_json(Path(plan['source']) / 'probe.json')
            evidence = r.replay_canonical(args, plan['binary'], plan['model'], plan['mode'], probe, out)
            d.save(out / 'replay.json', evidence)
            if not evidence.get('successful') or not evidence.get('supported'):
                raise r.BenchError('full canonical exact replay failed or unsupported')
        d.save(out / 'worker-complete.json', {'completed': True})
        return 0
    except BaseException as exc:
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        return 1


def aggregate(paths, output):
    roots = [p.expanduser().resolve() for p in paths]
    if len(set(roots)) != len(roots):
        raise r.BenchError('duplicate input window')
    measures, replays, plans, errors = {}, {}, [], []
    for root in roots:
        plan, summary = d.load_json(root / 'plan.json'), d.load_json(root / 'summary.json')
        validate_plan(plan)
        plans.append(plan)
        if not summary.get('complete') or not summary.get('final'):
            errors.append('incomplete:' + str(root))
        if plan['phase'] == 'measure':
            key = (plan['pair'], plan['mode'])
            if key in measures:
                raise r.BenchError('duplicate measurement slot, no retry-until-win')
            measures[key] = (root, d.load_json(root / 'probe.json'))
        else:
            if plan['source'] in replays:
                raise r.BenchError('duplicate replay source')
            replays[plan['source']] = d.load_json(root / 'replay.json')
    first = plans[0]
    if any(p['config_sha256'] != first['config_sha256'] or p['probe_assets'] != first['probe_assets']
           or p['source_hashes'] != first['source_hashes'] or p['protocol'] != first['protocol']
           or p['reuse']['hashes'] != first['reuse']['hashes'] or p['model'] != first['model'] for p in plans):
        raise r.BenchError('cohort/client/helper/contract/protocol/seed/target mismatch')
    expected = {(pair, mode) for pair in range(first['config']['pairs']) for mode in ARMS}
    if set(measures) != expected:
        errors.append('missing_declared_measurement_slots')
    if set(replays) != {str(root) for root, _ in measures.values()}:
        errors.append('missing_or_extra_replay_sources')
    states = []
    for key, (root, probe) in measures.items():
        replay = replays.get(str(root), {})
        if not probe.get('successful') or not replay.get('successful') or not replay.get('supported'):
            errors.append('invalid_measurement_or_exactness:' + str(key))
        errors.extend(probe.get('stability_errors', []))
        states.extend([probe['machine_before'], probe['machine_after'], *probe.get('machine_samples', [])])
    errors.extend(stable_states(states))
    comparisons = []
    for pair in range(first['config']['pairs']):
        group = {mode: measures[(pair, mode)][1] for mode in ARMS if (pair, mode) in measures}
        if len(group) != len(ARMS):
            continue
        baseline = group['mtp-legacy']
        if any(p['request_sequence'] != baseline['request_sequence'] for p in group.values()):
            errors.append('request_sequence_mismatch:' + str(pair))
        # Compare every attributed metric request across both native families as well.
        reference = {c['request_id']: c for c in baseline['responses'] if c.get('metrics')}
        for mode, probe in group.items():
            current = {c['request_id']: c for c in probe['responses'] if c.get('metrics')}
            if set(current) != set(reference) or any(not response_equal(current[k]['response'], reference[k]['response'])
                                                     for k in set(current) & set(reference)):
                errors.append('cross_family_metric_exactness:' + str(pair) + ':' + mode)
        for metric in ('decodeTokPerSec', 'speculative:predictableTokPerSec', 'speculative:novelTokPerSec'):
            natives = {family: group[family + '-legacy']['metric_values'][metric] for family in ('mtp', 'dflash')}
            for family in ('mtp', 'dflash'):
                value = group[family + '-persistent']['metric_values'][metric]
                comparisons.append({'pair': pair, 'family': family, 'metric': metric,
                    'feature': value, 'native': natives, 'family_ratio': value / natives[family],
                    'fastest_native_ratio': value / max(natives.values())})
    result = {'complete': not errors, 'errors': sorted(set(errors)), 'comparisons': comparisons,
              'numerical_mode': 'row-exact', 'performance_pass': False, 'promotion_pass': False,
              'scope': 'Matched canonical descriptive evidence across separately bounded windows, not jointly balanced confirmation; no reuse attribution without accepted tier receipts',
              'config': first['config'], 'windows': [str(root) for root in roots]}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    d.save(output / 'aggregate-summary.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--declare', action='store_true', help='CPU-only cohort declaration after the independent host rebuild')
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--mode', choices=ARMS)
    parser.add_argument('--pair', type=int)
    parser.add_argument('--source', type=Path, help='Separate exact replay of one complete measurement window')
    parser.add_argument('--aggregate', type=Path, nargs='+', help='CPU-only join; requires every declared measurement and replay')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        return worker(args.worker)
    if args.declare:
        if not args.config or args.config.exists() or not args.seed:
            parser.error('--declare requires a new --config and completed training --seed')
        manifest = d.load_json(d.KIT / 'manifest.json')
        config = {'version': 1, 'seed': str(args.seed.expanduser().resolve()),
                  'binary_sha256': r.digest(r.absolute_asset(manifest['binary'])),
                  'source_hashes': source_hashes(), 'protocol': PROTOCOL,
                  'cache_tag': 'canonical' + str(time.time_ns()), 'runs': 3, 'rungs': '512',
                  'pairs': 2, 'arms': list(ARMS),
                  'orders': [list(ARMS), list(reversed(ARMS))],
                  'scope': 'Two paired separately bounded windows per arm: diagnostic, never promotion'}
        validate_cohort(config)
        args.config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        d.save(args.config, config)
        return 0
    if args.aggregate:
        if not args.output:
            parser.error('--aggregate requires new --output')
        return aggregate(args.aggregate, args.output)
    if not args.config or not args.output or not 1 <= args.port <= 65535:
        parser.error('--config and new --output required; port must be 1..65535')
    if args.source and (args.mode is not None or args.pair is not None):
        parser.error('--source replay cannot override mode/pair')
    return d.supervise(args, lambda out: prepare(args, out), __file__, summarize)


if __name__ == '__main__':
    sys.exit(main())
