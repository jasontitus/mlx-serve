#!/usr/bin/env python3
"""One complete pinned canonical arm OR its exact replay, never an automatic sweep.

Reuse the host canonical proxy/report/replay and transfer frozen-store/supervisor
contracts. Default invocations retain a 285s worker / 290s absolute process-group
bound. Declare --unbounded to remove those bounds, retaining a 3600s full-probe
hang safeguard and the request/startup timeouts. Only complete protocols count.
Rebuild the independently implemented host before any GPU invocation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import lookup_similar_requests as s
d, r = s.d, s.r
from lookup_bench_evidence import stable_states, response_equal, bootstrap_effect

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
UNBOUNDED_PROTOCOL = dict(PROTOCOL, deadline_seconds=None, worker_stop_seconds=None,
                          server_args=dict(PROTOCOL['server_args'], probe_timeout=3600))
STATISTICS = {'method': 'family-quartet log-ratio percentile bootstrap', 'replications': 10000,
              'minimum_quartets': 4, 'confidence': 0.95, 'margin': 0,
              'latency_pooling': 'arithmetic mean of two physical boot summaries; primary TTFT uses mean of three client samples',
              'rate_pooling': 'geometric mean of two physical boot summaries; geometric rate estimate, not pooled tokens/time',
              'multiplicity': 'pointwise; no simultaneous or overall-workload win claim'}
METRICS = ('decodeTokPerSec', 'ttftMs', 'prefillTokPerSec',
           'speculative:predictableTokPerSec', 'speculative:novelTokPerSec',
           'context:512:decodeTokPerSec', 'context:512:ttftMs', 'context:512:prefillTokPerSec')
REPORT_METRICS = {'prefixCache': ('coldTtftMs', 'warmTtftMs'),
                  'batching': ('singleTokPerSec', 'aggregateTokPerSec')}


def declared_schedule(quartets):
    """A=native, B=adaptive; cross family precedence and orientation balanced."""
    slots = []
    for quartet in range(quartets):
        families = ('mtp', 'dflash') if quartet % 2 == 0 else ('dflash', 'mtp')
        tiers = ('legacy', 'persistent', 'persistent', 'legacy')
        if (quartet + quartet // 2) % 2:
            tiers = ('persistent', 'legacy', 'legacy', 'persistent')
        for family in families:
            for position, tier in enumerate(tiers):
                slots.append({'slot': len(slots), 'quartet': quartet, 'family': family,
                              'position': position, 'mode': family + '-' + tier})
    return slots


def cohort_protocol(config):
    budget = config.get('budget', 'bounded')
    if budget not in ('bounded', 'unbounded'):
        raise r.BenchError('canonical cohort budget changed or malformed')
    return UNBOUNDED_PROTOCOL if budget == 'unbounded' else PROTOCOL


def source_hashes():
    paths = [Path(__file__), Path(d.__file__), Path(s.__file__), Path(r.__file__),
             Path(s.evidence.__file__), Path(d.write_npy_i32.__code__.co_filename), s.CONTRACT]
    return {path.name: r.digest(path) for path in paths}


def validate_cohort(config):
    if not isinstance(config, dict) or config.get('source_hashes') != source_hashes():
        raise r.BenchError('canonical cohort helper/contract identity missing or changed; redeclare, never upgrade history')
    if (config.get('version') != 2
            or d.content_hash(config.get('protocol')) != d.content_hash(cohort_protocol(config))
            or type(config.get('runs')) is not int or config['runs'] != 3 or config.get('rungs') != '512'
            or type(config.get('quartets')) is not int or config['quartets'] not in (2, 4)
            or any(key in config for key in ('pairs', 'orders'))):
        raise r.BenchError('canonical cohort protocol changed or malformed; redeclare, never upgrade history')
    if config.get('measurement_isolation') not in ('declared', 'undeclared'):
        raise r.BenchError('canonical isolation declaration missing or malformed; redeclare, never upgrade history')
    if d.content_hash(config.get('statistics')) != d.content_hash(STATISTICS):
        raise r.BenchError('canonical statistics declaration missing or changed; redeclare, never upgrade history')
    identity = config.get('runtime_identity')
    if not isinstance(identity, dict) or not identity:
        raise r.BenchError('canonical runtime identity missing or malformed; redeclare, never upgrade history')
    if config.get('arms') != list(ARMS) or not isinstance(config.get('cache_tag'), str) or not config['cache_tag'].isalnum():
        raise r.BenchError('declare both native and both persistent arms and an alphanumeric cache tag')
    if d.content_hash(config.get('schedule')) != d.content_hash(declared_schedule(config['quartets'])):
        raise r.BenchError('canonical cohort boot slots changed or malformed')


def validate_plan(plan):
    config = plan.get('config')
    validate_cohort(config)
    protocol = config['protocol']
    if (plan.get('source_hashes') != config['source_hashes'] or d.content_hash(plan.get('protocol')) != d.content_hash(config['protocol'])
            or any(d.content_hash(plan.get(key)) != d.content_hash(protocol[key]) for key in ('numerical_mode', 'deadline_seconds', 'worker_stop_seconds'))
            or any(key not in plan for key in ('deadline_seconds', 'worker_stop_seconds'))):
        raise r.BenchError('canonical plan helper/contract/protocol identity mismatch')
    if any(d.content_hash(plan.get(key)) != d.content_hash(config[key])
           for key in ('measurement_isolation', 'statistics', 'quartets', 'schedule')):
        raise r.BenchError('canonical plan isolation/statistics/schedule identity mismatch')
    if d.content_hash(plan.get('runtime_identity')) != d.content_hash(config['runtime_identity']):
        raise r.BenchError('canonical plan runtime identity mismatch')
    server_args = plan.get('server_args')
    if (not isinstance(server_args, dict) or type(server_args.get('port')) is not int
            or not 1 <= server_args['port'] <= 65535
            or d.content_hash(server_args) != d.content_hash(dict(protocol['server_args'], port=server_args['port']))):
        raise r.BenchError('canonical plan server protocol changed or malformed')
    if (plan.get('phase') not in ('measure', 'replay') or type(plan.get('slot')) is not int
            or plan['slot'] not in range(len(config['schedule']))):
        raise r.BenchError('canonical plan phase/slot changed or malformed')
    slot = config['schedule'][plan['slot']]
    if any(type(plan.get(key)) is not type(value) or plan.get(key) != value for key, value in slot.items()):
        raise r.BenchError('canonical plan slot identity mismatch')
    if plan['phase'] == 'replay':
        identity = plan.get('source_identity')
        if (not isinstance(identity, dict) or set(identity) != {'plan.json', 'probe.json', 'arm/llmprobe.json'}
                or any(not isinstance(value, str) or not value for value in identity.values())):
            raise r.BenchError('canonical replay source identity missing or malformed')
    family, tier = plan['mode'].split('-')
    definitions = d.arm_definitions('reuse', family, [tier])
    d.declare_numerical_mode(definitions, True)
    if plan.get('tier') != tier or plan.get('arm_definitions') != definitions:
        raise r.BenchError('canonical plan arm controls changed or malformed')


def replay_source_identity(root):
    return {name: r.digest(root / name) for name in ('plan.json', 'probe.json', 'arm/llmprobe.json')}


def checked_runtime(binary, expected):
    actual = r.runtime_identity(binary)
    if d.content_hash(actual) != d.content_hash(expected):
        raise r.BenchError('canonical linked runtime or loading configuration changed; redeclare, never mix runtimes')
    return actual


def runtime_receipt_errors(out, plan):
    try:
        receipt = d.load_json(out / 'runtime-identity.json')
        expected = d.content_hash(plan['runtime_identity'])
        if any(d.content_hash(receipt.get(stage)) != expected for stage in ('before', 'after')):
            return ['runtime_identity_receipt_missing_or_changed']
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        return ['runtime_identity_receipt_unreadable:' + str(exc)]
    return []


def prepare(args, out):
    config = d.load_json(args.config)
    validate_cohort(config)
    protocol = config['protocol']
    manifest = d.load_json(d.KIT / 'manifest.json')
    model = next(m for m in manifest['models'] if m['id'] == 'qwen27b')
    if model['model_dir'] != TARGET or model['drafter_dir'] != DRAFTER:
        raise r.BenchError('target must be the declared Qwen27B with explicit incoai DFlash2')
    binary = r.absolute_asset(manifest['binary'])
    if r.digest(binary) != config['binary_sha256']:
        raise r.BenchError('cohort binary hash changed; redeclare after rebuilding, never mix binaries')
    checked_runtime(binary, config['runtime_identity'])
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
        if not summary.get('complete') or not summary.get('final') or previous.get('phase') != 'measure':
            raise r.BenchError('replay needs a completed full measurement, not a partial or replay run')
        if previous['config_sha256'] != r.digest(args.config) or previous['reuse']['hashes'] != seed['hashes']:
            raise r.BenchError('replay cohort or frozen history mismatch')
        slot_id = previous['slot']
    else:
        slot_id = args.slot
    if type(slot_id) is not int or slot_id not in range(len(config['schedule'])):
        raise r.BenchError('explicit --slot must belong to declared cohort')
    slot = config['schedule'][slot_id]
    mode = slot['mode']
    family, tier = mode.split('-')
    definitions = d.arm_definitions('reuse', family, [tier])
    d.declare_numerical_mode(definitions, True)
    plan.update(version=2, phase='replay' if source else 'measure', binary=binary,
        binary_sha256=config['binary_sha256'], config=config, config_sha256=r.digest(args.config),
        source_hashes=config['source_hashes'],
        runtime_identity=config['runtime_identity'],
        measurement_isolation=config['measurement_isolation'], statistics=config['statistics'],
        quartets=config['quartets'], schedule=config['schedule'],
        source=str(source) if source else None, **slot, tier=tier,
        source_identity=replay_source_identity(source) if source else None,
        numerical_mode='row-exact', arm_definitions=definitions, reuse=seed,
        probe_cmd=manifest['probe_cmd'], probe_assets={p: r.digest(p) for p in manifest['probe_cmd']},
        server_args=dict(protocol['server_args'], port=args.port),
        deadline_seconds=protocol['deadline_seconds'], worker_stop_seconds=protocol['worker_stop_seconds'],
        protocol=config['protocol'],
        training_policy='Only four predeclared held-out-transfer training requests; frozen clones, writes off; canonical answers never seed history',
        promotion_pass=False)
    validate_plan(plan)
    d.save(out / 'plan.json', plan)
    return plan


def summarize(out, plan, final=False, supervisor=None):
    errors, parse_errors = d.read_rows(out / 'errors.jsonl')
    boots, boot_errors = d.read_rows(out / 'boots.jsonl')
    runtime_errors = runtime_receipt_errors(out, plan)
    try:
        checked_runtime(plan['binary'], plan['runtime_identity'])
    except (r.BenchError, OSError, ValueError) as exc:
        runtime_errors.append('runtime_identity_changed:' + str(exc))
    complete = (out / 'worker-complete.json').is_file() and not errors and not parse_errors and not boot_errors
    complete = complete and not runtime_errors
    if final:
        complete = complete and supervisor is not None and supervisor.get('exit_code') == 0 and not supervisor.get('timed_out')
    if plan['phase'] == 'measure':
        complete = complete and len(boots) == 1 and all(b.get('completed') and b.get('assets_preserved')
                                                        and b.get('numerical_mode') == 'row-exact' for b in boots)
    result = {'complete': bool(complete), 'final': final, 'invalid_samples': len(errors) + len(parse_errors) + len(boot_errors) + len(runtime_errors),
              'samples': 1 if complete else 0, 'performance_pass': False, 'promotion_pass': False,
              'phase': plan['phase'], 'errors': errors + parse_errors + boot_errors + runtime_errors,
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
        runtime = {'started_at': time.time(), 'before': checked_runtime(plan['binary'], plan['runtime_identity'])}
        d.save(out / 'runtime-identity.json', runtime)
        if plan['phase'] == 'measure':
            folder = out / 'arm'
            folder.mkdir()
            command = r.controlled_probe_command(plan['probe_cmd'], folder, plan['config']['cache_tag'] + str(plan['quartet']))
            with d.owned_boot(out, folder, plan, args, plan['slot'], plan['tier']) as server:
                preflight = r.preflight_context(server, args, [], command)
                d.save(folder / 'context-preflight.json', preflight)
                probe = r.canonical_probe(server, args, plan['mode'], command, folder)
            d.save(out / 'probe.json', probe)
        else:
            if replay_source_identity(Path(plan['source'])) != plan['source_identity']:
                raise r.BenchError('replay source identity changed after preparation')
            previous = d.load_json(Path(plan['source']) / 'plan.json')
            validate_plan(previous)
            if (previous['phase'] != 'measure' or previous['config_sha256'] != plan['config_sha256']
                    or previous['slot'] != plan['slot'] or previous['mode'] != plan['mode']
                    or previous['reuse']['hashes'] != plan['reuse']['hashes']):
                raise r.BenchError('replay source cohort/protocol/slot/seed changed after preparation')
            probe = d.load_json(Path(plan['source']) / 'probe.json')
            evidence = r.replay_canonical(args, plan['binary'], plan['model'], plan['mode'], probe, out)
            d.save(out / 'replay.json', evidence)
            if replay_source_identity(Path(plan['source'])) != plan['source_identity']:
                raise r.BenchError('replay source identity changed during replay')
            if not evidence.get('successful') or not evidence.get('supported'):
                raise r.BenchError('full canonical exact replay failed or unsupported')
        runtime['after'] = checked_runtime(plan['binary'], plan['runtime_identity'])
        runtime['finished_at'] = time.time()
        d.save(out / 'runtime-identity.json', runtime)
        d.save(out / 'worker-complete.json', {'completed': True})
        return 0
    except BaseException as exc:
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        return 1



def observed_schedule(measures, config):
    """Require nonoverlapping real boot intervals in the frozen block order."""
    intervals, errors, states = [], [], []
    for slot_id, (root, _) in measures.items():
        slot = config['schedule'][slot_id]
        mode = slot['mode']
        boots, parse_errors = d.read_rows(root / 'boots.jsonl')
        if parse_errors or len(boots) != 1:
            errors.append('missing_or_malformed_boot:' + str(slot_id))
            continue
        boot = boots[0]
        before, after = boot.get('machine_before', {}), boot.get('machine_after', {})
        states.extend((before, after))
        start, end = before.get('timestamp'), after.get('timestamp')
        if (type(boot.get('pair')) is not int or boot['pair'] != slot_id or boot.get('arm') != mode.split('-')[1]
                or not boot.get('completed') or not boot.get('assets_preserved')
                or boot.get('numerical_mode') != 'row-exact'
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
                or end <= start):
            errors.append('invalid_boot_interval:' + str(slot_id))
            continue
        intervals.append({**slot, 'start': start, 'end': end})
    intervals.sort(key=lambda row: row['start'])
    if [row['slot'] for row in intervals] != [slot['slot'] for slot in config['schedule']]:
        errors.append('observed_boot_order_differs_from_declared_schedule')
    if any(left['end'] >= right['start'] for left, right in zip(intervals, intervals[1:])):
        errors.append('overlapping_or_ambiguous_boot_intervals')
    return {'proved': not errors, 'observed': intervals, 'errors': errors}, states


def full_metric_values(probe, report):
    values = {metric: probe.get('metric_values', {}).get(metric) for metric in METRICS}
    bench = report.get('bench') or {}
    samples = (bench.get('ttftMs') or {}).get('samples', [])
    values['ttftMs'] = (sum(samples) / len(samples) if isinstance(samples, list) and len(samples) == 3
                        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples)
                        else None)
    for section, metrics in REPORT_METRICS.items():
        for metric in metrics:
            values[section + ':' + metric] = (bench.get(section) or {}).get(metric)
    return values


def quartet_comparisons(quartet, family, slots, values):
    """One effect per family quartet; no physical boot/request pseudoreplication."""
    comparisons = []
    for metric in values[slots[0]['slot']]:
        by_tier = {tier: [values[slot['slot']][metric] for slot in slots
                          if slot['mode'] == family + '-' + tier]
                   for tier in ('legacy', 'persistent')}
        if any(len(vs) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                  for v in vs) for vs in by_tier.values()):
            continue
        latency = 'ttft' in metric.lower()
        pooled = {tier: sum(vs) / 2 if latency else math.exp(sum(math.log(v) for v in vs) / 2)
                  for tier, vs in by_tier.items()}
        native, feature = pooled['legacy'], pooled['persistent']
        comparisons.append({'quartet': quartet, 'family': family, 'metric': metric,
            'slots': [slot['slot'] for slot in slots], 'physical_values': by_tier,
            'feature': feature, 'native': native, 'higher_is_better': not latency,
            'pooling': STATISTICS['latency_pooling' if latency else 'rate_pooling'],
            'family_ratio': native / feature if latency else feature / native})
    return comparisons


def comparison_statistics(comparisons, qualified):
    """Resample complete family quartets, never physical boots or client repeats."""
    result = {'per_family': []}
    groups = {}
    for row in comparisons:
        groups.setdefault((row['family'], row['metric']), []).append(row)
    for (family, metric), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['quartet'])
        if len({row['quartet'] for row in rows}) != len(rows):
            raise r.BenchError('duplicate family quartet metric effect')
        effect = bootstrap_effect([row['family_ratio'] for row in rows], STATISTICS['margin'],
                                  STATISTICS['replications'], STATISTICS['minimum_quartets'])
        # The shared helper's generic block fields refer to quartets here, not boots.
        effect['quartets'] = effect.pop('boot_blocks', len(rows))
        if 'required_boot_blocks' in effect:
            effect['required_quartets'] = effect.pop('required_boot_blocks')
        ci = effect.get('ci95')
        certified = metric in METRICS
        lower_above_one = bool(ci and ci[0] > 1)
        sufficient = len(rows) >= STATISTICS['minimum_quartets']
        effect['statistical_verdict'] = effect['verdict']
        if not qualified or not sufficient:
            effect['verdict'] = 'descriptive_unqualified'
        elif not certified:
            effect['verdict'] = 'diagnostic_exactness_unsupported'
        result['per_family'].append({'family': family, 'metric': metric, **effect,
            'lower_ci_above_one': lower_above_one,
            'qualified_speedup': qualified and sufficient and certified and lower_above_one,
            'metric_exactness_supported': certified,
            'observed_regression': any(row['family_ratio'] < 1 for row in rows),
            'regression_ci_below_one': bool(ci and ci[1] < 1),
            'claim_scope': 'adjacent family-native/adaptive quartet, pointwise metric only; no reuse attribution'})
    return result

def aggregate(paths, output):
    roots = [p.expanduser().resolve() for p in paths]
    if not roots:
        raise r.BenchError('canonical aggregate requires measurement and replay windows')
    if len(set(roots)) != len(roots):
        raise r.BenchError('duplicate input window')
    measures, replays, replay_slots, plans, errors = {}, {}, {}, [], []
    replay_intervals = []
    for root in roots:
        plan, summary = d.load_json(root / 'plan.json'), d.load_json(root / 'summary.json')
        validate_plan(plan)
        plans.append(plan)
        errors.extend(runtime_receipt_errors(root, plan))
        if not summary.get('complete') or not summary.get('final'):
            errors.append('incomplete:' + str(root))
        if plan['phase'] == 'measure':
            key = plan['slot']
            if key in measures:
                raise r.BenchError('duplicate measurement slot, no retry-until-win')
            measures[key] = (root, d.load_json(root / 'probe.json'))
        else:
            if plan['source'] in replays:
                raise r.BenchError('duplicate replay source')
            replays[plan['source']] = d.load_json(root / 'replay.json')
            replay_slots[plan['source']] = (plan['slot'], plan['source_identity'])
            receipt = d.load_json(root / 'runtime-identity.json')
            start, end = receipt.get('started_at'), receipt.get('finished_at')
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
                    or end <= start):
                errors.append('missing_or_invalid_replay_interval:' + str(plan['slot']))
            else:
                replay_intervals.append((start, end))
    first = plans[0]
    checked_runtime(first['binary'], first['runtime_identity'])
    if any(p['config_sha256'] != first['config_sha256'] or p['probe_assets'] != first['probe_assets']
           or p['source_hashes'] != first['source_hashes'] or p['protocol'] != first['protocol']
           or d.content_hash(p['config']) != d.content_hash(first['config'])
           or p['binary'] != first['binary'] or p['runtime_identity'] != first['runtime_identity']
           or p['reuse']['hashes'] != first['reuse']['hashes'] or p['model'] != first['model'] for p in plans):
        raise r.BenchError('cohort/client/helper/contract/protocol/seed/target mismatch')
    expected = {slot['slot'] for slot in first['config']['schedule']}
    if set(measures) != expected:
        errors.append('missing_declared_measurement_slots')
    if set(replays) != {str(root) for root, _ in measures.values()}:
        errors.append('missing_or_extra_replay_sources')
    schedule, states = observed_schedule(measures, first['config'])
    measurement_end = max((row['end'] for row in schedule['observed']), default=math.inf)
    replay_intervals.sort()
    if any(start <= measurement_end for start, _ in replay_intervals):
        errors.append('replays_must_follow_all_measurement_slots')
    if any(left[1] >= right[0] for left, right in zip(replay_intervals, replay_intervals[1:])):
        errors.append('overlapping_replay_intervals')
    values, load_drift = {}, []
    for key, (root, probe) in measures.items():
        replay = replays.get(str(root), {})
        if replay_slots.get(str(root)) != (key, replay_source_identity(root)):
            errors.append('replay_source_slot_or_identity_mismatch:' + str(key))
        if not probe.get('successful') or not replay.get('successful') or not replay.get('supported'):
            errors.append('invalid_measurement_or_exactness:' + str(key))
        captures = probe.get('responses', [])
        if (len(captures) != 34 or len(probe.get('request_sequence', [])) != 34
                or len({c.get('request_index') for c in captures}) != 34
                or replay.get('captured_requests') != 34):
            errors.append('incomplete_34_request_protocol:' + str(key))
        results = replay.get('results', [])
        references = replay.get('reference_modes', [])
        if (not references or len(results) != 34 * len(references)
                or any(not row.get('reference_matches') for row in results)):
            errors.append('incomplete_full_request_replay:' + str(key))
        for metric in METRICS:
            if (probe.get('metric_request_counts', {}).get(metric) != 3
                    or sum(metric in c.get('metrics', []) for c in captures) != 3):
                errors.append('incomplete_metric_requests:' + str(key) + ':' + metric)
        report = d.load_json(root / 'arm' / 'llmprobe.json')
        errors.extend(r.probe_report_errors(report, [512], 3))
        errors.extend(r.probe_stability_errors(report))
        errors.extend(probe.get('report_errors', []))
        values[key] = full_metric_values(probe, report)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in values[key].values()):
            errors.append('invalid_full_workload_metrics:' + str(key))
        load_drift.append({**first['config']['schedule'][key],
                           'loadDrift': (report.get('bench') or {}).get('loadDrift')})
        errors.extend(probe.get('stability_errors', []))
        states.extend([probe.get('machine_before', {}), probe.get('machine_after', {}),
                       *probe.get('machine_samples', [])])
        if not probe.get('machine_samples'):
            errors.append('missing_machine_monitor_samples:' + str(key))
    errors.extend(stable_states(states))
    comparisons = []
    for quartet in range(first['config']['quartets']):
        for family in ('mtp', 'dflash'):
            slots = [slot for slot in first['config']['schedule']
                     if slot['quartet'] == quartet and slot['family'] == family]
            if any(slot['slot'] not in measures for slot in slots):
                continue
            group = [measures[slot['slot']][1] for slot in slots]
            baseline = group[0]
            if any(p['request_sequence'] != baseline['request_sequence'] for p in group):
                errors.append('request_sequence_mismatch:' + str((quartet, family)))
            reference = {c['request_id']: c for c in baseline['responses'] if c.get('metrics')}
            for slot, probe in zip(slots, group):
                current = {c['request_id']: c for c in probe['responses'] if c.get('metrics')}
                if set(current) != set(reference) or any(not response_equal(current[k]['response'], reference[k]['response'])
                                                         for k in set(current) & set(reference)):
                    errors.append('family_metric_exactness:' + str(slot['slot']))
            comparisons.extend(quartet_comparisons(quartet, family, slots, values))
    exclusions = list(errors) + schedule['errors']
    if first['config']['measurement_isolation'] != 'declared':
        exclusions.append('gpu_isolation_not_declared')
    if first['config']['quartets'] < STATISTICS['minimum_quartets']:
        exclusions.append('insufficient_independent_family_quartets')
    qualified = not exclusions
    statistics = comparison_statistics(comparisons, qualified)
    regressions = {kind: [row for row in rows if row['observed_regression'] or row['regression_ci_below_one']]
                   for kind, rows in statistics.items()}
    result = {'complete': not errors and schedule['proved'], 'errors': sorted(set(errors + schedule['errors'])),
              'comparisons': comparisons,
              'physical_observations': [{**first['config']['schedule'][slot], 'metrics': value}
                                        for slot, value in sorted(values.items())],
              'fastest_native': {'supported': False,
                  'reason': 'Family quartets are adjacent only within family; cross-family fastest-native effects are not drift-matched.'},
              'numerical_mode': 'row-exact', 'performance_pass': False, 'promotion_pass': False,
              'measurement_qualified': qualified, 'measurement_exclusions': sorted(set(exclusions)),
              'balanced_schedule': schedule, 'statistics': statistics,
              'full_workload_regressions': regressions, 'load_drift': load_drift,
              'reuse_attribution': False, 'blanket_canonical_correctness': False,
              'scope': 'Pointwise family-native/adaptive quartet claims only when qualified. Throughput is a geometric estimate, not pooled tokens/time. No fastest-native cross-family claim. Prefix/batching are diagnostic, not exactness-certified. No overall performance pass, promotion, or reuse attribution.',
              'config': first['config'], 'windows': [str(root) for root in roots]}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    d.save(output / 'aggregate-summary.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--declare', action='store_true', help='CPU-only cohort declaration after the independent host rebuild')
    parser.add_argument('--unbounded', action='store_true',
                        help='Declaration only: no whole-run deadline; retain 3600s full-probe hang safeguard')
    parser.add_argument('--isolation', choices=('declared', 'undeclared'),
                        help='Required at declaration: explicit exclusive-inference assertion, never inferred from timings')
    parser.add_argument('--quartets', type=int, choices=(2, 4),
                        help='Declaration only: quartets per family (4 qualifies; 2 is diagnostic only)')
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--slot', type=int, help='One physical boot slot from the frozen schedule')
    parser.add_argument('--source', type=Path, help='Separate exact replay of one complete measurement window')
    parser.add_argument('--aggregate', type=Path, nargs='+', help='CPU-only join; requires every declared measurement and replay')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.unbounded and not args.declare:
        parser.error('--unbounded is declaration-only; executions inherit the frozen cohort budget')
    if (args.isolation is not None or args.quartets is not None) and not args.declare:
        parser.error('--isolation and --quartets are declaration-only; executions inherit the frozen cohort')
    if args.worker:
        return worker(args.worker)
    if args.declare:
        if not args.config or args.config.exists() or not args.seed or args.isolation is None:
            parser.error('--declare requires a new --config, completed training --seed, and explicit --isolation')
        quartets = args.quartets if args.quartets is not None else 4
        manifest = d.load_json(d.KIT / 'manifest.json')
        config = {'version': 2, 'seed': str(args.seed.expanduser().resolve()),
                  'binary_sha256': r.digest(r.absolute_asset(manifest['binary'])),
                  'runtime_identity': r.runtime_identity(r.absolute_asset(manifest['binary'])),
                  'source_hashes': source_hashes(), 'budget': 'unbounded' if args.unbounded else 'bounded',
                  'protocol': UNBOUNDED_PROTOCOL if args.unbounded else PROTOCOL,
                  'cache_tag': 'canonical' + str(time.time_ns()), 'runs': 3, 'rungs': '512',
                  'measurement_isolation': args.isolation, 'statistics': STATISTICS,
                  'quartets': quartets, 'arms': list(ARMS), 'schedule': declared_schedule(quartets),
                  'scope': 'Frozen adjacent family-native/adaptive ABBA/BAAB quartets; four per family required for qualified pointwise claims; never automatic promotion'}
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
    if args.source and args.slot is not None:
        parser.error('--source replay cannot override slot')
    config = d.load_json(args.config)
    validate_cohort(config)
    protocol = config['protocol']
    return d.supervise(args, lambda out: prepare(args, out), __file__, summarize,
                       deadline_seconds=protocol['deadline_seconds'],
                       worker_stop_seconds=protocol['worker_stop_seconds'])


if __name__ == '__main__':
    sys.exit(main())
