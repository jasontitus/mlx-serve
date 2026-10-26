#!/usr/bin/env python3
"""Frozen full-workload component ablations; no build, training, or automatic sweep.

Declare one family/component with --declare --config NEW.json --seed TRAINING_RUN
--family dflash --component neural-width --isolation declared. Inspect that frozen
config, then explicitly execute --config NEW.json --output NEW_RUN. --plan-only
never launches inference. All four held-out cases, first exposure AND repeat, are
measured in every fresh boot. production-routing and suffix-backoff use adjacent
ABBA/BAAB primary quartets, as do *-fixed-cut and *-max-cut width contrasts.
Native references stay outside component quartets. The MTP lookup ablation retains six permutations.
There is no whole-run deadline. Per-request/boot hang safeguards remain.

Components: dflash neural-width, combined-width, lookup-increment; mtp lookup;
mtp/dflash production-routing (native/adaptive ABBA); pld suffix-backoff.
Neural-only MTP StairCut remains unsupported. No runtime is pre-certified here:
completed exact serial comparisons and execution receipts gate PLD evidence.
PLE is an embedding optimization, not a drafter or an arm of this experiment.
"""
import argparse
import json
import itertools
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import lookup_similar_requests as s
import lookup_gpu_telemetry as telemetry
import lookup_counterfactual_window as calibration

d, r, evidence = s.d, s.r, s.evidence
CASES = ('related-config', 'conflicting-config', 'related-code', 'unrelated-prose')
EXPOSURES = (-1, 0)
UNSUPPORTED = {
    'mtp/neural-width': 'No neural-only MTP StairCut runtime control; native MTP versus lookup is not this ablation.',
    'ple': 'PLE is an embedding optimization, not a speculative drafter; measured separately.',
}
WIDTH_QUARTETS = ('neural-fixed-cut', 'neural-max-cut', 'combined-fixed-cut', 'combined-max-cut')
PROTOCOL = {
    'version': 2, 'numerical_mode': 'row-exact', 'deadline_seconds': None,
    'worker_stop_seconds': None, 'server_args': {'ctx_size': 2048, 'ready_timeout': 120,
                                             'request_timeout': 300},
    'bootstrap_replications': 10000, 'practical_margin': .05,
    'unit': 'independent fresh-boot paired block; never requests, tokens, or speculative rounds',
    'workload': list(CASES), 'exposures': list(EXPOSURES),
    'scope': 'Full held-out transfer workload only; not canonical throughput or default promotion.',
    'cost_accounting': {
        'endpoint': 'Complete measured HTTP request time, including preparation, selection, cold bootstrap and width trials; no round exclusions',
        'offline_prior': 'Absent by default; supplied diagnostic verify/commit priors are not full-round training evidence',
        'learning': 'Fresh controller per boot; calibration and bounded width trials inside evaluation are charged to that arm',
        'selection': 'Included in endpoint and round timing, not separately inferred by subtracting kernel medians',
        'native_references': 'Width-quartet native boots are descriptive and not pooled into the isolated primary',
    },
}


def component_matrix(family, component, width):
    if type(width) is not int or not 2 <= width <= 32:
        raise r.BenchError('fixed width must be 2..32')
    allowed = {'dflash': ('neural-width', 'combined-width', 'lookup-increment', 'production-routing') + WIDTH_QUARTETS,
               'mtp': ('lookup', 'production-routing'), 'pld': ('suffix-backoff',)}
    if component not in allowed.get(family, ()):
        raise r.BenchError(UNSUPPORTED.get(family + '/' + component, 'unsupported family/component'))
    native = 'serial' if family == 'pld' else family + '-native'
    definitions = {native: {'family': 'serial' if family == 'pld' else family,
        'mode': 'serial' if family == 'pld' else family + '-legacy', 'tier': 'legacy',
        'selection': 'adaptive', 'actual': 'legacy', 'candidate_source': 'native',
        'meaning': 'Native family reference, unchanged speculation defaults; explicit row-exact numerical mode'}}

    def add(source, selection):
        name = family + '-' + source + '-' + selection
        control = 'fixed:' + str(width) if selection == 'fixed' else selection
        definitions[name] = {'family': family, 'mode': family + ('-local' if source == 'neural' else '-persistent'),
            'tier': 'local' if source == 'neural' else 'persistent', 'selection': control, 'actual': selection,
            'candidate_source': source, 'experiment': {'MLX_SERVE_LOOKUP_CANDIDATES': '0' if source == 'neural' else '1'},
            'meaning': ('DFlash neural proposals only; suffix setup/query/index/store disabled' if source == 'neural' else
                        'Local plus frozen independent persistent lookup; native family fallback remains')}
        return name

    contrasts = []
    def contrast(before, after, mechanism, primary=False):
        contrasts.append({'baseline': before, 'treatment': after, 'mechanism': mechanism,
                          'role': 'primary' if primary else 'native-reference' if before == native else 'secondary'})

    if component == 'production-routing':
        adaptive = add('combined', 'adaptive')
        for base in (native, adaptive):
            definitions[base]['comparison_arm'] = base
            replica = dict(definitions[base])
            if 'experiment' in replica:
                replica['experiment'] = dict(replica['experiment'])
            definitions[base + '-repeat'] = replica
        contrast(native, adaptive, 'production lookup admission versus native family; paired ABBA/BAAB blocks')
        contrasts[-1]['role'] = 'production-primary'
    elif component in WIDTH_QUARTETS:
        source, control, _ = component.split('-')
        baseline, cut = add(source, control), add(source, 'cut')
        for base in (native, baseline, cut):
            definitions[base]['comparison_arm'] = base
            replica = dict(definitions[base])
            if 'experiment' in replica:
                replica['experiment'] = dict(replica['experiment'])
            definitions[base + '-repeat'] = replica
        contrast(baseline, cut, 'isolated adjacent ABBA/BAAB width selection; identical candidate source', primary=True)
        for arm in (baseline, cut):
            contrast(native, arm, 'native-family descriptive reference outside the width quartet')
        definitions['serial'] = {'family': 'serial', 'mode': 'serial', 'tier': 'legacy',
            'selection': 'adaptive', 'actual': 'legacy', 'candidate_source': 'native',
            'meaning': 'Exact serial replay reference outside the timed width contrast; row-exact'}
    elif component in ('neural-width', 'combined-width'):
        source = 'neural' if component == 'neural-width' else 'combined'
        fixed, maximum, cut = [add(source, selection) for selection in ('fixed', 'max', 'cut')]
        for arm in (fixed, maximum, cut):
            contrast(native, arm, 'native-family comparison')
        contrast(fixed, cut, 'adaptive width versus frozen fixed width', primary=True)
        contrast(maximum, cut, 'adaptive width versus maximum legal width', primary=True)
        if source == 'combined':
            adaptive = add(source, 'adaptive')
            contrast(native, adaptive, 'native-family comparison')
            contrast(cut, adaptive, 'assistant admission with identical candidate sources')
    elif component == 'lookup-increment':
        neural, combined = add('neural', 'cut'), add('combined', 'cut')
        contrast(native, neural, 'native-family comparison')
        contrast(native, combined, 'native-family comparison')
        contrast(neural, combined, 'incremental suffix candidates at identical width policy', primary=True)
    elif component == 'suffix-backoff':
        for enabled in (False, True):
            arm = 'pld-backoff-' + ('on' if enabled else 'off')
            definitions[arm] = {'family': 'pld', 'mode': 'pld-persistent', 'tier': 'persistent',
                'selection': 'cut', 'actual': 'cut', 'candidate_source': 'combined', 'backoff': enabled,
                'experiment': {'MLX_SERVE_LOOKUP_CANDIDATES': '1', 'MLX_SERVE_LOOKUP_BACKOFF': '1' if enabled else '0'},
                'meaning': 'Standalone suffix lookup, identical frozen history and cut policy; bounded miss backoff ' + ('on' if enabled else 'off')}
            contrast(native, arm, 'standalone PLD descriptive serial reference outside the backoff quartet')
        contrast('pld-backoff-off', 'pld-backoff-on', 'adjacent ABBA/BAAB bounded suffix miss-backoff increment', primary=True)
        for base in (native, 'pld-backoff-off', 'pld-backoff-on'):
            definitions[base]['comparison_arm'] = base
            replica = dict(definitions[base])
            if 'experiment' in replica:
                replica['experiment'] = dict(replica['experiment'])
            definitions[base + '-repeat'] = replica
    else:
        cut, adaptive = add('combined', 'cut'), add('combined', 'adaptive')
        contrast(native, cut, 'lookup chain cut versus native MTP; NOT neural-only MTP StairCut')
        contrasts[-1]['role'] = 'forced-control'
        contrast(native, adaptive, 'production lookup admission versus row-exact native MTP; NOT neural-only MTP StairCut')
        contrasts[-1]['role'] = 'production-primary'
        contrast(cut, adaptive, 'lookup admission with identical candidate sources')
        contrasts[-1]['role'] = 'forced-control'
    d.declare_numerical_mode(definitions, True)
    return definitions, contrasts


def balanced_orders(arms, blocks):
    if type(blocks) is not int or blocks < 4 or blocks % 2:
        raise r.BenchError('require at least four blocks, an even count for balanced arm precedence')
    return [list(arms) if block % 2 == 0 else list(reversed(arms)) for block in range(blocks)]


def component_orders(family, component, arms, blocks):
    if component == 'suffix-backoff':
        if type(blocks) is not int or blocks < 4 or blocks % 2 or len(arms) != 6:
            raise r.BenchError('backoff ABBA requires six physical arms and at least four even-count blocks')
        native, a0, b0, native_repeat, a1, b1 = arms
        return [[native, a0, b0, b1, a1, native_repeat] if block % 2 == 0 else
                [native_repeat, b0, a0, a1, b1, native] for block in range(blocks)]
    if component in WIDTH_QUARTETS:
        if type(blocks) is not int or blocks < 4 or blocks % 2 or len(arms) != 7:
            raise r.BenchError('width ABBA requires seven physical arms and at least four even-count blocks')
        native, a0, b0, native_repeat, a1, b1, serial = arms
        return [[serial, native, a0, b0, b1, a1, native_repeat] if block % 2 == 0 else
                [native_repeat, b0, a0, a1, b1, native, serial] for block in range(blocks)]
    if component == 'production-routing':
        if type(blocks) is not int or blocks < 4 or blocks % 2 or len(arms) != 4:
            raise r.BenchError('ABBA requires four physical arms and at least four even-count blocks')
        a0, b0, a1, b1 = arms
        return [[a0, b0, b1, a1] if block % 2 == 0 else [b0, a0, a1, b1]
                for block in range(blocks)]
    if family == 'mtp':
        if type(blocks) is not int or blocks != 6 or len(arms) != 3:
            raise r.BenchError('MTP v2 requires exactly six full permutation blocks (144 requests)')
        return [list(order) for order in itertools.permutations(arms)]
    return balanced_orders(arms, blocks)


def price_declaration(path, source=None):
    if path is None:
        if source is not None:
            raise r.BenchError('--prices-source requires --prices')
        return None
    path = path.expanduser().resolve()
    table = d.load_json(path)
    if not isinstance(table, dict) or table.get('schema') != 3 or table.get('context_bucket') != 'floor-power-of-two-2048':
        raise r.BenchError('prices require schema3 and floor-power-of-two-2048; never relabel older tables')
    if 'step_ms' in table and (not d.finite(table['step_ms']) or table['step_ms'] <= 0):
        raise r.BenchError('invalid schema3 step_ms price')
    cells = 0
    for cls, kinds in table.items():
        if cls in ('schema', 'context_bucket', 'step_ms'):
            continue
        if (not cls.isdecimal() or
                (cls != '0' and (int(cls) < 2048 or int(cls) > 262144 or int(cls) & (int(cls) - 1))) or
                not isinstance(kinds, dict)):
            raise r.BenchError('invalid schema3 price class')
        for kind, variants in kinds.items():
            if kind not in ('plain', 'chain', 'tree', 'spec') or not isinstance(variants, dict):
                raise r.BenchError('invalid schema3 price kind')
            for variant, widths in variants.items():
                if variant not in ('generic', 'linear_gdn') or not isinstance(widths, dict):
                    raise r.BenchError('invalid schema3 executed variant')
                for width, value in widths.items():
                    if not width.isdecimal() or not 1 <= int(width) <= 64 or not d.finite(value) or value <= 0:
                        raise r.BenchError('invalid schema3 width or price')
                    cells += 1
    if not cells:
        raise r.BenchError('offline price table has no measured cells')
    if source is None:
        source = next((candidate for candidate in (path.with_suffix('.source.json'), path.parent / 'run.json')
                       if candidate.is_file()), None)
    if source is None:
        raise r.BenchError('bare price file has no provenance; supply --prices-source JSON')
    source = source.expanduser().resolve()
    metadata = d.load_json(source)
    if not isinstance(metadata, dict) or not metadata:
        raise r.BenchError('price provenance must be a nonempty JSON object')
    if metadata.get('purpose') != 'staircase_calibration':
        raise r.BenchError('unsupported price provenance: require bound training-only staircase calibration')
    if (metadata.get('version') != 1 or metadata.get('phase') != 'training'
            or not metadata.get('source_windows') or not metadata.get('scope')):
        raise r.BenchError('staircase calibration requires explicit training-only source windows and scope')
    for key in ('prices_sha256', 'contract_sha256', 'binary_sha256'):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise r.BenchError('staircase calibration requires valid ' + key + '; never upgrade historical provenance')
    for key in ('runtime_identity', 'model', 'model_content'):
        if not isinstance(metadata.get(key), dict) or not metadata[key]:
            raise r.BenchError('staircase calibration requires nonempty ' + key)
    price_hash = r.digest(path)
    if metadata['prices_sha256'] != price_hash:
        raise r.BenchError('offline prices changed or mismatched: prices_sha256 differs from exact price bytes')
    return {'path': str(path), 'sha256': price_hash, 'schema': 3,
            'context_bucket': table['context_bucket'], 'cells': cells,
            'source': {'path': str(source), 'sha256': r.digest(source), 'metadata': metadata},
            'scope': metadata['scope']}


def bind_prices(definitions, prices):
    if prices is not None:
        metadata = prices['source']['metadata']
        for spec in definitions.values():
            if spec['candidate_source'] != 'native':
                if (spec['family'] != 'dflash' or
                        metadata.get('preparation') != 'production-' + spec['candidate_source']):
                    raise r.BenchError('offline price preparation does not match the arm candidate source/family')
                spec['experiment']['MLX_SERVE_LOOKUP_PRICES'] = prices['path']


def validate_prices(plan):
    prices = plan.get('prices')
    if prices is None:
        return
    current = price_declaration(Path(prices['path']), Path(prices['source']['path']))
    if current != prices:
        raise r.BenchError('offline prices or source metadata changed after declaration')
    metadata = prices['source']['metadata']
    contract_hash = plan.get('source_hashes', {}).get(str(s.CONTRACT.resolve()))
    if metadata['contract_sha256'] != contract_hash:
        raise r.BenchError('offline price source contract_sha256 differs from frozen experiment')
    for key in ('binary_sha256', 'runtime_identity', 'model', 'model_content'):
        if metadata[key] != plan.get(key):
            raise r.BenchError('offline price source ' + key + ' differs from frozen experiment')
    if metadata.get('seed_hashes') != plan.get('reuse', {}).get('hashes') or not metadata.get('seed_hashes'):
        raise r.BenchError('offline price training seed_hashes differs from frozen experiment')


def source_hashes():
    paths = [Path(__file__), Path(s.__file__), Path(d.__file__), Path(r.__file__),
             Path(evidence.__file__), Path(telemetry.__file__), Path(calibration.__file__),
             Path(d.write_npy_i32.__code__.co_filename), s.CONTRACT]
    return {str(path.resolve()): r.digest(path) for path in paths}


def validate_config(config, current=True):
    if config.get('version') != 2 or config.get('protocol', {}).get('version') != 2:
        raise r.BenchError('historical component protocol rejected; redeclare, never upgrade old evidence')
    if config.get('hash') != d.content_hash({key: value for key, value in config.items() if key != 'hash'}):
        raise r.BenchError('component declaration hash changed')
    definitions, contrasts = component_matrix(config['family'], config['component'], config['fixed_width'])
    bind_prices(definitions, config.get('prices'))
    if (config.get('protocol') != PROTOCOL or config.get('arm_definitions') != definitions
            or config.get('arms') != list(definitions) or config.get('contrasts') != contrasts
            or config.get('orders') != component_orders(config['family'], config['component'], list(definitions), config['blocks'])
            or config.get('measurement_isolation') not in ('declared', 'contended', 'unknown')):
        raise r.BenchError('component protocol, controls, contrasts, or schedule changed')
    telemetry.validate_declaration(config.get('telemetry'), current=current)
    contract = config['contract']
    training = [s.render(contract, item) for item in contract['training']]
    evaluation = [s.render(contract, item) for item in contract['evaluation']]
    by_id = {case['id']: case for case in evaluation}
    if (len(evaluation) != 4 or set(by_id) != set(CASES) or config['cases'] != [by_id[key] for key in CASES]
            or config['training_cases'] != training
            or config['request_identities'] != s.unique_requests(training, evaluation)):
        raise r.BenchError('complete independent training/evaluation workload changed')
    if current:
        if config['source_hashes'] != source_hashes() or config['contract'] != d.load_json(s.CONTRACT):
            raise r.BenchError('helpers or workload changed; redeclare, never upgrade historical evidence')
        if r.digest(config['binary']) != config['binary_sha256']:
            raise r.BenchError('binary changed after declaration')
        validate_runtime(config)
        model = config['model']
        if (r.digest(Path(model['model_dir']) / 'tokenizer.json') != model['tokenizer_sha256']
                or r.inventory(Path(model['model_dir'])) != config['model_inventory']
                or (r.inventory(Path(model['drafter_dir'])) if model.get('drafter_dir') else None) != config['drafter_inventory']):
            raise r.BenchError('target, tokenizer, or drafter identity changed')
        seed = s.load_seed(Path(config['seed_root']), config)
        if seed != config['reuse']:
            raise r.BenchError('frozen independent training seed changed')


def declare(args):
    if args.config.exists():
        raise r.BenchError('declaration path already exists')
    definitions, contrasts = component_matrix(args.family, args.component, args.fixed_width)
    prices = price_declaration(args.prices, args.prices_source)
    bind_prices(definitions, prices)
    orders = component_orders(args.family, args.component, list(definitions), args.blocks)
    manifest = d.load_json(d.KIT / 'manifest.json')
    model = next(model for model in manifest['models'] if model['id'] == 'qwen27b')
    for spec in definitions.values():
        reason = r.unsupported(model, spec['mode'])
        if reason:
            raise r.BenchError(reason)
    contract = d.load_json(s.CONTRACT)
    training = [s.render(contract, item) for item in contract['training']]
    evaluation = [s.render(contract, item) for item in contract['evaluation']]
    binary = r.absolute_asset(manifest['binary'])
    config = {'version': 2, 'family': args.family, 'component': args.component,
        'fixed_width': args.fixed_width, 'blocks': args.blocks, 'arms': list(definitions),
        'arm_definitions': definitions, 'contrasts': contrasts, 'orders': orders,
        'protocol': PROTOCOL, 'source_hashes': source_hashes(), 'contract': contract,
        'telemetry': telemetry.declaration(args.macmon or 'macmon',
            1000 if args.telemetry_interval_ms is None else args.telemetry_interval_ms,
            3000 if args.telemetry_max_gap_ms is None else args.telemetry_max_gap_ms),
        'training_cases': training, 'cases': [next(case for case in evaluation if case['id'] == key) for key in CASES],
        'request_identities': s.unique_requests(training, evaluation), 'model': model,
        'model_inventory': r.inventory(Path(model['model_dir'])),
        'drafter_inventory': r.inventory(Path(model['drafter_dir'])) if model.get('drafter_dir') else None,
        'binary': binary, 'binary_sha256': r.digest(binary),
        'runtime_identity': r.runtime_identity(binary),
        'seed_root': str(args.seed.expanduser().resolve()), 'measurement_isolation': args.isolation,
        'unsupported_components': UNSUPPORTED, 'created_unix': time.time(),
        'prices': prices,
        'price_policy': ('Frozen schema3 offline prior plus online observations, identical file on every nonnative arm; native reference unchanged'
                         if prices else 'Online-only schema3 variant/width pricing; fresh controller each boot; no inherited price file'),
        'balance': ('Adjacent ABBA/BAAB quartets; fresh independent replicas, one statistical observation per complete quartet'
                    if args.component in ('production-routing', 'suffix-backoff') or args.component in WIDTH_QUARTETS else
                    'All six full permutations; each arm twice per position, each directed adjacency twice'
                    if args.family == 'mtp' else
                    'Adjacent forward/reverse blocks balance pairwise precedence; not position balance'),
        'claim_policy': 'All frozen contrasts reported; bootstrap independent blocks, descriptive unadjusted 95% intervals; no post-hoc arm selection or universal multiple-comparison claim'}
    if prices is not None:
        # Content-match calibration once; existing inventory guards detect
        # subsequent ordinary file changes without hashing weights at each boot.
        config['model_content'] = calibration.model_content(model)
    config['reuse'] = s.load_seed(args.seed, config)
    config['hash'] = d.content_hash(config)
    validate_config(config)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    d.save(args.config, config)
    args.config.chmod(0o400)
    return 0


def prepare(args, out):
    config = d.load_json(args.config)
    validate_config(config)
    plan = dict(config, config=config, config_path=str(args.config.resolve()),
                numerical_mode=PROTOCOL['numerical_mode'],
                deadline_seconds=None, worker_stop_seconds=None,
                server_args=dict(PROTOCOL['server_args'], port=args.port),
                total_requests=config['blocks'] * len(config['arms']) * len(CASES) * len(EXPOSURES))
    d.save(out / 'plan.json', plan)
    return plan


def validate_plan(plan):
    config = plan['config']
    validate_config(config)
    if any(plan.get(key) != value for key, value in config.items()):
        raise r.BenchError('plan differs from frozen component declaration')
    args = plan['server_args']
    if (type(args.get('port')) is not int or not 1 <= args['port'] <= 65535
            or args != dict(PROTOCOL['server_args'], port=args['port'])
            or plan.get('numerical_mode') != PROTOCOL['numerical_mode']
            or plan.get('deadline_seconds', 1) is not None or plan.get('worker_stop_seconds', 1) is not None
            or plan.get('total_requests') != config['blocks'] * len(config['arms']) * 8):
        raise r.BenchError('plan workload, numerical mode, or execution protocol changed')


def schedule_errors(plan, rows, boots):
    expected = [(block, arm, case['id'], rep) for block, order in enumerate(plan['orders'])
                for arm in order for case in plan['cases'] for rep in EXPOSURES]
    actual = [(row.get('pair'), row.get('arm'), row.get('case'), row.get('rep')) for row in rows]
    wanted_boots = [(block, arm) for block, order in enumerate(plan['orders']) for arm in order]
    errors = []
    if actual != expected:
        errors.append('incomplete_or_changed_full_workload_sequence')
    if [(boot.get('pair'), boot.get('arm')) for boot in boots] != wanted_boots:
        errors.append('incomplete_or_changed_boot_sequence')
    if any(not boot.get('completed') or not boot.get('assets_preserved')
           or boot.get('numerical_mode') != 'row-exact' for boot in boots):
        errors.append('boot_failed_or_assets_changed')
    return errors


def backoff_errors(row):
    events = row.get('backoff_events', [])
    enabled = row['arm_definition'].get('backoff') is True
    if not enabled:
        return ['backoff_events_with_control_disabled'] if events else []
    keys = ('window', 'left', 'plain_tokens', 'native_tokens', 'drain_tokens', 'cold_tokens', 'misses', 'resets')
    state = dict.fromkeys(keys, 0)
    errors = []
    for index, receipt in enumerate(events):
        prefix = 'backoff_' + str(index) + ':'
        if (not isinstance(receipt, dict) or receipt.get('family') != 'pld'
                or any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in keys)):
            errors.append(prefix + 'malformed_counters')
            continue
        event = receipt.get('event')
        expected = dict(state)
        if event in ('no_match', 'zero_accepted'):
            if state['left'] != 0:
                errors.append(prefix + 'lookup_during_skip_budget')
            expected['window'] = min(8, max(1, state['window'] * 2))
            expected['left'] = expected['window']
            expected['misses'] += 1
        elif event == 'accepted':
            if state['left'] != 0:
                errors.append(prefix + 'acceptance_during_skip_budget')
            expected.update(window=0, left=0, resets=state['resets'] + 1)
        elif event in ('native', 'drain', 'cold'):
            if state['left'] <= 0 or (event == 'native' and state['left'] <= 1) or (event != 'native' and state['left'] != 1):
                errors.append(prefix + 'emission_outside_exact_budget')
            expected['left'] -= 1
            expected['plain_tokens'] += 1
            expected[event + '_tokens'] += 1
        else:
            errors.append(prefix + 'unknown_event')
        actual = {key: receipt[key] for key in keys}
        if actual != expected:
            errors.append(prefix + 'window_or_cumulative_counter_transition')
        if actual['plain_tokens'] != sum(actual[key] for key in ('native_tokens', 'drain_tokens', 'cold_tokens')):
            errors.append(prefix + 'plain_counter_sum')
        state = actual
    if len(row.get('commits', [])) == 1 and state['plain_tokens'] > len(row['commits'][0].get('generated_ids', [])):
        errors.append('backoff_emissions_exceed_published_output')
    # A request may stop mid-window: unspent left is not a failure or an emission.
    return sorted(set(errors))


def component_receipt_errors(row):
    spec = row['arm_definition']
    errors = backoff_errors(row)
    for probe in row.get('width_probe_events', []):
        if spec['candidate_source'] == 'native' or spec['selection'] not in ('cut', 'adaptive'):
            errors.append('width_probe_in_nonlearning_control')
        if (not isinstance(probe, dict) or probe.get('family') != row['family']
                or probe.get('reason') != 'upper_width_learning'
                or probe.get('reservation_consumed') is not True
                or probe.get('max_forced_executions') != 2
                or type(probe.get('rows')) is not int or not 2 <= probe['rows'] <= 32
                or type(probe.get('class')) is not int
                or row['family'] == 'dflash' and probe.get('variant') not in ('generic', 'linear_gdn')):
            errors.append('invalid_width_learning_receipt')
    if spec['candidate_source'] == 'native':
        return errors
    controlled = 0
    for index, receipt in enumerate(row['rounds']):
        prefix = 'round_' + str(index) + ':'
        if receipt.get('lookup_candidates') is not (spec['candidate_source'] != 'neural'):
            errors.append(prefix + 'candidate_source_control_not_observed')
        if spec['candidate_source'] == 'neural' and any(receipt.get('selected_' + tier, 0) for tier in ('local', 'corpus', 'persistent')):
            errors.append(prefix + 'suffix_candidate_in_neural_only_arm')
        if receipt.get('variant') not in ('generic', 'linear_gdn'):
            errors.append(prefix + 'missing_executed_dispatch_variant')
        if row['family'] == 'mtp' and receipt.get('selection') == 'legacy':
            continue
        controlled += 1
        candidate, legal, actual = (receipt.get(key) for key in ('candidate_rows', 'legal_rows', 'rows'))
        copy = receipt.get('native_copy')
        native_copy = isinstance(copy, dict) and copy.get('selected') is True
        valid = all(type(value) is int for value in (candidate, legal, actual)) and 1 <= actual <= legal <= candidate
        if native_copy:
            errors.extend(prefix + error for error in d.native_copy_receipt_errors(receipt))
            valid = (all(type(value) is int for value in (candidate, legal, actual))
                     and 1 <= legal <= candidate <= 16 and 2 <= actual <= 16
                     and actual == copy.get('rows'))
        if not valid:
            errors.append(prefix + 'illegal_width')
        selection = spec['selection']
        if selection != 'adaptive':
            requested = int(selection.split(':')[1]) if selection.startswith('fixed:') else 0
            if receipt.get('requested_rows') != requested:
                errors.append(prefix + 'requested_width_mismatch')
            if ((requested and (not valid or actual != min(requested, legal)))
                    or (selection == 'max' and (not valid or actual != legal))):
                errors.append(prefix + 'forced_width_mismatch')
            if row['family'] == 'dflash' and receipt.get('assistant') is not True:
                errors.append(prefix + 'forced_assistant_not_observed')
    if row['family'] in ('dflash', 'pld') and not controlled:
        errors.append('missing_controlled_' + row['family'] + '_rounds')
    return errors


def validate_row(row, plan):
    errors = d.check_sample(row)
    spec = plan['arm_definitions'].get(row['arm'])
    case = next((case for case in plan['cases'] if case['id'] == row['case']), None)
    if not spec or not case:
        return errors + ['undeclared_arm_or_case']
    if row['arm_definition'] != spec or row['family'] != spec['family'] or row['mode'] != spec['mode']:
        errors.append('arm_definition_changed')
    if row['body'] != r.request_body(case, spec['mode'], None):
        errors.append('request_body_changed')
    expected_sample = f'{row["pair"]}-{row["arm"]}-{row["case"]}-{row["rep"]}'
    if row['sample'] != expected_sample or row.get('numerical_mode') != 'row-exact':
        errors.append('sample_or_numerical_identity_changed')
    return sorted(set(errors + s.quality(case, row) + component_receipt_errors(row)))


def comparison_groups(plan):
    groups = {}
    for arm, spec in plan['arm_definitions'].items():
        groups.setdefault(spec.get('comparison_arm', arm), []).append(arm)
    return groups


def contrast_effects(plan, rows):
    """One ratio per whole block; stratified cells also resample BLOCKS only."""
    indexed = {(row['pair'], row['arm'], row['case'], row['rep']): row for row in rows}
    groups = comparison_groups(plan)
    results = []
    for contrast in plan['contrasts']:
        before, after = contrast['baseline'], contrast['treatment']
        blocks, cells = [], {case + '/' + label: [] for case in CASES for label in ('first', 'repeat')}
        physical_boots = []
        for block in range(plan['blocks']):
            pairs = [([indexed.get((block, arm, case, rep)) for arm in groups[before]],
                      [indexed.get((block, arm, case, rep)) for arm in groups[after]])
                     for case in CASES for rep in EXPOSURES]
            if any(row is None for left, right in pairs for row in left + right):
                continue
            totals = [(sum(row['seconds'] for row in left), sum(row['seconds'] for row in right))
                      for left, right in pairs]
            blocks.append(sum(left for left, _ in totals) / sum(right for _, right in totals))
            for (case, rep), (left, right) in zip(((case, rep) for case in CASES for rep in EXPOSURES), totals):
                cells[case + ('/first' if rep == -1 else '/repeat')].append(left / right)
            physical = {'block': block}
            for label, members in (('baseline', groups[before]), ('treatment', groups[after])):
                physical[label] = {arm: sum(indexed[block, arm, case, rep]['seconds']
                                           for case in CASES for rep in EXPOSURES) for arm in members}
            physical_boots.append(physical)
        def estimate(values):
            return evidence.bootstrap_effect(values, PROTOCOL['practical_margin'], PROTOCOL['bootstrap_replications'], plan['blocks'])
        effect = estimate(blocks)
        strata = {key: estimate(values) for key, values in cells.items()}
        # Do not average away conflicting-config/prose regressions or first-exposure costs.
        guardrails = len(blocks) == plan['blocks'] and all(
            all(value >= 1 / 1.05 for value in values) for values in cells.values())
        results.append(dict(contrast, full_workload=effect, case_exposure=strata,
                            statistical_unit='complete four-boot ABBA/BAAB quartet' if
                                plan['component'] == 'production-routing' or
                                (plan['component'] in WIDTH_QUARTETS or plan['component'] == 'suffix-backoff') and contrast['role'] == 'primary'
                                else 'complete component block',
                            isolated_width_contrast=plan['component'] in WIDTH_QUARTETS and contrast['role'] == 'primary',
                            estimator='geometric mean of complete-block summed-duration ratios',
                            physical_boot_seconds=physical_boots,
                            every_cell_block_within_5pct=guardrails, performance_pass=False))
    return results


def mechanism_evidence(plan, rows):
    result = {}
    for arm, members in comparison_groups(plan).items():
        spec = plan['arm_definitions'][members[0]]
        own = [row for row in rows if row['arm'] in members]
        rounds = [receipt for row in own for receipt in s.published_rounds(row)
                  if receipt.get('commit_scope') == 'published']
        reduced = {}
        for receipt in rounds:
            copy = receipt.get('native_copy')
            if isinstance(copy, dict) and copy.get('selected') is True:
                continue
            if receipt.get('rows', 0) < receipt.get('legal_rows', 0):
                selection = receipt.get('selection', 'unknown')
                reduced[selection] = reduced.get(selection, 0) + 1
        forced_cut = reduced.get('cut', 0)
        accepted = sum(d.origin_evidence([receipt], 'neural')['accepted_origin_lower_bound'] for receipt in rounds)
        lookup_cells = []
        for row in own:
            if row['case'] in s.RELATED:
                transfer = s.transfer_evidence(row, plan['reuse'])
                lookup_cells.append(sum(span['external_accepted_matching_output_lower_bound']['persistent']
                                        for span in transfer['accepted_span_matches']))
        miss_cells = [row.get('backoff_events', [])[-1] if row.get('backoff_events') else {}
                      for row in own if row['case'] == 'unrelated-prose']
        backoff_observed = any(row.get('backoff_events') and
                               row['backoff_events'][-1].get('plain_tokens', 0) > 0 and
                               row['backoff_events'][-1].get('misses', 0) > 0 for row in own)
        backoff_pass = (spec.get('backoff') is not True or
            len(miss_cells) == plan['blocks'] * len(members) * 2 and
            all(cell.get('plain_tokens', 0) > 0 and cell.get('misses', 0) > 0 for cell in miss_cells))
        result[arm] = {'published_rounds': len(rounds),
            'native_copy_selected_rounds': sum(isinstance(receipt.get('native_copy'), dict) and
                                               receipt['native_copy'].get('selected') is True for receipt in rounds),
            'published_neural_acceptance_lower_bound': accepted,
            'width_reduced_rounds': sum(reduced.values()),
            'width_reduced_rounds_by_selection': reduced,
            'forced_cut_reduced_rounds': forced_cut,
            'width_learning_executions': sum(len(row.get('width_probe_events', [])) for row in own),
            'width_learning_cost': 'Included without subtraction in complete request durations; not a measured width win',
            'related_cell_persistent_training_output_lower_bounds': lookup_cells,
            'source_requirement_pass': (True if spec['candidate_source'] == 'native' else
                accepted > 0 if spec['candidate_source'] == 'neural' else
                len(lookup_cells) == plan['blocks'] * len(members) * len(s.RELATED) * 2 and all(value > 0 for value in lookup_cells)),
            'width_cut_required': spec['family'] != 'pld' and spec['actual'] == 'cut',
            'width_cut_requirement_pass': spec['family'] == 'pld' or spec['actual'] != 'cut' or forced_cut > 0,
            'backoff_required': spec.get('backoff') is True,
            'backoff_requirement_pass': backoff_pass, 'backoff_observed': backoff_observed,
            'unrelated_prose_backoff_endpoints': miss_cells}
    return result


def component_pass(effects, declared):
    primary = [(item['baseline'], item['treatment']) for item in declared if item['role'] == 'primary']
    observed = {(item['baseline'], item['treatment']): item for item in effects}
    return bool(primary) and all(observed.get(key, {}).get('performance_pass') is True for key in primary)


def claim_verdicts(effects, declared):
    observed = {(item['baseline'], item['treatment']): item for item in effects}
    def passed(role):
        keys = [(item['baseline'], item['treatment']) for item in declared if item['role'] == role]
        return bool(keys) and all(observed.get(key, {}).get('performance_pass') is True for key in keys)
    production = any(item['role'] == 'production-primary' for item in declared)
    return {'production_performance_pass': passed('production-primary') if production else None,
            'component_ablation_pass': passed('forced-control') if production else component_pass(effects, declared),
            'performance_pass': passed('production-primary') if production else component_pass(effects, declared)}


def validate_runtime(plan):
    validate_prices(plan)
    identity = r.runtime_identity(plan['binary'])
    if identity != plan.get('runtime_identity'):
        raise r.BenchError('executable, linked runtime, Metal assets, or loading configuration changed')
    return identity


def summarize(out, plan, final=False, supervisor=None):
    rows, parse_errors = d.read_rows(out / 'observations.jsonl')
    boots, boot_errors = d.read_rows(out / 'boots.jsonl')
    failures, failure_parse = d.read_rows(out / 'errors.jsonl')
    errors = list(parse_errors) + list(boot_errors) + list(failure_parse)
    invalid = []
    try:
        validate_plan(plan)
        for boot in boots:
            folder = out / (str(boot['pair']) + '-' + boot['arm'])
            for phase in ('before', 'after'):
                if d.load_json(folder / ('runtime-' + phase + '.json')) != plan['runtime_identity']:
                    errors.append('boot_runtime_identity_changed')
        for row in rows:
            if plan['family'] == 'pld':
                log_path = out / (str(row['pair']) + '-' + row['arm']) / (row['case'] + '-' + str(row['rep'])) / 'receipts.log'
                if r.receipts(log_path.read_text(), 'backoff') != row.get('backoff_events'):
                    errors.append('raw_backoff_receipts_differ_from_observation')
            problems = validate_row(row, plan)
            if problems:
                invalid.append({'sample': row.get('sample'), 'errors': problems})
    except (OSError, ValueError, KeyError, TypeError, r.BenchError) as exc:
        errors.append(str(exc))
    errors.extend(schedule_errors(plan, rows, boots))
    identities = {}
    for row in rows:
        identity = d.identity(row)
        case = row['case']
        if case in identities and identities[case] != identity:
            invalid.append({'sample': row['sample'], 'errors': ['exact_native_output_or_repeat_identity_mismatch']})
        identities[case] = identity
    measurement = evidence.transfer_measurement_errors(boots, plan['measurement_isolation'])
    hardware = telemetry.qualify(out,
        [f'{block}-{arm}-{case["id"]}-{rep}' for block, order in enumerate(plan['orders'])
         for arm in order for case in plan['cases'] for rep in EXPOSURES],
        [(block, arm) for block, order in enumerate(plan['orders']) for arm in order], plan.get('telemetry'),
        boot_states=[boot.get(key, {}) for boot in boots for key in ('machine_before', 'machine_after')])
    joined_boundaries = {item['sample']: item.get('boundary') for item in hardware['requests']}
    for row in rows:
        if row.get('request_boundary') is None or row.get('request_boundary') != joined_boundaries.get(row.get('sample')):
            hardware['errors'].append('observation_request_boundary_missing_or_changed')
    hardware['errors'] = sorted(set(hardware['errors']))
    hardware['qualified'] = not hardware['errors']
    measurement.extend('telemetry:' + error for error in hardware['errors'])
    complete = (not errors and not invalid and not failures and (out / 'worker-complete.json').is_file()
                and (not final or supervisor is not None and supervisor.get('exit_code') == 0 and not supervisor.get('timed_out')))
    safe = not invalid and not parse_errors and all(d.finite(row.get('seconds')) and row['seconds'] > 0 for row in rows)
    effects = contrast_effects(plan, rows) if safe else []
    mechanisms = mechanism_evidence(plan, rows) if safe else {}
    for effect in effects:
        treatment = mechanisms.get(effect['treatment'], {})
        baseline = mechanisms.get(effect['baseline'], {})
        # Multiarm forward/reverse blocks and out-of-quartet native references
        # remain descriptive: do not promote them as isolated width evidence.
        eligible = not (plan['component'] in ('neural-width', 'combined-width') or
                        (plan['component'] in WIDTH_QUARTETS or plan['component'] == 'suffix-backoff') and effect['role'] != 'primary')
        effect['performance_pass'] = bool(eligible and complete and not measurement and effect['every_cell_block_within_5pct']
            and effect['full_workload'].get('verdict') == 'win'
            and treatment.get('source_requirement_pass') and baseline.get('source_requirement_pass')
            and treatment.get('width_cut_requirement_pass') and treatment.get('backoff_requirement_pass')
            and baseline.get('backoff_requirement_pass'))
    result = {'version': 3, 'component': plan['component'], 'family': plan['family'], 'config_hash': plan['hash'],
        'complete': complete, 'samples': len(rows), 'expected_samples': plan['total_requests'],
        'invalid_samples': len(invalid), 'invalid': invalid, 'errors': errors, 'worker_failures': failures,
        'measurement_errors': measurement, 'speed_comparison_valid': complete and not measurement,
        'contrasts': effects, 'mechanisms': mechanisms,
        **claim_verdicts(effects, plan['contrasts']), 'telemetry': hardware,
        'primary_contrasts': [item for item in plan['contrasts'] if item['role'] in ('primary', 'production-primary')],
        'forced_control_results': [item for item in effects if item['role'] == 'forced-control'],
        'native_reference_results': [item for item in effects if item['role'] == 'native-reference'],
        'claim_scope': PROTOCOL['scope'], 'uncertainty': plan['claim_policy'],
        'cost_accounting': PROTOCOL['cost_accounting'],
        'unsupported_components': UNSUPPORTED, 'canonical_speedup_claim': False, 'promotion_pass': False,
        'supervisor': supervisor}
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
        with telemetry.Collector(out, plan['telemetry'], r.machine_state) as collector:
            for block, order in enumerate(plan['orders']):
                for arm in order:
                    folder = out / (str(block) + '-' + arm)
                    folder.mkdir()
                    d.save(folder / 'runtime-before.json', validate_runtime(plan))
                    spec = plan['arm_definitions'][arm]
                    with d.owned_boot(out, folder, plan, args, block, arm, telemetry=collector) as server:
                        for case in plan['cases']:
                            for rep in EXPOSURES:
                                row = d.record_sample(out, folder, server, args, spec, block, arm, case, rep,
                                                      telemetry=collector)
                                receipt_text = (folder / (case['id'] + '-' + str(rep)) / 'receipts.log').read_text()
                                row['backoff_events'] = r.receipts(receipt_text, 'backoff')
                                row['width_probe_events'] = r.receipts(receipt_text, 'width-probe')
                                row['errors'] = validate_row(row, plan)
                                row['phase'] = 'evaluation'
                                row['exposure'] = 'first' if rep == -1 else 'repeat'
                                row['warmup'] = False  # Both first exposure and repeat are measured.
                                row['errors'] = sorted(set(row['errors']))
                                r.append(out / 'observations.jsonl', row)
                                print(json.dumps({'sample': row['sample'], 'seconds': row['seconds'], 'errors': row['errors']}), flush=True)
                                if row['errors']:
                                    raise r.BenchError('invalid component response; no retries: ' + row['sample'])
                    d.save(folder / 'runtime-after.json', validate_runtime(plan))
        d.save(out / 'worker-complete.json', {'completed': True})
        return 0
    except BaseException as exc:
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prices', type=Path, help='Declaration-only reusable schema3 price prior; omit for online-only pricing')
    parser.add_argument('--prices-source', type=Path, help='Declaration-only provenance JSON; required without sibling .source.json or run.json')
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--family', choices=('dflash', 'mtp', 'pld'))
    parser.add_argument('--component', choices=('neural-width', 'combined-width', 'lookup-increment', 'lookup', 'suffix-backoff', 'production-routing') + WIDTH_QUARTETS)
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--fixed-width', type=int)
    parser.add_argument('--isolation', choices=('declared', 'contended', 'unknown'))
    parser.add_argument('--macmon', help='Declaration-only collector executable; default installed macmon')
    parser.add_argument('--telemetry-interval-ms', type=int, help='Declaration-only passive cadence; default 1000')
    parser.add_argument('--telemetry-max-gap-ms', type=int, help='Declaration-only coverage gap bound; default 3000')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        return worker(args.worker)
    if not args.config:
        parser.error('--config is required')
    if args.declare:
        if not args.seed or not args.family or not args.component or not args.isolation or args.plan_only or args.output:
            parser.error('--declare requires explicit --seed, --family, --component, --isolation, and no execution flags')
        args.blocks = (6 if args.family == 'mtp' or args.component == 'production-routing' else 4) if args.blocks is None else args.blocks
        args.fixed_width = 8 if args.fixed_width is None else args.fixed_width
        return declare(args)
    if any(getattr(args, key) is not None for key in ('seed', 'family', 'component', 'blocks', 'fixed_width', 'isolation', 'prices', 'prices_source', 'macmon', 'telemetry_interval_ms', 'telemetry_max_gap_ms')):
        parser.error('execution cannot override the frozen declaration')
    if not args.output or not 1 <= args.port <= 65535:
        parser.error('execution requires new --output and valid --port')
    return d.supervise(args, lambda out: prepare(args, out), __file__, summarize,
                       deadline_seconds=None, worker_stop_seconds=None)


if __name__ == '__main__':
    sys.exit(main())
