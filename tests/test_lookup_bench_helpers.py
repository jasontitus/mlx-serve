"""CPU-only consumer regressions for bounded benchmark qualification and identity."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import lookup_route_diagnostic as route
import lookup_similar_requests as transfer
import lookup_canonical_window as canonical


def machine_state():
    return {'power_source': 'AC', 'power_settings': 'fixed', 'thermal': 'No thermal warning',
            'pressure_level': 1, 'swap_used_bytes': 0, 'swapins': 0, 'swapouts': 0}


def canonical_probe_fixture(opt_in=True, vendor_disable=False):
    captures = []
    def append(request, metrics, identity, thinking=False):
        captures.append({'request_index': len(captures) + 4, 'request_id': identity,
                         'request': request, 'path': '/v1/chat/completions', 'http_status': 200,
                         'metrics': metrics, 'response': {'complete': True, 'text': 'exact'},
                         'raw_response': json.dumps({'choices': [{'message': {
                             'content': '4', 'reasoning_content': 'reason' if thinking else ''}}]})})
    setup = {'messages': [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}],
             'max_completion_tokens': 512, 'temperature': 0}
    append(dict(setup), [], 'detect', thinking=not opt_in)
    if opt_in:
        append(dict(setup, max_completion_tokens=2560, reasoning_effort='low',
                    tools=[{'function': {'name': 'get_weather'}}]), [], 'opt-in')
    append(dict(setup, reasoning_effort='none'), [], 'resolve', thinking=vendor_disable)
    if vendor_disable:
        append(dict(setup, reasoning_effort='none', chat_template_kwargs={'enable_thinking': False}),
               [], 'vendor')
    prompts = {'decode': 'retryWithBackoff<T>', 'drift': 'retryWithBackoff<T>',
               'prefill': 'The archive records the following entry.',
               'predictable': "You are a text normaliser. Output the user's text unchanged.",
               'novel': 'Invent an original, surprising paragraph of stream-of-consciousness prose.',
               'prefix': 'Operations handbook, section on shift handover.',
               'calibration': 'withRetryBudget<T>'}
    for slot in canonical.benchmark_slots():
        stage, index = slot['stage'], slot['index']
        prompt = prompts.get(stage, '')
        if stage == 'context':
            prompt = 'withRetryBudget<T>' if index % 2 == 0 else 'Count from 1 to 200, one number per line'
        elif stage == 'batch':
            prompt = 'Write a short descriptive paragraph. Subject number ' + str(index) + ':'
        tag_index = index // 2 if stage in ('prefix', 'context') else index
        tag = '[probe ' + stage + '-' + str(tag_index) + ']'
        role = 'system' if stage == 'predictable' else 'user'
        request = {'messages': [{'role': role, 'content': prompt + ' ' + tag}],
                   'max_completion_tokens': slot['max_tokens'], 'stream': True}
        identity = stage + ':' + str(0 if stage == 'prefix' else index)
        append(request, list(slot['metrics']), identity)
    return {'responses': captures, 'request_sequence': [c['request_id'] for c in captures],
            'metric_request_counts': {metric: 3 for metric in canonical.METRICS}}


class ResourceAccounting(unittest.TestCase):
    def test_cache_storage_deduplicates_hardlinks_and_rejects_missing_or_external_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            cache.mkdir()
            payload = cache / 'tokens.bin'
            payload.write_bytes(b'1234567')
            os.link(payload, cache / 'alias.bin')
            measured = route.r.storage_size(cache)
            self.assertTrue(measured['complete'])
            self.assertEqual(measured['logical_file_bytes'], 7)
            self.assertEqual(measured['unique_regular_files'], 1)
            self.assertEqual(measured['allocated_file_bytes'], payload.stat().st_blocks * 512)
            (root / 'outside').write_bytes(b'unrelated weights')
            (cache / 'external').symlink_to(root / 'outside')
            measured = route.r.storage_size(cache)
            self.assertFalse(measured['complete'])
            self.assertEqual(measured['logical_file_bytes'], 7)
            self.assertEqual(measured['symlinks_not_followed'], 1)
            self.assertFalse(route.r.storage_size(root / 'missing')['complete'])
            (root / 'linked-root').symlink_to(cache, target_is_directory=True)
            self.assertFalse(route.r.storage_size(root / 'linked-root')['complete'])
            self.assertFalse(route.r.storage_size(payload)['complete'])


class CanonicalWorkloadCoverage(unittest.TestCase):
    def test_complete_body_accepts_each_source_defined_setup_branch(self):
        for opt_in in (False, True):
            for vendor in (False, True):
                probe = canonical_probe_fixture(opt_in, vendor)
                result = canonical.workload_coverage(probe)
                self.assertTrue(result['complete'], result['errors'])
                self.assertEqual(result['benchmark_requests'], 34)
                self.assertEqual(result['setup_requests'], 2 + int(opt_in) + int(vendor))
                self.assertEqual(result['captured_requests'], 36 + int(opt_in) + int(vendor))

    def test_every_auxiliary_and_measured_stage_is_required(self):
        for index in range(len(canonical_probe_fixture()['responses'])):
            probe = canonical_probe_fixture()
            probe['responses'].pop(index)
            probe['request_sequence'] = [c['request_id'] for c in probe['responses']]
            self.assertFalse(canonical.workload_coverage(probe)['complete'], index)

    def test_count_preserving_corruption_and_declared_counts_do_not_prove_coverage(self):
        mutations = [
            lambda p: p['responses'][-1]['request'].update(max_completion_tokens=8),
            lambda p: p['responses'][5].update(metrics=[]),
            lambda p: p['responses'][0].update(metrics=['ttftMs']),
            lambda p: p['responses'][1]['request'].update(reasoning_effort='none'),
            lambda p: p['responses'][3].update(request_index=4),
            lambda p: p['responses'][3].update(http_status=500),
            lambda p: p['request_sequence'].pop(),
            lambda p: p['responses'][-1]['response'].update(complete=False),
        ]
        for mutate in mutations:
            probe = canonical_probe_fixture()
            mutate(probe)
            self.assertFalse(canonical.workload_coverage(probe)['complete'])

    def test_missing_optional_setup_is_rejected_when_response_requires_branch(self):
        for opt_in, vendor, remove in ((True, False, 1), (False, True, 2)):
            probe = canonical_probe_fixture(opt_in, vendor)
            probe['responses'].pop(remove)
            probe['request_sequence'] = [c['request_id'] for c in probe['responses']]
            self.assertFalse(canonical.workload_coverage(probe)['complete'])

    def test_summary_rechecks_structure_even_with_completion_marker(self):
        plan = CanonicalIdentity().plan()
        probe = canonical_probe_fixture()
        probe['responses'].pop()
        receipt = {'before': plan['runtime_identity'], 'after': plan['runtime_identity']}
        boots = [{'completed': True, 'assets_preserved': True, 'numerical_mode': 'row-exact'}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'worker-complete.json').touch()
            with mock.patch.object(route, 'load_json', side_effect=lambda path: probe if path.name == 'probe.json' else receipt), mock.patch.object(route, 'read_rows', side_effect=lambda path: (boots if path.name == 'boots.jsonl' else [], [])), mock.patch.object(route.r, 'runtime_identity', return_value=plan['runtime_identity']), mock.patch.object(route, 'save'):
                result = canonical.summarize(out, plan, final=True, supervisor={'exit_code': 0, 'timed_out': False})
        self.assertFalse(result['complete'])
        self.assertFalse(result['workload_coverage']['complete'])

    def test_worker_rejects_incomplete_body_before_writing_completion_marker(self):
        plan = CanonicalIdentity().plan()
        plan.update(probe_assets={}, probe_cmd=['probe'])
        probe = canonical_probe_fixture()
        probe['responses'].pop()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(mock.patch.object(route, 'load_json', return_value=plan))
            stack.enter_context(mock.patch.object(canonical, 'source_hashes', return_value=plan['source_hashes']))
            stack.enter_context(mock.patch.object(route.r, 'digest', return_value='hash'))
            stack.enter_context(mock.patch.object(route.r, 'runtime_identity', return_value=plan['runtime_identity']))
            stack.enter_context(mock.patch.object(route.r, 'controlled_probe_command', return_value=['probe']))
            stack.enter_context(mock.patch.object(route, 'owned_boot'))
            stack.enter_context(mock.patch.object(route.r, 'preflight_context', return_value={}))
            stack.enter_context(mock.patch.object(route.r, 'canonical_probe', return_value=probe))
            stack.enter_context(mock.patch.object(route.r, 'append'))
            stack.enter_context(mock.patch.object(canonical.signal, 'signal'))
            save = stack.enter_context(mock.patch.object(route, 'save'))
            self.assertEqual(canonical.worker(Path(directory)), 1)
        self.assertFalse(any(call.args[0].name == 'worker-complete.json' for call in save.call_args_list))
        coverage = next(call.args[1] for call in save.call_args_list if call.args[0].name == 'workload-coverage.json')
        self.assertFalse(coverage['complete'])


class NativeCopyReceipts(unittest.TestCase):
    def sample(self, known=False):
        ms, tokens = (12.0, 7.5) if known else (None, None)
        receipt = {
            'family': 'dflash', 'kind': 'tree', 'selection': 'adaptive',
            'lookup_candidates': True, 'candidate_rows': 4, 'legal_rows': 4,
            'requested_rows': 0, 'rows': 8, 'accepted': 7, 'committed': 8,
            'assistant': False, 'selected_local': 7, 'selected_neural': 0,
            'selected_corpus': 0, 'selected_persistent': 0,
            'lookup_order': 26, 'lookup_confidence': 1.0, 'selection_overhead_ms': None,
            'predicted_tokens': tokens, 'executed_geometry_hash': 123,
            'native_copy': {'rows': 8, 'backing_length': 26, 'geometry_hash': 123,
                            'selected': True, 'ms_per_token': ms, 'expected_tokens': tokens},
            'route': {'alternative_source': 'native_copy', 'decision': {
                'skip': True, 'reason': 'native_copy' if known else 'native_copy_unmeasured',
                'assistant_ms': 20.0, 'alternative_ms': ms, 'skipped': 0, 'interval': 12,
                'native_copy': True, 'native_copy_ms': ms, 'native_copy_tokens': tokens}}}
        return {'arm': 'persistent', 'family': 'dflash', 'arm_definition': {'actual': 'adaptive'},
                'response': {'http_status': 200, 'text': 'answer', 'finish_reason': 'stop',
                             'usage': {'prompt_tokens': 1, 'completion_tokens': 8, 'total_tokens': 9}},
                'commits': [{'prompt_ids': [1], 'generated_ids': list(range(2, 10)), 'finish_reason': 'stop'}],
                'rounds': [receipt], 'seconds': 1.0, 'body': {'max_tokens': 32}}

    def test_separate_copy_geometry_and_unknown_prediction_are_explicit(self):
        for known in (False, True):
            self.assertEqual(route.check_sample(self.sample(known)), [])
        sample = self.sample()
        sample['rounds'][0]['predicted_tokens'] = 8
        self.assertIn('round_0:native_copy_prediction_mismatch', route.check_sample(sample))

    def test_copy_claim_requires_matching_execution_identity_and_economics(self):
        changes = [('executed_geometry_hash', 456, 'native_copy_execution_mismatch'),
                   ('assistant', True, 'native_copy_execution_mismatch'),
                   ('rows', 7, 'native_copy_execution_mismatch'),
                   ('selected_neural', 1, 'native_copy_origin_mismatch'),
                   ('native_copy', None, 'missing_native_copy_identity')]
        for key, value, expected in changes:
            with self.subTest(field=key):
                sample = self.sample(True)
                sample['rounds'][0][key] = value
                self.assertIn('round_0:' + expected, route.check_sample(sample))
        sample = self.sample(True)
        sample['rounds'][0]['native_copy']['ms_per_token'] = 8
        self.assertIn('round_0:native_copy_economics_mismatch', route.check_sample(sample))
        sample = self.sample(True)
        sample['rounds'][0]['route']['decision']['alternative_ms'] = 99
        self.assertIn('round_0:native_copy_alternative_price_mismatch', route.check_sample(sample))

    def test_copy_cannot_masquerade_as_a_forced_or_ineligible_action(self):
        for field, value in [('selection', 'fixed'), ('lookup_candidates', False), ('requested_rows', 8)]:
            sample = self.sample()
            sample['rounds'][0][field] = value
            self.assertIn('round_0:native_copy_outside_adaptive_suffix_route', route.check_sample(sample))
        sample = self.sample()
        sample['rounds'][0]['native_copy']['backing_length'] = 1
        self.assertIn('round_0:invalid_native_copy_eligibility', route.check_sample(sample))


class RouteQualification(unittest.TestCase):
    def summarize(self, isolation, changed_state=None):
        arms = ['dflash-legacy', 'mtp-legacy']
        plan = {'baseline_arm': arms[0], 'timed_per_case_per_arm_per_boot': 1,
                'total_requests': 8, 'pairs': 2, 'arms': arms, 'cases': [{'id': 'case'}],
                'scenario': 'baselines', 'family': 'both', 'binary_sha256': 'binary',
                'model': {}, 'native_defaults': 'stock', 'pass_rule': 'diagnostic',
                'measurement_isolation': isolation,
                'arm_definitions': route.arm_definitions('baselines', 'both', arms)}
        rows = [{'sample': str((pair, arm, rep)), 'arm': arm, 'case': 'case',
                 'pair': pair, 'rep': rep, 'seconds': 2 if arm == arms[0] else 1,
                 'errors': [], 'rounds': [], 'commits': [], 'response': {}}
                for pair in range(2) for arm in arms for rep in (-1, 0)]
        boots = [{'arm': arm, 'completed': True, 'assets_preserved': True,
                  'power_before': 'AC', 'power_after': 'AC',
                  'machine_before': machine_state(), 'machine_after': machine_state()}
                 for pair in range(2) for arm in arms]
        if changed_state:
            boots[-1]['machine_after'].update(changed_state)
        documents = {'observations.jsonl': rows, 'boots.jsonl': boots}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'worker-complete.json').touch()
            with mock.patch.object(route, 'read_rows', side_effect=lambda path: (documents.get(path.name, []), [])), mock.patch.object(route, 'save'):
                return route.summarize(out, plan)

    def test_declared_stable_window_can_name_winner(self):
        result = self.summarize('declared')
        self.assertTrue(result['descriptive_timing_valid'])
        self.assertTrue(result['speed_comparison_valid'])
        self.assertEqual(result['baseline_comparison']['fastest_observed_by_case']['case'], 'mtp-legacy')

    def test_missing_contended_or_malformed_isolation_keeps_only_descriptive_timing(self):
        for isolation in (None, 'contended', True, [], {'declared': True}):
            with self.subTest(isolation=isolation):
                result = self.summarize(isolation)
                self.assertTrue(result['complete'])
                self.assertTrue(result['descriptive_timing_valid'])
                self.assertFalse(result['speed_comparison_valid'])
                self.assertFalse(result['performance_pass'])
                self.assertIsNone(result['baseline_comparison']['fastest_observed_by_case']['case'])
                self.assertIn('gpu_isolation_not_declared', result['measurement_exclusions'])
                self.assertEqual(result['summaries'][0]['pairs'][0]['latency_reduction_percent'], 50)

    def test_changed_stability_keeps_timings_but_rejects_winner(self):
        for state in ({'swapouts': 1}, {'thermal': 'CPU_Speed_Limit = 50'}, {'pressure_level': 2}):
            with self.subTest(state=state):
                result = self.summarize('declared', state)
                self.assertTrue(result['descriptive_timing_valid'])
                self.assertFalse(result['speed_comparison_valid'])
                self.assertIsNone(result['baseline_comparison']['fastest_observed_by_case']['case'])

    def test_transfer_wrapper_preserves_route_descriptive_field(self):
        result = self.summarize('contended')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'seed-fixtures').mkdir()
            (out / 'seed-fixtures/declaration.json').touch()
            with mock.patch.object(route, 'summarize', return_value=result), mock.patch.object(route, 'read_rows', return_value=([], [])), mock.patch.object(route, 'save'):
                wrapped = transfer.summarize(out, {'phase': 'training'})
        self.assertTrue(wrapped['descriptive_timing_valid'])
        self.assertFalse(wrapped['measurement_qualified'])
        self.assertFalse(wrapped['speed_comparison_valid'])


class CanonicalIdentity(unittest.TestCase):
    def cohort(self, unbounded=False):
        protocol = canonical.UNBOUNDED_PROTOCOL if unbounded else canonical.PROTOCOL
        return {'version': 2, 'source_hashes': canonical.source_hashes(), 'protocol': copy.deepcopy(protocol),
                'budget': 'unbounded' if unbounded else 'bounded',
                'measurement_isolation': 'declared', 'statistics': copy.deepcopy(canonical.STATISTICS),
                'runtime_identity': {'binary': 'hash', 'libraries': {'libmlx.dylib': 'mlx'},
                                     'metallibs': {'mlx.metallib': 'metal'}},
                'runs': 3, 'rungs': '512', 'quartets': 2, 'arms': list(canonical.ARMS),
                'cache_tag': 'test', 'schedule': canonical.declared_schedule(2)}

    def plan(self, unbounded=False):
        config = self.cohort(unbounded)
        definitions = route.arm_definitions('reuse', 'mtp', ['legacy'])
        route.declare_numerical_mode(definitions, True)
        return {'config': config, 'source_hashes': copy.deepcopy(config['source_hashes']),
                'protocol': copy.deepcopy(config['protocol']), 'numerical_mode': 'row-exact',
                'runtime_identity': copy.deepcopy(config['runtime_identity']),
                'binary': 'host', 'binary_sha256': 'hash',
                'measurement_isolation': config['measurement_isolation'],
                'statistics': copy.deepcopy(config['statistics']),
                'quartets': config['quartets'], 'schedule': copy.deepcopy(config['schedule']),
                'deadline_seconds': config['protocol']['deadline_seconds'],
                'worker_stop_seconds': config['protocol']['worker_stop_seconds'],
                'server_args': dict(config['protocol']['server_args'], port=11279),
                'phase': 'measure', **config['schedule'][0], 'tier': 'legacy',
                'source_identity': {'plan.json': 'hash', 'probe.json': 'hash', 'arm/llmprobe.json': 'hash'},
                'arm_definitions': definitions}

    def test_complete_current_cohort_and_plan_are_accepted(self):
        canonical.validate_plan(self.plan())

    def test_unbounded_measurement_and_replay_accept_full_protocol(self):
        for phase in ('measure', 'replay'):
            plan = self.plan(unbounded=True)
            plan['phase'] = phase
            canonical.validate_plan(plan)

    def test_unbounded_budget_cannot_omit_or_change_declared_timeouts(self):
        for key, value in (('deadline_seconds', 290), ('worker_stop_seconds', 285),
                           ('deadline_seconds', 0), ('worker_stop_seconds', False)):
            with self.subTest(key=key, value=value):
                plan = self.plan(unbounded=True)
                plan[key] = value
                with self.assertRaisesRegex(route.r.BenchError, 'identity mismatch'):
                    canonical.validate_plan(plan)
        for key in ('deadline_seconds', 'worker_stop_seconds'):
            plan = self.plan(unbounded=True)
            del plan[key]
            with self.assertRaisesRegex(route.r.BenchError, 'identity mismatch'):
                canonical.validate_plan(plan)
        for timeout in (240, 0, None, 3601):
            plan = self.plan(unbounded=True)
            plan['server_args']['probe_timeout'] = timeout
            with self.assertRaisesRegex(route.r.BenchError, 'server protocol changed'):
                canonical.validate_plan(plan)
            config = self.cohort(unbounded=True)
            config['protocol']['server_args']['probe_timeout'] = timeout
            with self.assertRaisesRegex(route.r.BenchError, 'protocol changed'):
                canonical.validate_cohort(config)
        config = self.cohort(unbounded=True)
        config['budget'] = 'bounded'
        with self.assertRaisesRegex(route.r.BenchError, 'protocol changed'):
            canonical.validate_cohort(config)

    def test_isolation_statistics_and_schedule_must_be_explicit_and_frozen(self):
        for key in ('measurement_isolation', 'statistics', 'runtime_identity'):
            config = self.cohort()
            del config[key]
            with self.assertRaisesRegex(route.r.BenchError, 'redeclare, never upgrade history'):
                canonical.validate_cohort(config)
        for isolation in (None, True, 'contended', []):
            config = self.cohort()
            config['measurement_isolation'] = isolation
            with self.assertRaisesRegex(route.r.BenchError, 'isolation declaration'):
                canonical.validate_cohort(config)
        for key, value in (('measurement_isolation', 'undeclared'), ('statistics', {}),
                           ('quartets', 4), ('schedule', [])):
            plan = self.plan()
            plan[key] = value
            with self.assertRaisesRegex(route.r.BenchError, 'schedule identity mismatch'):
                canonical.validate_plan(plan)
        config = self.cohort()
        config['quartets'] = 4
        config['schedule'] = canonical.declared_schedule(4)
        canonical.validate_cohort(config)

    def test_declaration_requires_explicit_isolation_and_rejects_old_cli(self):
        for argv in (['canonical', '--declare'], ['canonical', '--isolation', 'declared'],
                     ['canonical', '--quartets', '4'], ['canonical', '--pairs', '4'],
                     ['canonical', '--mode', 'mtp-legacy', '--pair', '0']):
            with mock.patch.object(canonical.sys, 'argv', argv), mock.patch.object(canonical.sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    canonical.main()

    def test_schedule_mutation_and_historical_protocol_are_rejected(self):
        for mutate in (lambda c: c['schedule'].reverse(),
                       lambda c: c['schedule'].pop(),
                       lambda c: c['schedule'].__setitem__(1, c['schedule'][0]),
                       lambda c: c['schedule'][0].update(mode='dflash-legacy'),
                       lambda c: c['schedule'][0].update(slot=True)):
            config = self.cohort()
            mutate(config)
            with self.assertRaisesRegex(route.r.BenchError, 'boot slots'):
                canonical.validate_cohort(config)
        config = self.cohort()
        config.update(version=1, pairs=2, orders=[])
        with self.assertRaisesRegex(route.r.BenchError, 'redeclare, never upgrade history'):
            canonical.validate_cohort(config)

    def test_slot_identity_cannot_alias_another_replica(self):
        for key, value in (('slot', True), ('slot', -1), ('slot', 16), ('slot', 3),
                           ('position', 3), ('quartet', 1), ('family', 'dflash')):
            plan = self.plan()
            plan[key] = value
            with self.assertRaises(route.r.BenchError):
                canonical.validate_plan(plan)

    def test_runtime_identity_is_required_and_frozen_in_plan(self):
        for identity in (None, {}, [], 'old-binary-hash'):
            config = self.cohort()
            config['runtime_identity'] = identity
            with self.assertRaisesRegex(route.r.BenchError, 'runtime identity missing'):
                canonical.validate_cohort(config)
        plan = self.plan()
        plan['runtime_identity']['libraries']['libmlx.dylib'] = 'changed'
        with self.assertRaisesRegex(route.r.BenchError, 'plan runtime identity mismatch'):
            canonical.validate_plan(plan)

    def test_runtime_mutation_with_unchanged_executable_is_rejected(self):
        expected = self.cohort()['runtime_identity']
        for key in ('libraries', 'metallibs'):
            changed = copy.deepcopy(expected)
            changed[key][next(iter(changed[key]))] = 'mutated'
            with mock.patch.object(route.r, 'runtime_identity', return_value=changed):
                with self.assertRaisesRegex(route.r.BenchError, 'linked runtime or loading configuration changed'):
                    canonical.checked_runtime('host', expected)

    def test_prepare_rejects_runtime_mutation_before_loading_seed(self):
        config = self.cohort()
        config['binary_sha256'] = 'hash'
        manifest = {'binary': 'host', 'models': [{'id': 'qwen27b', 'model_dir': canonical.TARGET,
                                                 'drafter_dir': canonical.DRAFTER}]}
        with mock.patch.object(route, 'load_json', side_effect=[config, manifest]) as load, mock.patch.object(route.r, 'absolute_asset', return_value='host'), mock.patch.object(route.r, 'digest', return_value='hash'), mock.patch.object(canonical, 'source_hashes', return_value=config['source_hashes']), mock.patch.object(route.r, 'runtime_identity', return_value={'libraries': 'changed'}):
            with self.assertRaisesRegex(route.r.BenchError, 'linked runtime or loading configuration changed'):
                canonical.prepare(argparse.Namespace(config=Path('cohort.json')), Path('out'))
        self.assertEqual(load.call_count, 2)

    def test_worker_checks_runtime_before_and_after_measurement_and_replay(self):
        for phase in ('measure', 'replay'):
            for mutation_stage in ('none', 'before', 'after'):
                with self.subTest(phase=phase, mutation_stage=mutation_stage), tempfile.TemporaryDirectory() as directory:
                    out = Path(directory)
                    plan = self.plan()
                    plan.update(phase=phase, probe_assets={}, probe_cmd=['probe'],
                                source='source', config_sha256='config', reuse={'hashes': {}}, model={})
                    previous = dict(plan, phase='measure')
                    original = plan['runtime_identity']
                    changed = dict(original, libraries={'libmlx.dylib': 'changed'})
                    identities = [changed if mutation_stage == 'before' else original,
                                  changed if mutation_stage == 'after' else original]
                    documents = {out / 'plan.json': plan, Path('source/plan.json'): previous,
                                 Path('source/probe.json'): canonical_probe_fixture()}
                    with mock.patch.object(canonical, 'source_hashes', return_value=plan['source_hashes']), mock.patch.object(route, 'load_json', side_effect=lambda path: documents[path]), mock.patch.object(route.r, 'runtime_identity', side_effect=identities) as runtime, mock.patch.object(route.r, 'digest', return_value='hash'), mock.patch.object(route.r, 'controlled_probe_command', return_value=['probe']), mock.patch.object(route, 'owned_boot') as boot, mock.patch.object(route.r, 'preflight_context', return_value={}), mock.patch.object(route.r, 'canonical_probe', return_value=canonical_probe_fixture()), mock.patch.object(route.r, 'replay_canonical', return_value={'successful': True, 'supported': True}) as replay, mock.patch.object(route, 'save') as save, mock.patch.object(route.r, 'append'), mock.patch.object(canonical.signal, 'signal'):
                        status = canonical.worker(out)
                    self.assertEqual(status, 0 if mutation_stage == 'none' else 1)
                    self.assertEqual(runtime.call_count, 1 if mutation_stage == 'before' else 2)
                    self.assertEqual(any(call.args[0].name == 'worker-complete.json' for call in save.call_args_list),
                                     mutation_stage == 'none')
                    if mutation_stage == 'before':
                        boot.assert_not_called()
                        replay.assert_not_called()

    def test_summary_requires_runtime_receipts_and_unchanged_live_runtime(self):
        plan = self.plan()
        plan['phase'] = 'replay'
        expected = plan['runtime_identity']
        for receipt, live in (({}, expected), ({'before': expected}, expected),
                              ({'before': expected, 'after': expected}, {'binary': 'changed'}),
                              ({'before': expected, 'after': expected}, expected)):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                (out / 'worker-complete.json').touch()
                with mock.patch.object(route, 'read_rows', return_value=([], [])), mock.patch.object(route, 'load_json', return_value=receipt), mock.patch.object(route.r, 'runtime_identity', return_value=live), mock.patch.object(route, 'save'):
                    result = canonical.summarize(out, plan, final=True, supervisor={'exit_code': 0, 'timed_out': False})
            self.assertEqual(result['complete'], receipt == {'before': expected, 'after': expected} and live == expected)

    def test_prepare_rejects_changed_declared_helper_before_loading_manifest(self):
        config = self.cohort()
        config['source_hashes']['lookup_bench_evidence.py'] = 'changed'
        with mock.patch.object(route, 'load_json', return_value=config) as load:
            with self.assertRaisesRegex(route.r.BenchError, 'identity missing or changed'):
                canonical.prepare(argparse.Namespace(config=Path('cohort.json')), Path('out'))
        self.assertEqual(load.call_count, 1)

    def test_missing_historical_binding_and_changed_or_malformed_protocol_are_rejected(self):
        config = self.cohort()
        del config['source_hashes']
        with self.assertRaisesRegex(route.r.BenchError, 'identity missing or changed'):
            canonical.validate_cohort(config)
        for key, value in (('runs', 2), ('rungs', 512), ('quartets', True), ('protocol', None)):
            with self.subTest(key=key):
                config = self.cohort()
                config[key] = value
                with self.assertRaisesRegex(route.r.BenchError, 'protocol changed or malformed'):
                    canonical.validate_cohort(config)
        config = self.cohort()
        config['protocol']['server_args']['row_exact'] = 1
        with self.assertRaisesRegex(route.r.BenchError, 'protocol changed or malformed'):
            canonical.validate_cohort(config)

    def test_worker_rejects_malformed_protocol_before_any_boot(self):
        plan = self.plan()
        plan['server_args'] = None
        with mock.patch.object(route, 'load_json', return_value=plan), mock.patch.object(route, 'owned_boot') as boot, mock.patch.object(route.r, 'append'), mock.patch.object(canonical.signal, 'signal'), mock.patch.object(canonical.traceback, 'print_exc'):
            self.assertEqual(canonical.worker(Path('out')), 1)
        boot.assert_not_called()

    def test_aggregate_rejects_changed_sources_before_reading_probe(self):
        plan = self.plan()
        plan['source_hashes']['lookup_route_diagnostic.py'] = 'changed'
        with mock.patch.object(route, 'load_json', side_effect=[plan, {'complete': True, 'final': True}]) as load:
            with self.assertRaisesRegex(route.r.BenchError, 'identity mismatch'):
                canonical.aggregate([Path('window')], Path('aggregate'))
        self.assertEqual(load.call_count, 2)


class CanonicalClaims(unittest.TestCase):
    def schedule(self, mutation=None):
        config = {'schedule': canonical.declared_schedule(4)}
        measures, documents = {}, {}
        for slot in config['schedule']:
            index, mode = slot['slot'], slot['mode']
            root = Path(str(index))
            measures[index] = (root, {})
            documents[root / 'boots.jsonl'] = [{
                'pair': index, 'arm': mode.split('-')[1], 'completed': True,
                'assets_preserved': True, 'numerical_mode': 'row-exact',
                'machine_before': dict(machine_state(), timestamp=index * 10 + 1),
                'machine_after': dict(machine_state(), timestamp=index * 10 + 9)}]
        if mutation:
            mutation(documents)
        with mock.patch.object(route, 'read_rows', side_effect=lambda path: (documents[path], [])):
            return canonical.observed_schedule(measures, config)[0]

    def test_actual_boot_timestamps_prove_balance_not_input_path_order(self):
        self.assertTrue(self.schedule()['proved'])

    def test_missing_reordered_overlapping_or_incomplete_boots_cannot_prove_balance(self):
        def reorder(documents):
            first, second = documents[Path('0/boots.jsonl')][0], documents[Path('1/boots.jsonl')][0]
            for key in ('machine_before', 'machine_after'):
                first[key], second[key] = second[key], first[key]
        mutations = [
            reorder,
            lambda docs: docs[Path('0/boots.jsonl')].clear(),
            lambda docs: docs[Path('0/boots.jsonl')][0]['machine_after'].update(timestamp=12),
            lambda docs: docs[Path('0/boots.jsonl')][0]['machine_before'].pop('timestamp'),
            lambda docs: docs[Path('0/boots.jsonl')][0].update(completed=False),
            lambda docs: docs[Path('0/boots.jsonl')][0].update(pair=3),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(self.schedule(mutation)['proved'])

    def comparisons(self, quartets=4, metric='decodeTokPerSec'):
        return [{'quartet': quartet, 'family': 'mtp', 'metric': metric,
                 'family_ratio': 2.0} for quartet in range(quartets)]

    def test_only_family_quartets_support_pointwise_claims(self):
        stats = canonical.comparison_statistics(self.comparisons(), True)
        family = stats['per_family'][0]
        self.assertTrue(family['qualified_speedup'])
        self.assertTrue(family['lower_ci_above_one'])
        self.assertEqual(family['quartets'], 4)
        self.assertNotIn('fastest_native', stats)

    def test_replica_effects_cannot_be_counted_as_extra_quartets(self):
        comparisons = self.comparisons()
        comparisons.append(dict(comparisons[0]))
        with self.assertRaisesRegex(route.r.BenchError, 'duplicate family quartet'):
            canonical.comparison_statistics(comparisons, True)

    def test_schedule_crosses_orientation_and_family_precedence(self):
        schedule = canonical.declared_schedule(4)
        self.assertEqual([slot['slot'] for slot in schedule], list(range(32)))
        for family in ('mtp', 'dflash'):
            combinations = set()
            for quartet in range(4):
                block = [slot for slot in schedule if slot['quartet'] == quartet]
                slots = [slot for slot in block if slot['family'] == family]
                self.assertEqual([slot['position'] for slot in slots], [0, 1, 2, 3])
                self.assertEqual([slot['slot'] for slot in slots],
                                 list(range(slots[0]['slot'], slots[0]['slot'] + 4)))
                pattern = ''.join('A' if slot['mode'].endswith('-legacy') else 'B' for slot in slots)
                self.assertIn(pattern, ('ABBA', 'BAAB'))
                combinations.add((block[0]['family'] == family, pattern))
            self.assertEqual(len(combinations), 4)

    def test_latency_arithmetic_and_rate_geometric_pooling(self):
        slots = canonical.declared_schedule(4)[:4]
        values = {slot['slot']: {'ttftMs': latency, 'decodeTokPerSec': rate}
                  for slot, latency, rate in zip(slots, (1, 4, 8, 9), (1, 4, 16, 9))}
        rows = {row['metric']: row for row in canonical.quartet_comparisons(0, 'mtp', slots, values)}
        self.assertEqual(rows['ttftMs']['native'], 5)
        self.assertEqual(rows['ttftMs']['feature'], 6)
        self.assertAlmostEqual(rows['ttftMs']['family_ratio'], 5 / 6)
        self.assertAlmostEqual(rows['decodeTokPerSec']['native'], 3)
        self.assertAlmostEqual(rows['decodeTokPerSec']['feature'], 8)
        self.assertAlmostEqual(rows['decodeTokPerSec']['family_ratio'], 8 / 3)
        self.assertEqual(rows['decodeTokPerSec']['physical_values']['legacy'], [1, 9])

    def test_two_quartets_unqualified_or_auxiliary_metrics_never_claim(self):
        for comparisons, qualified in ((self.comparisons(2), True),
                                      (self.comparisons(), False),
                                      (self.comparisons(metric='batching:aggregateTokPerSec'), True)):
            stats = canonical.comparison_statistics(comparisons, qualified)
            self.assertFalse(stats['per_family'][0]['qualified_speedup'])
        two = canonical.comparison_statistics(self.comparisons(2), True)['per_family'][0]
        self.assertNotIn('ci95', two)
        self.assertEqual(two['required_quartets'], 4)

    def test_uncertainty_uses_quartets_and_unity_is_not_a_win(self):
        comparisons = self.comparisons()
        for row, ratio in zip(comparisons, (0.5, 0.5, 2, 2)):
            row['family_ratio'] = ratio
        effect = canonical.comparison_statistics(comparisons, True)['per_family'][0]
        self.assertLess(effect['ci95'][0], 1)
        self.assertGreater(effect['ci95'][1], 1)
        self.assertFalse(effect['qualified_speedup'])
        for row in comparisons:
            row['family_ratio'] = 1
        self.assertFalse(canonical.comparison_statistics(comparisons, True)['per_family'][0]['lower_ci_above_one'])

    def test_full_metrics_include_latency_prefix_and_batch_diagnostics(self):
        probe = {'metric_values': {metric: 10 for metric in canonical.METRICS}}
        report = {'bench': {'ttftMs': {'samples': [1, 5, 24]},
                            'prefixCache': {'coldTtftMs': 8, 'warmTtftMs': 4},
                            'batching': {'singleTokPerSec': 20, 'aggregateTokPerSec': 40}}}
        values = canonical.full_metric_values(probe, report)
        self.assertEqual(len(values), 12)
        self.assertEqual(values['ttftMs'], 10)
        self.assertEqual(values['prefixCache:warmTtftMs'], 4)
        self.assertEqual(values['batching:aggregateTokPerSec'], 40)

    def aggregate_fixture(self, directory, mutate=None, live_runtime=None, mutate_paths=None,
                          setup_flags=(True, False)):
        base = Path(directory).resolve()
        config = CanonicalIdentity().cohort(unbounded=True)
        config.update(quartets=4, schedule=canonical.declared_schedule(4))
        documents, boots, paths = {}, {}, []
        captures = canonical_probe_fixture(*setup_flags)['responses']
        for slot in config['schedule']:
            index, mode = slot['slot'], slot['mode']
            root, replay_root = base / str(index), base / ('replay-' + str(index))
            family, tier = mode.split('-')
            plan = CanonicalIdentity().plan(unbounded=True)
            definitions = route.arm_definitions('reuse', family, [tier])
            route.declare_numerical_mode(definitions, True)
            plan.update(config=config, quartets=4, schedule=config['schedule'], **slot,
                        tier=tier, arm_definitions=definitions, config_sha256='config',
                        probe_assets={'client': 'hash'}, reuse={'hashes': {}}, model={})
            multiplier = 2 if tier == 'persistent' else 1
            metric_values = {metric: 10 / multiplier if 'ttft' in metric.lower() else 10 * multiplier
                             for metric in canonical.METRICS}
            probe = {'successful': True, 'responses': copy.deepcopy(captures),
                     'request_sequence': [c['request_id'] for c in captures],
                     'metric_values': metric_values,
                     'metric_request_counts': {metric: 3 for metric in canonical.METRICS},
                     'machine_before': machine_state(), 'machine_after': machine_state(),
                     'machine_samples': [machine_state()], 'stability_errors': [], 'report_errors': []}
            bench = {metric: {'median': metric_values[metric], 'samples': [metric_values[metric]] * 3}
                     for metric in ('decodeTokPerSec', 'ttftMs', 'prefillTokPerSec')}
            bench.update(contextScaling=[{'targetTokens': 512, 'runs': 3, 'inputTokens': 512,
                                          **{metric: metric_values['context:512:' + metric]
                                             for metric in ('decodeTokPerSec', 'ttftMs', 'prefillTokPerSec')}}],
                         speculative={'predictableTokPerSec': metric_values['speculative:predictableTokPerSec'],
                                      'novelTokPerSec': metric_values['speculative:novelTokPerSec'], 'offScript': 0},
                         prefixCache={'coldTtftMs': 10 * multiplier, 'warmTtftMs': 5 * multiplier},
                         batching={'singleTokPerSec': 10, 'aggregateTokPerSec': 40, 'streams': 4},
                         loadDrift={'verdict': 'steady', 'driftPct': 0})
            documents[root / 'plan.json'] = plan
            documents[root / 'summary.json'] = {'complete': True, 'final': True}
            documents[root / 'probe.json'] = probe
            documents[root / 'arm/llmprobe.json'] = {'bench': bench}
            documents[replay_root / 'plan.json'] = dict(plan, phase='replay', source=str(root))
            documents[replay_root / 'summary.json'] = {'complete': True, 'final': True}
            documents[replay_root / 'replay.json'] = {
                'successful': True, 'supported': True, 'captured_requests': len(captures),
                'reference_modes': ['serial'],
                'results': [{'reference_matches': 1} for _ in captures]}
            boots[root / 'boots.jsonl'] = [{
                'pair': index, 'arm': tier, 'completed': True, 'assets_preserved': True,
                'numerical_mode': 'row-exact',
                'machine_before': dict(machine_state(), timestamp=index * 10 + 1),
                'machine_after': dict(machine_state(), timestamp=index * 10 + 9)}]
            paths.extend((root, replay_root))
            for window in (root, replay_root):
                documents[window / 'runtime-identity.json'] = {
                    'before': copy.deepcopy(config['runtime_identity']),
                    'after': copy.deepcopy(config['runtime_identity'])}
                if window == replay_root:
                    documents[window / 'runtime-identity.json'].update(
                        started_at=1000 + index * 10, finished_at=1009 + index * 10)
        if mutate:
            mutate(documents, boots, config)
        if mutate_paths:
            mutate_paths(paths)
        # Keep guard tests cheap; actual bootstrap uncertainty is exercised above.
        def effect(ratios, margin, replications, minimum):
            ratio = ratios[0]
            return {'boot_blocks': len(ratios), 'ratios': ratios, 'ci95': [ratio, ratio],
                    'verdict': 'win' if ratio > 1 else 'inconclusive'}
        with mock.patch.object(route, 'load_json', side_effect=lambda path: documents[path]), mock.patch.object(route, 'read_rows', side_effect=lambda path: (boots[path], [])), mock.patch.object(route, 'save') as save, mock.patch.object(canonical, 'bootstrap_effect', side_effect=effect), mock.patch.object(canonical, 'source_hashes', return_value=config['source_hashes']), mock.patch.object(route.r, 'digest', return_value='hash'), mock.patch.object(route.r, 'runtime_identity', return_value=config['runtime_identity'] if live_runtime is None else live_runtime):
            status = canonical.aggregate(list(reversed(paths)), base / 'aggregate')
        return status, save.call_args.args[1]

    def test_aggregate_rejects_mutated_live_runtime_even_with_valid_saved_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(route.r.BenchError, 'linked runtime or loading configuration changed'):
                self.aggregate_fixture(directory, live_runtime={'libraries': 'mutated'})

    def test_qualified_aggregate_keeps_regressions_separate_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            status, result = self.aggregate_fixture(directory)
        self.assertEqual(status, 0)
        self.assertTrue(result['measurement_qualified'])
        self.assertTrue(result['balanced_schedule']['proved'])
        self.assertFalse(result['promotion_pass'])
        self.assertFalse(result['performance_pass'])
        self.assertFalse(result['reuse_attribution'])
        family = {row['metric']: row for row in result['statistics']['per_family'] if row['family'] == 'mtp'}
        self.assertTrue(family['decodeTokPerSec']['qualified_speedup'])
        self.assertTrue(family['ttftMs']['qualified_speedup'])
        self.assertFalse(family['prefixCache:coldTtftMs']['qualified_speedup'])
        self.assertTrue(family['prefixCache:coldTtftMs']['regression_ci_below_one'])
        self.assertIn('prefixCache:coldTtftMs', {row['metric'] for row in result['full_workload_regressions']['per_family']})
        self.assertEqual(len(result['load_drift']), 32)
        self.assertEqual(len(result['physical_observations']), 32)
        self.assertEqual({row['quartets'] for row in result['statistics']['per_family']}, {4})
        self.assertFalse(result['fastest_native']['supported'])

    def test_aggregate_replays_every_capture_for_variable_setup_branches(self):
        for setup in ((False, False), (True, True)):
            with self.subTest(setup=setup), tempfile.TemporaryDirectory() as directory:
                status, result = self.aggregate_fixture(directory, setup_flags=setup)
            self.assertEqual(status, 0)
            self.assertTrue(result['measurement_qualified'])

    def test_replay_cannot_cover_only_the_benchmark_body(self):
        def mutate(documents, boots, config):
            replay = next(value for path, value in documents.items() if path.name == 'replay.json')
            replay.update(captured_requests=34, results=replay['results'][-34:])
        with tempfile.TemporaryDirectory() as directory:
            status, result = self.aggregate_fixture(directory, mutate)
        self.assertEqual(status, 1)
        self.assertFalse(result['measurement_qualified'])
        self.assertTrue(any(error.startswith('incomplete_full_request_replay:') for error in result['errors']))

    def test_missing_or_duplicate_physical_windows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            status, result = self.aggregate_fixture(directory, mutate_paths=lambda paths: paths.pop(0))
        self.assertEqual(status, 1)
        self.assertFalse(result['measurement_qualified'])
        self.assertIn('missing_declared_measurement_slots', result['errors'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(route.r.BenchError, 'duplicate input window'):
                self.aggregate_fixture(directory, mutate_paths=lambda paths: paths.append(paths[0]))

    def test_replay_cannot_substitute_another_same_arm_replica(self):
        def mutate(documents, boots, config):
            plan = next(value for path, value in documents.items()
                        if path.parent.name == 'replay-0' and path.name == 'plan.json')
            plan.update(config['schedule'][3])
        with tempfile.TemporaryDirectory() as directory:
            status, result = self.aggregate_fixture(directory, mutate)
        self.assertEqual(status, 1)
        self.assertFalse(result['measurement_qualified'])
        self.assertIn('replay_source_slot_or_identity_mismatch:0', result['errors'])

    def test_duplicate_physical_slot_cannot_hide_in_distinct_paths(self):
        def mutate(documents, boots, config):
            plan = next(value for path, value in documents.items()
                        if path.parent.name == '3' and path.name == 'plan.json')
            plan.update(config['schedule'][0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(route.r.BenchError, 'duplicate measurement slot'):
                self.aggregate_fixture(directory, mutate)

    def test_replay_source_bytes_and_adjacency_are_required(self):
        def replay_document(documents, name):
            return next(value for path, value in documents.items()
                        if path.parent.name == 'replay-0' and path.name == name)
        mutations = [
            lambda docs, boots, config: replay_document(docs, 'plan.json')['source_identity'].update(
                {'probe.json': 'changed'}),
            lambda docs, boots, config: replay_document(docs, 'runtime-identity.json').update(
                started_at=20, finished_at=25),
            lambda docs, boots, config: replay_document(docs, 'runtime-identity.json').pop('started_at'),
            lambda docs, boots, config: replay_document(docs, 'runtime-identity.json').update(
                finished_at=1011),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                status, result = self.aggregate_fixture(directory, mutation)
            self.assertEqual(status, 1)
            self.assertFalse(result['measurement_qualified'])

    def test_aggregate_guards_preserve_diagnostics_but_remove_all_claims(self):
        def first(documents, suffix):
            return next(value for path, value in documents.items() if str(path).endswith('/0/' + suffix))
        def undeclared(documents, boots, config):
            config['measurement_isolation'] = 'undeclared'
            for path, plan in documents.items():
                if path.name == 'plan.json':
                    plan['measurement_isolation'] = 'undeclared'
        mutations = [
            undeclared,
            lambda docs, boots, config: next(iter(boots.values()))[0]['machine_after'].update(timestamp=12),
            lambda docs, boots, config: first(docs, 'probe.json')['responses'].pop(),
            lambda docs, boots, config: first(docs, 'probe.json')['machine_after'].update(swapouts=1),
            lambda docs, boots, config: first(docs, 'arm/llmprobe.json')['bench']['loadDrift'].update(verdict='degraded'),
            lambda docs, boots, config: first(docs, 'arm/llmprobe.json')['bench']['batching'].update(streams=3),
            lambda docs, boots, config: next(value for path, value in docs.items() if path.name == 'replay.json')['results'].pop(),
            lambda docs, boots, config: first(docs, 'runtime-identity.json')['after']['libraries'].update({'libmlx.dylib': 'mutated'}),
            lambda docs, boots, config: first(docs, 'runtime-identity.json').pop('before'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                _, result = self.aggregate_fixture(directory, mutation)
                self.assertFalse(result['measurement_qualified'])
                self.assertTrue(result['measurement_exclusions'])
                self.assertFalse(any(row['qualified_speedup'] for rows in result['statistics'].values() for row in rows))


class SupervisorBudgets(unittest.TestCase):
    def supervise(self, *, unbounded=False, failure=None):
        child = mock.Mock(pid=12345, returncode=0)
        clock = mock.Mock(return_value=100)
        waits = []
        def child_wait(timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                clock.return_value = 500
                if failure is not None:
                    raise failure
            return child.returncode
        child.wait.side_effect = child_wait
        budgets = {'deadline_seconds': None, 'worker_stop_seconds': None} if unbounded else {}
        def summary(out, plan, final=False, supervisor=None):
            complete = final and supervisor['exit_code'] == 0 and not supervisor['timed_out']
            return {'complete': complete, 'performance_pass': False, 'invalid_samples': 0,
                    'samples': int(complete)}
        summarize = mock.Mock(side_effect=summary)
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(mock.patch.object(route.time, 'monotonic', clock))
            stack.enter_context(mock.patch.object(route.signal, 'signal'))
            timer = stack.enter_context(mock.patch.object(route.signal, 'setitimer'))
            groups = stack.enter_context(mock.patch.object(route, 'kill_group'))
            spawn = stack.enter_context(mock.patch.object(route.subprocess, 'Popen', return_value=child))
            stack.enter_context(mock.patch.object(route.traceback, 'print_exc'))
            args = argparse.Namespace(output=Path(directory) / 'run', plan_only=False)
            status = route.supervise(args, lambda out: dict(budgets), __file__, summarize, **budgets)
            evidence = route.load_json(args.output / 'bounded-run.json')
        return status, waits, timer, groups, spawn, evidence

    def test_explicit_unbounded_wait_can_finish_after_290_seconds(self):
        status, waits, timer, groups, spawn, evidence = self.supervise(unbounded=True)
        self.assertEqual(status, 0)
        self.assertEqual(waits, [None, 1])
        timer.assert_called_once_with(route.signal.ITIMER_REAL, 0)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        groups.assert_any_call(12345, route.signal.SIGKILL)
        self.assertEqual(evidence['elapsed_seconds'], 400)
        self.assertIsNone(evidence['deadline_seconds'])
        self.assertIsNone(evidence['worker_stop_seconds'])
        self.assertFalse(evidence['timed_out'])

    def test_default_still_arms_290_second_alarm_and_285_second_wait(self):
        status, waits, timer, groups, spawn, evidence = self.supervise()
        self.assertEqual(status, 0)
        self.assertEqual(waits[0], 285)
        self.assertEqual(timer.call_args_list[0], mock.call(route.signal.ITIMER_REAL, 290))
        self.assertEqual(evidence['deadline_seconds'], 290)
        self.assertEqual(evidence['worker_stop_seconds'], 285)

    def test_timeout_cannot_pass_and_terminates_owned_group(self):
        status, waits, timer, groups, spawn, evidence = self.supervise(
            failure=route.subprocess.TimeoutExpired('worker', 285))
        self.assertEqual(status, 1)
        self.assertTrue(evidence['timed_out'])
        groups.assert_any_call(12345, route.signal.SIGTERM)
        groups.assert_any_call(12345, route.signal.SIGKILL)

    def test_unbounded_interrupt_still_reaps_and_kills_owned_group(self):
        status, waits, timer, groups, spawn, evidence = self.supervise(
            unbounded=True, failure=KeyboardInterrupt())
        self.assertEqual(status, 1)
        self.assertEqual(waits, [None, 1])
        groups.assert_any_call(12345, route.signal.SIGKILL)
        timer.assert_called_once_with(route.signal.ITIMER_REAL, 0)

    def test_zero_or_mixed_budgets_are_not_unbounded(self):
        for deadline, worker in ((0, 0), (None, 285), (290, None), (False, False),
                                 (285, 290), (float('inf'), 285)):
            with self.subTest(deadline=deadline, worker=worker):
                with self.assertRaisesRegex(route.r.BenchError, 'positive worker'):
                    route.supervise(None, None, __file__, None,
                                    deadline_seconds=deadline, worker_stop_seconds=worker)


class TransferAggregationIdentity(unittest.TestCase):
    def test_schema2_fails_before_new_compatibility_fields_are_indexed(self):
        with mock.patch.object(route, 'load_json', side_effect=[{'gate_schema': 2}, {'gate_schema': 2}]):
            with self.assertRaisesRegex(route.r.BenchError, 'gate-schema 3'):
                transfer.aggregate([Path('old-window')], None)

    def test_schema3_without_source_binding_is_not_upgraded(self):
        with mock.patch.object(route, 'load_json', side_effect=[{'gate_schema': 3}, {'gate_schema': 3}]):
            with self.assertRaisesRegex(route.r.BenchError, 'complete source/contract/protocol identity'):
                transfer.aggregate([Path('old-window')], None)

    def test_different_helper_contract_or_protocol_cannot_mix(self):
        plan = {'gate_schema': 3, 'phase': 'evaluation', 'family': 'mtp', 'binary_sha256': 'binary',
                'model': {}, 'request_identities': [], 'pairs': 2, 'arms': ['legacy', 'persistent'],
                'both_native_baselines': True, 'pass_rule': 'schema3', 'measurement_isolation': 'declared',
                'request_helper_sha256': 'request', 'source_script_sha256': 'source',
                'shared_driver_sha256': 'driver', 'evidence_helper_sha256': 'evidence',
                'seed_helper_sha256': 'seed-helper', 'contract_sha256': 'contract', 'contract': {},
                'feature_selection': 'adaptive', 'price_trace': False, 'server_args': {'ctx_size': 2048},
                'warmup_per_case_per_arm_per_boot': 1, 'timed_per_case_per_arm_per_boot': 1,
                'deadline_seconds': 290, 'worker_stop_seconds': 285, 'reuse': {'hashes': {}},
                'seed_root': 'seed', 'cases': []}
        summary = {'gate_schema': 3, 'final': True, 'complete': True, 'speed_comparison_valid': True,
                   'power_sources': ['AC']}
        for key in ('source_script_sha256', 'shared_driver_sha256', 'request_helper_sha256',
                    'evidence_helper_sha256', 'seed_helper_sha256', 'contract_sha256', 'contract', 'server_args'):
            with self.subTest(key=key):
                changed = copy.deepcopy(plan)
                changed[key] = 'changed'
                with mock.patch.object(route, 'load_json', side_effect=[plan, changed, summary, summary]), mock.patch.object(transfer, 'load_seed'), mock.patch.object(route, 'read_rows', return_value=([], [])), mock.patch.object(route.r, 'digest', return_value='digest'):
                    with self.assertRaisesRegex(route.r.BenchError, 'helper/contract/protocol'):
                        transfer.aggregate([Path('first-window'), Path('second-window')], None)


if __name__ == '__main__':
    unittest.main()
