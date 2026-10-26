"""CPU-only consumer regressions for bounded benchmark qualification and identity."""
import argparse
from contextlib import ExitStack
import copy
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
        return {'source_hashes': canonical.source_hashes(), 'protocol': copy.deepcopy(protocol),
                'budget': 'unbounded' if unbounded else 'bounded',
                'measurement_isolation': 'declared', 'statistics': copy.deepcopy(canonical.STATISTICS),
                'runtime_identity': {'binary': 'hash', 'libraries': {'libmlx.dylib': 'mlx'},
                                     'metallibs': {'mlx.metallib': 'metal'}},
                'runs': 3, 'rungs': '512', 'pairs': 2, 'arms': list(canonical.ARMS),
                'cache_tag': 'test', 'orders': [list(canonical.ARMS), list(reversed(canonical.ARMS))]}

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
                'pairs': config['pairs'], 'orders': copy.deepcopy(config['orders']),
                'deadline_seconds': config['protocol']['deadline_seconds'],
                'worker_stop_seconds': config['protocol']['worker_stop_seconds'],
                'server_args': dict(config['protocol']['server_args'], port=11279),
                'phase': 'measure', 'mode': 'mtp-legacy', 'pair': 0, 'tier': 'legacy',
                'arm_definitions': definitions}

    def test_complete_current_cohort_and_plan_are_accepted(self):
        canonical.validate_plan(self.plan())
        self.assertTrue({'lookup_canonical_window.py', 'lookup_similar_requests.py',
                         'lookup_route_diagnostic.py', 'bench_lookup_staircut.py',
                         'lookup_bench_evidence.py', 'prepare_lookup_offline.py',
                         'lookup_transfer_contract.json'} <= set(canonical.source_hashes()))

    def test_unbounded_measurement_and_replay_keep_full_protocol(self):
        for phase in ('measure', 'replay'):
            plan = self.plan(unbounded=True)
            plan['phase'] = phase
            canonical.validate_plan(plan)
            self.assertIsNone(plan['deadline_seconds'])
            self.assertIsNone(plan['worker_stop_seconds'])
            expected = dict(canonical.PROTOCOL['server_args'], probe_timeout=3600, port=11279)
            self.assertEqual(plan['server_args'], expected)

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
                           ('pairs', 4), ('orders', [])):
            plan = self.plan()
            plan[key] = value
            with self.assertRaisesRegex(route.r.BenchError, 'schedule identity mismatch'):
                canonical.validate_plan(plan)
        config = self.cohort()
        config['pairs'] = 4
        config['orders'] *= 2
        canonical.validate_cohort(config)

    def test_declaration_requires_isolation_and_defaults_to_four_pairs(self):
        for pairs in (None, 2):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'cohort.json'
                argv = ['canonical', '--declare', '--config', str(path), '--seed', directory,
                        '--isolation', 'declared', '--unbounded']
                if pairs is not None:
                    argv += ['--pairs', str(pairs)]
                with mock.patch.object(canonical.sys, 'argv', argv), mock.patch.object(route, 'load_json', return_value={'binary': 'host'}), mock.patch.object(route.r, 'absolute_asset', return_value='host'), mock.patch.object(route.r, 'digest', return_value='hash'), mock.patch.object(route.r, 'runtime_identity', return_value={'binary': 'hash', 'libraries': {'mlx': 'hash'}}), mock.patch.object(route, 'save') as save:
                    self.assertEqual(canonical.main(), 0)
                config = save.call_args.args[1]
                self.assertEqual(config['pairs'], 4 if pairs is None else pairs)
                self.assertEqual(len(config['orders']), config['pairs'])
                self.assertEqual(config['measurement_isolation'], 'declared')
                self.assertEqual(config['statistics'], canonical.STATISTICS)
                self.assertEqual(config['runtime_identity'], {'binary': 'hash', 'libraries': {'mlx': 'hash'}})
        for argv in (['canonical', '--declare'], ['canonical', '--isolation', 'declared'],
                     ['canonical', '--pairs', '4']):
            with mock.patch.object(canonical.sys, 'argv', argv), mock.patch.object(canonical.sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    canonical.main()

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
                                 Path('source/probe.json'): {}}
                    with mock.patch.object(canonical, 'source_hashes', return_value=plan['source_hashes']), mock.patch.object(route, 'load_json', side_effect=lambda path: documents[path]), mock.patch.object(route.r, 'runtime_identity', side_effect=identities) as runtime, mock.patch.object(route.r, 'digest', return_value='hash'), mock.patch.object(route.r, 'controlled_probe_command', return_value=['probe']), mock.patch.object(route, 'owned_boot') as boot, mock.patch.object(route.r, 'preflight_context', return_value={}), mock.patch.object(route.r, 'canonical_probe', return_value={}), mock.patch.object(route.r, 'replay_canonical', return_value={'successful': True, 'supported': True}) as replay, mock.patch.object(route, 'save') as save, mock.patch.object(route.r, 'append'), mock.patch.object(canonical.signal, 'signal'):
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
        for key, value in (('runs', 2), ('rungs', 512), ('pairs', True), ('protocol', None)):
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
        config = {'orders': [list(canonical.ARMS), list(reversed(canonical.ARMS))] * 2}
        measures, documents = {}, {}
        for index, (pair, mode) in enumerate((pair, mode) for pair, order in enumerate(config['orders']) for mode in order):
            root = Path(str(index))
            measures[(pair, mode)] = (root, {})
            documents[root / 'boots.jsonl'] = [{
                'pair': pair, 'arm': mode.split('-')[1], 'completed': True,
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

    def comparisons(self, pairs=4, metric='decodeTokPerSec'):
        return [{'pair': pair, 'family': 'mtp', 'metric': metric,
                 'family_ratio': 2.0, 'fastest_native_ratio': 0.8} for pair in range(pairs)]

    def test_separate_family_and_fastest_native_pointwise_claims(self):
        stats = canonical.comparison_statistics(self.comparisons(), True)
        family, fastest = stats['per_family'][0], stats['fastest_native'][0]
        self.assertTrue(family['qualified_speedup'])
        self.assertTrue(family['lower_ci_above_one'])
        self.assertEqual(family['boot_blocks'], 4)
        self.assertFalse(fastest['qualified_speedup'])
        self.assertTrue(fastest['regression_ci_below_one'])
        self.assertTrue(fastest['observed_regression'])

    def test_two_pairs_unqualified_or_auxiliary_metrics_never_claim(self):
        for comparisons, qualified in ((self.comparisons(2), True),
                                      (self.comparisons(), False),
                                      (self.comparisons(metric='batching:aggregateTokPerSec'), True)):
            stats = canonical.comparison_statistics(comparisons, qualified)
            self.assertFalse(stats['per_family'][0]['qualified_speedup'])
        two = canonical.comparison_statistics(self.comparisons(2), True)['per_family'][0]
        self.assertNotIn('ci95', two)
        self.assertEqual(two['required_boot_blocks'], 4)

    def test_uncertainty_uses_paired_blocks_and_unity_is_not_a_win(self):
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
        report = {'bench': {'prefixCache': {'coldTtftMs': 8, 'warmTtftMs': 4},
                            'batching': {'singleTokPerSec': 20, 'aggregateTokPerSec': 40}}}
        values = canonical.full_metric_values(probe, report)
        self.assertEqual(len(values), 12)
        self.assertEqual(values['ttftMs'], 10)
        self.assertEqual(values['prefixCache:warmTtftMs'], 4)
        self.assertEqual(values['batching:aggregateTokPerSec'], 40)

    def aggregate_fixture(self, directory, mutate=None, live_runtime=None):
        base = Path(directory).resolve()
        config = CanonicalIdentity().cohort(unbounded=True)
        config.update(pairs=4, orders=config['orders'] * 2)
        documents, boots, paths = {}, {}, []
        groups = [('decodeTokPerSec', 'ttftMs'), ('prefillTokPerSec',),
                  ('speculative:predictableTokPerSec',), ('speculative:novelTokPerSec',),
                  ('context:512:decodeTokPerSec', 'context:512:ttftMs', 'context:512:prefillTokPerSec')]
        captures = [{'request_id': str(index), 'request_index': index, 'metrics': [],
                     'response': {'complete': True, 'text': 'exact'}} for index in range(34)]
        for index, metrics in enumerate(groups):
            for rep in range(3):
                captures[index * 3 + rep]['metrics'] = list(metrics)
        for index, (pair, mode) in enumerate((pair, mode) for pair, order in enumerate(config['orders']) for mode in order):
            root, replay_root = base / str(index), base / ('replay-' + str(index))
            family, tier = mode.split('-')
            plan = CanonicalIdentity().plan(unbounded=True)
            definitions = route.arm_definitions('reuse', family, [tier])
            route.declare_numerical_mode(definitions, True)
            plan.update(config=config, pairs=4, orders=config['orders'], pair=pair, mode=mode,
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
            bench = {metric: {'median': metric_values[metric], 'samples': [10] * 3}
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
                'successful': True, 'supported': True, 'captured_requests': 34,
                'reference_modes': ['serial'],
                'results': [{'reference_matches': 1} for _ in captures]}
            boots[root / 'boots.jsonl'] = [{
                'pair': pair, 'arm': tier, 'completed': True, 'assets_preserved': True,
                'numerical_mode': 'row-exact',
                'machine_before': dict(machine_state(), timestamp=index * 10 + 1),
                'machine_after': dict(machine_state(), timestamp=index * 10 + 9)}]
            paths.extend((root, replay_root))
            for window in (root, replay_root):
                documents[window / 'runtime-identity.json'] = {
                    'before': copy.deepcopy(config['runtime_identity']),
                    'after': copy.deepcopy(config['runtime_identity'])}
        if mutate:
            mutate(documents, boots, config)
        # Keep guard tests cheap; actual bootstrap uncertainty is exercised above.
        def effect(ratios, margin, replications, minimum):
            ratio = ratios[0]
            return {'boot_blocks': len(ratios), 'ratios': ratios, 'ci95': [ratio, ratio],
                    'verdict': 'win' if ratio > 1 else 'inconclusive'}
        with mock.patch.object(route, 'load_json', side_effect=lambda path: documents[path]), mock.patch.object(route, 'read_rows', side_effect=lambda path: (boots[path], [])), mock.patch.object(route, 'save') as save, mock.patch.object(canonical, 'bootstrap_effect', side_effect=effect), mock.patch.object(route.r, 'runtime_identity', return_value=config['runtime_identity'] if live_runtime is None else live_runtime):
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
        self.assertEqual(len(result['load_drift']), 16)

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
