"""CPU-only consumer regressions for bounded benchmark qualification and identity."""
import argparse
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
    def cohort(self):
        return {'source_hashes': canonical.source_hashes(), 'protocol': copy.deepcopy(canonical.PROTOCOL),
                'runs': 3, 'rungs': '512', 'pairs': 2, 'arms': list(canonical.ARMS),
                'cache_tag': 'test', 'orders': [list(canonical.ARMS), list(reversed(canonical.ARMS))]}

    def plan(self):
        config = self.cohort()
        definitions = route.arm_definitions('reuse', 'mtp', ['legacy'])
        route.declare_numerical_mode(definitions, True)
        return {'config': config, 'source_hashes': copy.deepcopy(config['source_hashes']),
                'protocol': copy.deepcopy(config['protocol']), 'numerical_mode': 'row-exact',
                'deadline_seconds': 290, 'worker_stop_seconds': 285,
                'server_args': dict(canonical.PROTOCOL['server_args'], port=11279),
                'phase': 'measure', 'mode': 'mtp-legacy', 'pair': 0, 'tier': 'legacy',
                'arm_definitions': definitions}

    def test_complete_current_cohort_and_plan_are_accepted(self):
        canonical.validate_plan(self.plan())
        self.assertTrue({'lookup_canonical_window.py', 'lookup_similar_requests.py',
                         'lookup_route_diagnostic.py', 'bench_lookup_staircut.py',
                         'lookup_bench_evidence.py', 'prepare_lookup_offline.py',
                         'lookup_transfer_contract.json'} <= set(canonical.source_hashes()))

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
