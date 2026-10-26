"""Deterministic component-v2 telemetry contracts; no subprocesses or inference."""
import copy
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import lookup_component_bench as c
import lookup_gpu_telemetry as t


def declaration():
    return dict(t.RULES, binary='/macmon', binary_sha256='a' * 64, version='macmon frozen',
                command=['/macmon', 'pipe', '--soc-info', '-i', '1000'],
                interval_ms=1000, max_gap_ms=3000)


def point(value):
    return {'monotonic': value, 'wall_unix': 1704067200 + value}


def machine():
    return {'power_source': 'ac', 'pressure_level': 1, 'swap_used_bytes': 0,
            'swapins': 0, 'swapouts': 0, 'power_settings': 'frozen AC settings',
            'thermal': 'No thermal warning'}


def endpoint(value):
    return {'collection_start': point(value), 'collection_end': point(value + .1), 'machine': machine()}


def raw_sample(second):
    return {'timestamp': datetime.fromtimestamp(1704067200 + second, timezone.utc).isoformat(), 'gpu_freq_mhz': 900,
            'gpu_scaled_ratio': .8, 'temp': {'gpu_temp_avg': 60}, 'gpu_power': 35,
            'fans': [{'name': 'fan0', 'rpm': 1500, 'max_rpm': 4000}]}


class TelemetryQualification(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.root = self.out / 'telemetry'
        self.root.mkdir()
        self.config = declaration()
        self.state = {'declaration': self.config, 'errors': [], 'started': point(0),
                      'stopped': point(11), 'stop_requested': point(9),
                      'returncode': -15, 'owned_stop': True}
        self.samples = [dict(point(i), raw_line=json.dumps(raw_sample(i)) + '\n',
                             metrics=t.metrics(raw_sample(i))) for i in range(11)]
        self.boundaries = [{'sample': 'request', 'start': point(2.2), 'end': point(4.8),
                            'before': endpoint(2), 'after': endpoint(4.8),
                            'endpoint_collection_seconds': .2, 'error': None}]
        self.phases = [dict(pair=0, arm='native', phase=name, **endpoint(value))
                       for name, value in zip(('prelaunch', 'ready', 'pre_shutdown', 'post_shutdown'),
                                              (.5, 1, 6, 8))]

    def persist(self):
        t.save(self.root / 'collector.json', self.state)
        for filename, rows in (('receipts.jsonl', self.samples), ('requests.jsonl', self.boundaries),
                               ('boot-phases.jsonl', self.phases)):
            (self.root / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.root / 'macmon.stdout.jsonl').write_text(''.join(row['raw_line'] for row in self.samples))

    def result(self):
        self.persist()
        return t.qualify(self.out, ['request'], [(0, 'native')], self.config)

    def test_request_join_preserves_raw_frequency_temperature_power_and_fans(self):
        result = self.result()
        self.assertTrue(result['qualified'], result['errors'])
        joined = result['requests'][0]['samples']
        self.assertEqual([row['monotonic'] for row in joined], [2, 3, 4, 5])
        self.assertEqual(joined[1]['metrics']['gpu_frequency_mhz'], 900)
        self.assertEqual(joined[1]['metrics']['gpu_temperature_c'], 60)
        self.assertEqual(joined[1]['metrics']['gpu_power_w'], 35)
        self.assertEqual(joined[1]['metrics']['fans'][0]['rpm'], 1500)
        self.assertEqual(joined[1]['metrics']['device_frequency_limits_status'], 'unsupported')

    def test_missing_malformed_dead_and_gapped_fail_closed(self):
        baseline = copy.deepcopy((self.samples, self.state, self.boundaries, self.phases))
        for failure in ('missing', 'malformed', 'dead', 'gap', 'source_gap', 'endpoint', 'request', 'phase'):
            self.samples, self.state, self.boundaries, self.phases = copy.deepcopy(baseline)
            if failure == 'missing':
                self.samples = []
            elif failure == 'malformed':
                self.samples[3]['raw_line'] = '{malformed\n'
            elif failure == 'dead':
                self.state['owned_stop'] = False
                self.state['returncode'] = 1
            elif failure == 'gap':
                self.samples = [row for row in self.samples if row['monotonic'] not in (3, 4, 5)]
            elif failure == 'source_gap':
                self.samples[3]['metrics']['source_wall_unix'] += 10
            elif failure == 'endpoint':
                self.boundaries[0]['after'] = dict(endpoint(4.8), error='machine collection failed')
            elif failure == 'request':
                self.boundaries[0]['error'] = 'timeout'
            elif failure == 'phase':
                self.phases.pop()
            with self.subTest(failure=failure):
                self.assertFalse(self.result()['qualified'])

    def test_absent_artifact_and_raw_receipt_mutation_fail(self):
        self.persist()
        (self.root / 'macmon.stdout.jsonl').write_text('changed')
        self.assertFalse(t.qualify(self.out, ['request'], [(0, 'native')], self.config)['qualified'])
        (self.root / 'collector.json').unlink()
        self.assertFalse(t.qualify(self.out, ['request'], [(0, 'native')], self.config)['qualified'])

    def test_each_request_needs_bracket_and_internal_sample(self):
        boundary = self.boundaries[0]
        for start, end in ((-1, 2), (9, 11), (2.1, 2.9)):
            changed = dict(boundary, start=point(start), end=point(end))
            self.assertTrue(t.join_request(changed, self.samples, 3)['errors'])
        second = dict(boundary, sample='second', start=point(6.2), end=point(8.8))
        joined = t.join_request(second, self.samples, 3)
        self.assertEqual([row['monotonic'] for row in joined['samples']], [6, 7, 8, 9])

    def test_unsupported_sensors_are_not_normal_defaults(self):
        for key, value in (('fans', []), ('gpu_freq_mhz', None), ('gpu_power', float('nan')),
                           ('temp', {'gpu_temp_avg': None})):
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                t.metrics(dict(raw_sample(0), **{key: value}))

    def test_active_residency_is_optional_and_never_replaced_by_scaled_ratio(self):
        for value, status in ((0, 'available'), (.25, 'available'), (1, 'available'), (None, 'unsupported')):
            normalized = t.metrics(dict(raw_sample(0), gpu_active_ratio=value))
            self.assertEqual(normalized['gpu_utilization_active_ratio'], value)
            self.assertEqual(normalized['gpu_utilization_active_ratio_status'], status)
            self.assertEqual(normalized['gpu_utilization_scaled_ratio'], .8)
        absent = t.metrics(raw_sample(0))
        self.assertIsNone(absent['gpu_utilization_active_ratio'])
        self.assertEqual(absent['gpu_utilization_active_ratio_status'], 'not_reported')
        for value in (-.01, 1.01, True, '0.5', float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                t.metrics(dict(raw_sample(0), gpu_active_ratio=value))

    def test_all_endpoint_states_share_existing_boot_invariants(self):
        for target in ('before', 'after', 'prelaunch', 'ready', 'pre_shutdown', 'post_shutdown'):
            for key, value in (('pressure_level', 2), ('power_source', 'battery'),
                               ('thermal', 'Thermal warning'), ('thermal', None),
                               ('power_settings', None), ('swapins', 1), ('swapouts', 1),
                               ('swap_used_bytes', 4096)):
                points = {**{name: self.boundaries[0][name] for name in ('before', 'after')},
                          **{phase['phase']: phase for phase in self.phases}}
                snapshot = points[target]['machine']
                original = dict(snapshot)
                snapshot[key] = value
                with self.subTest(endpoint=target, key=key, value=value):
                    self.assertFalse(self.result()['qualified'])
                snapshot.clear()
                snapshot.update(original)
        # Consistently changed endpoint states must also disagree with normal boot envelopes.
        for endpoint_ in self.phases + [self.boundaries[0]['before'], self.boundaries[0]['after']]:
            endpoint_['machine']['power_source'] = 'battery'
        self.persist()
        result = t.qualify(self.out, ['request'], [(0, 'native')], self.config, boot_states=[machine(), machine()])
        self.assertIn('machine_state:power_source_changed', result['errors'])


class ProductionEndpointGate(unittest.TestCase):
    def test_transient_and_missing_endpoint_states_block_otherwise_winning_production(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'telemetry'
            root.mkdir()
            config = declaration()
            definitions, contrasts = c.component_matrix('mtp', 'lookup', 8)
            plan = {'family': 'mtp', 'component': 'lookup', 'blocks': 6,
                    'arms': list(definitions), 'arm_definitions': definitions, 'contrasts': contrasts,
                    'orders': c.component_orders('mtp', 'lookup', list(definitions), 6),
                    'cases': [{'id': case} for case in c.CASES], 'telemetry': config,
                    'measurement_isolation': 'declared', 'runtime_identity': {'fixture': 'runtime'},
                    'hash': 'synthetic-gate-fixture', 'claim_policy': 'synthetic-gate-fixture',
                    'total_requests': 144}
            boots, rows, boundaries, phases = [], [], [], []
            for block, order in enumerate(plan['orders']):
                for arm in order:
                    base = len(boots) * 22
                    folder = out / f'{block}-{arm}'
                    folder.mkdir()
                    for phase in ('before', 'after'):
                        t.save(folder / f'runtime-{phase}.json', plan['runtime_identity'])
                    boots.append({'pair': block, 'arm': arm, 'completed': True, 'assets_preserved': True,
                                  'numerical_mode': 'row-exact',
                                  'machine_before': machine(), 'machine_after': machine()})
                    phases.extend(dict(pair=block, arm=arm, phase=name, **endpoint(base + offset))
                                  for name, offset in zip(('prelaunch', 'ready', 'pre_shutdown', 'post_shutdown'),
                                                          (.5, 1, 19, 20)))
                    for index, (case, rep) in enumerate((case, rep) for case in c.CASES for rep in c.EXPOSURES):
                        start = base + 2.2 + 2 * index
                        end = start + 1.6
                        sample = f'{block}-{arm}-{case}-{rep}'
                        boundary = {'sample': sample, 'start': point(start), 'end': point(end),
                                    'before': endpoint(start - .2), 'after': endpoint(end),
                                    'endpoint_collection_seconds': .2, 'error': None}
                        boundaries.append(boundary)
                        rows.append({'sample': sample, 'pair': block, 'arm': arm, 'case': case,
                                     'rep': rep, 'seconds': 1, 'request_boundary': boundary})
            samples = [dict(point(i), raw_line=json.dumps(raw_sample(i)) + '\n',
                            metrics=t.metrics(raw_sample(i))) for i in range(398)]
            t.save(root / 'collector.json', {'declaration': config, 'errors': [], 'started': point(0),
                'stopped': point(399), 'stop_requested': point(396), 'returncode': -15, 'owned_stop': True})
            (root / 'macmon.stdout.jsonl').write_text(''.join(row['raw_line'] for row in samples))
            (root / 'receipts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in samples))
            (out / 'boots.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in boots))
            (out / 'errors.jsonl').write_text('')
            t.save(out / 'worker-complete.json', {'completed': True})
            def summarize():
                for path, data in ((root / 'requests.jsonl', boundaries), (root / 'boot-phases.jsonl', phases),
                                   (out / 'observations.jsonl', rows)):
                    path.write_text(''.join(json.dumps(row) + '\n' for row in data))
                return c.summarize(out, plan)
            effects = [dict(contrast, full_workload={'verdict': 'win'}, every_cell_block_within_5pct=True)
                       for contrast in contrasts]
            mechanisms = {arm: {'source_activated': True, 'width_cut_activated': True, 'backoff_activated': True}
                          for arm in definitions}
            # Isolate the measurement gate; schedule, original swap guards, telemetry joins,
            # summary effect gates, and final production verdict run their real implementations.
            with mock.patch.object(c, 'validate_plan'), mock.patch.object(c, 'validate_row', return_value=[]), \
                    mock.patch.object(c.d, 'identity', return_value='exact-fixture'), \
                    mock.patch.object(c, 'contrast_effects', side_effect=lambda *args: copy.deepcopy(effects)), \
                    mock.patch.object(c, 'mechanism_evidence', return_value=mechanisms):
                self.assertEqual(c.evidence.transfer_measurement_errors(boots, 'declared'), [])
                clean = summarize()
                self.assertTrue(clean['production_performance_pass'], clean['measurement_errors'])
                targets = {'request-before': boundaries[0]['before'], 'request-after': boundaries[0]['after'],
                           **{phase['phase']: phase for phase in phases[:4]}}
                for target, snapshot in targets.items():
                    original = copy.deepcopy(snapshot['machine'])
                    for key, value in (('pressure_level', 2), ('power_source', 'battery'),
                                       ('thermal', 'Thermal warning'), ('thermal', None),
                                       ('power_settings', None)):
                        snapshot['machine'] = dict(original, **{key: value})
                        with self.subTest(endpoint=target, key=key, value=value):
                            result = summarize()
                            self.assertTrue(result['complete'])
                            self.assertFalse(result['production_performance_pass'])
                            self.assertFalse(result['performance_pass'])
                            self.assertFalse(result['speed_comparison_valid'])
                            self.assertTrue(any(error.startswith('telemetry:machine_state:')
                                                for error in result['measurement_errors']))
                    snapshot['machine'] = {}
                    with self.subTest(endpoint=target, missing='entire machine state'):
                        result = summarize()
                        self.assertTrue(result['complete'])
                        self.assertFalse(result['production_performance_pass'])
                        self.assertIn('telemetry:machine_state:thermal_unknown', result['measurement_errors'])
                    snapshot['machine'] = original
                self.assertTrue(summarize()['production_performance_pass'])


class RequestBoundaries(unittest.TestCase):
    def exercise(self, failure=False, collector_enabled=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'server.log'
            log.write_text('')
            server = SimpleNamespace(log_path=log, numerical_mode='row-exact', assert_owned=lambda: None)
            clock = [0]
            def snapshot():
                clock[0] += 2
                return machine()
            def request(*args):
                clock[0] += 5
                if failure:
                    raise RuntimeError('request failed')
                return {'total_seconds': 5, 'text': 'response'}
            collector = t.Collector(root, declaration(), snapshot) if collector_enabled else None
            if collector:
                collector.root.mkdir()
            original_http = c.r.HTTP
            with mock.patch.object(t, 'stamp', side_effect=lambda: point(clock[0])), \
                    mock.patch.object(c.r, 'request_body', return_value={}), \
                    mock.patch.object(c.r, 'request', side_effect=request), \
                    mock.patch.object(c.r, 'settled_log', return_value=''), \
                    mock.patch.object(c.r, 'parse_stats', return_value={'commits': [], 'rounds': []}), \
                    mock.patch.object(c.d, 'check_sample', return_value=[]):
                args = (root, root, server, SimpleNamespace(request_timeout=1),
                        {'family': 'mtp', 'mode': 'mtp-legacy'}, 0, 'native', {'id': 'case'}, -1)
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'request failed'):
                        c.d.record_sample(*args, telemetry=collector)
                    row = None
                else:
                    row = c.d.record_sample(*args, telemetry=collector)
            self.assertIs(c.r.HTTP, original_http)
            if collector:
                boundary = json.loads((root / 'case--1/request-boundary.json').read_text())
                self.assertEqual(boundary['start']['monotonic'], 2)
                self.assertEqual(boundary['end']['monotonic'], 7)
                self.assertEqual(boundary['endpoint_collection_seconds'], 4)
                self.assertEqual(boundary['after']['collection_end']['monotonic'], 9)
                self.assertEqual(t.read_jsonl(collector.root / 'requests.jsonl'), [boundary])
                if failure:
                    self.assertIn('request failed', boundary['error'])
                else:
                    self.assertEqual(row['seconds'], 5)
                    self.assertEqual(row['response']['total_seconds'], 5)
                    self.assertEqual(row['request_boundary'], boundary)
            else:
                self.assertNotIn('request_boundary', row)
                self.assertFalse((root / 'telemetry').exists())

    def test_collection_overhead_is_outside_response_latency(self):
        self.exercise()

    def test_request_exception_preserves_post_state_and_restores_http(self):
        self.exercise(failure=True)

    def test_existing_driver_needs_no_collector(self):
        self.exercise(collector_enabled=False)


class BootBoundaries(unittest.TestCase):
    def test_failure_still_records_shutdown_and_preserves_original_swap_envelope(self):
        class Server:
            numerical_mode = 'row-exact'
            proc = SimpleNamespace(returncode=0)
            command = ['server']
            env = {'HOME': '/owned'}
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        plan = {'arm_definitions': {'native': {'mode': 'mtp-legacy', 'selection': 'adaptive'}},
                'binary': '/binary', 'model': {}, 'numerical_mode': 'row-exact'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = mock.Mock()
            before, after = machine(), dict(machine(), swapins=8)
            with mock.patch.object(c.d, 'prepare_boot_assets', return_value=(root / 'store', None, {})), \
                    mock.patch.object(c.d, 'finish_boot_assets', return_value={'unchanged': True, 'seeds_unchanged': True}), \
                    mock.patch.object(c.r, 'machine_state', side_effect=[before, after]), \
                    mock.patch.object(c.r, 'Server', Server):
                with self.assertRaisesRegex(RuntimeError, 'request'):
                    with c.d.owned_boot(root, root, plan, SimpleNamespace(), 0, 'native', telemetry=collector):
                        raise RuntimeError('request')
            self.assertEqual(collector.phase.call_args_list,
                [mock.call(0, 'native', phase) for phase in
                 ('prelaunch', 'ready', 'pre_shutdown', 'post_shutdown')])
            boots = t.read_jsonl(root / 'boots.jsonl')
            self.assertFalse(boots[0]['completed'])
            self.assertEqual(boots[0]['machine_before']['swapins'], 0)
            self.assertEqual(boots[0]['machine_after']['swapins'], 8)
            self.assertTrue(c.evidence.transfer_measurement_errors(boots, 'declared'))


class CollectorOwnership(unittest.TestCase):
    def test_one_process_is_terminated_waited_and_closed_on_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = mock.Mock(pid=123, returncode=-15, stdout=io.BytesIO())
            proc.poll.return_value = None
            thread = mock.Mock()
            thread.is_alive.return_value = False
            with mock.patch.object(t, 'validate_declaration'), \
                    mock.patch.object(t.subprocess, 'Popen', return_value=proc) as launch, \
                    mock.patch.object(t.threading, 'Thread', return_value=thread):
                collector = t.Collector(directory, declaration(), machine)
                collector.first.wait = mock.Mock(return_value=True)
                with self.assertRaisesRegex(RuntimeError, 'body'):
                    with collector:
                        raise RuntimeError('body')
            launch.assert_called_once_with(declaration()['command'], stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=collector.stderr)
            proc.terminate.assert_called_once()
            proc.wait.assert_called_once_with(timeout=5)
            self.assertTrue(proc.stdout.closed)
            self.assertTrue(collector.stderr.closed)
            self.assertTrue(collector.state['owned_stop'])
            self.assertIsNotNone(collector.state['stopped'])

    def test_start_failure_persists_and_stop_timeout_kills_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = t.Collector(directory, declaration(), machine)
            with mock.patch.object(t, 'validate_declaration', side_effect=ValueError('identity changed')):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    collector.__enter__()
            self.assertTrue(collector.state['errors'])
            self.assertTrue((collector.root / 'collector.json').is_file())
            self.assertIsNotNone(collector.state['stopped'])
        with tempfile.TemporaryDirectory() as directory:
            collector = t.Collector(directory, declaration(), machine)
            collector.root.mkdir()
            collector.proc = mock.Mock(returncode=-9, stdout=io.BytesIO())
            collector.proc.poll.return_value = None
            collector.proc.wait.side_effect = [subprocess.TimeoutExpired('macmon', 5), None]
            collector.__exit__(None, None, None)
            collector.proc.kill.assert_called_once()
            self.assertIn('collector_stop_timeout', collector.state['errors'])

    def test_reader_persists_raw_bytes_receipt_clock_and_parse_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = t.Collector(directory, declaration(), machine)
            collector.root.mkdir()
            raw = (json.dumps(raw_sample(0)) + '\n{broken\n').encode()
            collector.proc = SimpleNamespace(stdout=io.BytesIO(raw))
            with mock.patch.object(t, 'stamp', side_effect=[point(0), point(1)]):
                collector._read()
            self.assertEqual((collector.root / 'macmon.stdout.jsonl').read_bytes(), raw)
            receipts = t.read_jsonl(collector.root / 'receipts.jsonl')
            self.assertEqual(receipts[0]['metrics']['gpu_power_w'], 35)
            self.assertIn('error', receipts[1])
            self.assertIn('collector_exited_before_owned_stop', collector.state['errors'])


if __name__ == '__main__':
    unittest.main()
