"""Component-v2-only passive macmon ownership and fail-closed request joins.

No thermal stabilization policy or inference warmups. macmon's JSON schema is
frozen here, not guessed from missing fields; unsupported sensors fail closed.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time

from lookup_bench_evidence import stable_states


REQUIRED_FIELDS = ['timestamp', 'gpu_freq_mhz', 'gpu_scaled_ratio',
                   'temp.gpu_temp_avg', 'gpu_power', 'fans']
RULES = {'schema': 1, 'required_fields': REQUIRED_FIELDS,
         'optional_fields': {'gpu_active_ratio': 'report separately with availability; never substitute scaled ratio'},
         'machine_invariants': 'stable_states across original boot envelopes and every phase/request endpoint',
         'join': 'receipt-monotonic bracket plus in-request sample; no interpolation',
         'unsupported_fields': 'fail qualification; never substitute normal values',
         'frequency_limits': 'raw soc metadata when available; unsupported otherwise',
         'startup_timeout_seconds': 10, 'shutdown_timeout_seconds': 5,
         'thermal_stabilization': 'not declared; hardware characterization required'}


def stamp():
    return {'monotonic': time.monotonic(), 'wall_unix': time.time()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def declaration(binary='macmon', interval_ms=1000, max_gap_ms=3000):
    if (type(interval_ms) is not int or interval_ms <= 0 or
            type(max_gap_ms) is not int or max_gap_ms < interval_ms):
        raise ValueError('telemetry cadence/gap must be positive integers; gap >= cadence')
    found = shutil.which(str(binary))
    if not found:
        raise ValueError('component v2 requires an installed macmon executable')
    path = Path(found).resolve()
    version = subprocess.run([str(path), '--version'], check=True, capture_output=True,
                             text=True, timeout=10).stdout.strip()
    if not version:
        raise ValueError('macmon version missing')
    return dict(RULES, binary=str(path), binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                version=version, command=[str(path), 'pipe', '--soc-info', '-i', str(interval_ms)],
                interval_ms=interval_ms, max_gap_ms=max_gap_ms)


def validate_declaration(config, current=False):
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in RULES.items()):
        raise ValueError('telemetry protocol changed; redeclare')
    interval, gap = config.get('interval_ms'), config.get('max_gap_ms')
    if (type(interval) is not int or interval <= 0 or type(gap) is not int or gap < interval
            or not isinstance(config.get('binary'), str) or not config['binary']
            or not isinstance(config.get('version'), str) or not config['version']
            or not isinstance(config.get('binary_sha256'), str) or len(config['binary_sha256']) != 64
            or config.get('command') != [config['binary'], 'pipe', '--soc-info', '-i', str(interval)]):
        raise ValueError('invalid frozen telemetry identity or cadence')
    if current and declaration(config['binary'], interval, gap) != config:
        raise ValueError('macmon command/version/binary changed; redeclare')


def metrics(raw):
    if not isinstance(raw, dict):
        raise ValueError('macmon sample is not an object')
    timestamp = datetime.fromisoformat(raw['timestamp'].replace('Z', '+00:00'))
    if timestamp.tzinfo is None:
        raise ValueError('macmon timestamp lacks timezone')
    values = {'gpu_frequency_mhz': raw['gpu_freq_mhz'],
              'gpu_utilization_scaled_ratio': raw['gpu_scaled_ratio'],
              'gpu_temperature_c': raw['temp']['gpu_temp_avg'],
              'gpu_power_w': raw['gpu_power']}
    if any(not finite(v) or v < 0 for v in values.values()):
        raise ValueError('missing/unsupported/nonfinite GPU sensor')
    if values['gpu_utilization_scaled_ratio'] > 1:
        raise ValueError('GPU utilization outside [0,1]')
    active = raw.get('gpu_active_ratio')
    if active is not None and (not finite(active) or not 0 <= active <= 1):
        raise ValueError('malformed GPU active residency ratio')
    values['gpu_utilization_active_ratio'] = active
    values['gpu_utilization_active_ratio_status'] = (
        'available' if active is not None else 'unsupported' if 'gpu_active_ratio' in raw else 'not_reported')
    fans = raw['fans']
    if not isinstance(fans, list) or not fans:
        raise ValueError('fans unsupported or absent')
    for fan in fans:
        if (not isinstance(fan, dict) or not isinstance(fan.get('name'), str)
                or not finite(fan.get('rpm')) or fan['rpm'] < 0):
            raise ValueError('malformed fan sensor')
    return dict(values, fans=fans, soc=raw.get('soc'),
                device_frequency_limits_status='raw soc metadata' if raw.get('soc') else 'unsupported',
                source_wall_unix=timestamp.timestamp())


class Collector:
    def __init__(self, out, config, machine_state):
        self.root = Path(out) / 'telemetry'
        self.config, self.machine_state = config, machine_state
        self.proc = self.thread = self.stderr = None
        self.stopping = False
        self.first = threading.Event()
        self.state = {'declaration': config, 'errors': [], 'started': None, 'stopped': None,
                      'returncode': None, 'owned_stop': False}

    def persist(self):
        save(self.root / 'collector.json', self.state)

    def _read(self):
        try:
            with (self.root / 'macmon.stdout.jsonl').open('wb') as raw:
                for line in self.proc.stdout:
                    received = stamp()
                    raw.write(line)
                    raw.flush()
                    record = dict(received, raw_line=line.decode('utf-8', errors='replace'))
                    try:
                        record['metrics'] = metrics(json.loads(line))
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        record['error'] = str(exc)
                    append(self.root / 'receipts.jsonl', record)
                    self.first.set()
            if not self.stopping:
                self.state['errors'].append('collector_exited_before_owned_stop')
        except BaseException as exc:
            self.state['errors'].append('collector_reader_failed: ' + str(exc))
        finally:
            self.first.set()

    def __enter__(self):
        self.root.mkdir()
        self.state['started'] = stamp()
        self.persist()
        try:
            validate_declaration(self.config, current=True)
            self.stderr = (self.root / 'macmon.stderr.log').open('wb')
            # Inherit the cohort process group: supervisor cancellation owns this child too.
            self.proc = subprocess.Popen(self.config['command'], stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=self.stderr)
            self.state['pid'] = self.proc.pid
            self.persist()
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            if not self.first.wait(self.config['startup_timeout_seconds']):
                raise ValueError('collector_startup_timeout')
            if self.proc.poll() is not None or self.state['errors']:
                raise ValueError('collector died during startup')
            return self
        except BaseException as exc:
            self.state['errors'].append('collector_start_failed: ' + str(exc))
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        self.state['stop_requested'] = stamp()
        trailing_error = None
        # Trailing passive sample brackets final cleanup; outside request latency.
        try:
            if self.proc is not None and self.proc.poll() is None and self.thread is not None:
                self.first.clear()
                if not self.first.wait(self.config['max_gap_ms'] / 1000):
                    self.state['errors'].append('collector_trailing_sample_timeout')
        except BaseException as error:
            # Cancellation during trailing collection must still reap our child.
            self.state['errors'].append('collector_trailing_sample_interrupted: ' + str(error))
            trailing_error = error
        self.stopping = True
        try:
            if self.proc is not None:
                if self.proc.poll() is None:
                    self.state['owned_stop'] = True
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=self.config['shutdown_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        self.state['errors'].append('collector_stop_timeout')
                        self.proc.kill()
                        self.proc.wait()
                else:
                    self.state['errors'].append('collector_dead_before_cleanup')
                self.state['returncode'] = self.proc.returncode
            if self.thread is not None:
                self.thread.join(timeout=self.config['shutdown_timeout_seconds'])
                if self.thread.is_alive():
                    self.state['errors'].append('collector_reader_cleanup_timeout')
            if self.proc is not None and self.proc.stdout is not None:
                self.proc.stdout.close()
        finally:
            if self.stderr is not None:
                self.stderr.close()
            self.state['stopped'] = stamp()
            self.persist()
        if trailing_error is not None:
            raise trailing_error

    def endpoint(self):
        point = {'collection_start': stamp()}
        try:
            point['machine'] = self.machine_state()
            state = point['machine']
            if (not isinstance(state, dict) or state.get('power_source') not in ('ac', 'battery')
                    or any(not finite(state.get(key)) for key in
                           ('pressure_level', 'swap_used_bytes', 'swapins', 'swapouts'))):
                point['error'] = 'machine endpoint has missing or unsupported fields'
        except Exception as exc:
            point['error'] = type(exc).__name__ + ': ' + str(exc)
        point['collection_end'] = stamp()
        return point

    def phase(self, pair, arm, phase):
        point = dict(pair=pair, arm=arm, phase=phase, **self.endpoint())
        append(self.root / 'boot-phases.jsonl', point)
        return point

    def start_request(self, sample):
        return {'sample': sample, 'before': self.endpoint(), 'start': stamp()}

    def finish_request(self, boundary, error=None):
        boundary['end'] = stamp()  # Before machine collection, receipt settling or response persistence.
        boundary['after'] = self.endpoint()
        boundary['error'] = error
        boundary['endpoint_collection_seconds'] = sum(
            boundary[k]['collection_end']['monotonic'] - boundary[k]['collection_start']['monotonic']
            for k in ('before', 'after'))
        append(self.root / 'requests.jsonl', boundary)
        return boundary


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def join_request(boundary, samples, max_gap_seconds):
    start, end = boundary['start']['monotonic'], boundary['end']['monotonic']
    if not finite(start) or not finite(end) or end < start:
        raise ValueError('malformed request interval')
    before = [i for i, point in enumerate(samples) if point['monotonic'] <= start]
    after = [i for i, point in enumerate(samples) if point['monotonic'] >= end]
    errors = []
    if not before or not after:
        return {'sample': boundary['sample'], 'errors': ['request_not_bracketed'], 'samples': []}
    selected = samples[before[-1]:after[0] + 1]
    if not any(start <= point['monotonic'] <= end for point in selected):
        errors.append('no_in_request_sample')
    if any(b['monotonic'] - a['monotonic'] > max_gap_seconds for a, b in zip(selected, selected[1:])):
        errors.append('request_telemetry_gap')
    return {'sample': boundary['sample'], 'errors': errors, 'samples': selected}


def qualify(out, expected_samples, expected_boots, config, *, boot_states=()):
    root = Path(out) / 'telemetry'
    errors, requests = [], []
    try:
        validate_declaration(config)
        state = json.loads((root / 'collector.json').read_text())
        if (state.get('declaration') != config or not state.get('started') or not state.get('stopped')
                or not state.get('owned_stop') or state.get('returncode') not in (0, -15)):
            errors.append('collector_missing_dead_or_changed')
        errors.extend(state.get('errors', []))
        samples = read_jsonl(root / 'receipts.jsonl')
        if not samples:
            errors.append('missing_telemetry_samples')
        raw = (root / 'macmon.stdout.jsonl').read_bytes()
        if raw != ''.join(point['raw_line'] for point in samples).encode():
            errors.append('raw_telemetry_receipt_mismatch')
        for point in samples:
            if point.get('error') or point.get('metrics') != metrics(json.loads(point['raw_line'])):
                errors.append('malformed_telemetry_sample')
            if not finite(point['monotonic']) or not finite(point['wall_unix']):
                errors.append('malformed_telemetry_receipt_clock')
            elif abs(point['wall_unix'] - point['metrics']['source_wall_unix']) > config['max_gap_ms'] / 1000:
                errors.append('source_receipt_clock_gap')
        gap = config['max_gap_ms'] / 1000
        if any(not 0 < b['monotonic'] - a['monotonic'] <= gap
               or not 0 < b['metrics']['source_wall_unix'] - a['metrics']['source_wall_unix'] <= gap
               for a, b in zip(samples, samples[1:])):
            errors.append('cohort_telemetry_gap_or_reordered_clock')
        boundaries = read_jsonl(root / 'requests.jsonl')
        if [item['sample'] for item in boundaries] != expected_samples:
            errors.append('missing_or_changed_request_boundaries')
        phases = read_jsonl(root / 'boot-phases.jsonl')
        if [(p['pair'], p['arm'], p['phase']) for p in phases] != [
                (pair, arm, phase) for pair, arm in expected_boots
                for phase in ('prelaunch', 'ready', 'pre_shutdown', 'post_shutdown')]:
            errors.append('missing_or_changed_boot_phases')
        endpoints = phases + [b[k] for b in boundaries for k in ('before', 'after')]
        states = [*boot_states, *(point.get('machine', {}) for point in endpoints)]
        if any(not isinstance(state, dict) for state in states):
            errors.append('machine_endpoint_collection_failed')
        else:
            errors.extend('machine_state:' + error for error in stable_states(states))
        for point in endpoints:
            if point.get('error') or not isinstance(point.get('machine'), dict) or not point['machine']:
                errors.append('machine_endpoint_collection_failed')
            elif (point['machine'].get('power_source') not in ('ac', 'battery')
                    or any(not finite(point['machine'].get(key)) for key in
                           ('pressure_level', 'swap_used_bytes', 'swapins', 'swapouts'))):
                errors.append('machine_endpoint_collection_failed')
            for key in ('collection_start', 'collection_end'):
                if not all(finite(point[key][clock]) for clock in ('wall_unix', 'monotonic')):
                    errors.append('malformed_endpoint_clock')
            if point['collection_end']['monotonic'] < point['collection_start']['monotonic']:
                errors.append('reversed_endpoint_collection_clock')
        for boundary in boundaries:
            if boundary.get('error'):
                errors.append('request_failed')
            for key in ('start', 'end'):
                if not all(finite(boundary[key][clock]) for clock in ('wall_unix', 'monotonic')):
                    errors.append('malformed_request_clock')
            if not (boundary['before']['collection_end']['monotonic'] <= boundary['start']['monotonic']
                    <= boundary['end']['monotonic'] <= boundary['after']['collection_start']['monotonic']):
                errors.append('request_endpoint_order_changed')
            joined = join_request(boundary, samples, gap)
            joined['boundary'] = boundary
            requests.append(joined)
            errors.extend(joined['errors'])
        if samples and phases and (samples[0]['monotonic'] > phases[0]['collection_start']['monotonic']
                or samples[-1]['monotonic'] < phases[-1]['collection_end']['monotonic']):
            errors.append('cohort_endpoints_not_bracketed')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append('missing_or_malformed_telemetry: ' + str(exc))
    return {'qualified': not errors, 'errors': sorted(set(errors)), 'requests': requests,
            'thermal_stabilization': RULES['thermal_stabilization']}
