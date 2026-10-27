#!/usr/bin/env python3
"""Bounded held-out chat transfer diagnostic; never builds or chains GPU runs.

Prepare once (four thinking-off Qwen27B training requests), then evaluate a NEW
output directory against that frozen seed. Default evaluation is MTP legacy vs
persistent, two balanced pairs, one first exposure and one timed repeat per case
(32 HTTP requests). --pairs 4 confirms; two pairs never promote. --family dflash
uses DFlash2 with its own native baseline. --arms legacy,local or legacy,corpus
isolates controls. Existing lookup_route_diagnostic.py --scenario reuse remains
the exact-answer replay mechanism sanity check, not this transfer experiment.
--both-native-baselines adds the opposite native family to the same balanced
window (48 requests at two pairs). --cases selects a predeclared subset (two
cases = 24 requests with both native baselines); --aggregate combines completed
windows without running inference. Full-workload pass requires all four cases.
--cross-family-reference compares a matching previous opposite-family run but
labels separate-window latency evidence as not jointly balanced confirmation.
--row-exact explicitly controls the numerical mode on EVERY arm; these are not
unchanged fastest-default MTP timings. Omit it for the separate stock comparison.
--row-exact --serial-oracle is correctness-only: one related case (related-code
by default), one feature, serial plus both native families, one first exposure
and one repeat each (exactly eight requests, one <=290-second window). It uses
the same frozen training history, never generates replacement training seeds,
and cannot promote or enter performance aggregation.
--feature-selection cut|fixed:8|fixed:16 is restricted to that serial oracle.
It controls only the feature arm, never the serial/native references; receipt
checks prove the requested selection and legal width. It is a measurement
control, not a production routing change or performance claim.

Examples (each command is a separate <=290-second window; do not autochain):
  python tests/lookup_similar_requests.py --prepare --output /tmp/transfer-train
  python tests/lookup_similar_requests.py --seed /tmp/transfer-train --output /tmp/transfer-mtp
  python tests/lookup_similar_requests.py --seed /tmp/transfer-train --family dflash --output /tmp/transfer-dflash
  python tests/lookup_similar_requests.py --seed /tmp/transfer-train --pairs 4 --output /tmp/transfer-confirm
--plan-only never starts a model. No generated Python is executed.
"""
import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback

import lookup_route_diagnostic as d
r = d.r
import lookup_bench_evidence as evidence
from lookup_bench_evidence import transfer_gate_decisions, transfer_phase_pass
CONTRACT = Path(__file__).with_name('lookup_transfer_contract.json')
RELATED = ('related-config', 'related-code')


def controlled_receipt_errors(row):
    """Supplement shared selection checks for feature arms named by tier."""
    selection = row['arm_definition']['selection']
    if selection == 'adaptive':
        return []
    expected = 'fixed' if selection.startswith('fixed:') else 'cut'
    width = int(selection.split(':')[1]) if expected == 'fixed' else None
    errors = []
    controlled = 0
    for index, receipt in enumerate(row['rounds']):
        # Native MTP fallback is legal, but cannot prove forced width.
        if row['family'] == 'mtp' and receipt.get('selection') == 'legacy':
            continue
        controlled += 1
        prefix = 'round_' + str(index) + ':'
        if receipt.get('selection') != expected:
            errors.append(prefix + 'controlled_selection_mismatch')
        candidate, legal, actual = (receipt.get(key) for key in ('candidate_rows', 'legal_rows', 'rows'))
        legal_width = (all(type(value) is int for value in (candidate, legal, actual))
                       and 1 <= actual <= legal <= candidate)
        if not legal_width:
            errors.append(prefix + 'controlled_illegal_width')
        if receipt.get('requested_rows') != (width or 0):
            errors.append(prefix + 'controlled_requested_rows_mismatch')
        if width is not None and (not legal_width or actual != min(width, legal)):
            errors.append(prefix + 'controlled_actual_width_mismatch')
        if row['family'] == 'dflash' and receipt.get('assistant') is not True:
            errors.append(prefix + 'controlled_missing_assistant')
    if not controlled:
        errors.append('missing_controlled_round_receipts')
    return errors


def render(contract, item):
    case = dict(item)
    if item['kind'] == 'config':
        prompt = contract['config_prompt']
        for key, value in [('NAME', item['name']), ('PORT', item['port']),
                           ('LEVEL', item['log_level']), ('PATH', item['health_path'])]:
            prompt = prompt.replace(key, str(value))
    elif item['kind'] == 'code':
        prompt = contract['code_prompt'].replace('FUNCTION', item['function']).replace('FALLBACK', item['fallback'])
    else:
        prompt = item['prompt']
    case.pop('prompt', None)
    case.update({key: value for key, value in contract['request_contract'].items() if key != 'endpoint'})
    case['messages'] = [{'role': 'system', 'content': contract['system']}, {'role': 'user', 'content': prompt}]
    return case


def semantic_hash(case):
    body = r.request_body(case, 'mtp-legacy', None)
    for key in ('enable_mtp', 'enable_drafter', 'enable_pld'):
        body.pop(key)
    return d.content_hash(body)


def unique_requests(training, evaluation):
    records = [{'phase': phase, 'case': case['id'], 'rendered_request_sha256': semantic_hash(case)}
               for phase, cases in [('training', training), ('evaluation', evaluation)] for case in cases]
    if len(records) != 8 or len({x['rendered_request_sha256'] for x in records}) != len(records):
        raise r.BenchError('expected eight distinct rendered training/evaluation requests')
    return records


def ordered_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field: ' + key)
        result[key] = value
    return result


def quality(case, row):
    failures = []
    response = row['response']
    text = response.get('text') or ''
    if row['body'].get('enable_thinking') is not False or row['body'].get('reasoning_effort') != 'none':
        failures.append('reasoning_not_explicitly_disabled')
    if '<think>' in text or '</think>' in text:
        failures.append('reasoning_markup_in_output')
    for event in response.get('events', []):
        if isinstance(event, dict):
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                if delta.get('reasoning_content') or delta.get('reasoning'):
                    failures.append('reasoning_stream_emitted')
    if case['kind'] in ('config', 'code') and response.get('finish_reason') != 'stop':
        failures.append('truncated_or_unfinished_structured_answer')
    if case['kind'] == 'config':
        expected = {'service': case['name'],
                    'server': {'host': '127.0.0.1', 'port': case['port'], 'protocol': 'http'},
                    'logging': {'level': case['log_level'], 'format': 'json'},
                    'health': {'path': case['health_path'], 'interval_seconds': 30, 'timeout_seconds': 5},
                    'retry': {'attempts': 3, 'backoff_seconds': 2}}
        try:
            actual = json.loads(text, object_pairs_hook=ordered_object)
            if d.content_hash(actual) != d.content_hash(expected):
                failures.append('JSON_exact_requested_fields_or_values_mismatch')
            if not isinstance(actual, dict) or list(actual) != list(expected):
                failures.append('JSON_requested_field_order_mismatch')
        except (ValueError, TypeError) as exc:
            failures.append('invalid_JSON: ' + str(exc))
    elif case['kind'] == 'code':
        try:
            tree = ast.parse(text)
            compile(tree, '<model-answer>', 'exec', dont_inherit=True)  # Validate only; never execute.
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == case['function']]
            if len(functions) != 1:
                failures.append('missing_or_duplicate_required_Python_function')
            else:
                fn = functions[0]
                positional = fn.args.posonlyargs + fn.args.args
                if len(positional) != 1 or positional[0].arg != 'path' or fn.args.vararg or fn.args.kwarg:
                    failures.append('Python_function_signature_mismatch')
            if not any(isinstance(node, ast.Import) and any(alias.name == 'json' for alias in node.names)
                       for node in ast.walk(tree)):
                failures.append('Python_missing_json_import')
            if not any(isinstance(node, ast.With) for node in ast.walk(tree)):
                failures.append('Python_missing_with_statement')
        except (SyntaxError, ValueError) as exc:
            failures.append('invalid_Python: ' + str(exc))
    return sorted(set(failures))


def published_rounds(row):
    """Clip acceptance to output actually published, excluding each root token."""
    result = []
    for receipt in row['rounds']:
        item = dict(receipt)
        item['accepted'] = (min(max(0, receipt.get('accepted', 0)), max(0, receipt.get('committed', 0) - 1))
                            if receipt.get('commit_scope') == 'published' else 0)
        result.append(item)
    return result


def transfer_evidence(row, seed):
    rounds = published_rounds(row)
    origins = {tier: d.origin_evidence(rounds, tier) for tier in ('local', 'corpus', 'persistent')}
    evidence = {'origins': origins, 'training_output_matches': [], 'accepted_span_matches': [],
                'per_token_origin_proven': False,
                'scope': 'Published accepted positions and training OUTPUT substring equality; receipts lack accepted-path origins. External-origin lower bounds are aggregate pigeonhole bounds, not per-token attribution.'}
    if len(row['commits']) != 1:
        return evidence
    commit = row['commits'][0]
    generated = commit['generated_ids']
    # SequenceMatcher is bounded by the declared 192-token outputs, never the corpus.
    matched_positions = set()
    for source in seed.get('seed_identity', []):
        training = source['identity']['generated_ids']
        for block in difflib.SequenceMatcher(None, generated, training, autojunk=False).get_matching_blocks():
            if block.size < 2:
                continue
            matched_positions.update(range(block.a, block.a + block.size))
            evidence['training_output_matches'].append({'training_case': source['case'],
                'evaluation_generated_start': block.a, 'training_generated_start': block.b,
                'tokens': generated[block.a:block.a + block.size], 'length': block.size})
    for index, receipt in enumerate(rounds):
        count = receipt['accepted']
        anchor = receipt.get('anchor_pos')
        if not count or type(anchor) is not int:
            continue
        start = anchor - len(commit['prompt_ids']) + 1
        end = start + count
        if start < 0 or end > len(generated):
            continue
        covered = sum(pos in matched_positions for pos in range(start, end))
        if not covered:
            continue
        matches = []
        for match in evidence['training_output_matches']:
            lo = max(start, match['evaluation_generated_start'])
            hi = min(end, match['evaluation_generated_start'] + match['length'])
            if lo >= hi:
                continue
            tokens = generated[lo:hi]
            prompt_occurrences = sum(
                source['identity']['prompt_ids'][pos:pos + len(tokens)] == tokens
                for source in seed['seed_identity']
                for pos in range(len(source['identity']['prompt_ids']) - len(tokens) + 1))
            matches.append({'training_case': match['training_case'], 'evaluation_generated_start': lo,
                            'training_generated_start': match['training_generated_start'] + lo - match['evaluation_generated_start'],
                            'tokens': tokens, 'training_prompt_occurrences': prompt_occurrences})
        evidence['accepted_span_matches'].append({'round': index, 'generated_start': start,
            'accepted_published_tokens': count, 'matching_training_output_positions': covered,
            'matches': matches,
            'external_accepted_matching_output_lower_bound': {
                tier: max(0, d.origin_evidence([receipt], tier)['accepted_origin_lower_bound'] - (count - covered))
                for tier in ('corpus', 'persistent')}})
    return evidence


def freeze_training(out, plan):
    rows, errors = d.read_rows(out / 'observations.jsonl')
    if errors or len(rows) != 4 or any(row['errors'] for row in rows):
        raise r.BenchError('only four completed, valid training responses may seed fixtures')
    if [row['case'] for row in rows] != [case['id'] for case in plan['cases']]:
        raise r.BenchError('training response identities differ from declared training requests')
    commits = [row['commits'][0] for row in rows]
    for commit in commits:
        if commit.get('version') != 1 or commit.get('finish_reason') != 'stop' or any(
                not commit.get(key) or any(type(token) is not int or not 0 <= token < d.DOC_SEP for token in commit[key])
                for key in ('prompt_ids', 'generated_ids')):
            raise r.BenchError('training must contain complete published token IDs')
    seed_dir = out / 'seed-fixtures'
    seed_dir.mkdir()
    corpus, store = seed_dir / 'corpus', seed_dir / 'persistent'
    corpus.mkdir()
    store.mkdir()
    tokenizer_sha = plan['model']['tokenizer_sha256']
    tokens = [token for commit in commits for token in commit['prompt_ids'] + commit['generated_ids'] + [d.DOC_SEP]]
    d.write_npy_i32(corpus / 'tokens.npy', tokens)
    d.write_npy_i32(corpus / 'sa.npy', sorted(range(len(tokens)), key=lambda pos: tokens[pos:pos + d.MAX_ORDER]))
    d.save(corpus / 'meta.json', {'tokenizer_sha256': tokenizer_sha, 'max_order': d.MAX_ORDER,
        'doc_sep': d.DOC_SEP, 'n_tokens': len(tokens), 'documents': len(commits),
        'tokens_sha256': r.digest(corpus / 'tokens.npy'), 'sa_sha256': r.digest(corpus / 'sa.npy'),
        'commits_sha256': d.content_hash(commits)})
    (store / 'tokens.bin').write_bytes(b'MLXLOOK1' + bytes.fromhex(tokenizer_sha) + struct.pack('<' + 'I' * len(tokens), *tokens))
    (store / 'store.lock').touch(exist_ok=False)
    validation = r.validate_seed_store(store, tokenizer_sha, commits)
    source_files = [out / 'plan.json', out / 'observations.jsonl', out / '0-legacy/launch.json']
    for row in rows:
        sample = out / '0-legacy' / (row['case'] + '-0')
        source_files.extend(sample / name for name in ('request.json', 'response.json', 'response.sse', 'http.json', 'receipts.log'))
    declaration = {'classification': 'held-out similar requests; completed training outputs only; never evaluated answers',
        'format_basis': 'existing production MLXLOOK1 tokenizer header plus published prompt/generated documents, DOC_SEP, and store.lock; corpus uses existing NPY writer and MAX_ORDER suffix array',
        'production_append_used': False, 'production_append_reason': 'Train on owned lookup-disabled native baseline with writes off, then validate exact published format without contaminating model-owned learning state',
        'tokenizer_sha256': tokenizer_sha, 'model_dir': plan['model']['model_dir'],
        'training_binary_sha256': plan['binary_sha256'], 'request_identities': plan['request_identities'],
        'source': str(out / 'observations.jsonl'), 'source_sha256': r.digest(out / 'observations.jsonl'),
        'source_artifacts': {str(path.relative_to(out)): r.digest(path) for path in source_files},
        'training_output_root': str(out), 'seed_identity': [
            {'case': row['case'], 'identity': d.identity(row), 'request_sha256': row['request_sha256'],
             'rendered_request_sha256': row['rendered_request_sha256'],
             'request_identity_sha256': row['request_identity_sha256'], 'commit_sha256': d.content_hash(row['commits'][0])}
            for row in rows],
        'commits': commits, 'documents': len(commits), 'measured_outputs_used_for_seed': False,
        'persistent_validation': validation, 'frozen_before_evaluation': True, 'frozen_unix': time.time(),
        'paths': {'corpus': str(corpus), 'persistent': str(store)},
        'hashes': {'corpus': d.tree_hashes(corpus), 'persistent': d.tree_hashes(store)}}
    d.save(seed_dir / 'declaration.json', declaration)
    d.freeze(seed_dir)
    return declaration


def load_seed(path, plan):
    root = path.expanduser().resolve()
    source_plan = d.load_json(root / 'plan.json')
    source_summary = d.load_json(root / 'summary.json')
    if (source_plan.get('phase') != 'training' or not source_summary.get('complete')
            or source_summary.get('invalid_samples') or not (root / 'worker-complete.json').is_file()):
        raise r.BenchError('seed requires a completed, valid bounded training run')
    seed = d.load_json(root / 'seed-fixtures/declaration.json')
    if seed.get('documents') != 4 or seed.get('measured_outputs_used_for_seed') is not False or not seed.get('frozen_before_evaluation'):
        raise r.BenchError('not a complete frozen training-only fixture')
    if seed['tokenizer_sha256'] != plan['model']['tokenizer_sha256'] or seed['model_dir'] != plan['model']['model_dir']:
        raise r.BenchError('training tokenizer/target identity mismatch')
    if seed['request_identities'] != plan['request_identities']:
        raise r.BenchError('training/evaluation contract identities changed')
    for relative, expected in seed['source_artifacts'].items():
        if r.digest(root / relative) != expected:
            raise r.BenchError('training source changed: ' + relative)
    rows, errors = d.read_rows(root / 'observations.jsonl')
    if errors or len(rows) != 4 or [row['case'] for row in rows] != [case['id'] for case in plan['training_cases']]:
        raise r.BenchError('training source request membership mismatch')
    for case, row, declared in zip(plan['training_cases'], rows, seed['seed_identity']):
        if d.check_sample(row) or quality(case, row) or d.identity(row) != declared['identity']:
            raise r.BenchError('invalid training source response: ' + case['id'])
        if semantic_hash(case) != declared['rendered_request_sha256'] or d.content_hash(row['body']) != declared['request_sha256']:
            raise r.BenchError('training request identity mismatch')
    for tier, directory in seed['paths'].items():
        if d.tree_hashes(Path(directory)) != seed['hashes'][tier]:
            raise r.BenchError('frozen fixture changed: ' + tier)
        if any(file.stat().st_mode & 0o222 for file in [Path(directory), *Path(directory).rglob('*')]):
            raise r.BenchError('fixture is writable: ' + tier)
    r.validate_seed_store(Path(seed['paths']['persistent']), seed['tokenizer_sha256'], seed['commits'])
    if not (Path(seed['paths']['persistent']) / 'store.lock').is_file():
        raise r.BenchError('frozen persistent fixture is missing store.lock')
    return seed


def fastest_native_comparison(plan, result, rows, invalid):
    cross = plan.get('cross_family_reference')
    same_window = len(plan['native_baseline_arms']) == 2
    available = same_window or bool(cross)
    same_power = (result['same_power_phase'] and
                  (not cross or result['power_sources'] == cross['power_sources']))
    valid = bool(available and result['speed_comparison_valid'] and same_power)
    comparisons = []
    for arm in plan['feature_arms']:
        for case in plan['cases']:
            for pair in range(plan['pairs']):
                for rep in (-1, 0):
                    group = [row for row in rows if row['pair'] == pair and row['case'] == case['id']
                             and row['rep'] == rep and row['arm'] in plan['native_baseline_arms'] + [arm]]
                    if any(row['sample'] in invalid for row in group):
                        continue
                    feature = [row for row in group if row['arm'] == arm]
                    natives = {row['family']: row['seconds'] for row in group
                               if row['arm'] in plan['native_baseline_arms']}
                    if cross and not same_window:
                        previous = [row for row in cross['native_timings'] if row['pair'] == pair
                                    and row['case'] == case['id'] and row['rep'] == rep]
                        if len(previous) == 1:
                            natives[cross['family']] = previous[0]['seconds']
                    if len(feature) != 1 or set(natives) != {'mtp', 'dflash'}:
                        continue
                    fastest = min(natives, key=natives.get)
                    comparisons.append({'arm': arm, 'case': case['id'], 'pair': pair,
                        'first_exposure': rep < 0, 'native_seconds': natives,
                        'fastest_native_family': fastest, 'feature_family': plan['family'],
                        'feature_seconds': feature[0]['seconds'],
                        'latency_reduction_vs_fastest_native_percent': 100 * (1 - feature[0]['seconds'] / natives[fastest])})
    cases = [case['id'] for case in plan['cases']]
    passes = {}
    for arm in plan['feature_arms']:
        phases = []
        for first in (True, False):
            phase = [{'case': case, 'pairs': [
                {'pair': item['pair'], 'latency_reduction_percent': item['latency_reduction_vs_fastest_native_percent']}
                for item in comparisons if item['arm'] == arm and item['case'] == case
                and item['first_exposure'] == first]} for case in cases]
            phases.append(transfer_phase_pass(phase, cases, plan['pairs'], RELATED))
        passes[arm] = bool(valid and all(phases))
    return {'available': available, 'same_window_balanced': same_window, 'same_power_phase': same_power,
            'numerical_mode': plan.get('numerical_mode', 'stock'),
            'reference_label': ('Numerical-mode-controlled native references; not unchanged fastest defaults'
                                if plan.get('numerical_mode') == 'row-exact' else 'Stock native-default references'),
            'comparison_valid': valid, 'comparisons': comparisons, 'arm_threshold_pass': passes,
            'scope': ('Both native families and feature share the same alternating boot schedule.'
                      if same_window else
                      'Prior opposite-family native timings are descriptive, separately windowed comparisons; not jointly balanced confirmation.' if cross else
                      'Only own-family native baseline measured; no fastest-across-families claim.')}


def summarize(out, plan, final=False, supervisor=None):
    result = d.summarize(out, plan, final=final, supervisor=supervisor)
    rows, _ = d.read_rows(out / 'observations.jsonl')
    result.update(performance_pass=False, promotion_pass=False, confirmation_pass=False,
                  diagnostic_pass=False, arm_performance_pass={}, activation={},
                  scope='held-out similar requests, boot-paired end-to-end latency; first exposure separate; no throughput or per-token-origin claim')
    if plan['phase'] == 'training':
        result['complete'] = result['complete'] and (out / 'seed-fixtures/declaration.json').is_file()
        result['speed_comparison_valid'] = False
        result['training_fixture_frozen'] = (out / 'seed-fixtures/declaration.json').is_file()
        d.save(out / 'summary.json', result)
        return result
    cross = plan.get('cross_family_reference')
    cross_errors = []
    if cross:
        for row in rows:
            reference = cross['identities'].get(row['case'])
            if reference is not None and d.identity(row) != reference:
                cross_errors.append({'sample': row['sample'], 'errors': ['cross_family_exact_identity_mismatch']})
    result['cross_family_validation'] = {'provided': bool(cross), 'errors': cross_errors,
        'scope': 'matching target, binary, seed, requests and generation limits; text/prompt IDs/generated IDs/usage/finish'}
    if cross_errors:
        by_sample = {item['sample']: item for item in result['sample_validation']}
        for item in cross_errors:
            by_sample[item['sample']]['errors'].extend(item['errors'])
        result['invalid_samples'] = sum(bool(item['errors']) for item in by_sample.values())
        result['speed_comparison_valid'] = False
    invalid = {item['sample'] for item in result['sample_validation'] if item['errors']}
    result['fastest_native_baseline'] = fastest_native_comparison(plan, result, rows, invalid)
    first_exposure = []
    for arm in plan['arms'][1:]:
        for case in plan['cases']:
            comparisons = []
            for pair in range(plan['pairs']):
                group = [row for row in rows if row['pair'] == pair and row['case'] == case['id']
                         and row['rep'] == -1 and row['arm'] in ('legacy', arm)]
                if len(group) == 2 and all(row['sample'] not in invalid for row in group):
                    base = next(row['seconds'] for row in group if row['arm'] == 'legacy')
                    feature = next(row['seconds'] for row in group if row['arm'] == arm)
                    comparisons.append({'pair': pair, 'legacy_seconds': base, 'feature_seconds': feature,
                                        'latency_reduction_percent': 100 * (1 - feature / base)})
            first_exposure.append({'arm': arm, 'case': case['id'], 'pairs': comparisons})
    result['first_exposure_comparisons'] = first_exposure
    result['sample_transfer_evidence'] = [{'sample': row['sample'], 'first_exposure': row['rep'] < 0,
                                         **row.get('transfer', {})} for row in rows]
    result['unseeded_control_classification'] = [{'sample': row['sample'], 'arm': row['arm'],
        'first_exposure': row['rep'] < 0, 'origins': row.get('transfer', {}).get('origins', {})}
        for row in rows if row['case'] == 'unrelated-prose']
    required_related = [case['id'] for case in plan['cases'] if case['id'] in RELATED]
    for arm in plan['feature_arms']:
        tier = arm if arm in ('corpus', 'persistent') else 'local'
        evidence = []
        for pair in range(plan['pairs']):
            for case in required_related:
                for first in (True, False):
                    group = [row for row in rows if row['pair'] == pair and row['arm'] == arm and row['case'] == case
                             and (row['rep'] < 0) == first]
                    valid = bool(group) and all(row['sample'] not in invalid for row in group)
                    origins = d.origin_evidence([receipt for row in group for receipt in published_rounds(row)], tier)
                    output_matches = sum(
                        item['external_accepted_matching_output_lower_bound'].get(tier, 0)
                        for row in group for item in row.get('transfer', {}).get('accepted_span_matches', []))
                    evidence.append({'pair': pair, 'case': case, 'first_exposure': first,
                                     'valid_samples': valid, 'origins': origins,
                                     'external_accepted_matching_training_output_lower_bound': output_matches})
        activated = bool(evidence) and all(
            item['valid_samples'] and item['origins']['accepted_origin_lower_bound'] > 0
            and (tier == 'local' or item['external_accepted_matching_training_output_lower_bound'] > 0)
            for item in evidence)
        result['activation'][arm] = {'activated': activated, 'required_tier': tier, 'by_boot_case': evidence,
            'required_related_cases': required_related, 'activation_required_in_this_window': bool(required_related),
            'output_span_attribution_policy': 'shared accepted output spans are reported when provable; aggregate origins alone never prove a specific token source'}
        cases = [case['id'] for case in plan['cases']]
        checks = [item for item in result['summaries'] if item['arm'] == arm]
        timed_pass = transfer_phase_pass(checks, cases, plan['pairs'], RELATED)
        cold = [item for item in first_exposure if item['arm'] == arm]
        cold_pass = transfer_phase_pass(cold, cases, plan['pairs'], RELATED)
        result['arm_performance_pass'][arm] = bool(
            result['speed_comparison_valid'] and (activated or not required_related)
            and timed_pass and cold_pass)
    result['covered_cases'] = [case['id'] for case in plan['cases']]
    result['full_workload_covered'] = set(result['covered_cases']) == {case['id'] for case in plan['evaluation_cases']}
    result.update(transfer_gate_decisions(
        result['arm_performance_pass'], result['fastest_native_baseline']['arm_threshold_pass'],
        plan['pairs'], result['full_workload_covered'],
        result['fastest_native_baseline']['same_window_balanced'],
        valid=result['speed_comparison_valid'] and not plan.get('serial_oracle')
        and plan.get('gate_schema') == 3))
    result['request_identity_policy'] = plan['request_identity_policy']
    result['numerical_mode'] = plan.get('numerical_mode', 'stock')
    result['native_reference_label'] = plan['native_defaults']
    result['correctness_only'] = plan.get('serial_oracle', False)
    if plan.get('serial_oracle'):
        result['correctness_pass'] = bool(
            result['complete'] and not result['invalid_samples']
            and all(result['activation'][arm]['activated'] for arm in plan['feature_arms']))
        result.update(speed_comparison_valid=False, performance_pass=False, confirmation_pass=False,
                      diagnostic_pass=False, full_workload_diagnostic_pass=False,
                      fastest_native_confirmation_pass=False,
                      arm_performance_pass={arm: False for arm in plan['feature_arms']},
                      scope='Correctness-only serial oracle, explicit row-exact on every arm; no stock or fastest-default speed claim.')
        result['serial_oracle'] = {
            'reference_arm': 'serial', 'numerical_mode': plan['numerical_mode'],
            'request_count_cap': plan['request_count_cap'], 'passed': result['correctness_pass'],
            'identity_fields': ['prompt_ids', 'generated_ids', 'text', 'usage', 'finish_reason'],
            'samples': [{'sample': row['sample'], 'arm': row['arm'],
                         'first_exposure': row['rep'] < 0,
                         'identity_sha256': d.content_hash({
                             'numerical_mode': row.get('numerical_mode', 'stock'),
                             'published': d.identity(row)})} for row in rows]}
        result['fastest_native_baseline']['comparison_valid'] = False
        result['fastest_native_baseline']['arm_threshold_pass'] = {arm: False for arm in plan['feature_arms']}
        result['fastest_native_baseline']['same_window_balanced'] = False
        result['fastest_native_baseline']['scope'] = 'Correctness-only single window; timings are descriptive, not a speed comparison.'
    d.save(out / 'summary.json', result)
    return result


def worker(out):
    plan = d.load_json(out / 'plan.json')
    args = argparse.Namespace(**plan['server_args'])
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if r.digest(plan['binary']) != plan['binary_sha256']:
            raise r.BenchError('binary changed after declaration')
        sources = {'source_script_sha256': __file__, 'shared_driver_sha256': d.__file__,
                   'request_helper_sha256': r.__file__, 'evidence_helper_sha256': evidence.__file__,
                   'seed_helper_sha256': d.write_npy_i32.__code__.co_filename}
        if any(plan.get(key) != r.digest(Path(path)) for key, path in sources.items()):
            raise r.BenchError('helper changed after declaration or historical source identity missing')
        if plan['phase'] == 'evaluation':
            load_seed(Path(plan['seed_root']), plan)
        for pair, order in enumerate(plan['orders']):
            for arm in order:
                folder = out / (str(pair) + '-' + arm)
                folder.mkdir()
                spec = plan['arm_definitions'][arm]
                with d.owned_boot(out, folder, plan, args, pair, arm) as server:
                    d.save(folder / 'owned.json', {'pid': server.proc.pid, 'process_group': os.getpgrp()})
                    for case in plan['cases']:
                        repetitions = (0,) if plan['phase'] == 'training' else (-1, 0)
                        for rep in repetitions:
                            row = d.record_sample(out, folder, server, args, spec, pair, arm, case, rep)
                            row['phase'] = plan['phase']
                            row['request_sha256'] = d.content_hash(row['body'])
                            row['rendered_request_sha256'] = semantic_hash(case)
                            row['request_identity_sha256'] = d.content_hash({'run': str(out), 'phase': plan['phase'],
                                'numerical_mode': plan.get('numerical_mode', 'stock'),
                                'pair': pair, 'arm': arm, 'case': case['id'], 'rep': rep, 'body_sha256': row['request_sha256']})
                            row['errors'] = sorted(set(row['errors'] + quality(case, row) + controlled_receipt_errors(row)))
                            row['transfer'] = (transfer_evidence(row, plan.get('reuse') or {})
                                               if not row['errors'] else {'evidence_unavailable': 'invalid sample'})
                            r.append(out / 'observations.jsonl', row)
                            summarize(out, plan)
                            print(json.dumps({'sample': row['sample'], 'seconds': row['seconds'], 'errors': row['errors']}), flush=True)
                            if row['errors']:
                                raise r.BenchError('invalid response; no retry: ' + row['sample'])
        if plan['phase'] == 'training':
            freeze_training(out, plan)
        d.save(out / 'worker-complete.json', {'completed': True})
        result = summarize(out, plan)
        if plan.get('serial_oracle') and not result.get('correctness_pass'):
            raise r.BenchError('serial oracle failed exact output, quality, frozen-history or genuine reuse activation gates')
        return 0
    except BaseException as exc:
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        summarize(out, plan)
        traceback.print_exc()
        return 1


def build_plan(args, out):
    if args.feature_selection != 'adaptive' and not args.serial_oracle:
        raise r.BenchError('--feature-selection other than adaptive requires --serial-oracle')
    if args.serial_oracle and (not args.row_exact or args.prepare or args.cross_family_reference):
        raise r.BenchError('--serial-oracle requires --row-exact evaluation without a cross-family reference')
    if args.price_trace and (not args.serial_oracle or args.family != 'dflash'):
        raise r.BenchError('--price-trace is only a perturbed DFlash serial-oracle diagnostic')
    contract = d.load_json(args.contract)
    training = [render(contract, item) for item in contract['training']]
    evaluation = [render(contract, item) for item in contract['evaluation']]
    identities = unique_requests(training, evaluation)
    manifest = d.load_json(d.KIT / 'manifest.json')
    model = next(model for model in manifest['models'] if model['id'] == 'qwen27b')
    if r.digest(Path(model['model_dir']) / 'tokenizer.json') != model['tokenizer_sha256']:
        raise r.BenchError('current tokenizer differs from staged Qwen27B tokenizer')
    binary = r.absolute_asset(manifest['binary'])
    arms = ['legacy'] if args.prepare else args.arms.split(',')
    feature_arms = arms[1:]
    if args.serial_oracle and len(feature_arms) != 1:
        raise r.BenchError('--serial-oracle requires exactly one feature arm')
    native_arms = ['legacy']
    definitions = d.arm_definitions('reuse', args.family, arms)
    if args.feature_selection != 'adaptive':
        for arm in feature_arms:
            spec = definitions[arm]
            spec['selection'] = args.feature_selection
            spec['actual'] = 'fixed' if args.feature_selection.startswith('fixed:') else 'cut'
            width_policy = ('adaptive width cut' if args.feature_selection == 'cut' else
                            'fixed width ' + args.feature_selection.split(':')[1] + ' capped by legal rows')
            spec['meaning'] = (
                ('always assistant; merge ' + arm + ' lookup then ' + width_policy)
                if args.family == 'dflash' else
                (arm + ' lookup-chain ' + width_policy + '; native fallback or controlled root-only round on miss')
            ) + '; correctness-only measurement control, not adaptive admission or production routing'
    both_native = (args.both_native_baselines or args.serial_oracle) and not args.prepare
    if both_native:
        opposite = ('dflash' if args.family == 'mtp' else 'mtp') + '-legacy'
        definitions.update(d.arm_definitions('baselines', args.family, [opposite]))
        arms.insert(1, opposite)
        native_arms.append(opposite)
    if args.serial_oracle:
        arms.insert(0, 'serial')
        definitions['serial'] = {'mode': 'serial', 'family': 'serial', 'selection': 'adaptive',
                                 'actual': 'legacy', 'meaning': 'serial oracle; MTP, drafter and PLD disabled'}
    numerical_mode = d.declare_numerical_mode(definitions, args.row_exact)
    for spec in definitions.values():
        unsupported = r.unsupported(model, spec['mode'])
        if unsupported:
            raise r.BenchError(unsupported)
    pairs = 1 if args.prepare or args.serial_oracle else args.pairs
    selected = args.cases.split(',') if args.cases else (
        ['related-code'] if args.serial_oracle else [case['id'] for case in evaluation])
    if len(set(selected)) != len(selected) or not selected or not set(selected) <= {case['id'] for case in evaluation}:
        raise r.BenchError('--cases must name unique declared evaluation cases')
    if args.serial_oracle and (len(selected) != 1 or selected[0] not in RELATED):
        raise r.BenchError('--serial-oracle is capped at one related case and eight requests')
    cases = training if args.prepare else [case for case in evaluation if case['id'] in selected]
    plan = {'version': 1, 'scenario': 'transfer', 'phase': 'training' if args.prepare else 'evaluation',
        'gate_schema': 3, 'measurement_isolation': args.isolation,
        'family': args.family, 'binary': binary, 'binary_sha256': r.digest(binary), 'model': model,
        'numerical_mode': numerical_mode, 'serial_oracle': args.serial_oracle,
        'price_trace': args.price_trace,
        'feature_selection': args.feature_selection,
        'experiment_kind': 'correctness-only serial oracle' if args.serial_oracle else 'paired transfer latency',
        'contract': contract, 'contract_sha256': r.digest(args.contract),
        'manifest_sha256': r.digest(d.KIT / 'manifest.json'),
        'source_script_sha256': r.digest(Path(__file__)), 'shared_driver_sha256': r.digest(Path(d.__file__)),
        'request_helper_sha256': r.digest(Path(r.__file__)),
        'evidence_helper_sha256': r.digest(Path(evidence.__file__)),
        'seed_helper_sha256': r.digest(Path(d.write_npy_i32.__code__.co_filename)),
        'training_cases': training, 'evaluation_cases': evaluation, 'cases': cases, 'request_identities': identities,
        'request_identity_policy': 'Eight distinct rendered semantic requests: four train plus four held-out variants. Repeats deliberately share semantic/body hashes; every phase/pair/arm/case/rep has a unique invocation-identity hash. No artificial request variation inside timing pairs.',
        'baseline_arm': 'serial' if args.serial_oracle else 'legacy',
        'pairs': pairs, 'arms': arms, 'arm_definitions': definitions,
        'feature_arms': feature_arms, 'native_baseline_arms': native_arms,
        'both_native_baselines': both_native,
        'orders': [arms[:] if pair % 2 == 0 else list(reversed(arms)) for pair in range(pairs)],
        'warmup_per_case_per_arm_per_boot': 0 if args.prepare else 1,
        'timed_per_case_per_arm_per_boot': 1, 'total_requests': 4 if args.prepare else pairs * len(arms) * len(cases) * 2,
        'server_args': {'port': args.port, 'ctx_size': 2048, 'ready_timeout': 35, 'request_timeout': 25},
        'reuse': None, 'seed_root': str(args.seed.expanduser().resolve()) if args.seed else None,
        'cross_family_reference': None, 'deadline_seconds': 290, 'worker_stop_seconds': 285,
        'native_defaults': ('Numerical-mode-controlled native MTP and DFlash references, explicit row-exact on every arm; not unchanged fastest defaults. '
                            if args.row_exact else 'Stock native numerical defaults. ') +
                           'Native MTP prompt lookup and DFlash copy defaults unchanged; MTP engagement via spec-stats, native fallback need not emit round receipts; prefix cache disabled',
        'isolation': 'Owned child only, fresh HOME/store per boot, frozen read-only clones, persistent writes off for every request; evaluation outputs never appended',
        'pass_rule': 'Gate schema 3: family-improvement and fastest-native gates are independent. Every pair: >=5% reduction on related-config/code and <=5% regression on conflicting-config/unrelated-prose, separately cold/timed. Require published accepted tier/training-output lower bounds on related cases, exact text/tokens/usage/finish, frozen train/eval split, matching numerical mode and complete same-power schedule. Performance qualification requires declared exclusive GPU inference and known, unchanged before/after boot power source/settings, thermal state, normal memory pressure and swap usage/counters, without thermal warnings or throttling; contended or unknown evidence is descriptive only. Fastest-native additionally requires both native families in the same balanced window and identical thresholds versus their minimum latency. All four cases required for confirmation; two pairs diagnose, four confirm. Unqualified performance/confirmation require fastest-native; promotion always false. Historical evidence is never upgraded.',
        'preflight_tests': False, 'cpu_tests': False, 'build': False, 'canonical': False,
        'promotion_pass': False, 'created_unix': time.time()}
    if args.serial_oracle:
        plan['pass_rule'] = (
            'Correctness only, exactly eight full requests: one related case, one first exposure and one repeat '
            'on serial, native MTP, native DFlash and one feature. Explicit row-exact activation on every boot; '
            'published prompt/generated IDs, text, usage and finish equal serial; full structured-output quality '
            'and frozen-history checks unchanged; positive accepted external training-output lower bound for '
            'external feature on both exposures. No latency threshold, performance pass, or stock-default claim.')
        plan['request_count_cap'] = 8
    if not args.prepare:
        plan['reuse'] = load_seed(args.seed, plan)
    if args.cross_family_reference:
        root = args.cross_family_reference.expanduser().resolve()
        reference_plan = d.load_json(root / 'plan.json')
        reference_summary = d.load_json(root / 'summary.json')
        if not reference_summary.get('complete') or reference_summary.get('invalid_samples') or not reference_summary.get('speed_comparison_valid'):
            raise r.BenchError('cross-family reference must be complete, valid, and power-controlled')
        if (reference_plan['family'] == args.family or reference_plan['binary_sha256'] != plan['binary_sha256']
                or reference_plan['model'] != model or reference_plan['request_identities'] != identities
                or reference_plan.get('numerical_mode', 'stock') != numerical_mode
                or reference_plan['reuse']['hashes'] != plan['reuse']['hashes']):
            raise r.BenchError('cross-family reference target/binary/requests/seed not matching')
        rows, errors = d.read_rows(root / 'observations.jsonl')
        refs = {}
        for row in rows:
            if row['arm'] == 'legacy':
                if row['errors'] or (row['case'] in refs and refs[row['case']] != d.identity(row)):
                    raise r.BenchError('inconsistent cross-family native reference')
                refs[row['case']] = d.identity(row)
        if errors or not {case['id'] for case in cases} <= set(refs):
            raise r.BenchError('incomplete cross-family native references')
        plan['cross_family_reference'] = {'path': str(root), 'family': reference_plan['family'], 'identities': refs,
            'numerical_mode': numerical_mode,
            'plan_sha256': r.digest(root / 'plan.json'), 'observations_sha256': r.digest(root / 'observations.jsonl'),
            'power_sources': reference_summary['power_sources'],
            'native_timings': [{key: row[key] for key in ('pair', 'case', 'rep', 'seconds')}
                               for row in rows if row['arm'] == 'legacy'],
            'summary_sha256': r.digest(root / 'summary.json')}
    d.save(out / 'plan.json', plan)
    d.save(out / 'declared-cases.json', {'training': training, 'evaluation': evaluation, 'request_identities': identities})
    (out / 'executed-similar.py').write_bytes(Path(__file__).read_bytes())
    (out / 'executed-diagnostic.py').write_bytes(Path(d.__file__).read_bytes())
    return plan


def aggregate(paths, output):
    """CPU-only combination of declared windows, never an inference sweep."""
    roots = [path.expanduser().resolve() for path in paths]
    if len(set(roots)) != len(roots):
        raise r.BenchError('duplicate aggregate input window')
    plans = [d.load_json(root / 'plan.json') for root in roots]
    summaries = [d.load_json(root / 'summary.json') for root in roots]
    # Reject archived schemas before indexing fields introduced by qualification.
    if any(plan.get('gate_schema') != 3 for plan in plans) or any(
            summary.get('gate_schema') != 3 for summary in summaries):
        raise r.BenchError('aggregate requires qualified gate-schema 3 evidence; historical runs remain unchanged')
    first = plans[0]
    compatibility = ('family', 'binary_sha256', 'model', 'request_identities', 'pairs', 'arms',
                     'both_native_baselines', 'pass_rule', 'request_helper_sha256', 'measurement_isolation',
                     'source_script_sha256', 'shared_driver_sha256', 'evidence_helper_sha256', 'seed_helper_sha256',
                     'contract_sha256', 'contract', 'feature_selection', 'price_trace', 'server_args',
                     'warmup_per_case_per_arm_per_boot', 'timed_per_case_per_arm_per_boot',
                     'deadline_seconds', 'worker_stop_seconds')
    if any(any(key not in plan for key in compatibility) for plan in plans):
        raise r.BenchError('aggregate requires complete source/contract/protocol identity; historical runs remain unchanged')
    errors, artifacts, identities = [], [], {}
    covered = set()
    power = set()
    for root, plan, summary in zip(roots, plans, summaries):
        if plan.get('phase') != 'evaluation' or plan.get('serial_oracle'):
            raise r.BenchError('aggregation accepts performance evaluation windows only, not serial oracles')
        if plan.get('numerical_mode', 'stock') != first.get('numerical_mode', 'stock'):
            raise r.BenchError('aggregate windows differ in numerical mode; stock and row-exact cannot mix')
        if (any(d.content_hash(plan[key]) != d.content_hash(first[key]) for key in compatibility)
                or plan['reuse']['hashes'] != first['reuse']['hashes']):
            raise r.BenchError('aggregate windows differ in target/binary/seed/requests/arms/pairs/gates/helper/contract/protocol')
        load_seed(Path(plan['seed_root']), plan)
        if (not summary.get('final') or not summary.get('complete') or summary.get('invalid_samples')
                or not summary.get('speed_comparison_valid')):
            errors.append('incomplete_or_invalid_window:' + str(root))
        rows, parse_errors = d.read_rows(root / 'observations.jsonl')
        if parse_errors:
            errors.append('invalid_observations:' + str(root))
        for row in rows:
            if row.get('numerical_mode', 'stock') != plan.get('numerical_mode', 'stock'):
                errors.append('observation_numerical_mode_mismatch:' + row['sample'])
            if row['case'] in identities and identities[row['case']] != d.identity(row):
                errors.append('cross_window_exact_identity_mismatch:' + row['case'])
            identities.setdefault(row['case'], d.identity(row))
        covered.update(case['id'] for case in plan['cases'])
        power.update(summary.get('power_sources', []))
        artifacts.append({'root': str(root), 'cases': [case['id'] for case in plan['cases']],
                          'hashes': {name: r.digest(root / name) for name in ('plan.json', 'summary.json', 'observations.jsonl')}})
    expected = {case['id'] for case in first['evaluation_cases']}
    full = covered == expected
    same_power = len(power) == 1 and 'unknown' not in power
    valid = not errors and same_power and full
    family_passes = {arm: bool(valid and all(
        summary['arm_family_improvement_pass'].get(arm, False) for summary in summaries))
        for arm in first['feature_arms']}
    fastest_passes = {arm: bool(valid and all(
        summary['arm_fastest_native_pass'].get(arm, False) for summary in summaries))
        for arm in first['feature_arms']}
    gates = transfer_gate_decisions(family_passes, fastest_passes, first['pairs'], full,
                                    first['both_native_baselines'], valid=valid)
    diagnostic, confirmed = gates['diagnostic_pass'], gates['confirmation_pass']
    result = {'complete': valid, 'errors': sorted(set(errors)), 'windows': artifacts,
              'numerical_mode': first.get('numerical_mode', 'stock'),
              'native_reference_label': first['native_defaults'],
              'covered_cases': sorted(covered), 'missing_cases': sorted(expected - covered),
              'full_workload_covered': full, 'same_power_phase': same_power, 'power_sources': sorted(power),
              **gates, 'canonical_speedup_claim': False,
              'scope': 'Completed compatible bounded windows, all four cases; independent family and fastest-native gates, independent cold/timed gates. No inference or automatic chaining.',
              'pass_rule': first['pass_rule']}
    out = (output or d.KIT / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-similar-aggregate-' + str(os.getpid()))).expanduser().resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    d.save(out / 'aggregate-summary.json', result)
    print(json.dumps({'output': str(out), 'complete': valid, 'diagnostic_pass': diagnostic,
                      'confirmation_pass': confirmed, 'missing_cases': result['missing_cases']}), flush=True)
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepare', action='store_true', help='Explicit GPU training phase: four distinct chat requests, then freeze')
    parser.add_argument('--seed', type=Path, help='Completed --prepare output directory; never an evaluation run')
    parser.add_argument('--contract', type=Path, default=CONTRACT)
    parser.add_argument('--family', choices=('mtp', 'dflash'), default='mtp')
    parser.add_argument('--arms', default='legacy,persistent', help='legacy first; unique local/corpus/persistent controls')
    parser.add_argument('--pairs', type=int, choices=(2, 4), default=2)
    parser.add_argument('--isolation', choices=('declared', 'contended'), default='contended',
                        help='Declared means the operator confirms no other GPU inference; unknown defaults to contended')
    parser.add_argument('--cross-family-reference', type=Path)
    parser.add_argument('--cases', help='Comma-separated evaluation subset; output cap stays 192; full pass requires all four cases')
    parser.add_argument('--aggregate', type=Path, nargs='+', help='CPU-only aggregate of completed evaluation windows')
    parser.add_argument('--both-native-baselines', action='store_true',
                        help='Include opposite-family native baseline in the same balanced window; 48 requests at two pairs')
    parser.add_argument('--row-exact', action='store_true',
                        help='Explicit model-lifetime row-exact mode on ALL arms; not unchanged fastest-default arithmetic')
    parser.add_argument('--serial-oracle', action='store_true',
                        help='With --row-exact: correctness only, one related case/feature, both natives plus serial, eight requests')
    parser.add_argument('--feature-selection', choices=('adaptive', 'cut', 'fixed:8', 'fixed:16'),
                        default='adaptive', help='Feature only; forced widths/cut require serial oracle, never performance evidence')
    parser.add_argument('--price-trace', action='store_true',
                        help='DFlash serial-oracle only: emit evaluated cut prices; perturbed timings are never performance evidence')
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.feature_selection != 'adaptive' and not args.serial_oracle:
        parser.error('--feature-selection other than adaptive requires --serial-oracle')
    if args.worker:
        return worker(args.worker)
    if args.aggregate:
        return aggregate(args.aggregate, args.output)
    arms = args.arms.split(',')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    if args.prepare and (args.seed or args.cross_family_reference or args.both_native_baselines or args.cases or args.serial_oracle):
        parser.error('--prepare always trains all four requests and cannot include evaluation comparisons')
    if not args.prepare and not args.seed:
        parser.error('evaluation requires an explicit completed --seed training directory')
    if len(arms) < 2 or arms[0] != 'legacy' or len(set(arms)) != len(arms) or any(arm not in ('legacy', 'local', 'corpus', 'persistent') for arm in arms):
        parser.error('--arms requires legacy first and unique local/corpus/persistent controls')
    if args.output is None:
        args.output = d.KIT / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-similar-' + ('training' if args.prepare else args.family) + '-' + str(os.getpid()))
    return d.supervise(args, lambda out: build_plan(args, out), __file__, summarize)


if __name__ == '__main__':
    sys.exit(main())
