"""CPU-only tests; importing the driver does not load models or external helpers."""
import copy
import contextlib
import io
import unittest
import json
import tempfile
from pathlib import Path
import lookup_counterfactual_window as cf


def receipt(phase):
    return {
        'phase': phase, 'action': 'assistant16', 'first_offer': 0,
        'identity': {key: [11] * 32 for key in cf.IDENTITY_KEYS},
        'prepared': dict({key: [17] * 32 for key in ('target_kv', 'target_ssm', 'assistant')},
                         preparation='merged16', assistant_horizon=16, suffix_candidates=True, context_length=384),
        'offer': {'candidates': [23] * 32, 'probabilities': [29] * 32},
        'selected': {'candidates': [31] * 32, 'probabilities': [37] * 32},
        'kernel': {'rows': 16, 'variant': 'linear_gdn', 'kind': 'tree', 'class': 0, 'dispatch': [41] * 32,
                   'geometry': {'selected_parents': list(range(-1, 15)), 'selected_depths': list(range(16)),
                                'committed_rows': list(range(16)), 'committed_count': 3, 'kv_compacted': False}},
        'timing': {'verify_commit_ns': 23000, 'full_round_ns': None, 'diagnostic_readback_ns': 9000},
        'warmup_proven': phase == 'measured', 'published': [53, 59, 61],
        'published_len': 3, 'accepted_drafts': 2,
    }


class CounterfactualReceipts(unittest.TestCase):
    def pair(self):
        return [receipt('warmup'), receipt('measured')]

    def test_transport_or_inference_failure_does_not_prove_request_limit(self):
        limit = json.dumps({'error': {'type': 'diagnostic_request_limit', 'code': 409}})
        self.assertTrue(cf.diagnostic_limit_refused({'http_status': 409, 'error': limit}))
        for response in ({'error': 'timeout'}, {'http_status': 500, 'error': limit},
                         {'http_status': 409, 'error': 'connection reset'},
                         {'http_status': 409, 'error': '{"error":{"type":"inference_error","code":409}}'}):
            self.assertFalse(cf.diagnostic_limit_refused(response))

    def test_identical_first_offer_with_actual_dispatch(self):
        measured = cf.validate_pair(self.pair(), 'assistant16')
        self.assertIsNone(measured['timing']['full_round_ns'])
        self.assertTrue(measured['warmup_proven'])

    def test_complete_learner_identity_is_required(self):
        for key in ('learner', 'request_lookup'):
            with self.subTest(key=key):
                pair = self.pair()
                pair[1]['identity'][key][0] += 1
                with self.assertRaisesRegex(ValueError, 'identity'):
                    cf.validate_pair(pair, 'assistant16')

    def test_missing_request_trust_is_not_legacy_same_state_proof(self):
        pair = self.pair()
        for row in pair:
            del row['identity']['request_lookup']
        with self.assertRaises((KeyError, ValueError)):
            cf.validate_pair(pair, 'assistant16')

    def test_equal_malformed_request_trust_digests_do_not_prove_identity(self):
        pair = self.pair()
        for row in pair:
            row['identity']['request_lookup'] = [11] * 31
        with self.assertRaisesRegex(ValueError, 'digest'):
            cf.validate_pair(pair, 'assistant16')

    def test_valid_prefix_tokens_do_not_replace_actual_kv(self):
        pair = self.pair()
        pair[1]['identity']['target_kv'][1] += 1
        with self.assertRaisesRegex(ValueError, 'identity'):
            cf.validate_pair(pair, 'assistant16')

    def test_width_and_actual_kernel_variant_both_matter(self):
        for key, value in [('rows', 8), ('variant', 'generic'), ('dispatch', [43] * 32)]:
            pair = self.pair()
            pair[1]['kernel'][key] = value
            with self.assertRaises(ValueError):
                cf.validate_pair(pair, 'assistant16')

    def test_staircase_actions_require_exact_executed_width(self):
        for action, width in (('assistant1', 1), ('lookup2', 2), ('assistant9', 9), ('lookup15', 15)):
            with self.subTest(action=action):
                pair = self.pair()
                for row in pair:
                    row.update(action=action, published_len=1, accepted_drafts=0)
                    row['kernel']['rows'] = width
                    row['kernel']['geometry']['committed_count'] = 1
                    row['prepared']['assistant_horizon'] = 16 if action.startswith('assistant') else 0
                self.assertEqual(cf.validate_pair(pair, action)['kernel']['rows'], width)
                pair[1]['kernel']['rows'] = width + 1
                with self.assertRaisesRegex(ValueError, 'forced width'):
                    cf.validate_pair(pair, action)

    def test_invalid_staircase_actions_fail_closed(self):
        for action in ('serial', 'assistant0', 'lookup17', 'assistant01', 'lookup1x'):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'invalid diagnostic action'):
                cf.action_width(action)

    def test_no_search_for_a_later_matching_offer(self):
        pair = self.pair()
        pair[1]['first_offer'] = 1
        with self.assertRaisesRegex(ValueError, 'first offer'):
            cf.validate_pair(pair, 'assistant16')

    def test_warmup_does_not_count_without_dispatch(self):
        pair = self.pair()
        pair[0]['kernel']['dispatch'] = [0] * 32
        with self.assertRaisesRegex(ValueError, 'dispatch'):
            cf.validate_pair(pair, 'assistant16')

    def test_reached_output_and_probabilities_cannot_drift(self):
        for key in ('offer', 'selected', 'prepared'):
            pair = self.pair()
            nested = next(iter(pair[1][key]))
            pair[1][key][nested][0] += 1
            with self.assertRaises(ValueError):
                cf.validate_pair(pair, 'assistant16')
        pair = self.pair()
        pair[1]['published'][1] = 67
        with self.assertRaisesRegex(ValueError, 'output'):
            cf.validate_pair(pair, 'assistant16')

    def test_no_partial_or_extra_request_pair(self):
        for pair in ([receipt('warmup')], self.pair() + [copy.deepcopy(receipt('measured'))]):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                cf.validate_pair(pair, 'assistant16')

    def test_changed_runtime_or_incomplete_capture_cannot_price_a_width(self):
        identity = {'libraries': {'libmlx.dylib': 'original'}, 'kernels': {'mlx.metallib': 'original'}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'runtime_identity': identity}
            for changed in ('libraries', 'kernels'):
                after = copy.deepcopy(identity)
                after[changed][next(iter(after[changed]))] = 'replaced'
                cf.save(root / 'runtime-identity.json', {'before': identity, 'after': after})
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'runtime libraries or kernels'):
                    cf.validate_runtime_evidence(root, plan)
            cf.save(root / 'runtime-identity.json', {'before': identity})
            with self.assertRaisesRegex(ValueError, 'runtime libraries or kernels'):
                cf.validate_runtime_evidence(root, plan)
            cf.save(root / 'runtime-identity.json', {'before': identity, 'after': identity})
            cf.validate_runtime_evidence(root, plan)
            with self.assertRaisesRegex(ValueError, 'redeclare historical'):
                cf.validate_runtime_evidence(root, {})

    def test_production_preparations_retain_horizon_and_exact_requested_width(self):
        for preparation in ('production-neural', 'production-combined'):
            with self.subTest(preparation=preparation):
                pair = self.pair()
                for row in pair:
                    row['prepared'].update(preparation=preparation, assistant_horizon=7,
                                           suffix_candidates=preparation == 'production-combined')
                measured = cf.validate_pair(pair, 'assistant16', preparation)
                self.assertEqual(measured['prepared']['assistant_horizon'], 7)
                pair[1]['kernel']['rows'] = 8
                with self.assertRaisesRegex(ValueError, 'forced width'):
                    cf.validate_pair(pair, 'assistant16', preparation)

    def test_preparation_mismatch_and_neural_lookup_actions_fail_closed(self):
        for action in ('serial', 'assistant1', 'assistant16'):
            cf.validate_preparation('production-neural', action)
        for action in ('lookup1', 'lookup16'):
            with self.assertRaisesRegex(ValueError, 'rejects lookup'):
                cf.validate_preparation('production-neural', action)
        with self.assertRaisesRegex(ValueError, 'preparation mismatch'):
            cf.validate_pair(self.pair(), 'assistant16', 'production-neural')
        pair = self.pair()
        for row in pair:
            row['prepared']['suffix_candidates'] = False
        with self.assertRaisesRegex(ValueError, 'candidate-source'):
            cf.validate_pair(pair, 'assistant16')
        pair = self.pair()
        pair[1]['prepared']['assistant_horizon'] = 7
        with self.assertRaisesRegex(ValueError, 'horizon'):
            cf.validate_pair(pair, 'assistant16')

    def test_combined_lookup_requires_suffixes_but_no_assistant_horizon(self):
        pair = self.pair()
        for row in pair:
            row['action'] = 'lookup16'
            row['prepared'].update(preparation='production-combined', assistant_horizon=0)
        cf.validate_pair(pair, 'lookup16', 'production-combined')
        pair[1]['prepared']['assistant_horizon'] = 7
        with self.assertRaisesRegex(ValueError, 'assistant horizon'):
            cf.validate_pair(pair, 'lookup16', 'production-combined')
        pair[1]['prepared']['assistant_horizon'] = 0
        pair[1]['prepared']['suffix_candidates'] = False
        with self.assertRaisesRegex(ValueError, 'candidate-source'):
            cf.validate_pair(pair, 'lookup16', 'production-combined')

    def test_commit_geometry_and_context_are_required(self):
        for field, value in (('committed_count', 2), ('kv_compacted', True),
                             ('committed_rows', [0, 2, 1])):
            pair = self.pair()
            pair[1]['kernel']['geometry'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'geometry|path'):
                cf.validate_pair(pair, 'assistant16')
        pair = self.pair()
        pair[1]['prepared']['context_length'] = 0
        with self.assertRaisesRegex(ValueError, 'context length'):
            cf.validate_pair(pair, 'assistant16')


class TrainingPriceExport(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'prices.json'
        self.source = self.output.with_suffix('.source.json')
        self.plan = {
            'action': 'assistant16', 'preparation': 'merged16', 'binary_sha256': 'a' * 64,
            'runtime_identity': {'libraries': {'mlx': 'b' * 64}, 'kernels': {'metal': 'c' * 64}},
            'model': {'id': 'test-target'}, 'model_content': {'target': {'weights': 'd' * 64}},
            'semantic_request_sha256': 'e' * 64, 'contract_sha256': 'f' * 64,
            'seed_hashes': {'store': '1' * 64}, 'case_phase': 'training',
            'case': {'id': 'train-config-a', 'prompt': 'training only'},
            'training_cases': [{'id': 'train-config-a', 'prompt': 'training only'}],
        }
        self.plan['request_identities'] = [
            {'phase': 'training', 'case': 'train-config-a', 'rendered_request_sha256': 'e' * 64},
            {'phase': 'evaluation', 'case': 'heldout', 'rendered_request_sha256': '2' * 64},
        ]
        self.plan['contract_json'] = json.dumps({
            'training': self.plan['training_cases'],
            'evaluation': [{'id': 'heldout', 'prompt': 'held out request'}],
            'request_contract': {},
        }, indent=2) + '\n'
        self.plan['contract_sha256'] = cf.hashlib.sha256(self.plan['contract_json'].encode('utf-8')).hexdigest()
        self.identity = {'prompt_ids': [1, 2], 'generated_ids': [53, 59, 61],
                         'text': 'complete', 'finish_reason': 'stop',
                         'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}
        self.oracle = self.window('serial', action='serial')
        self.feature = self.window('assistant16')

    def window(self, name, *, action='assistant16', variant='linear_gdn', cls=0):
        root = self.root / name
        root.mkdir()
        plan = copy.deepcopy(self.plan)
        plan['action'] = action
        cf.save(root / 'plan.json', plan)
        cf.save(root / 'summary.json', {'complete': True})
        cf.save(root / 'runtime-identity.json',
                {'before': plan['runtime_identity'], 'after': plan['runtime_identity']})
        cf.save(root / 'samples.json', [{'phase': phase, 'identity': self.identity}
                                       for phase in ('warmup', 'measured')])
        pair = []
        if action != 'serial':
            for phase in ('warmup', 'measured'):
                row = receipt(phase)
                row.update(action=action, published_len=1, accepted_drafts=0)
                row['kernel'].update(rows=cf.action_width(action), variant=variant)
                row['kernel']['class'] = cls
                row['kernel']['geometry']['committed_count'] = 1
                row['prepared']['assistant_horizon'] = 16 if action.startswith('assistant') else 0
                row['timing'].update(verify_commit_ns=1000000 if phase == 'warmup' else 23000,
                                     diagnostic_readback_ns=9000000, full_round_ns=99000000)
                pair.append(row)
        cf.save(root / 'receipts.json', pair)
        return root

    def change(self, root, name, modify):
        path = root / (name + '.json')
        value = cf.load(path)
        modify(value)
        cf.save(path, value)

    def export(self, windows=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cf.export_prices(windows or [self.feature], self.oracle, self.output)

    def rejected(self, windows=None, message=None):
        with self.assertRaisesRegex((ValueError, KeyError), message or '.'):
            self.export(windows)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.source.exists())

    def test_export_measured_verify_commit_only_with_adjacent_provenance(self):
        self.assertEqual(self.export(), 0)
        self.assertEqual(cf.load(self.output), {
            'schema': 3, 'context_bucket': 'floor-power-of-two-2048',
            '0': {'tree': {'linear_gdn': {'16': 0.023}}},
        })
        source = cf.load(self.source)
        for key, value in {'version': 1, 'purpose': 'staircase_calibration', 'phase': 'training',
                           'scope': 'same-state first-offer verify/commit only; no throughput claim',
                           'throughput_claim': False}.items():
            self.assertEqual(source[key], value)
        for key in ('binary_sha256', 'runtime_identity', 'model', 'seed_hashes', 'contract_sha256'):
            self.assertEqual(source[key], self.plan[key])
        self.assertEqual(source['prices_sha256'], cf.hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(source['preparation'], 'merged16')
        row = cf.normalized(cf.load(self.feature / 'receipts.json')[1])
        self.assertEqual(source['source_windows'], [{
            'path': str(self.feature), 'plan_sha256': cf.sha(self.feature / 'plan.json'),
            'receipts_sha256': cf.sha(self.feature / 'receipts.json'),
            'action': 'assistant16', 'case': 'train-config-a',
            'prepared': row['prepared'], 'geometry': row['kernel']['geometry'],
            'accepted_drafts': 0, 'committed_tokens': 1,
        }])
        self.assertNotIn('step_ms', cf.load(self.output))

    def test_sidecar_binding_covers_exact_bytes_not_reserialized_json(self):
        self.export()
        bound = cf.load(self.source)['prices_sha256']
        original = self.output.read_bytes()
        self.output.write_bytes(original + b' ')
        self.assertNotEqual(bound, cf.sha(self.output))
        self.assertEqual(json.loads(original), cf.load(self.output))

    def test_oracle_preparation_cannot_be_relabelled_or_mixed(self):
        for preparation in ('production-neural', 'production-combined'):
            self.change(self.oracle, 'plan', lambda plan: plan.update(preparation=preparation))
            self.rejected(message='serial oracle mismatch: preparation')

    def test_missing_historical_preparation_is_not_upgraded(self):
        self.change(self.feature, 'plan', lambda plan: plan.pop('preparation'))
        self.rejected(message='predeclared preparation')

    def test_export_rejects_incompatible_neural_lookup_action(self):
        self.change(self.feature, 'plan', lambda plan: plan.update(preparation='production-neural', action='lookup16'))
        self.rejected(message='rejects lookup')

    def test_neural_export_requires_matching_receipt_and_oracle_preparation(self):
        for root in (self.feature, self.oracle):
            self.change(root, 'plan', lambda plan: plan.update(preparation='production-neural'))
        self.rejected(message='receipt preparation mismatch')
        rows = cf.load(self.feature / 'receipts.json')
        for row in rows:
            row['prepared'].update(preparation='production-neural', assistant_horizon=7, suffix_candidates=False)
        cf.save(self.feature / 'receipts.json', rows)
        self.export()
        source = cf.load(self.source)
        self.assertEqual(source['preparation'], 'production-neural')
        self.assertEqual(source['source_windows'][0]['prepared']['assistant_horizon'], 7)
        self.assertFalse(source['source_windows'][0]['prepared']['suffix_candidates'])

    def test_actual_executable_variants_are_separate_even_at_same_width(self):
        generic = self.window('generic16', variant='generic')
        self.export([self.feature, generic])
        self.assertEqual(cf.load(self.output)['0']['tree'], {
            'linear_gdn': {'16': 0.023}, 'generic': {'16': 0.023}})

    def test_one_row_is_not_inferred_plain_and_small_width_not_inferred_generic(self):
        one = self.window('assistant1', action='assistant1', variant='linear_gdn')
        self.export([one])
        self.assertEqual(cf.load(self.output)['0'], {'tree': {'linear_gdn': {'1': 0.023}}})

    def test_linear_topology_cannot_export_into_the_chain_verifier_price_table(self):
        rows = cf.load(self.feature / 'receipts.json')
        for row in rows:
            row['kernel']['kind'] = 'chain'
        cf.save(self.feature / 'receipts.json', rows)
        self.rejected(message='tree-verifier classification')

    def test_no_pooling_across_sources_or_undeclared_replicates(self):
        for name, action in (('lookup16', 'lookup16'), ('replicate16', 'assistant16')):
            other = self.window(name, action=action)
            with self.subTest(action=action):
                self.rejected([self.feature, other], 'conflicting price cell')
        self.rejected([self.feature, self.feature], 'duplicate source window')

    def test_evaluation_or_relabelled_case_is_rejected_including_oracle(self):
        for root in (self.feature, self.oracle):
            original = cf.load(root / 'plan.json')
            for change in ({'case_phase': 'evaluation'},
                           {'case': {'id': 'heldout', 'prompt': 'training only'}},
                           {'training_cases': []}):
                with self.subTest(root=root.name, change=change):
                    cf.save(root / 'plan.json', dict(original, **change))
                    self.rejected(message='evaluation|training case')
            cf.save(root / 'plan.json', original)

    def test_relabelling_completed_evaluation_pair_does_not_create_training_evidence(self):
        for root in (self.feature, self.oracle):
            plan = cf.load(root / 'plan.json')
            plan['case'] = {'id': 'heldout', 'prompt': 'held out request'}
            plan['semantic_request_sha256'] = '2' * 64
            plan['case_phase'] = 'training'
            plan['training_cases'].append(plan['case'])
            cf.save(root / 'plan.json', plan)
        # Output, receipts, binary/runtime/seed/contract identity all remain identical.
        self.rejected(message='frozen contract')
        # Relabelling request identities as well cannot override contract membership.
        for root in (self.feature, self.oracle):
            plan = cf.load(root / 'plan.json')
            plan['request_identities'][1]['phase'] = 'training'
            cf.save(root / 'plan.json', plan)
        self.rejected(message='frozen contract')

    def test_training_case_and_semantic_hash_must_bind_to_one_training_identity(self):
        original = cf.load(self.feature / 'plan.json')
        for identities in ([], original['request_identities'] * 2,
                           [{'phase': 'evaluation', 'case': 'train-config-a',
                             'rendered_request_sha256': 'e' * 64}],
                           [{'phase': 'training', 'case': 'train-config-a',
                             'rendered_request_sha256': '2' * 64}],
                           [{'phase': 'training', 'case': 'different-training-case',
                             'rendered_request_sha256': 'e' * 64}]):
            with self.subTest(identities=identities):
                plan = copy.deepcopy(original)
                plan['request_identities'] = identities
                cf.save(self.feature / 'plan.json', plan)
                self.rejected(message='frozen training request identity')

    def test_missing_or_modified_frozen_contract_never_qualifies_historical_windows(self):
        original = cf.load(self.feature / 'plan.json')
        for snapshot in (None, original['contract_json'] + ' '):
            plan = copy.deepcopy(original)
            plan['contract_json'] = snapshot
            cf.save(self.feature / 'plan.json', plan)
            with self.subTest(snapshot=snapshot):
                self.rejected(message='frozen contract')

    def test_every_frozen_declaration_identity_must_match_serial_oracle(self):
        original = cf.load(self.feature / 'plan.json')
        for key in cf.DECLARATION_KEYS:
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[key] = {'drift': True} if isinstance(changed[key], dict) else '9' * 64
                cf.save(self.feature / 'plan.json', changed)
                self.rejected(message='mismatch|runtime libraries|predeclared preparation')
        cf.save(self.feature / 'plan.json', original)

    def test_runtime_prepost_evidence_is_required_for_both_windows(self):
        for root in (self.feature, self.oracle):
            path = root / 'runtime-identity.json'
            original = cf.load(path)
            cf.save(path, {'before': original['before']})
            self.rejected(message='runtime libraries')
            cf.save(path, original)

    def test_incomplete_windows_and_sample_phases_are_rejected(self):
        for root in (self.feature, self.oracle):
            self.change(root, 'summary', lambda x: x.update(complete=False))
            self.rejected(message='incomplete')
            self.change(root, 'summary', lambda x: x.update(complete=True))
            original = cf.load(root / 'samples.json')
            for samples in (original[:1], [original[1], original[1]]):
                cf.save(root / 'samples.json', samples)
                self.rejected(message='warmup')
            cf.save(root / 'samples.json', original)
        self.change(self.feature, 'receipts', lambda x: x.pop())
        self.rejected(message='exactly one')

    def test_nonpositive_nonfinite_and_noninteger_timings_are_rejected(self):
        original = cf.load(self.feature / 'receipts.json')
        for value in (0, -1, float('nan'), float('inf'), -float('inf'), 1.5, True, 10 ** 400):
            with self.subTest(value=value):
                pair = copy.deepcopy(original)
                pair[1]['timing']['verify_commit_ns'] = value
                pair[1]['timing']['full_round_ns'] = None
                cf.save(self.feature / 'receipts.json', pair)
                self.rejected(message='verify/commit')

    def test_actual_variant_dispatch_and_context_class_cannot_be_guessed(self):
        original = cf.load(self.feature / 'receipts.json')
        for key, value in (('variant', None), ('dispatch', [0] * 32),
                           ('class', 1024), ('class', 2049), ('class', True),
                           ('class', float('nan'))):
            pair = copy.deepcopy(original)
            for row in pair:
                row['kernel'][key] = value
            cf.save(self.feature / 'receipts.json', pair)
            with self.subTest(key=key, value=value):
                self.rejected()

    def test_cross_window_state_preparation_and_offer_must_match(self):
        other = self.window('assistant8', action='assistant8')
        original = cf.load(other / 'receipts.json')
        for part, key in (('identity', 'learner'), ('prepared', 'target_kv'),
                          ('prepared', 'assistant'), ('offer', 'probabilities')):
            pair = copy.deepcopy(original)
            for row in pair:
                row[part][key][0] += 1
            cf.save(other / 'receipts.json', pair)
            with self.subTest(part=part, key=key):
                self.rejected([self.feature, other], 'mismatch|differs')

    def test_exact_output_and_first_publication_must_match(self):
        original = cf.load(self.feature / 'samples.json')
        for sample in original:
            sample['identity']['generated_ids'][0] = 99
        cf.save(self.feature / 'samples.json', original)
        self.rejected(message='exact-output')
        cf.save(self.feature / 'samples.json', cf.load(self.oracle / 'samples.json'))
        pair = cf.load(self.feature / 'receipts.json')
        for row in pair:
            row['published'][0] = 99
        cf.save(self.feature / 'receipts.json', pair)
        self.rejected(message='first published')

    def test_price_or_provenance_is_never_overwritten(self):
        for existing in (self.output, self.source):
            existing.write_text('retain exactly\n')
            with self.assertRaises(FileExistsError):
                self.export()
            self.assertEqual(existing.read_text(), 'retain exactly\n')
            other = self.source if existing == self.output else self.output
            self.assertFalse(other.exists())
            existing.unlink()

    def test_declared_context_class_is_retained_without_width_inference(self):
        contextual = self.window('context2048', cls=2048)
        self.export([self.feature, contextual])
        table = cf.load(self.output)
        self.assertEqual(table['0'], table['2048'])


if __name__ == '__main__':
    unittest.main()
