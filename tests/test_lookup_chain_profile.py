"""CPU-only production-training profile boundaries; no inference or external tools."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lookup_chain_profile as p


def receipt(width=8, milliseconds=10, **changes):
    row = {'family': 'mtp', 'kind': 'chain', 'variant': 'generic', 'assistant': False,
           'selection': 'fixed', 'admissible': True, 'timing_scope': 'settled_round',
           'priced': True, 'requested_rows': width, 'rows': width, 'class': 0,
           'verify_commit_ms': milliseconds}
    row.update(changes)
    return row


def request(process, sweep, receipts, width=8, name=None, errors=None):
    return {'arm': 'width' + str(width), 'sample': name or str(process), 'rounds': receipts,
            'requested_width': width, 'process_instance': process, 'sweep': sweep,
            'errors': errors or []}


def consumer_plan():
    return {'binary_sha256': 'binary', 'runtime_identity': {'library': 'frozen'},
            'model': {'tokenizer_sha256': 'tokenizer', 'model_dir': '/target'},
            'model_inventory': 'inventory', 'numerical_mode': 'row-exact',
            'contract_sha256': 'contract', 'request_identities': [{'phase': 'training'}],
            'training_cases': [{'id': 'train'}], 'reuse': {'hashes': {'persistent': 'seed'}},
            'contract': {'request_contract': {'temperature': 0}}, 'server_args': {'ctx_size': 4096}}


class PopulationTests(unittest.TestCase):
    def test_priced_true_does_not_prove_warm_and_cold_is_actual_execution(self):
        rows = [request(0, 0, [receipt(rows=3), receipt(milliseconds=100), receipt(milliseconds=10)]),
                request(0, 0, [receipt(milliseconds=12)], name='repeat'),
                request(1, 1, [receipt(milliseconds=1000), receipt(milliseconds=14)])]
        table, members, cells = p.population(rows)
        self.assertEqual(table['0']['chain']['generic'], {'8': 12})
        self.assertEqual([m['excluded'] for m in members], [
            'short_offer_or_clipped_width', 'first_actual_valid_execution', None, None,
            'first_actual_valid_execution', None])
        self.assertEqual(cells[0]['count'], 3)
        self.assertEqual(cells[0]['median_absolute_deviation_ms'], 2)
        self.assertEqual(cells[0]['processes'], 2)

    def test_cold_context_and_width_are_independent_within_process(self):
        rows = [request(0, 0, [receipt(), receipt(), receipt(**{'class': 2048}), receipt(**{'class': 2048})]),
                request(0, 0, [receipt(9), receipt(9)], width=9),
                request(1, 1, [receipt(), receipt()])]
        table, members, _ = p.population(rows)
        self.assertEqual(sum(m['excluded'] == 'first_actual_valid_execution' for m in members), 4)
        self.assertNotIn('2048', table)  # No second independent sweep: remains absent.
        self.assertNotIn('9', table['0']['chain']['generic'])

    def test_unpriced_actual_execution_can_be_cold_but_never_a_price(self):
        rows = [request(i, i, [receipt(priced=False), receipt(milliseconds=20 + i)]) for i in (0, 1)]
        table, members, _ = p.population(rows)
        self.assertEqual(table['0']['chain']['generic']['8'], 20.5)
        self.assertEqual([m['excluded'] for m in members], ['first_actual_valid_execution', None] * 2)

    def test_unsupported_clipped_absent_failed_and_wrong_namespace_stay_absent(self):
        rows = [request(0, 0, [receipt(16, rows=15)], width=16),
                request(1, 0, [receipt(15, rows=14)], width=15),
                request(2, 0, []), request(3, 0, [receipt()], errors=['HTTP failure'])]
        for key, value in [('kind', 'tree'), ('variant', 'linear_gdn'), ('assistant', True),
                           ('selection', 'adaptive'), ('admissible', False),
                           ('timing_scope', 'overlapped_draft_unseparated'), ('verify_commit_ms', None)]:
            rows.append(request(len(rows), 0, [receipt(**{key: value})]))
        table, members, cells = p.population(rows)
        self.assertEqual(table, {'schema': 3, 'context_bucket': 'floor-power-of-two-2048'})
        self.assertFalse(cells)
        self.assertTrue(all(member['excluded'] for member in members))
        self.assertEqual(members[0]['excluded'], 'unsupported_width')
        self.assertEqual(members[2]['excluded'], 'absent_proposal')
        self.assertEqual(members[3]['excluded'], 'failed_request')
        self.assertEqual(p.eligible_receipt(receipt(milliseconds=float('inf')), 8), 'invalid_settled_verify_commit_ms')
        self.assertEqual(p.eligible_receipt(receipt(milliseconds=float('nan')), 8), 'invalid_settled_verify_commit_ms')

    def test_schema_namespace_never_relabels_tree_or_extrapolates_width(self):
        table = {'schema': 3, 'context_bucket': 'floor-power-of-two-2048', '0': {'chain': {'generic': {'15': 10}}}}
        p.validate_table(table)
        for bad in [dict(table, **{'0': {'tree': {'generic': {'15': 10}}}}),
                    dict(table, **{'0': {'chain': {'linear_gdn': {'15': 10}}}}),
                    dict(table, **{'0': {'chain': {'generic': {'16': 10}}}}),
                    dict(table, **{'0': {'chain': {'generic': {'15': False}}}}),
                    dict(table, **{'2049': {'chain': {'generic': {'15': 10}}}})]:
            with self.assertRaises(ValueError):
                p.validate_table(bad)

    def test_training_label_cannot_hide_evaluation_prompts(self):
        contract = {'system': 'source only', 'request_contract': {'endpoint': 'chat', 'max_tokens': 768},
                    'training': [{'id': 'train' + str(i), 'kind': 'coding', 'prompt': 'train ' + str(i)} for i in range(4)],
                    'evaluation': [{'id': 'eval' + str(i), 'kind': 'coding', 'prompt': 'eval ' + str(i)} for i in range(4)]}
        training = [p.s.render(contract, item) for item in contract['training']]
        evaluation = [p.s.render(contract, item) for item in contract['evaluation']]
        raw = p.encode(contract).decode()
        plan = {'profile_contract': p.CONTRACT, 'phase': 'training', 'contract_bytes': raw,
                'contract_sha256': p.sha_bytes(raw.encode()), 'contract': contract,
                'training_cases': training, 'cases': evaluation}
        with self.assertRaisesRegex(ValueError, 'evaluation data'):
            p.validate_training(plan)


class ProfileConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'prices.json'
        self.output = self.root / 'evaluation'
        self.output.mkdir()
        self.plan = consumer_plan()
        self.table = {'schema': 3, 'context_bucket': 'floor-power-of-two-2048',
                      '0': {'chain': {'generic': {'8': 12}}}}
        self.provenance = {'profile_contract': p.CONTRACT, 'phase': 'training',
            'bindings': p.binding(self.plan), 'source_hashes': {}, 'source_run': str(self.root / 'training')}
        self.save()
        self.source_patch = patch.object(p, 'sources', return_value=[])
        self.export_patch = patch.object(p, 'export_data', return_value=(self.table, self.provenance))
        self.source_patch.start()
        self.export_patch.start()
        self.addCleanup(self.export_patch.stop)
        self.addCleanup(self.source_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def save(self, source=None):
        self.path.write_bytes(p.encode(self.table))
        self.path.with_suffix('.source.json').write_bytes(p.encode(source or dict(
            self.provenance, prices_sha256=p.sha_bytes(self.path.read_bytes()))))

    def test_selection_uses_frozen_validated_bytes_not_mutable_original(self):
        original = self.path.read_bytes()
        identity = p.freeze_profile(self.path, self.plan, os.path.relpath(self.output))
        environment = p.r.clean_environment('persistent', experiment={'MLX_SERVE_LOOKUP_PRICES': identity['path']})
        self.path.write_bytes(b'{}')
        selected = Path(environment['MLX_SERVE_LOOKUP_PRICES'])
        self.assertEqual(selected.read_bytes(), original)
        self.assertEqual(identity['sha256'], p.sha_bytes(original))
        p.assert_profile_frozen(identity)
        selected.chmod(0o600)
        selected.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'frozen profile'):
            p.assert_profile_frozen(identity)

    def test_tampered_bytes_and_evaluation_sources_rejected(self):
        self.path.write_bytes(self.path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'tampered'):
            p.freeze_profile(self.path, self.plan, self.output)
        self.save(dict(self.provenance, phase='evaluation', prices_sha256=p.sha_bytes(p.encode(self.table))))
        with self.assertRaisesRegex(ValueError, 'evaluation'):
            p.freeze_profile(self.path, self.plan, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_runtime_target_mode_seed_and_contract_rejected(self):
        for field, value in [('runtime_identity', {'library': 'other'}), ('model_inventory', 'other'),
                             ('numerical_mode', 'stock'), ('contract_sha256', 'other'),
                             ('reuse', {'hashes': {'persistent': 'evaluation-output'}})]:
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                p.freeze_profile(self.path, plan, self.output)

    def test_forged_source_membership_rejected(self):
        source = dict(self.provenance, prices_sha256=p.sha_bytes(self.path.read_bytes()), membership=['invented'])
        self.save(source)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            p.freeze_profile(self.path, self.plan, self.output)

    def test_malformed_and_duplicate_price_json_rejected(self):
        for raw in (b'{', b'{"schema":3,"schema":3}', b'{"x":NaN}'):
            self.path.write_bytes(raw)
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                p.freeze_profile(self.path, self.plan, self.output)

    def test_raw_artifact_tampering_is_rejected_before_population_export(self):
        run = self.root / 'raw'
        run.mkdir()
        raw = run / 'receipts.log'
        raw.write_text('original')
        (run / 'completed.json').write_bytes(p.encode({'complete': True, 'artifacts': {'receipts.log': p.r.digest(raw)}}))
        raw.write_text('modified')
        # Exercise the real exporter rather than the provenance resolver seam above.
        self.export_patch.stop()
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            p.export_data(run)


if __name__ == '__main__':
    unittest.main()
