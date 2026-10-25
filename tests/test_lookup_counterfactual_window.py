"""CPU-only tests; importing the driver does not load models or external helpers."""
import copy
import unittest
import json
import lookup_counterfactual_window as cf


def receipt(phase):
    return {
        'phase': phase, 'action': 'assistant16', 'first_offer': 0,
        'identity': {key: [11] * 32 for key in cf.IDENTITY_KEYS},
        'prepared': {key: [17] * 32 for key in ('target_kv', 'target_ssm', 'assistant')},
        'offer': {'candidates': [23] * 32, 'probabilities': [29] * 32},
        'selected': {'candidates': [31] * 32, 'probabilities': [37] * 32},
        'kernel': {'rows': 16, 'variant': 'linear_gdn', 'kind': 'chain', 'class': 0, 'dispatch': [41] * 32},
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


if __name__ == '__main__':
    unittest.main()
