#!/usr/bin/env python3
"""Offline kit boundary regressions; stdlib only, no model or network service needed."""
import socket
import json
from pathlib import Path
import tempfile
import struct
import unittest
from copy import deepcopy

from bench_lookup_staircut import BenchError, port_free
from prepare_lookup_offline import checkpoint_audit, ReadinessError, safetensors_header


class PortLifecycle(unittest.TestCase):
    def test_live_listener_is_not_reusable(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaises(BenchError):
                port_free(listener.getsockname()[1])

    def test_owned_server_can_rebind_after_active_close(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = listener.accept()
                accepted.close()  # Server is the active closer: its port enters TIME_WAIT.
                self.assertEqual(client.recv(1), b"")
        port_free(port)
        with socket.socket() as replacement:
            replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            replacement.bind(("127.0.0.1", port))
            replacement.listen(1)
            with self.assertRaises(BenchError):
                port_free(port)


class ComparisonValidity(unittest.TestCase):
    @staticmethod
    def fixture(mode="mtp-corpus"):
        from bench_lookup_staircut import paired_schedule
        state = {"power_source": "ac", "power_settings": "lowpowermode0", "thermal": "no thermal warning",
                 "pressure_level": 1, "swap_used_bytes": 0, "swapins": 0, "swapouts": 0}
        events, requests = [], []
        for slot in paired_schedule([mode], 1):
            serial = slot["mode"].startswith("serial")
            rounds = [{"version": 1, "rows": 4, "selected_local": 1, "selected_corpus": 2,
                       "selected_persistent": 0}] if not serial else []
            metrics = {"family_engaged": not serial, "runtime_disabled": False,
                       "lookup_rounds": int(not serial), "rounds": rounds, "commits": []}
            response = {"text": "March", "usage": {"completion_tokens": 2},
                        "finish_reason": "length", "complete": True}
            probe = {"machine_before": deepcopy(state), "machine_after": deepcopy(state),
                     "metrics": deepcopy(metrics), "successful": True, "report_errors": [],
                     "request_sequence": ["decode", "prefill"],
                     "metric_values": {"decodeTokPerSec": 12 if slot["slot"] in ("feature", "cut") else 10,
                                       "prefillTokPerSec": 100},
                     "matched_replay": {"successful": True, "supported": True, "requests": 2},
                     "responses": [{"request_id": metric, "response": deepcopy(response),
                                    "metrics": [metric + "TokPerSec"], "rounds": deepcopy(rounds)}
                                   for metric in ("decode", "prefill")]}
            event = {**slot, "attempt": 1, "status": "complete", "session_id": "one",
                     "canonical_probe": probe}
            events.append(event)
            requests.append({"arm_id": event["arm_id"], "mode": event["mode"], "boot": 0,
                             "attempt": 1, "stage": "warm", "repetition": 0, "case": "budget",
                             "kind": "budget2", "session_id": "one", "request": {"temperature": 0},
                             "text": "March", "usage": {"completion_tokens": 2}, "finish_reason": "length",
                             "exactness_contract_supported": True, "metrics": metrics,
                             "machine_before": deepcopy(state), "machine_after": deepcopy(state)})
        return events, requests

    def reduce(self, events, requests, isolation="declared"):
        from lookup_bench_evidence import reduce_arms
        arms, _ = reduce_arms(events, requests, isolation, {})
        return next(a for a in arms if a["slot"] == "feature"), arms

    def test_supported_matched_controls_pass_without_borrowing_budget_engagement(self):
        events, requests = self.fixture()
        for row in requests:
            row["metrics"]["family_engaged"] = False
        feature, _ = self.reduce(events, requests)
        self.assertTrue(feature["correctness_valid"], feature)
        self.assertTrue(feature["canonical_comparison_valid"], feature)

    def test_synthetic_local_only_corpus_cannot_validate_feature_attribution(self):
        events, requests = self.fixture()
        for event in events:
            for capture in event["canonical_probe"]["responses"]:
                capture["rounds"] = [{"version": 1, "rows": 4, "selected_local": 3,
                                      "selected_corpus": 0, "selected_persistent": 0}]
        feature, _ = self.reduce(events, requests)
        self.assertFalse(feature["canonical_comparison_valid"])
        self.assertFalse(feature["feature_attribution_valid"])
        self.assertTrue(feature["incremental_comparison_valid"])  # Legitimate no-hit overhead cell.

    def test_synthetic_unsupported_only_sampling_cannot_pass_correctness(self):
        events, requests = self.fixture()
        for row in requests:
            row["exactness_contract_supported"] = False
            if not row["mode"].startswith("serial"):
                row["text"] = "unequal sampled output"
        feature, _ = self.reduce(events, requests)
        self.assertFalse(feature["correctness_valid"])
        self.assertFalse(feature["canonical_comparison_valid"])
        self.assertFalse(feature["incremental_comparison_valid"])

    def test_unstable_protocol_retains_correctness_but_rejects_whole_block(self):
        from bench_lookup_staircut import probe_report_errors, probe_stability_errors
        report = {"bench": {name: {"median": 10, "samples": [10]}
                             for name in ("decodeTokPerSec", "ttftMs", "prefillTokPerSec")}}
        report["bench"]["contextScaling"] = [{"targetTokens": 512, "runs": 1,
                                               "inputTokens": 512, "decodeTokPerSec": 10}]
        report["bench"]["speculative"] = {"predictableTokPerSec": 30, "novelTokPerSec": 10, "offScript": 0}
        report["bench"]["prefixCache"] = {"coldTtftMs": 20, "warmTtftMs": 20}
        report["bench"]["batching"] = {"streams": 4, "singleTokPerSec": 10, "aggregateTokPerSec": 10}
        for verdict, drift in (("improved", 25), ("degraded", -25)):
            report["bench"]["loadDrift"] = {"verdict": verdict, "driftPct": drift}
            self.assertEqual(probe_report_errors(report, [512], 1), [])
            events, requests = self.fixture()
            next(e for e in events if e["slot"] == "aa")["canonical_probe"]["stability_errors"] = probe_stability_errors(report)
            _, arms = self.reduce(events, requests)
            self.assertTrue(all(a["execution_valid"] and a["correctness_valid"] for a in arms))
            self.assertTrue(all(not a["stability_valid"] for a in arms))
            self.assertTrue(all(not a["incremental_comparison_valid"] for a in arms))
        report["bench"]["loadDrift"]["driftPct"] = None
        self.assertIn("canonical loadDrift missing or malformed", probe_report_errors(report, [512], 1))

    def test_auxiliary_protocol_cannot_be_silently_truncated(self):
        from bench_lookup_staircut import probe_report_errors
        for field in ("prefixCache", "batching"):
            errors = probe_report_errors({"bench": {field: None}}, [512], 3)
            self.assertIn("incomplete canonical " + field, errors)
        errors = probe_report_errors({"bench": {"batching": {
            "streams": 2, "singleTokPerSec": 10, "aggregateTokPerSec": 15}}}, [512], 3)
        self.assertIn("canonical batching requires all four streams", errors)

    def test_filtered_or_missing_copy_samples_cannot_certify_reuse_metric(self):
        from bench_lookup_staircut import probe_report_errors
        for off_script in (None, 1):
            report = {"bench": {"speculative": {"predictableTokPerSec": 30, "novelTokPerSec": 10,
                                                "offScript": off_script}}}
            self.assertIn("canonical predictable workload has missing or off-script samples",
                          probe_report_errors(report, [], 1))
        for value in (None, 0, float("inf"), float("nan"), True):
            report = {"bench": {"speculative": {"predictableTokPerSec": value,
                                                "novelTokPerSec": 10, "offScript": 0}}}
            self.assertIn("missing successful speculative:predictableTokPerSec",
                          probe_report_errors(report, [], 1))

    def test_missing_legacy_cannot_produce_incremental_win(self):
        events, requests = self.fixture()
        events = [e for e in events if e["slot"] != "legacy"]
        requests = [r for r in requests if not r["arm_id"].endswith(":legacy")]
        feature, _ = self.reduce(events, requests)
        self.assertFalse(feature["incremental_comparison_valid"])

    def test_attribution_is_independent_for_each_metric(self):
        events, requests = self.fixture()
        feature_event = next(e for e in events if e["slot"] == "feature")
        feature_event["canonical_probe"]["responses"][0]["rounds"] = [
            {"version": 1, "rows": 4, "selected_local": 3, "selected_corpus": 0}]
        feature, _ = self.reduce(events, requests)
        cells = {c["metric"]: c for c in feature["metric_cells"]}
        self.assertFalse(cells["decodeTokPerSec"]["feature_attribution_valid"])
        self.assertTrue(cells["prefillTokPerSec"]["feature_attribution_valid"])

    def test_pressure_swap_power_and_session_reject_complete_block(self):
        for key, value in (("pressure_level", 2), ("swap_used_bytes", 1024),
                           ("swapins", 1), ("swapouts", 1), ("power_source", "battery"),
                           ("thermal", "thermal warning")):
            with self.subTest(key=key):
                events, requests = self.fixture()
                events[1]["canonical_probe"]["machine_after"][key] = value
                _, arms = self.reduce(events, requests)
                self.assertTrue(all(not a["stability_valid"] for a in arms))
        events, requests = self.fixture()
        events[1]["session_id"] = "resumed"
        feature, _ = self.reduce(events, requests)
        self.assertFalse(feature["incremental_comparison_valid"])

    def test_exact_canonical_replay_not_short_fixture_controls_correctness(self):
        events, requests = self.fixture()
        next(e for e in events if e["slot"] == "feature")["canonical_probe"]["matched_replay"]["successful"] = False
        feature, _ = self.reduce(events, requests)
        self.assertFalse(feature["correctness_valid"])
        self.assertFalse(feature["incremental_comparison_valid"])

    def test_equal_work_tag_mismatch_retained_as_descriptive_invalidity(self):
        events, requests = self.fixture()
        next(e for e in events if e["slot"] == "legacy")["canonical_probe"]["request_sequence"] = ["different-tag"]
        feature, _ = self.reduce(events, requests)
        self.assertTrue(feature["correctness_valid"])
        self.assertFalse(feature["incremental_comparison_valid"])
        self.assertIn("canonical_request_sequence_mismatch", feature["canonical_comparison_labels"])

    def test_boot_blocks_not_requests_are_statistical_unit(self):
        from lookup_bench_evidence import bootstrap_effect
        self.assertEqual(bootstrap_effect([1.2, 1.2], .05, 1000, 4)["verdict"],
                         "inconclusive_insufficient_blocks")
        effect = bootstrap_effect([1.2] * 4, .05, 1000, 4)
        self.assertEqual(effect["boot_blocks"], 4)
        self.assertEqual(effect["verdict"], "win")
        self.assertEqual(bootstrap_effect([1.01] * 4, .05, 1000, 4)["verdict"], "inconclusive")


    def test_two_declared_canonical_blocks_cannot_become_confirmation(self):
        from lookup_bench_evidence import incremental_effects, reduce_arms
        events, requests = self.fixture()
        arms, _ = reduce_arms(events, requests, "declared", {})
        feature = next(arm for arm in arms if arm["slot"] == "feature")
        effects = incremental_effects(arms + [deepcopy(feature)],
                                      {"boot_repetitions": 2, "bootstrap_replications": 100})
        self.assertTrue(effects)
        for cell in effects:
            self.assertNotEqual(cell["legacy_primary"]["verdict"], "win")
            self.assertEqual(cell["legacy_primary"]["required_boot_blocks"], 4)


class TransferGateDistinctions(unittest.TestCase):
    def gate(self, family=True, fastest=False, pairs=2, full=True, same_window=True, valid=True):
        from lookup_bench_evidence import transfer_gate_decisions
        return transfer_gate_decisions({"persistent": family}, {"persistent": fastest},
                                       pairs, full, same_window, valid)

    def test_family_improvement_is_not_fastest_native_improvement(self):
        result = self.gate(pairs=4)
        self.assertTrue(result["family_confirmation_pass"])
        self.assertFalse(result["fastest_native_confirmation_pass"])
        self.assertFalse(result["performance_pass"])
        self.assertFalse(result["promotion_pass"])

    def test_two_pairs_diagnose_but_four_confirm(self):
        diagnostic = self.gate(fastest=True)
        self.assertTrue(diagnostic["family_diagnostic_pass"])
        self.assertTrue(diagnostic["fastest_native_diagnostic_pass"])
        self.assertFalse(diagnostic["confirmation_pass"])
        confirmed = self.gate(fastest=True, pairs=4)
        self.assertTrue(confirmed["confirmation_pass"])
        self.assertFalse(confirmed["promotion_pass"])

    def test_separate_window_native_evidence_cannot_confirm(self):
        result = self.gate(fastest=True, pairs=4, same_window=False)
        self.assertTrue(result["family_confirmation_pass"])
        self.assertFalse(result["fastest_native_diagnostic_pass"])
        self.assertFalse(result["confirmation_pass"])

    def test_missing_cases_and_invalid_evidence_cannot_confirm(self):
        self.assertFalse(self.gate(fastest=True, pairs=4, full=False)["confirmation_pass"])
        for key in ("family_diagnostic_pass", "fastest_native_diagnostic_pass", "performance_pass"):
            self.assertFalse(self.gate(fastest=True, pairs=4, valid=False)[key])
        self.assertFalse(self.gate(family=False, fastest=True, pairs=4)["performance_pass"])

    def test_cold_control_regression_is_not_hidden_by_timed_win(self):
        from lookup_bench_evidence import transfer_phase_pass
        cases = ["related-code", "conflicting-config"]
        timed = [{"case": case, "pairs": [
            {"pair": pair, "latency_reduction_percent": 5 if case == "related-code" else -5}
            for pair in range(2)]} for case in cases]
        self.assertTrue(transfer_phase_pass(timed, cases, 2, ["related-code"]))
        cold = deepcopy(timed)
        cold[1]["pairs"][0]["latency_reduction_percent"] = -5.1
        self.assertFalse(transfer_phase_pass(cold, cases, 2, ["related-code"]))
        cold[1]["pairs"][0]["latency_reduction_percent"] = 50
        cold[0]["pairs"][1]["latency_reduction_percent"] = 4.9
        self.assertFalse(transfer_phase_pass(cold, cases, 2, ["related-code"]))

    def test_missing_duplicate_and_nonfinite_pairs_are_rejected(self):
        from lookup_bench_evidence import transfer_phase_pass
        rows = [{"case": "related-code", "pairs": [
            {"pair": 0, "latency_reduction_percent": 10},
            {"pair": 1, "latency_reduction_percent": 10}]}]
        for bad in (float("nan"), float("inf"), None):
            broken = deepcopy(rows)
            broken[0]["pairs"][0]["latency_reduction_percent"] = bad
            self.assertFalse(transfer_phase_pass(broken, ["related-code"], 2, ["related-code"]))
        rows[0]["pairs"][1]["pair"] = 0
        self.assertFalse(transfer_phase_pass(rows, ["related-code"], 2, ["related-code"]))
        self.assertFalse(transfer_phase_pass([], ["related-code"], 2, ["related-code"]))


class RequestedControls(unittest.TestCase):
    def test_serial_only_schedule_preserves_the_requested_kernel_binding(self):
        from bench_lookup_staircut import paired_schedule
        schedule = paired_schedule(["serial-dflash"], 2)
        self.assertEqual([s["mode"] for s in schedule], ["serial-dflash", "serial-dflash"])
        both = paired_schedule(["serial", "serial-dflash"], 2)
        self.assertEqual({s["mode"] for s in both}, {"serial", "serial-dflash"})
        self.assertEqual(len(both), 4)

    def test_equal_work_identity_keeps_generation_controls(self):
        from lookup_bench_evidence import request_identity
        request = {"prompt": "same", "temperature": 0, "enable_thinking": False}
        base = request_identity("/v1/completions", request)
        self.assertEqual(base, request_identity("/v1/completions", {**request, "enable_mtp": True}))
        self.assertNotEqual(base, request_identity("/v1/completions", {**request, "enable_thinking": True}))


class ExplicitNumericalMode(unittest.TestCase):
    def test_launch_evidence_does_not_serialize_inherited_credentials(self):
        from bench_lookup_staircut import evidence_environment
        environment = {
            "HOME": "/private/owned-server",
            "MLX_SERVE_ROW_EXACT": "1",
            "MLX_SERVE_LOOKUP_SELECTION": "adaptive",
            "HF_HUB_OFFLINE": "1",
            "INCO_API_KEY": "provider-secret",
            "HF_TOKEN": "hub-secret",
            "HF_CUSTOM_TOKEN": "unknown-hub-secret",
            "MLX_SERVE_API_KEY": "server-secret",
            "UNFAMILIAR_CREDENTIAL": "future-provider-secret",
        }
        self.assertEqual(evidence_environment(environment), {
            "HOME": "/private/owned-server", "MLX_SERVE_ROW_EXACT": "1",
            "MLX_SERVE_LOOKUP_SELECTION": "adaptive", "HF_HUB_OFFLINE": "1",
        })
        self.assertEqual(environment["INCO_API_KEY"], "provider-secret")

    def test_diagnostic_price_trace_is_never_inherited(self):
        import os
        from unittest.mock import patch
        from bench_lookup_staircut import clean_environment
        key = "MLX_SERVE_LOOKUP_PRICE_TRACE"
        with patch.dict(os.environ, {key: "1"}):
            self.assertNotIn(key, clean_environment("persistent"))
            self.assertEqual(clean_environment("persistent", experiment={key: "1"})[key], "1")
        with self.assertRaises(BenchError):
            clean_environment("persistent", experiment={key: "true"})

    def test_counterfactual_preparation_must_be_declared_and_supported(self):
        import os
        from unittest.mock import patch
        from bench_lookup_staircut import clean_environment
        key = "MLX_SERVE_LOOKUP_COUNTERFACTUAL_PREPARATION"
        with patch.dict(os.environ, {key: "production-neural"}):
            self.assertNotIn(key, clean_environment("persistent"))
        for value in ("merged16", "production-neural", "production-combined"):
            self.assertEqual(clean_environment("persistent", experiment={key: value})[key], value)
        for value in ("", "production", "neural", "merged8", "1"):
            with self.subTest(value=value), self.assertRaises(BenchError):
                clean_environment("persistent", experiment={key: value})

    def test_environment_requires_declared_exact_opt_in(self):
        import os
        from unittest.mock import patch
        from bench_lookup_staircut import clean_environment
        with patch.dict(os.environ, {"MLX_SERVE_ROW_EXACT": "1", "MLX_SERVE_UNDECLARED_KERNEL": "1"}):
            stock = clean_environment("legacy")
            self.assertNotIn("MLX_SERVE_ROW_EXACT", stock)
            self.assertNotIn("MLX_SERVE_UNDECLARED_KERNEL", stock)
            exact = clean_environment("legacy", experiment={"MLX_SERVE_ROW_EXACT": "1"})
            self.assertEqual(exact["MLX_SERVE_ROW_EXACT"], "1")
            self.assertEqual(exact["MLX_SERVE_LOOKUP"], "0")
        for value in ("0", "true", "", "2"):
            with self.subTest(value=value), self.assertRaises(BenchError):
                clean_environment("legacy", experiment={"MLX_SERVE_ROW_EXACT": value})
        with self.assertRaises(BenchError):
            clean_environment("legacy", experiment={"MLX_SERVE_UNDECLARED_KERNEL": "1"})

    def test_activation_must_match_declared_mode(self):
        from bench_lookup_staircut import numerical_mode
        marker = "[row-exact] explicit numerical mode enabled"
        self.assertEqual(numerical_mode({}, "Model ready\n"), "stock")
        self.assertEqual(numerical_mode({"MLX_SERVE_ROW_EXACT": "1"}, marker + "\nModel ready\n"), "row-exact")
        for log in ("Model ready\n", marker + " requested\nModel ready\n"):
            with self.subTest(log=log), self.assertRaises(BenchError):
                numerical_mode({"MLX_SERVE_ROW_EXACT": "1"}, log)
        with self.assertRaises(BenchError):
            numerical_mode({}, marker + "\nModel ready\n")

    def test_all_exact_arms_share_mode_without_fake_drafter(self):
        from types import SimpleNamespace
        from bench_lookup_staircut import Server, request_body
        args = SimpleNamespace(port=11279, ctx_size=2048)
        model = {"model_dir": "/target", "drafter_dir": "/drafter"}
        case = {"prompt": "same frozen request", "max_tokens": 192, "temperature": 0, "seed": 0}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("serial", "mtp-legacy", "dflash-legacy", "mtp-persistent", "dflash-persistent"):
                with self.subTest(mode=mode):
                    server = Server(args, "/binary", model, mode, Path(tmp) / (mode + ".log"),
                                    experiment={"MLX_SERVE_ROW_EXACT": "1"})
                    self.assertEqual(server.env["MLX_SERVE_ROW_EXACT"], "1")
                    self.assertEqual("--drafter" in server.command, mode.startswith("dflash-"))
                    self.assertEqual("--no-drafter" in server.command, not mode.startswith("dflash-"))
                    if mode == "serial":
                        self.assertIn("--no-mtp", server.command)
                        self.assertIn("--no-pld", server.command)
                        self.assertEqual(server.env["MLX_SERVE_LOOKUP"], "0")
                        body = request_body(case, mode, None)
                        for key in ("enable_mtp", "enable_drafter", "enable_pld"):
                            self.assertIs(body[key], False)


class ControlledCandidates(unittest.TestCase):
    def test_candidate_identity_mismatch_cannot_be_called_cut_only(self):
        from lookup_bench_evidence import candidate_match
        left = {"responses": [{"request_id": "request", "metrics": ["decode"],
                               "rounds": [{"family": "mtp", "kind": "chain", "class": 0,
                                           "anchor_pos": 8, "prefix_hash": "prefix",
                                           "candidate_hash": "offered", "candidate_rows": 8,
                                           "candidate_probability_hash": "calibrated",
                                           "legal_rows": 8}]}]}
        right = deepcopy(left)
        self.assertTrue(candidate_match(left, right, "decode")["valid"])
        right["responses"][0]["rounds"][0]["candidate_hash"] = "different-offered-set"
        self.assertFalse(candidate_match(left, right, "decode")["valid"])
        self.assertFalse(candidate_match(left, left, "unrelated-metric")["valid"])

    def test_probability_changes_invalidate_a_same_token_cut_ablation(self):
        from lookup_bench_evidence import candidate_match
        probe = {"responses": [{"request_id": "request", "metrics": ["decode"],
                 "rounds": [{"family": "dflash", "kind": "tree", "class": 0,
                            "anchor_pos": 8, "prefix_hash": 123, "candidate_hash": 456,
                            "candidate_probability_hash": 789, "candidate_rows": 8,
                            "legal_rows": 8}]}]}
        changed = deepcopy(probe)
        changed["responses"][0]["rounds"][0]["candidate_probability_hash"] = 790
        self.assertFalse(candidate_match(probe, changed, "decode")["valid"])


class StreamingRequests(unittest.TestCase):
    def consume(self, case, events, *, done=True, mode="mtp-persistent", cap=None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from types import SimpleNamespace
        from bench_lookup_staircut import request, request_body

        received = []
        wire = "".join("data: " + (event if isinstance(event, str) else json.dumps(event)) +
                       "\n\n" for event in events)
        if done:
            wire += "data: [DONE]\n\n"
        encoded = wire.encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                received.append((self.path, json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                # Split across SSE lines and JSON boundaries, not just content chunks.
                for offset in range(0, len(encoded), 7):
                    self.wfile.write(encoded[offset:offset + 7])
                    self.wfile.flush()

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            try:
                result = request(SimpleNamespace(base=f"http://127.0.0.1:{http.server_port}"),
                                 request_body(case, mode, cap), timeout=2)
            finally:
                http.shutdown()
                thread.join()
        self.assertEqual(len(received), 1)
        return result, received[0]

    @staticmethod
    def chat_case():
        return {"messages": [{"role": "system", "content": "Output only the answer."},
                             {"role": "user", "content": "Name the third month."}],
                "max_tokens": 32, "temperature": 0, "seed": 0,
                "enable_thinking": False, "reasoning_effort": "none"}

    def test_fragmented_chat_content_separate_usage_and_explicit_thinking_off(self):
        events = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": None, "reasoning_content": "not visible"}}]},
            {"choices": [{"delta": {"content": "Mar"}, "text": "not chat content"}]},
            {"choices": [{"delta": {"content": "ch"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                                     "total_tokens": 14}, "timings": {"decode": 0.1}},
        ]
        result, (path, body) = self.consume(self.chat_case(), events, cap=16)
        self.assertNotIn("error", result)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["messages"], self.chat_case()["messages"])
        self.assertNotIn("prompt", body)
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["max_tokens"], 16)
        self.assertEqual((body["temperature"], body["seed"]), (0, 0))
        self.assertEqual((body["enable_mtp"], body["enable_drafter"], body["enable_pld"]),
                         (True, False, False))
        self.assertEqual((body["top_p"], body["top_k"], body["min_p"],
                          body["repeat_penalty"], body["presence_penalty"],
                          body["frequency_penalty"]), (1.0, 0, 0.0, 1.0, 0.0, 0.0))
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(result["text"], "March")
        self.assertEqual(result["emitted_pieces"], ["Mar", "ch"])
        self.assertEqual(result["events"], events)
        self.assertEqual(result["usage"], events[-1]["usage"])
        self.assertEqual(result["server_timings"], {"decode": 0.1})
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["http_status"], 200)
        self.assertIsNotNone(result["ttft_seconds"])
        self.assertLessEqual(result["ttft_seconds"], result["total_seconds"])
        self.assertIsNone(result["token_ids"])

    def test_chat_rejects_incomplete_and_abnormal_finishes(self):
        content = {"choices": [{"delta": {"content": "March"}}]}
        finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        usage = {"choices": [], "usage": {"completion_tokens": 2}}
        variants = [
            ("missing_done", [content, finish, usage], False, "incomplete SSE"),
            ("missing_usage", [content, finish], True, "incomplete SSE"),
            ("missing_finish", [content, usage], True, "incomplete SSE"),
            ("abnormal_finish", [content, {"choices": [{"finish_reason": "error"}]}, usage],
             True, "abnormal generation finish"),
            ("finish_details", [content, {"choices": [{"finish_reason": "stop",
                "finish_details": {"error": "failed"}}]}, usage], True, "abnormal generation finish"),
            ("malformed_json", [content, "{broken"], True, "JSONDecodeError"),
            ("server_error", [content, {"error": {"message": "failed"}}], True, "failed"),
        ]
        for label, events, done, error in variants:
            with self.subTest(label=label):
                result, _ = self.consume(self.chat_case(), events, done=done)
                self.assertIn(error, result["error"])
                self.assertEqual(result["text"], "March")

    def test_role_and_null_content_do_not_start_ttft(self):
        result, _ = self.consume(self.chat_case(), [
            {"choices": [{"delta": {"role": "assistant", "content": None,
                                   "reasoning_content": "hidden"}}]},
            {"choices": [{"delta": None, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 0}},
        ])
        self.assertNotIn("error", result)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["emitted_pieces"], [])
        self.assertIsNone(result["ttft_seconds"])

    def test_completion_consumer_and_implicit_thinking_defaults_remain_unchanged(self):
        case = {"prompt": "Name a month.", "max_tokens": 2, "temperature": 0, "seed": 0}
        events = [{"choices": [{"text": "Mar", "delta": {"content": "not completion"}}]},
                  {"choices": [{"text": "ch", "finish_reason": "length"}],
                   "usage": {"completion_tokens": 2}}]
        for mode, flags in (("dflash-legacy", (False, True, False)),
                            ("pld-local", (False, False, True)),
                            ("serial", (False, False, False))):
            with self.subTest(mode=mode):
                result, (path, body) = self.consume(case, events, mode=mode)
                self.assertNotIn("error", result)
                self.assertEqual(path, "/v1/completions")
                self.assertEqual(body["prompt"], case["prompt"])
                self.assertNotIn("messages", body)
                self.assertNotIn("enable_thinking", body)
                self.assertNotIn("reasoning_effort", body)
                self.assertEqual((body["enable_mtp"], body["enable_drafter"],
                                  body["enable_pld"]), flags)
                self.assertEqual(result["text"], "March")
                self.assertEqual(result["emitted_pieces"], ["Mar", "ch"])
                self.assertEqual(result["usage"], {"completion_tokens": 2})
                self.assertEqual(result["finish_reason"], "length")


class ReceiptEvidence(unittest.TestCase):
    def test_mtp_lookup_only_route_is_engaged(self):
        from bench_lookup_staircut import parse_stats
        evidence = parse_stats("[spec-stats] mode=mtp attempts=0 accepts=0 lookup=3/9/6", "mtp")
        self.assertTrue(evidence["family_engaged"])
        self.assertEqual(evidence["lookup_rounds"], 3)

    def test_successful_stream_retains_events_and_rejects_missing_done(self):
        from bench_lookup_staircut import response_receipt
        raw = 'data: {"choices":[{"text":"March","finish_reason":"length"}],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n'
        result = response_receipt(raw, True)
        self.assertEqual(result["text"], "March")
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["events"]), 1)
        self.assertFalse(response_receipt(raw.replace("data: [DONE]", ""), True)["complete"])

    def test_commit_ids_not_retokenized_text_define_token_identity(self):
        from lookup_bench_evidence import response_equal
        receipt = {"complete": True, "text": "same", "finish_reason": "stop",
                   "usage": {"completion_tokens": 1},
                   "commit": {"prompt_ids": [1], "generated_ids": [2], "finish_reason": "stop"}}
        changed = deepcopy(receipt)
        changed["commit"]["generated_ids"] = [3]
        self.assertFalse(response_equal(receipt, changed))

    def test_overlapping_log_windows_do_not_mix_request_origins(self):
        from lookup_bench_evidence import attach_canonical_receipts
        captures = [{"request": {"messages": [{"content": "[probe same-0] copy"}]},
                     "response": {}, "commits": []},
                    {"request": {"messages": [{"content": "[probe same-1] code"}]},
                     "response": {}, "commits": []}]
        commits = [{"prompt_hash": 11, "prompt_ids": [1], "generated_ids": [2],
                    "finish_reason": "length"},
                   {"prompt_hash": 22, "prompt_ids": [3], "generated_ids": [4],
                    "finish_reason": "length"}]
        rounds = [{"prompt_hash": 11, "selected_local": 1},
                  {"prompt_hash": 22, "selected_corpus": 1}]
        attach_canonical_receipts(captures, commits, rounds,
            {"11": "user: [probe same-0] copy", "22": "user: [probe same-1] code"})
        self.assertEqual(captures[0]["rounds"], [rounds[0]])
        self.assertEqual(captures[1]["rounds"], [rounds[1]])
        self.assertEqual(captures[0]["response"]["commit"]["generated_ids"], [2])

    def test_repeated_prompt_with_different_token_results_is_ambiguous(self):
        from lookup_bench_evidence import attach_canonical_receipts
        capture = {"request": {"prompt": "same prompt"}, "response": {}}
        commits = [{"prompt_hash": 11, "prompt_ids": [1], "generated_ids": [token],
                    "finish_reason": "length"} for token in (2, 3)]
        attach_canonical_receipts([capture], commits,
                                 [{"prompt_hash": 11, "selected_corpus": 1}],
                                 {"11": "same prompt"})
        self.assertTrue(capture["concurrent_or_receipt_ambiguous"])
        self.assertNotIn("commit", capture["response"])
        self.assertEqual(capture["rounds"], [])

    def test_seed_validation_rejects_partial_wrong_or_unseparated_documents(self):
        from bench_lookup_staircut import validate_seed_store
        sha = "01" * 32
        commits = [{"version": 1, "prompt_ids": [11, 12], "generated_ids": [13]},
                   {"version": 1, "prompt_ids": [21], "generated_ids": [22]}]
        header = b"MLXLOOK1" + bytes.fromhex(sha)
        ids = [11, 12, 13, 1 << 30, 21, 22, 1 << 30]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def put(values, prefix=header):
                (root / "tokens.bin").write_bytes(prefix + struct.pack("<" + "I" * len(values), *values))
            put(ids)
            self.assertEqual(validate_seed_store(root, sha, commits)["documents"], 2)
            for values in (ids[:4], ids[:-1], [11, 12, 99, 1 << 30, 21, 22, 1 << 30]):
                put(values)
                with self.assertRaises(BenchError):
                    validate_seed_store(root, sha, commits)
            put(ids, b"MLXLOOK1" + bytes(32))
            with self.assertRaises(BenchError):
                validate_seed_store(root, sha, commits)

    def test_tag_instrumentation_changes_only_initializer_not_timing_clock(self):
        from bench_lookup_staircut import controlled_probe_command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "node_modules" / "llmprobe" / "bin" / "dist" / "llmprobe.mjs"
            source.parent.mkdir(parents=True)
            original = 'const metadata = {version: "0.6.17"};\nvar CACHE_BUST_TAG = Date.now().toString(36);\nconst elapsed = Date.now() - started;\n'
            source.write_text(original)
            out = root / "arm"
            out.mkdir()
            command = controlled_probe_command(["/local/node", str(source)], out, "blocktag")
            derived = Path(command[1]).read_text()
            self.assertEqual(derived, original.replace("Date.now().toString(36)", '"blocktag"'))
            self.assertEqual(source.read_text(), original)
            source.write_text(original + "var CACHE_BUST_TAG = Date.now().toString(36);")
            with self.assertRaises(BenchError):
                controlled_probe_command(["/local/node", str(source)], out, "blocktag")


class CanonicalWorkloadCoverage(unittest.TestCase):
    def test_copy_and_novel_metrics_exclude_warmup_and_drift_requests(self):
        from bench_lookup_staircut import canonical_metric_evidence
        names = ("decode", "prefill", "predictable", "novel")
        builders = [{"id": "canonical-" + name, "prompt": name.upper() + "_WORKLOAD"} for name in names]
        captures = [{"request": {"messages": [{"role": "user", "content": name.upper() + "_WORKLOAD"}]}}
                    for name, count in zip(names, (4, 3, 3, 3)) for _ in range(count)]
        captures.extend(deepcopy(captures[:2]))
        report = {"bench": {metric: {"median": 10} for metric in
                            ("decodeTokPerSec", "ttftMs", "prefillTokPerSec")}}
        report["bench"]["speculative"] = {"predictableTokPerSec": 30, "novelTokPerSec": 8}
        evidence = canonical_metric_evidence(report, captures, builders, 2)
        for metric, indices in (("decodeTokPerSec", [2, 3]),
                                ("speculative:predictableTokPerSec", [8, 9]),
                                ("speculative:novelTokPerSec", [11, 12])):
            self.assertEqual([i for i, c in enumerate(captures) if metric in c["metrics"]], indices)
            self.assertEqual(evidence["metric_request_counts"][metric], 2)
        self.assertEqual(evidence["metric_values"]["speculative:predictableTokPerSec"], 30)


class CanonicalMetricReplay(unittest.TestCase):
    @staticmethod
    def fixture():
        from lookup_bench_evidence import request_identity
        captures, commits, rows = [], [], []
        labels = [["decodeTokPerSec", "ttftMs"], ["prefillTokPerSec"],
                  ["context:65536:decodeTokPerSec", "context:65536:ttftMs",
                   "context:65536:prefillTokPerSec"], []]
        for index, metrics in enumerate(labels):
            body = {"prompt": f"actual request {index}", "temperature": 0}
            commit = {"prompt_hash": index, "prompt_ids": [10, index],
                      "generated_ids": [20, index], "finish_reason": "length"}
            response = {"complete": True, "text": "same visible text", "finish_reason": "length",
                        "usage": {"completion_tokens": 2}, "commit": deepcopy(commit)}
            capture = {"path": "/v1/completions", "request": body,
                       "request_id": request_identity("/v1/completions", body),
                       "request_index": index, "response": response, "metrics": metrics}
            captures.append(capture)
            commits.append(commit)
            for reference in ("serial", "mtp-legacy"):
                rows.append({**deepcopy(capture), "reference": reference})
        values = {metric: 10 for group in labels for metric in group}
        return {"responses": captures, "metrics": {"commits": commits, "rounds": []},
                "metric_values": values, "metric_request_counts": {m: 1 for m in values}}, rows

    @staticmethod
    def classify(probe, rows):
        from lookup_bench_evidence import classify_canonical_replay
        return classify_canonical_replay(probe, rows, ["serial", "mtp-legacy"], "mtp-local")

    def test_unmeasured_batch_difference_and_missing_aux_receipt_are_not_certified(self):
        probe, rows = self.fixture()
        rows[-1]["response"]["commit"]["generated_ids"] = [99]
        rows[-2]["response"].pop("commit")
        result = self.classify(probe, rows)
        self.assertTrue(result["successful"], result)
        self.assertTrue(result["supported"])
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["captured_requests"], 4)
        self.assertFalse(result["blanket_canonical_correctness"])
        self.assertFalse(result["auxiliary_diagnostics"]["supported"])
        self.assertEqual(result["auxiliary_diagnostics"]["observed_differences_or_missing_receipts"], 2)
        self.assertTrue(all(not r["supported"] for r in result["results"] if not r["in_metric_scope"]))

    def test_every_measured_request_needs_every_reference_including_long_context(self):
        for missing in (0, 2, 4, 5):
            with self.subTest(missing=missing):
                probe, rows = self.fixture()
                del rows[missing]
                result = self.classify(probe, rows)
                self.assertFalse(result["successful"], result)
                self.assertFalse(result["supported"])

    def test_long_context_token_difference_cannot_borrow_short_fixture_proof(self):
        probe, rows = self.fixture()
        rows[4]["response"]["commit"]["generated_ids"] = [20, 999]
        result = self.classify(probe, rows)
        self.assertFalse(result["successful"], result)
        context = [r for r in result["results"] if "context:65536:ttftMs" in r["metrics"]]
        self.assertTrue(any(not r["emission_exact"] for r in context))

    def test_reference_commit_never_substitutes_for_missing_canonical_commit(self):
        probe, rows = self.fixture()
        probe["responses"][2]["response"].pop("commit")
        del probe["metrics"]["commits"][2]
        result = self.classify(probe, rows)
        self.assertFalse(result["successful"], result)
        self.assertFalse(result["supported"])
        self.assertNotIn("commit", probe["responses"][2]["response"])

    def test_missing_metric_and_partial_repetition_coverage_fail(self):
        probe, rows = self.fixture()
        probe["responses"][2]["metrics"] = []
        result = self.classify(probe, rows)
        self.assertFalse(result["successful"], result)
        self.assertEqual(len(result["coverage_errors"]), 3)
        probe, rows = self.fixture()
        probe["metric_request_counts"]["decodeTokPerSec"] = 2
        result = self.classify(probe, rows)
        self.assertFalse(result["supported"], result)
        self.assertEqual(result["coverage_errors"][0]["expected"], 2)

    def test_tools_and_thinking_disambiguate_only_from_actual_rendered_ids(self):
        from lookup_bench_evidence import attach_canonical_receipts, request_identity
        probe, rows = self.fixture()
        # Same user text, three real renderings: default thinking, tools, disabled thinking.
        bodies = [{"messages": [{"role": "user", "content": "same user text"}]},
                  {"messages": [{"role": "user", "content": "same user text"}],
                   "tools": [{"type": "function", "function": {"name": "weather"}}]},
                  {"messages": [{"role": "user", "content": "same user text"}],
                   "reasoning_effort": "none"}]
        captures, commits, references, decoded = [], [], [], {}
        for index, body in enumerate(bodies, 10):
            commit = {"prompt_hash": index, "prompt_ids": [100, index],
                      "generated_ids": [200, index], "finish_reason": "length"}
            response = {"complete": True, "text": "same", "finish_reason": "length",
                        "usage": {"completion_tokens": 2}}
            capture = {"path": "/v1/chat/completions", "request": body,
                       "request_id": request_identity("/v1/chat/completions", body),
                       "request_index": index, "response": response, "metrics": []}
            captures.append(capture)
            commits.append(commit)
            decoded[str(index)] = f"rendered {index}: same user text"
            for reference in ("serial", "mtp-legacy"):
                references.append({**deepcopy(capture), "reference": reference,
                                   "response": {**deepcopy(response), "commit": deepcopy(commit)}})
        attach_canonical_receipts(captures, commits, [], decoded)
        self.assertTrue(all(c["concurrent_or_receipt_ambiguous"] for c in captures))
        probe["responses"].extend(captures)
        probe["metrics"]["commits"].extend(commits)
        result = self.classify(probe, rows + references)
        self.assertTrue(result["successful"], result)
        for capture, commit in zip(captures, commits):
            self.assertEqual(capture["response"]["commit"], commit)
            self.assertFalse(capture["concurrent_or_receipt_ambiguous"])
            self.assertTrue(capture["content_receipt_ambiguous"])
            self.assertEqual(capture["receipt_match_basis"], "reference_rendered_prompt_ids")
        # Render IDs are not generated-token evidence: a canonical token
        # disagreement stays a disagreement even with identical visible text.
        probe["metrics"]["commits"][-1]["generated_ids"] = [999]
        result = self.classify(probe, rows + references)
        auxiliary = [r for r in result["results"] if r["request_index"] == 12]
        self.assertTrue(all(not r["emission_exact"] for r in auxiliary))
        self.assertTrue(result["successful"])  # Still explicitly outside metric scope.

    def test_conflicting_reference_renderings_preserve_ambiguity(self):
        probe, rows = self.fixture()
        rows[1]["response"]["commit"]["prompt_ids"] = [404]
        result = self.classify(probe, rows)
        self.assertFalse(result["supported"])
        self.assertTrue(probe["responses"][0]["concurrent_or_receipt_ambiguous"])
        self.assertTrue(probe["responses"][0]["reference_prompt_ambiguous"])
        self.assertNotIn("commit", probe["responses"][0]["response"])

    def test_auxiliary_replay_http_error_is_still_fatal(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from urllib.error import HTTPError
        from bench_lookup_staircut import replay_canonical
        probe, _ = self.fixture()
        probe["responses"] = [probe["responses"][-1]]
        error = HTTPError("http://127.0.0.1/v1/completions", 500, "generation failed", {}, None)
        with tempfile.TemporaryDirectory() as directory, patch(
                "bench_lookup_staircut.Server", autospec=True) as owned, patch(
                "bench_lookup_staircut.HTTP.open", side_effect=error):
            server = owned.return_value
            server.__enter__.return_value = server
            server.base = "http://127.0.0.1"
            server.log_path = Path(directory) / "server.log"
            server.log_path.touch()
            with self.assertRaises(HTTPError):
                replay_canonical(SimpleNamespace(request_timeout=1, row_exact_tree=False),
                                 None, {}, "mtp-local", probe, Path("/unused"))

    def test_reference_modes_are_unique_without_dropping_legacy(self):
        from lookup_bench_evidence import canonical_reference_modes
        self.assertEqual(canonical_reference_modes("serial-dflash"), ["serial-dflash"])
        self.assertEqual(canonical_reference_modes("serial"), ["serial"])
        self.assertEqual(canonical_reference_modes("dflash-local"), ["serial-dflash", "dflash-legacy"])
        self.assertEqual(canonical_reference_modes("mtp-local"), ["serial", "mtp-legacy"])


class CheckpointSelection(unittest.TestCase):
    @staticmethod
    def shard(path, *, metadata=None):
        header = json.dumps({"__metadata__": metadata, "model.embed_tokens.weight": {
            "dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))

    def test_null_metadata_and_unused_export_do_not_break_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.shard(root / "selected.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.embed_tokens.weight": "selected.safetensors"}}))
            (root / "stray.safetensors").write_bytes(b"partial unused export")
            served, _, audit = checkpoint_audit(root)
            self.assertEqual(served["model.embed_tokens.weight"]["shape"], [2, 2])
            self.assertEqual([Path(s["path"]).name for s in audit["shards"]],
                             ["selected.safetensors"])

    def test_selected_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            self.shard(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ReadinessError):
                safetensors_header(path)


class ProbeErrors(unittest.TestCase):
    def test_only_empty_discovery_bad_request_is_expected(self):
        from bench_lookup_staircut import probe_http_failed
        self.assertFalse(probe_http_failed({"http_status": 400,
                                           "request_kind": "empty_discovery"}))
        for evidence in (
            {"http_status": 400, "request_kind": "generation"},
            {"http_status": 500, "request_kind": "empty_discovery"},
            {"proxy_error": "connection reset", "request_kind": "generation"},
        ):
            with self.subTest(evidence=evidence):
                self.assertTrue(probe_http_failed(evidence))


class DrafterContract(unittest.TestCase):
    def test_nested_fallback_ordering_and_selector_pairs(self):
        from prepare_lookup_offline import audit_dflash_contract
        root = {"block_size": 8, "hidden_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "head_dim": 8, "intermediate_size": 32,
                "rms_norm_eps": 1e-6,
                "dflash_config": {"mask_token_id": 19, "target_layer_ids": [1, 3],
                                  "selector_rank": 8, "selector_top_k": 4}}
        self.assertTrue(audit_dflash_contract(root, 4, "fixture"))
        for change in ({"target_layer_ids": [3, 1]}, {"target_layer_ids": [1, 1]},
                       {"selector_top_k": 1}, {"conv_kernel_size": 3},
                       {"mask_token_id": -1}, {"block_size": None}):
            bad = {**root, "dflash_config": {**root["dflash_config"], **change}}
            with self.subTest(change=change), self.assertRaises(ReadinessError):
                audit_dflash_contract(bad, 4, "fixture")


class TransferMeasurementQualification(unittest.TestCase):
    def test_contention_and_machine_changes_exclude_confirmation(self):
        from lookup_bench_evidence import transfer_measurement_errors, transfer_gate_decisions
        normal = {"power_source": "ac", "power_settings": "lowpowermode 0",
                  "thermal": "No thermal warning level has been recorded",
                  "pressure_level": 1, "swap_used_bytes": 0, "swapins": 0, "swapouts": 0}
        boots = [{"machine_before": dict(normal), "machine_after": dict(normal)}]
        self.assertEqual(transfer_measurement_errors(boots, "declared"), [])
        cases = [(boots, "contended"), ([{}], "declared")]
        for key, changed in (("thermal", "CPU_Speed_Limit = 80"), ("swapouts", 1),
                             ("pressure_level", 2), ("power_source", "battery")):
            altered = deepcopy(boots)
            altered[0]["machine_after"][key] = changed
            cases.append((altered, "declared"))
        for evidence, isolation in cases:
            with self.subTest(evidence=evidence, isolation=isolation):
                exclusions = transfer_measurement_errors(evidence, isolation)
                self.assertTrue(exclusions)
                decision = transfer_gate_decisions({"persistent": True}, {"persistent": True},
                                                  4, True, True, valid=not exclusions)
                self.assertFalse(decision["family_confirmation_pass"])
                self.assertFalse(decision["performance_pass"])


if __name__ == "__main__":
    unittest.main()
