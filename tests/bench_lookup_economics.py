#!/usr/bin/env python3
"""Offline within-family row economics, NOT llmprobe or a throughput benchmark.

Each action gets a fresh owned process, the same immutable starting store, and
an explicitly recorded priming sequence. All legal root-included widths at the
first offered decision are measured, not inferred from the chosen action.
Training and held-out boots are disjoint. Invalid/missing/incomparable receipts
remain evidence, but never contribute to price accuracy or decision regret.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import statistics
import struct
import sys
import time

from lookup_bench_evidence import selected_origin, stable_states
import bench_lookup_staircut as runner

VERSION = 1
MAX_ROWS = 64
CONTEXT_BUCKET = "floor-power-of-two-2048"


def identity(value):
    return hashlib.sha256(runner.canonical(value).encode()).hexdigest()


def finite(value, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def receipt_errors(receipt, action):
    """Do not coerce caps into exact widths or treat censored emissions as yield."""
    if not isinstance(receipt, dict):
        return ["missing_round_receipt"]
    errors = []
    required = ("version", "family", "kind", "class", "anchor_pos", "prefix_hash",
                "candidate_hash", "candidate_probability_hash", "candidate_rows", "legal_rows",
                "requested_rows", "selection",
                "rows", "accepted", "committed", "draft_ms", "verify_commit_ms",
                "round_ms", "priced")
    errors += ["missing_" + key for key in required if receipt.get(key) is None]
    if errors:
        return errors
    if receipt["version"] != VERSION:
        errors.append("unsupported_receipt_version")
    if receipt.get("context_bucket") != CONTEXT_BUCKET:
        errors.append("unsupported_context_bucket")
    for key in ("class", "anchor_pos", "candidate_rows", "legal_rows", "requested_rows", "rows",
                "accepted", "committed"):
        if type(receipt[key]) is not int or receipt[key] < 0:
            errors.append("invalid_" + key)
    if errors:
        return errors
    if receipt["family"] not in ("mtp", "pld", "dflash", "serial"):
        errors.append("unsupported_family")
    if receipt["kind"] not in ("chain", "tree"):
        errors.append("unsupported_kind")
    if not 1 <= receipt["rows"] <= receipt["legal_rows"] <= receipt["candidate_rows"] <= MAX_ROWS:
        errors.append("illegal_row_geometry")
    if receipt["accepted"] >= receipt["rows"]:
        errors.append("invalid_raw_acceptance")
    if receipt["committed"] != receipt["accepted"] + 1:
        errors.append("emission_censored")
    if receipt.get("admissible") is False:
        errors.append("inadmissible_action")
    if receipt["priced"] is not True:
        errors.append("unpriced_observation")
    for key in ("draft_ms", "verify_commit_ms", "round_ms"):
        if not finite(receipt[key], positive=key != "draft_ms"):
            errors.append("invalid_" + key)
    if not errors and receipt["round_ms"] + 0.05 < receipt["draft_ms"] + receipt["verify_commit_ms"]:
        errors.append("round_cost_excludes_candidate_construction_or_verify_commit")
    if action.startswith("fixed:"):
        width = int(action.split(":")[1])
        if receipt["selection"] != "fixed" or receipt["requested_rows"] != width:
            errors.append("selection_not_applied")
        if receipt["rows"] != width:
            errors.append("requested_width_not_measured_exactly")
    elif action in ("adaptive", "cut", "max"):
        if receipt["selection"] != action or receipt["requested_rows"] != 0:
            errors.append("selection_not_applied")
        if action == "max" and receipt["rows"] != receipt["legal_rows"]:
            errors.append("max_did_not_measure_full_offer")
    return errors


def first_offer_origin_errors(profile, receipt, action):
    """A declared lookup witness must reach that tier in the full first offer."""
    required = profile.get("economics_required_origin")
    if not required or action != "max":
        return []
    if required not in ("local", "corpus", "persistent"):
        return ["invalid_required_first_offer_origin"]
    if not isinstance(receipt, dict) or not selected_origin([receipt], required):
        return ["required_first_offer_origin_missing:" + required]
    return []


def first_offer_coverage(rows):
    return [{"mode": row["mode"], "profile": row["profile"], "repetition": row["repetition"],
             "rows": row["round"]["rows"], "accepted": row["round"]["accepted"],
             "selected_origins": {tier: row["round"].get("selected_" + tier, 0)
                                  for tier in ("local", "corpus", "persistent", "neural")},
             "lookup_exercised": selected_origin([row["round"]], "all")}
            for row in rows if row["action"] == "max"]


def offer_key(receipt):
    return tuple(receipt.get(key) for key in
                 ("family", "kind", "class", "anchor_pos", "prefix_hash",
                  "candidate_hash", "candidate_probability_hash", "candidate_rows", "legal_rows"))


def no_assistant_root(receipt):
    """The sole candidate-identity exception: a genuinely unassisted root."""
    return (receipt.get("root_no_assistant") is True
            and receipt.get("assistant") is False and receipt.get("selected_neural") == 0
            and receipt.get("draft_ms") == 0 and receipt.get("rows") == 1
            and receipt.get("candidate_rows") == 1 and receipt.get("legal_rows") == 1
            and receipt.get("accepted") == 0)


def curve_key(row):
    r = row["round"]
    return runner.canonical([row["mode"], row["profile"], r["kind"], r["class"], r["rows"],
                             "no-assistant" if no_assistant_root(r) else "candidate-paid"])


def causal_block_errors(block):
    errors = []
    if any(row.get("isolation") != "declared" for row in block):
        errors.append("gpu_isolation_not_declared")
    states = [row.get(key) or {} for row in block for key in ("machine_before", "machine_after")]
    errors.extend(stable_states(states))
    return errors


def fit_curves(rows, widths, causal=False):
    """Train only on complete matched training blocks, never held-out rows."""
    groups, cells = {}, {}
    for row in rows:
        if row["phase"] == "train":
            groups.setdefault((row["mode"], row["profile"], row["repetition"]), []).append(row)
    for (mode, profile, _repetition), block in groups.items():
        if not compare_block(block, widths[mode + "/" + profile], {})["comparable"]:
            continue
        if causal and causal_block_errors(block):
            continue
        for row in block:
            if row["action"].startswith("fixed:"):
                cells.setdefault(curve_key(row), []).append(row["round"])
    return {key: {"n": len(values), "context_bucket": CONTEXT_BUCKET,
                  "verify_ms": statistics.mean(r["verify_commit_ms"] for r in values),
                  "draft_ms": statistics.mean(r["draft_ms"] for r in values),
                  "round_ms": statistics.mean(r["round_ms"] for r in values),
                  "tokens": statistics.mean(r["accepted"] + 1 for r in values)}
            for key, values in cells.items()}


def interval(values, seed=1701):
    """Percentile bootstrap over independent matched boot blocks, not rounds."""
    if not values:
        return {"n_blocks": 0, "mean": None, "ci95": None}
    result = {"n_blocks": len(values), "mean": statistics.mean(values), "ci95": None}
    if len(values) > 1:
        rng = random.Random(seed)
        samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        result["ci95"] = [samples[49], samples[1949]]
    return result


def prediction_summary(pairs):
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "calibration": []}
    errors = [p - actual for p, actual in pairs]
    ordered = sorted(pairs)
    bins = []
    width = max(1, math.ceil(len(ordered) / 5))
    for start in range(0, len(ordered), width):
        cell = ordered[start:start + width]
        bins.append({"n": len(cell), "predicted_mean": statistics.mean(p for p, _ in cell),
                     "observed_mean": statistics.mean(a for _, a in cell)})
    return {"n": len(pairs), "mae": statistics.mean(abs(e) for e in errors),
            "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
            "bias": statistics.mean(errors), "calibration": bins}


def compare_block(block, expected_widths, curves):
    """A measured oracle is only over the COMPLETE, identical offered action set."""
    errors = []
    actions = {row["action"]: row for row in block}
    required = ["fixed:" + str(width) for width in expected_widths] + ["adaptive", "cut", "max"]
    if len(actions) != len(block):
        errors.append("duplicate_actions")
    for action in required:
        if action not in actions:
            errors.append("missing_action:" + action)
        elif (actions[action].get("errors") or receipt_errors(actions[action].get("round"), action)
              or not complete_receipt(actions[action].get("commit"))):
            errors.append("invalid_action:" + action)
    if errors:
        return {"comparable": False, "errors": errors}
    reference = actions["max"]["round"]
    for action in required:
        receipt = actions[action]["round"]
        root = no_assistant_root(receipt)
        keys = ("family", "kind", "class", "anchor_pos", "prefix_hash")
        if (any(receipt.get(key) != reference.get(key) for key in keys)
                or (not root and offer_key(receipt) != offer_key(reference))):
            return {"comparable": False, "errors": ["different_first_offered_prefix_candidates_kind_or_class:" + action]}
    if not no_assistant_root(actions["fixed:1"]["round"]):
        return {"comparable": False, "errors": ["missing_real_no_assistant_root_alternative"]}
    prompt_ids = {runner.canonical(actions[action].get("commit", {}).get("prompt_ids")) for action in required}
    outputs = {runner.canonical([actions[action].get("commit", {}).get("generated_ids"),
                                 actions[action].get("commit", {}).get("finish_reason")]) for action in required}
    if len(prompt_ids) != 1 or len(outputs) != 1:
        return {"comparable": False, "errors": ["different_prompt_or_committed_token_stream"]}
    priming = {runner.canonical(actions[action].get("priming_commits")) for action in required}
    if len(priming) != 1:
        return {"comparable": False, "errors": ["different_prescribed_priming_token_stream"]}
    fixed = [actions["fixed:" + str(width)] for width in expected_widths]
    if not fixed or expected_widths != list(range(1, reference["legal_rows"] + 1)):
        return {"comparable": False, "errors": ["incomplete_legal_width_sweep"]}
    costs = {row["action"]: row["round"]["round_ms"] / (row["round"]["accepted"] + 1) for row in fixed}
    winner = min(costs, key=costs.get)
    best = costs[winner]
    policies = {}
    for policy in ("adaptive", "cut"):
        receipt = actions[policy]["round"]
        selected = "fixed:" + str(receipt["rows"])
        same_regime = (no_assistant_root(receipt)
                       == no_assistant_root(actions[selected]["round"]))
        realized = receipt["round_ms"] / (receipt["accepted"] + 1)
        policies.update({policy + "_action": selected,
            policy + "_winner_agreement": selected == winner and same_regime,
            policy + "_regret_ms_per_token": realized - best,
            policy + "_relative_regret": realized / best - 1,
            policy + "_selected_width_regret_ms_per_token": costs[selected] - best if same_regime else None,
            policy + "_realized_ms_per_token": realized})
    predicted = {}
    for row in fixed:
        cell = curves.get(curve_key(row))
        if cell:
            predicted[row["action"]] = cell["round_ms"] / cell["tokens"]
    predicted_winner = min(predicted, key=predicted.get) if len(predicted) == len(fixed) else None
    return {"comparable": True, "errors": [], "n_measured_widths": len(fixed),
            "observed_ms_per_raw_token": costs, "observed_winner": winner,
            **policies,
            "predicted_winner": predicted_winner,
            "predicted_winner_agreement": predicted_winner == winner if predicted_winner else None,
            "predicted_winner_regret_ms_per_token": costs[predicted_winner] - best if predicted_winner else None,
            "oracle_scope": "single noisy measured first-decision alternatives including true no-assistant root; signed realized regret includes policy/candidate overhead and measurement noise, not true optimality or throughput"}


def summarize(rows, widths, curves, causal=False):
    groups = {}
    for row in rows:
        if row["phase"] == "heldout":
            groups.setdefault((row["mode"], row["profile"], row["repetition"]), []).append(row)
    comparisons = []
    valid_rows = []
    for (mode, profile, repetition), block in groups.items():
        comparison = compare_block(block, widths[mode + "/" + profile], curves)
        comparison["causal_exclusions"] = causal_block_errors(block)
        comparison["eligible_for_evidence"] = (comparison["comparable"]
            and (not causal or not comparison["causal_exclusions"]))
        comparisons.append({"mode": mode, "profile": profile, "repetition": repetition, **comparison})
        if comparison["eligible_for_evidence"]:
            valid_rows.extend(block)
    metrics = {key: [] for key in ("verify_ms", "round_ms", "tokens", "host_verify_ms", "host_tokens")}
    block_predictions = {}
    heldout_predictions = []
    for row in valid_rows:
        r = row["round"]
        cell = curves.get(curve_key(row))
        samples = []
        if cell:
            for name, actual in (("verify_ms", r["verify_commit_ms"]), ("round_ms", r["round_ms"]),
                                 ("tokens", r["accepted"] + 1)):
                samples.append((name, cell[name], actual))
        if finite(r.get("predicted_verify_ms"), True):
            samples.append(("host_verify_ms", r["predicted_verify_ms"], r["verify_commit_ms"]))
        if finite(r.get("predicted_tokens"), True):
            samples.append(("host_tokens", r["predicted_tokens"], r["accepted"] + 1))
        block_key = (row["mode"] + "/" + row["profile"], row["repetition"])
        for name, predicted, actual in samples:
            metrics[name].append((predicted, actual))
            block_predictions.setdefault(block_key, {}).setdefault(name, []).append((predicted, actual))
        heldout_predictions.append({"mode": row["mode"], "profile": row["profile"],
            "repetition": row["repetition"], "action": row["action"],
            "first_offer": offer_key(r), "cost_regime": "no-assistant" if no_assistant_root(r) else "candidate-paid",
            "training_n": cell["n"] if cell else 0, "frozen_prediction": cell,
            "observed": {"verify_ms": r["verify_commit_ms"], "round_ms": r["round_ms"], "tokens": r["accepted"] + 1},
            "host_snapshot": {"verify_ms": r.get("predicted_verify_ms"), "tokens": r.get("predicted_tokens")}})
    profiles = {}
    for comparison in comparisons:
        profiles.setdefault(comparison["mode"] + "/" + comparison["profile"], []).append(comparison)
    uncertainty = {}
    for profile, blocks in profiles.items():
        valid = [b for b in blocks if b["eligible_for_evidence"]]
        uncertainty[profile] = {"attempted_blocks": len(blocks), "comparable_blocks": len(valid),
            **{key: interval([float(b[key]) for b in valid if b.get(key) is not None]) for key in
               ("adaptive_winner_agreement", "adaptive_regret_ms_per_token", "adaptive_relative_regret",
                "cut_winner_agreement", "cut_regret_ms_per_token", "cut_relative_regret",
                "predicted_winner_agreement", "predicted_winner_regret_ms_per_token")}}
    prediction_uncertainty = {}
    for profile in profiles:
        prediction_uncertainty[profile] = {}
        for metric in metrics:
            block_errors = [prediction_summary(values[metric]) for (name, _rep), values in block_predictions.items()
                            if name == profile and metric in values]
            prediction_uncertainty[profile][metric] = {
                statistic: interval([value[statistic] for value in block_errors])
                for statistic in ("mae", "bias")}
    priced_blocks = [c for c in comparisons if c["eligible_for_evidence"] and c["predicted_winner"] is not None]
    coverage = first_offer_coverage(valid_rows)
    return {"version": VERSION, "diagnostic_only": True, "canonical_speedup_claim": False,
            "prediction_scope": "declared_isolation_observed_stable_blocks" if causal else "noncausal_diagnostic_only",
            "causal_economics_valid": causal and bool(priced_blocks) and len(priced_blocks) == len(comparisons),
            "n_evidence_blocks": sum(c["eligible_for_evidence"] for c in comparisons),
            "n_heldout_blocks": len(comparisons), "n_comparable_blocks": sum(c["comparable"] for c in comparisons),
            "first_offer_coverage": coverage,
            "lookup_first_offer_exercised": any(c["lookup_exercised"] for c in coverage),
            "prediction_errors": {key: prediction_summary(value) for key, value in metrics.items()},
            "prediction_boot_block_uncertainty": prediction_uncertainty,
            "frozen_prediction_evidence_valid": bool(priced_blocks) and len(priced_blocks) == len(comparisons),
            "n_completely_predicted_blocks": len(priced_blocks),
            "heldout_predictions": heldout_predictions,
            "comparisons": comparisons, "boot_block_uncertainty": uncertainty,
            "uncertainty_method": "2000 deterministic percentile bootstrap resamples within profile over eligible fresh-process matched action blocks; one block has no CI; causal scope additionally requires declared isolation and shared observed power/thermal/pressure/swap stability; no direct GPU monitoring or llmprobe loadDrift certification",
            "generalization_scope": "held-out boot repetitions of prescribed prompts, not unseen prompts",
            "host_prediction_unavailable": {key: sum(not finite(row["round"].get(field), True) for row in valid_rows)
                for key, field in (("verify_ms", "predicted_verify_ms"), ("tokens", "predicted_tokens"))},
            "no_assistant": [row for row in (valid_rows if causal else rows)
                             if row.get("round") and no_assistant_root(row["round"])]}


def snapshot_files(path, tokenizer_sha):
    if path is None:
        return []
    if not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
        raise runner.BenchError("store snapshot must be a real directory without symlinks")
    files = [{"path": str(p.relative_to(path)), "sha256": runner.digest(p), "bytes": p.stat().st_size}
             for p in sorted(path.rglob("*")) if p.is_file()]
    if not any(p["path"] == "tokens.bin" for p in files):
        raise runner.BenchError("store snapshot lacks tokens.bin; prepare a designated seed snapshot first")
    documents = validate_snapshot_tokens(path / "tokens.bin", tokenizer_sha)
    for entry in files:
        if entry["path"] == "tokens.bin":
            entry.update(documents=documents, tokenizer_sha256=tokenizer_sha)
    return files


def validate_snapshot_tokens(path, tokenizer_sha):
    """Validate the host's durable token-log schema, not just nonempty bytes."""
    length = path.stat().st_size
    if length < 48 or (length - 40) % 4:
        raise runner.BenchError("snapshot token log is empty, truncated, or unaligned")
    documents, in_document = 0, False
    with path.open("rb") as stream:
        header = stream.read(40)
        if header[:8] != b"MLXLOOK1" or header[8:] != bytes.fromhex(tokenizer_sha):
            raise runner.BenchError("snapshot token log magic/tokenizer SHA mismatch")
        while True:
            block = stream.read(65536)
            if not block:
                break
            if len(block) % 4:
                raise runner.BenchError("snapshot token log changed or became truncated while reading")
            for (token,) in struct.iter_unpack("<I", block):
                if token == 1 << 30:
                    if not in_document:
                        raise runner.BenchError("snapshot contains an empty document separator")
                    documents += 1
                    in_document = False
                elif token >= 1 << 30:
                    raise runner.BenchError("snapshot contains a token outside the host token-ID domain")
                else:
                    in_document = True
    if in_document or not documents:
        raise runner.BenchError("snapshot requires complete nonempty documents and a final separator")
    return documents


def corpus_for_mode(mode, corpus):
    return corpus if mode.endswith(("-corpus", "-all")) else None


def store_open_errors(text, path, tokenizer_sha, token_count):
    """Absence of a warning is NOT proof the requested frozen history opened."""
    if "[lookup] persistent store unavailable" in text or "lookup disabled: tokenizer identity unavailable" in text:
        return ["requested_persistent_store_unavailable"]
    receipts = runner.receipts(text, "store")
    matched = [r for r in receipts if r.get("version") == 1 and r.get("opened") is True
               and r.get("read_only") is True and r.get("path") == str(path)
               and r.get("tokenizer_sha256") == tokenizer_sha and r.get("token_count") == token_count]
    return [] if matched else ["missing_positive_matching_readonly_store_open_receipt"]


def complete_receipt(commit):
    return (isinstance(commit, dict) and commit.get("version") == VERSION
            and isinstance(commit.get("prompt_ids"), list) and bool(commit["prompt_ids"])
            and isinstance(commit.get("generated_ids"), list) and bool(commit["generated_ids"])
            and all(type(t) is int and t >= 0 for t in commit["prompt_ids"] + commit["generated_ids"])
            and commit.get("finish_reason") in ("stop", "length"))


def execute(args, binary, model, out, mode, profile, phase, repetition, action, corpus, snapshot, prices=None):
    arm = out / "arms" / phase / mode / profile["id"] / str(repetition) / action.replace(":", "-")
    arm.mkdir(parents=True)
    store = None
    starting = []
    if snapshot is not None and mode.endswith(("-persistent", "-all")):
        starting = snapshot_files(snapshot, model["tokenizer_sha256"])
        if starting != args.snapshot_identity:
            raise runner.BenchError("starting snapshot changed since identity-bearing preflight")
        store = arm / "store"
        shutil.copytree(snapshot, store)
        if snapshot_files(store, model["tokenizer_sha256"]) != starting:
            raise runner.BenchError("private starting-store copy differs from snapshot")
    actual_mode = mode
    experiment = {"MLX_SERVE_LOOKUP_SELECTION": action,
                  "MLX_SERVE_LOOKUP_STORE_WRITE": "0", "MLX_SERVE_LOOKUP_RECEIPTS": "1"}
    if prices is not None:
        experiment["MLX_SERVE_LOOKUP_PRICES"] = str(prices)
    row = {"version": VERSION, "mode": mode, "profile": profile["id"], "phase": phase,
           "repetition": repetition, "action": action, "errors": [], "round": None,
           "starting_store": starting, "priming_count": args.priming, "priming_commits": [],
           "isolation": args.isolation, "machine_before": None, "machine_after": None,
           "log": str(arm / "server.log")}
    try:
        with runner.Server(args, binary, model, actual_mode, arm / "server.log", corpus_for_mode(mode, corpus), store,
                           experiment=experiment) as server:
            runner.atomic_json(arm / "launch.json", {"command": server.command, "environment":
                {k: v for k, v in server.env.items() if k.startswith(("MLX_", "HF_", "TRANSFORMERS_"))},
                "pid": server.proc.pid, "starting_store": starting})
            for prime in range(args.priming):
                server.assert_owned()
                offset = server.log_path.stat().st_size
                result = runner.request(server, runner.request_body(profile, "serial", None), args.request_timeout)
                text = runner.settled_log(server, offset, timeout=args.receipt_timeout)
                commits = runner.receipts(text, "commit")
                runner.append(out / "priming.jsonl", {"mode": mode, "profile": profile["id"],
                    "phase": phase, "repetition": repetition, "action": action, "prime": prime,
                    "result": result, "commits": commits})
                if result.get("error") or len(commits) != 1 or not complete_receipt(commits[0]):
                    raise runner.BenchError("prescribed priming request failed or has no settled commit receipt")
                row["priming_commits"].append(commits[0])
            server.assert_owned()
            row["machine_before"] = runner.machine_state()
            offset = server.log_path.stat().st_size
            result = runner.request(server, runner.request_body(profile, actual_mode, None), args.request_timeout)
            text = runner.settled_log(server, offset, timeout=args.receipt_timeout)
            row["machine_after"] = runner.machine_state()
            rounds, commits = runner.receipts(text, "round"), runner.receipts(text, "commit")
            row.update(result=result, rounds=rounds, commit=commits[0] if len(commits) == 1 else None,
                       round=rounds[0] if rounds else None)
            if result.get("error"):
                row["errors"].append("request_failed:" + result["error"])
            if not complete_receipt(row["commit"]):
                row["errors"].append("missing_or_invalid_settled_commit")
            elif row["commit"]["prompt_ids"] != profile["prompt_ids"]:
                row["errors"].append("different_actual_prompt_token_stream")
            row["errors"] += receipt_errors(row["round"], action)
            row["errors"] += first_offer_origin_errors(profile, row["round"], action)
            if row["round"] and row["round"].get("kind") != ("tree" if mode.startswith("dflash-") else "chain"):
                row["errors"].append("unexpected_verify_kind")
            if row["round"] and profile["max_tokens"] <= row["round"].get("legal_rows", MAX_ROWS):
                row["errors"].append("insufficient_output_budget_for_complete_offer")
            row["cost_scope"] = ("no-assistant root verify+commit, no candidate construction"
                if row["round"] and no_assistant_root(row["round"])
                else "settled candidate construction plus selected verification+commit")
            if store is not None:
                process_text = server.log_since(0)
                row["store_open_receipts"] = runner.receipts(process_text, "store")
                token_file = next(entry for entry in starting if entry["path"] == "tokens.bin")
                row["errors"] += store_open_errors(process_text, store, model["tokenizer_sha256"],
                                                    (token_file["bytes"] - 40) // 4)
    except (OSError, ValueError, runner.BenchError) as exc:
        row["errors"].append(type(exc).__name__ + ":" + str(exc))
    if store is not None and snapshot_files(store, model["tokenizer_sha256"]) != starting:
        row["errors"].append("write_disabled_starting_history_changed")
    runner.append(out / "observations.jsonl", row)
    return row


def prepare_profiles(args, server, cases):
    profiles = []
    for case in cases:
        for target in args.contexts:
            prompt = case["prompt"]
            tokens = runner.http_json(server.base, "/tokenize", {"content": prompt}, args.request_timeout)["tokens"]
            # Append no generated output and truncate nothing: pad BEFORE the original workload.
            filler = "Background context record: the archive contains numbered maintenance notes.\n"
            copies = 0
            while len(tokens) < target:
                copies += max(1, math.ceil((target - len(tokens)) / 12))
                prompt = filler * copies + "\n" + case["prompt"]
                tokens = runner.http_json(server.base, "/tokenize", {"content": prompt}, args.request_timeout)["tokens"]
                if len(tokens) + args.max_tokens > args.ctx_size:
                    raise runner.BenchError("context profile exceeds capacity; no truncation: " + case["id"])
            if len(tokens) + args.max_tokens > args.ctx_size:
                raise runner.BenchError("prompt plus full output budget exceeds context capacity")
            profiles.append({**case, "id": case["id"] + "-ctx" + str(target), "prompt": prompt,
                             "prompt_ids": tokens, "prompt_tokens": len(tokens), "context_target_min": target,
                             "max_tokens": args.max_tokens})
    return profiles


def price_files(out, curves):
    if any(cell.get("context_bucket") != CONTEXT_BUCKET for cell in curves.values()):
        raise runner.BenchError("price curves require current runtime context-bucket evidence")
    tables, plain = {}, {}
    for key, cell in curves.items():
        mode, _profile, kind, cls, rows, cost_regime = json.loads(key)
        tables.setdefault(mode, {}).setdefault(str(cls), {}).setdefault(kind, {}).setdefault(str(rows), []).append(cell)
        if rows == 1 and cost_regime == "no-assistant":
            plain.setdefault(mode, {}).setdefault(str(cls), []).append(cell)
    paths = {}
    for mode, table in tables.items():
        result = {"schema": 2, "context_bucket": CONTEXT_BUCKET}
        for cls, kinds in table.items():
            result[cls] = {kind: {rows: sum(c["verify_ms"] * c["n"] for c in cells) / sum(c["n"] for c in cells)
                                  for rows, cells in widths.items()} for kind, widths in kinds.items()}
            roots = plain.get(mode, {}).get(cls, [])
            if roots:
                result[cls]["plain"] = {"1": sum(c["verify_ms"] * c["n"] for c in roots) / sum(c["n"] for c in roots)}
        path = out / (mode + "-training-prices.json")
        runner.atomic_json(path, result)
        paths[mode] = path
    return paths


def discovery_summary(config, rows, widths, errors, run_identity, failure=None):
    """Discovery establishes available actions, never held-out economic evidence."""
    observed = {row["mode"] + "/" + row["profile"]: row for row in rows
                if row["phase"] == "discovery"}
    outcomes = {}
    for mode in config["modes"]:
        for case in config["cases"]:
            for context in config["contexts"]:
                key = mode + "/" + case + "-ctx" + str(context)
                row = observed.get(key)
                receipt = (row or {}).get("round") or {}
                reasons = errors.get(key, [])
                unsupported = config["unsupported"].get(mode)
                status = ("unsupported" if unsupported else "failed" if reasons
                          else "discovered" if key in widths else "not_run")
                outcomes[key] = {
                    "status": status, "errors": reasons,
                    "unsupported_reason": unsupported,
                    "legal_widths": widths.get(key, []),
                    "selected_origins": {tier: receipt.get("selected_" + tier, 0)
                                         for tier in ("local", "corpus", "persistent", "neural")}
                                        if receipt else None}
    complete = bool(outcomes) and all(o["status"] == "discovered" for o in outcomes.values())
    return {"run_identity": run_identity, "phase": "discovery",
            "discovery_only": config["discovery_only"],
            "profile_outcomes": outcomes, "legal_widths": widths,
            "unsupported_modes": config["unsupported"], "unsupported_profiles": errors,
            "discovery_complete": complete and failure is None,
            "comparability_scope": "discovery only; no matched training or held-out economics",
            "canonical_speedup_claim": False, "causal_economics_valid": False,
            "failure": failure}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path("~/.mlx-serve/bench/lookup-staircut"))
    parser.add_argument("--model", default="qwen27b")
    parser.add_argument("--modes", default="dflash-local", help="lookup modes; dflash=tree, mtp/pld=chain")
    parser.add_argument("--cases", default="copy,code,prose")
    parser.add_argument("--contexts", default="0", help="minimum actual prompt token counts, comma separated; reached classes come from receipts")
    parser.add_argument("--train-repetitions", type=int, default=3)
    parser.add_argument("--heldout-repetitions", type=int, default=3)
    parser.add_argument("--priming", type=int, default=3, help="identical no-drafting full-workload kernel priming in each family-bound process; leaves lookup learners cold")
    parser.add_argument("--max-tokens", type=int, default=128, help="full output budget, must exceed maximum legal offer")
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--store-snapshot", type=Path, help="immutable prepared designated-history store, required for persistent/all")
    parser.add_argument("--port", type=int, default=11279)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--receipt-timeout", type=float, default=10)
    parser.add_argument("--isolation", choices=("contended", "declared"), default="contended")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--discovery-only", action="store_true",
                        help="prepare actual profiles and discover max offers only; no training or held-out actions")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        args.contexts = [int(x) for x in args.contexts.split(",")]
    except ValueError:
        parser.error("contexts must be comma-separated nonnegative token counts")
    if (not args.contexts or min(args.contexts) < 0 or len(set(args.contexts)) != len(args.contexts)
            or min(args.train_repetitions, args.heldout_repetitions) < 2 or args.priming < 0
            or args.max_tokens <= MAX_ROWS or args.ctx_size <= args.max_tokens
            or not 1 <= args.port <= 65535
            or min(args.ready_timeout, args.request_timeout, args.receipt_timeout) <= 0):
        parser.error("use >=2 disjoint train/heldout repetitions, unique contexts >=0, priming >=0, max-tokens >64, sufficient context, valid port/timeouts")
    args.kit = args.kit.expanduser().resolve()
    manifest_path = args.kit / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise runner.BenchError("unsupported manifest")
    model = next((dict(m) for m in manifest["models"] if m["id"] == args.model), None)
    if model is None:
        raise runner.BenchError("unknown prepared model")
    binary = runner.absolute_asset(manifest["binary"])
    if not os.access(binary, os.X_OK):
        raise runner.BenchError("prepared binary is not executable")
    model["model_dir"] = runner.absolute_asset(model["model_dir"], directory=True)
    if model.get("drafter_dir"):
        model["drafter_dir"] = runner.absolute_asset(model["drafter_dir"], directory=True)
    token_sha = runner.digest(Path(model["model_dir"]) / "tokenizer.json")
    if model.get("tokenizer_sha256") != token_sha:
        raise runner.BenchError("prepared tokenizer fingerprint changed")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in runner.MODES or m.startswith("serial") or m.endswith("-legacy") for m in modes):
        raise runner.BenchError("select unique lookup-enabled family modes, not serial/legacy")
    snapshot = args.store_snapshot.expanduser().resolve() if args.store_snapshot else None
    if any(m.endswith(("-persistent", "-all")) for m in modes) and snapshot is None:
        raise runner.BenchError("persistent/all requires --store-snapshot; no implicit repeated-history growth")
    snapshot_identity = snapshot_files(snapshot, token_sha)
    args.snapshot_identity = snapshot_identity
    workloads_path = Path(runner.absolute_asset(manifest["workloads"]))
    workloads = json.loads(workloads_path.read_text())
    ids = args.cases.split(",")
    cases = [case for name in ids for case in workloads["cases"] if case["id"] == name]
    if len(cases) != len(ids) or len(set(ids)) != len(ids):
        raise runner.BenchError("cases must be unique prepared workload IDs")
    if any(c["temperature"] != 0 for c in cases):
        raise runner.BenchError("controlled first-decision economics currently requires greedy workloads")
    config = {"version": VERSION, "script_sha256": runner.digest(Path(__file__)),
              "runner_sha256": runner.digest(Path(runner.__file__)), "manifest_sha256": runner.digest(manifest_path),
              "evidence_reducer_sha256": runner.digest(Path(runner.__file__).with_name("lookup_bench_evidence.py")),
              "binary_sha256": runner.digest(binary), "workloads_sha256": runner.digest(workloads_path),
              "model": model, "model_inventory": runner.inventory(model["model_dir"]),
              "drafter_inventory": runner.inventory(model["drafter_dir"]) if model.get("drafter_dir") else None,
              "tokenizer_sha256": token_sha, "modes": modes, "cases": ids, "contexts": args.contexts,
              "max_tokens": args.max_tokens, "ctx_size": args.ctx_size, "train_repetitions": args.train_repetitions,
              "heldout_repetitions": args.heldout_repetitions, "priming": args.priming,
              "starting_store": snapshot_identity, "isolation": args.isolation,
              "discovery_only": args.discovery_only,
              "timeouts": [args.ready_timeout, args.request_timeout, args.receipt_timeout],
              "policy": "fresh process each action; identical no-drafting kernel priming; fresh acceptance learner; heldout plain and chain/tree price priors measured only in disjoint training boots; immutable starting history; all legal fixed widths; alternate action order; first offered decision only",
              "offline": "runner offline env, no proxies, loopback owned server; OS sandbox required for enforced network denial",
              "isolation_evidence": "user declaration only; memory is not GPU isolation evidence",
              "causal_stability_policy": "declared isolation plus shared stable_states over every measured-action endpoint in the complete block; unknown/changed power, thermal, pressure, swap usage or swap counters exclude whole block from hardware training and causal statistics; pressure must be normal; no loadDrift measurement",
              "unsupported": {mode: runner.unsupported(model, mode) for mode in modes if runner.unsupported(model, mode)}}
    run_identity = identity(config)
    print(json.dumps({"configuration_identity": run_identity, "config": config,
                      "run_identity_status": "finalized after owned tokenizer/corpus preflight"}, indent=2), flush=True)
    if args.dry_run:
        return 0
    out = (args.output or args.kit / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-economics-" + args.model)).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    if any(out.iterdir()):
        raise runner.BenchError("economics output must be new and empty; no cross-session resume")
    print("Evidence: " + str(out), flush=True)
    print("Command: " + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]), flush=True)
    runner.atomic_json(out / "run.json", {"configuration_identity": run_identity, "config": config, "created": time.time()})
    runner.atomic_json(out / "machine-start.json", runner.machine_state())
    rows, widths = [], {}
    supported = [mode for mode in modes if mode not in config["unsupported"]]
    discovery_errors = {}
    try:
        if not supported:
            summary = discovery_summary(config, rows, widths, discovery_errors, run_identity)
            runner.atomic_json(out / "discovery-summary.json", summary)
            runner.atomic_json(out / "summary.json", summary)
            return 1
        corpus = None
        with runner.Server(args, binary, model, "serial", out / "prepare.log") as server:
            profiles = prepare_profiles(args, server, cases)
            if any(mode.endswith(("-corpus", "-all")) for mode in supported):
                from prepare_lookup_offline import ensure_corpus
                corpus = ensure_corpus(model, manifest, server.base, args.kit)
        runner.atomic_json(out / "profiles.json", profiles)
        config["profile_token_stream_sha256"] = identity(profiles)
        config["corpus_files"] = ([{"path": str(p.relative_to(corpus)), "sha256": runner.digest(p)}
                                    for p in sorted(Path(corpus).rglob("*")) if p.is_file()] if corpus else [])
        run_identity = identity(config)
        runner.atomic_json(out / "run.json", {"run_identity": run_identity, "config": config, "created": time.time()})
        print("Run identity: " + run_identity, flush=True)
        for mode in supported:
            for profile in profiles:
                discovery = execute(args, binary, model, out, mode, profile, "discovery", 0, "max", corpus, snapshot)
                rows.append(discovery)
                if discovery["errors"] or discovery["round"]["legal_rows"] <= 1:
                    discovery_errors[mode + "/" + profile["id"]] = discovery["errors"] or ["no_nontrivial_first_offer"]
                    continue
                widths[mode + "/" + profile["id"]] = list(range(1, discovery["round"]["legal_rows"] + 1))
        runner.atomic_json(out / "legal-widths.json", widths)
        discovery_result = discovery_summary(config, rows, widths, discovery_errors, run_identity)
        runner.atomic_json(out / "discovery-summary.json", discovery_result)
        if args.discovery_only or not discovery_result["discovery_complete"]:
            runner.atomic_json(out / "summary.json", discovery_result)
            print(json.dumps({"evidence": str(out), "phase": "discovery",
                              "discovery_complete": discovery_result["discovery_complete"],
                              "canonical_speedup_claim": False}), flush=True)
            return 0 if discovery_result["discovery_complete"] else 1
        curves, hardware_curves, paths = {}, {}, {}
        for phase, repetitions in (("train", args.train_repetitions), ("heldout", args.heldout_repetitions)):
            for repetition in range(repetitions):
                for mode in supported:
                    for profile in profiles:
                        if mode + "/" + profile["id"] not in widths:
                            continue
                        actions = ["fixed:" + str(w) for w in widths[mode + "/" + profile["id"]]] + ["adaptive", "cut", "max"]
                        if repetition % 2:
                            actions.reverse()
                        for action in actions:
                            rows.append(execute(args, binary, model, out, mode, profile, phase, repetition,
                                                action, corpus, snapshot, paths.get(mode)))
            if phase == "train":
                curves = fit_curves(rows, widths)
                hardware_curves = fit_curves(rows, widths, causal=True)
                runner.atomic_json(out / "diagnostic-training-curves.json", curves)
                runner.atomic_json(out / "hardware-training-curves.json", hardware_curves)
                loaded_curves = hardware_curves if args.isolation == "declared" else curves
                runner.atomic_json(out / "training-curves.json", loaded_curves)
                paths = price_files(out, loaded_curves)
                runner.atomic_json(out / "frozen-price-identity.json", {mode: runner.digest(path) for mode, path in paths.items()})
        summary = summarize(rows, widths, curves)
        causal_summary = summarize(rows, widths, hardware_curves, causal=True)
        summary["causal_economics_valid"] = (causal_summary["causal_economics_valid"]
                                             and not config["unsupported"] and not discovery_errors)
        summary["causal_economics"] = causal_summary
        summary["run_identity"] = run_identity
        summary["unsupported_modes"] = config["unsupported"]
        summary["unsupported_profiles"] = discovery_errors
        for prediction in summary["heldout_predictions"]:
            runner.append(out / "heldout-predictions.jsonl", {"run_identity": run_identity, **prediction})
        for comparison in summary["comparisons"]:
            runner.append(out / "comparisons.jsonl", {"run_identity": run_identity, **comparison})
        for prediction in causal_summary["heldout_predictions"]:
            runner.append(out / "causal-heldout-predictions.jsonl", {"run_identity": run_identity, **prediction})
        for comparison in causal_summary["comparisons"]:
            runner.append(out / "causal-comparisons.jsonl", {"run_identity": run_identity, **comparison})
        runner.atomic_json(out / "summary.json", summary)
        print(json.dumps({"evidence": str(out), "n_heldout_blocks": summary["n_heldout_blocks"],
                          "n_comparable_blocks": summary["n_comparable_blocks"], "canonical_speedup_claim": False}), flush=True)
        return 0 if (summary["n_comparable_blocks"] == summary["n_heldout_blocks"]
                     and summary["n_comparable_blocks"] > 0 and summary["frozen_prediction_evidence_valid"]
                     and not config["unsupported"] and not discovery_errors
                     and (args.isolation != "declared" or summary["causal_economics_valid"])
                     and not any(row["errors"] for row in rows)) else 1
    except BaseException as exc:
        if not any(row["phase"] == "train" for row in rows):
            summary = discovery_summary(config, rows, widths, discovery_errors, run_identity,
                                        {"type": type(exc).__name__, "message": str(exc)})
            runner.atomic_json(out / "discovery-summary.json", summary)
            runner.atomic_json(out / "summary.json", summary)
        runner.atomic_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                            "run_identity": run_identity, "observations_retained": len(rows)})
        raise
    finally:
        runner.atomic_json(out / "machine-end.json", runner.machine_state())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, runner.interrupted)
    try:
        sys.exit(main())
    except (runner.BenchError, OSError, ValueError, KeyboardInterrupt) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
