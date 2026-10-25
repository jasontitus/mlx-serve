#!/usr/bin/env python3
"""Untimed, offline lookup runtime gates (never a performance benchmark).

Example: python3 tests/validate_lookup_runtime.py --model qwen27b --output /tmp/lookup-runtime
Only newly owned localhost children and private output stores are touched. All
requested gates must execute: unsupported hardware, admission refusal, missing
receipts or failure to reach a branch is a failure, never a skipped pass.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

import bench_lookup_staircut as bench


class ValidationError(bench.BenchError):
    pass


def require(value, message):
    if not value:
        raise ValidationError(message)


def receipts(text, kind):
    marker = "[lookup-" + kind + "] "
    result = []
    for line in text.splitlines():
        if marker in line:
            row = json.loads(line.split(marker, 1)[1])
            require(row.get("version") == 1, "unsupported receipt version")
            result.append(row)
    return result


def settled(server, offset, count=1):
    deadline = time.monotonic() + server.args.request_timeout
    while time.monotonic() < deadline:
        server.assert_owned()
        text = server.log_since(offset)
        if len(receipts(text, "commit")) >= count:
            if getattr(server, "validation_store", None) is not None:
                lock_released(server.validation_store)
            return server.log_since(offset)
        time.sleep(0.02)
    raise ValidationError("generator teardown receipt did not arrive")


def lock_released(store):
    with (store / "store.lock").open("a+b") as handle:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
                return
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise ValidationError("persistent store retained its lifetime lock") from exc
                time.sleep(0.01)


def successful(result):
    require(not result.get("error") and result.get("http_status") == 200,
            "request failed (resource refusals are not passes): " + bench.canonical(result))
    require(result.get("finish_reason") in ("stop", "length"), "abnormal finish")


def one(server, body):
    server.assert_owned()
    offset = server.log_path.stat().st_size
    result = bench.request(server, body, server.args.request_timeout)
    successful(result)
    text = settled(server, offset)
    commits = receipts(text, "commit")
    require(len(commits) == 1, "sequential request must have exactly one commit receipt")
    commit = commits[0]
    require(isinstance(commit.get("prompt_ids"), list) and commit["prompt_ids"], "missing actual prompt IDs")
    require(isinstance(commit.get("generated_ids"), list), "missing generated token IDs")
    require(commit.get("finish_reason") == result["finish_reason"], "receipt/API finish mismatch")
    return {"response": result, "commit": commit, "rounds": receipts(text, "round"), "log": text}


def equivalent(reference, actual):
    for field in ("prompt_ids", "generated_ids", "finish_reason"):
        require(reference["commit"][field] == actual["commit"][field], "serial/spec token receipt mismatch: " + field)
    for field in ("text", "usage", "finish_reason"):
        require(reference["response"][field] == actual["response"][field], "serial/spec API mismatch: " + field)


def class_for(length):
    if length < 2048:
        return 0
    return min(1 << (length.bit_length() - 1), 262144)


def token_count(server, prompt):
    return len(bench.http_json(server.base, "/tokenize", {"content": prompt},
                               timeout=server.args.request_timeout)["tokens"])


def boundary_prompts(server, fixture):
    """Find actual tokenized text on either side, not estimated filler counts."""
    unit = fixture["context_unit"]
    result = []
    for boundary in fixture["context_boundaries"]:
        low, high = 0, boundary * 2
        while low < high:
            mid = (low + high) // 2
            prompt = unit * mid + "\n" + fixture["prompt"]
            if token_count(server, prompt) < boundary:
                low = mid + 1
            else:
                high = mid
        for repeats in (low - 1, low + 1):
            require(repeats >= 0, "boundary fixture too long")
            prompt = unit * repeats + "\n" + fixture["prompt"]
            count = token_count(server, prompt)
            require(count + fixture["max_tokens"] < server.args.ctx_size, "context gate exceeds owned capacity")
            result.append(("context-%d-%d" % (boundary, count), prompt, count))
        require(result[-2][2] < boundary < result[-1][2], "actual prompts do not straddle context boundary")
    return result


def body(prompt, mode, seed, temperature, max_tokens):
    case = dict(prompt=prompt, seed=seed, temperature=temperature, max_tokens=max_tokens)
    return bench.request_body(case, mode, None)


def owned_server(args, binary, model, mode, log, store=None, concurrent=1):
    experiment = {"MLX_SERVE_LOOKUP_RECEIPTS": "1", "MLX_SERVE_LOOKUP_SELECTION": "max"}
    server = bench.Server(args, binary, model, mode, log, store=store, experiment=experiment)
    server.validation_store = store
    index = server.command.index("--max-concurrent")
    server.command[index + 1] = str(concurrent)
    # Enforce the offline contract even if a child has an unexpected updater.
    sandbox = shutil.which("sandbox-exec")
    require(sys.platform == "darwin" and sandbox, "blocked: sandbox-exec required to deny nonlocal child networking")
    server.command = [sandbox, "-p", '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'] + server.command
    return server


def wait_live_round(server, offset, future):
    deadline = time.monotonic() + server.args.request_timeout
    while time.monotonic() < deadline:
        text = server.log_since(offset)
        if receipts(text, "round"):
            require(not future.done() and not receipts(text, "commit"), "request ended before controlled overlap/fault injection")
            return
        require(not future.done(), "no live lookup round reached; cannot exercise requested branch")
        time.sleep(0.002)
    raise ValidationError("no live lookup round reached")


def storage_gates(args, binary, model, fixture, out, identity):
    mode = args.family + "-persistent"
    header = b"MLXLOOK1" + bytes.fromhex(identity)
    valid = header + struct.pack("<4I", 1, 2, 3, 1 << 30)
    broken = {"corrupt": b"BADLOOK1" + valid[8:], "headerless": valid[40:],
              "truncated": valid[:-1], "wrong-tokenizer": b"MLXLOOK1" + bytes(b ^ 0xff for b in bytes.fromhex(identity)) + valid[40:]}
    request_body = body(fixture["prompt"], mode, 7, 0.0, fixture["max_tokens"])
    serial_body = body(fixture["prompt"], "serial", 7, 0.0, fixture["max_tokens"])
    baseline_mode = "serial-dflash" if args.family == "dflash" else "serial"
    with owned_server(args, binary, model, baseline_mode, out / "storage-reference.log") as server:
        reference = one(server, serial_body)
        long_reference = one(server, body(fixture["prompt"], "serial", 19, 0.0, fixture["overlap_tokens"]))
    for name, data in broken.items():
        store = out / ("store-" + name)
        store.mkdir()
        path = store / "tokens.bin"
        path.write_bytes(data)
        with owned_server(args, binary, model, mode, out / (name + ".log"), store) as server:
            for _ in range(2):
                actual = one(server, request_body)
                equivalent(reference, actual)
                require("persistent store unavailable:" in actual["log"], "negative store was not rejected")
                require("TokenizerMismatch" in actual["log"] if name == "wrong-tokenizer" else "MalformedStore" in actual["log"], "wrong storage rejection reason")
                require(path.read_bytes() == data, "rejected store modified")
                lock_released(store)
        bench.append(out / "gates.jsonl", dict(gate=name, status="pass"))

    store = out / "store-live"
    store.mkdir()
    path = store / "tokens.bin"
    path.write_bytes(valid)
    with owned_server(args, binary, model, mode, out / "storage-live.log", store, concurrent=2) as server:
        long_body = body(fixture["prompt"], mode, 19, 0.0, fixture["overlap_tokens"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            offset = server.log_path.stat().st_size
            first = pool.submit(bench.request, server, long_body, args.request_timeout)
            wait_live_round(server, offset, first)
            second = pool.submit(bench.request, server, request_body, args.request_timeout)
            first_response = first.result(timeout=args.request_timeout)
            second_response = second.result(timeout=args.request_timeout)
            successful(first_response)
            successful(second_response)
            text = settled(server, offset, count=2)
            require("persistent store unavailable: StoreBusy" in text, "second-slot StoreBusy fallback not reached")
            commits = receipts(text, "commit")
            require(len(commits) == 2, "overlap did not teardown both generators")
            # Join both unordered teardown receipts to their exact serial
            # streams, then compare each caller's own complete HTTP response.
            unmatched = list(commits)
            for expected, response in ((long_reference, first_response), (reference, second_response)):
                match = next((c for c in unmatched if all(
                    c.get(field) == expected["commit"][field]
                    for field in ("prompt_ids", "generated_ids", "finish_reason"))), None)
                require(match is not None, "overlap commit stream differs from matching serial request")
                unmatched.remove(match)
                equivalent(expected, {"response": response, "commit": match})
            require(not unmatched, "unexpected extra overlap commit receipt")
            lock_released(store)
            equivalent(reference, one(server, request_body))
            lock_released(store)
            bench.append(out / "gates.jsonl", dict(gate="second-slot-StoreBusy", status="pass"))

            before = path.read_bytes()
            offset = server.log_path.stat().st_size
            failed_append = pool.submit(bench.request, server, long_body, args.request_timeout)
            wait_live_round(server, offset, failed_append)
            # Private deliberate log-length mismatch: appendTurn must refuse before publishing a turn.
            with path.open("ab") as handle:
                handle.write(b"\x00")
                handle.flush()
                os.fsync(handle.fileno())
            append_response = failed_append.result(timeout=args.request_timeout)
            successful(append_response)
            text = settled(server, offset)
            append_commits = receipts(text, "commit")
            require(len(append_commits) == 1, "append-failure request must have exactly one commit receipt")
            equivalent(long_reference, {"response": append_response, "commit": append_commits[0]})
            require("MalformedStore" in text and "[lookup] saving committed tokens failed:" in text, "append failure did not reach generator save failure")
            require(path.read_bytes() == before + b"\x00", "failed append unexpectedly published data")
            lock_released(store)
            path.write_bytes(before)
            equivalent(reference, one(server, request_body))
            lock_released(store)
            bench.append(out / "gates.jsonl", dict(gate="append-failure-next-request", status="pass"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path("~/.mlx-serve/bench/lookup-staircut"))
    parser.add_argument("--model", default="qwen27b")
    parser.add_argument("--family", choices=("dflash", "mtp", "pld"), default="dflash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).parent / "fixtures/lookup_staircut/validation.json")
    parser.add_argument("--port", type=int, default=11279)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    args = parser.parse_args()
    os.umask(0o077)
    require(args.ctx_size > 0 and 1 <= args.port <= 65535 and args.ready_timeout > 0 and args.request_timeout > 0, "invalid capacity/port/timeout")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads((args.kit.expanduser() / "manifest.json").read_text())
        require(manifest.get("version") == 1, "unsupported prepared manifest")
        model = next((m for m in manifest["models"] if m["id"] == args.model), None)
        require(model is not None, "unknown local model profile")
        model = dict(model)
        binary = bench.absolute_asset(manifest["binary"])
        model["model_dir"] = bench.absolute_asset(model["model_dir"], directory=True)
        if model.get("drafter_dir"):
            model["drafter_dir"] = bench.absolute_asset(model["drafter_dir"], directory=True)
        mode = args.family + "-local"
        require(bench.unsupported(model, mode) is None, "blocked: " + str(bench.unsupported(model, mode)))
        identity = bench.digest(Path(model["model_dir"]) / "tokenizer.json")
        require(model.get("tokenizer_sha256") in (None, identity), "prepared tokenizer identity changed")
        fixture = json.loads(args.fixture.read_text())
        require(fixture.get("version") == 1 and len(set(fixture["seeds"])) >= 3, "requires version1 fixture and multiple distinct seeds")
        baseline_mode = "serial-dflash" if args.family == "dflash" else "serial"
        references = {}
        with owned_server(args, binary, model, baseline_mode, args.output / "serial.log") as server:
            prompts = [("short", fixture["prompt"], token_count(server, fixture["prompt"]))] + boundary_prompts(server, fixture)
            for name, prompt, count in prompts:
                # Same-seed exactness is only promised for the row-exact keyed DFlash tree.
                temperatures = (0.0, 0.8) if args.family == "dflash" else (0.0,)
                for temperature in temperatures:
                    for seed in fixture["seeds"]:
                        b = body(prompt, "serial", seed, temperature, fixture["max_tokens"])
                        references[(name, seed, temperature)] = one(server, b)
        classes = set()
        lookup_classes = set()
        with owned_server(args, binary, model, mode, args.output / "lookup.log") as server:
            for name, prompt, count in prompts:
                for temperature in ((0.0, 0.8) if args.family == "dflash" else (0.0,)):
                    for seed in fixture["seeds"]:
                        actual = one(server, body(prompt, mode, seed, temperature, fixture["max_tokens"]))
                        equivalent(references[(name, seed, temperature)], actual)
                        rounds = actual["rounds"]
                        require(rounds, "no actual lookup verification receipt")
                        lookup_classes.update(r["class"] for r in rounds if r.get("selected_local", 0) > 0)
                        if args.family == "dflash":
                            require(any(r.get("kind") == "tree" for r in rounds), "merged tree route not reached")
                        require(all(r.get("class") == class_for(r["anchor_pos"] + 1) for r in rounds), "wrong price context class")
                        classes.update(r["class"] for r in rounds)
                        bench.append(args.output / "requests.jsonl", dict(case=name, seed=seed, temperature=temperature, prompt_tokens=count, evidence=actual))
        expected = {class_for(count + 1) for _, _, count in prompts}
        require(expected <= classes, "actual rounds did not reach both sides of requested context classes")
        require(expected <= lookup_classes, "no selected local lookup at a requested context class")
        bench.append(args.output / "gates.jsonl", dict(gate="seeds-context-token-equivalence", classes=sorted(classes), selected_lookup_classes=sorted(lookup_classes), status="pass", sampled_contract="keyed exact DFlash only; MTP/PLD sampled distribution unvalidated"))
        storage_gates(args, binary, model, fixture, args.output, identity)
        bench.atomic_json(args.output / "summary.json", dict(version=1, status="pass", kind="untimed-runtime-validation", model=args.model, family=args.family, classes=sorted(classes), in_process_transitions="run lookup validation Zig tests separately"))
        return 0
    except (OSError, ValueError, KeyError, bench.BenchError, TimeoutError) as exc:
        bench.atomic_json(args.output / "summary.json", dict(version=1, status="failed-or-blocked", error=str(exc), kind="untimed-runtime-validation"))
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
