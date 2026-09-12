"""Measure real SGLang rollout -> Megatron LoRA update -> SGLang handoff.

Start the serving engine on the same 24 GPUs and arrange for the worker launcher
to wait for initialize_trainer.json before starting torchrun. This controller
uses SGLang's existing memory and adapter HTTP APIs.
Its adapter transport is HF safetensors through the supplied shared directory;
it does not claim to measure Miles' newer CUDA IPC streaming protocol.

The smoke reward is the first generated digit divided by nine (zero if absent).
It tests a real group-relative policy update, not useful model-quality training.
"""

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Pick a single digit from 0 to 9. Answer:"
BASE_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _request(base, route, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            if not content:
                return {}
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Older SGLang control endpoints return a plain-text 200.
                return {"message": content.decode()}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{route}: HTTP {error.code}: {error.read().decode()}") from error


def _phase(report, directory, name, function):
    started = time.monotonic()
    result = function()
    record = {"phase": name, "seconds": time.monotonic() - started}
    if isinstance(result, dict) and "_timing_details" in result:
        record.update(result["_timing_details"])
    report["phases"].append(record)
    _write(directory / "controller-report.json", report)
    print(
        "RL_CONTROLLER " + json.dumps({key: value for key, value in record.items() if key != "tms_rank_records"}),
        flush=True,
    )
    return result


def _wait_engine(base, timeout=1800):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            return _request(base, "/get_server_info", timeout=10)
        except (OSError, RuntimeError):
            time.sleep(2)
    raise TimeoutError("SGLang did not become ready")


def _wait_ranks(directory, pattern, timeout=14400):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        paths = [directory / pattern.format(rank=rank) for rank in range(24)]
        if all(path.exists() for path in paths):
            return [json.loads(path.read_text()) for path in paths]
        time.sleep(1)
    raise TimeoutError(f"Some trainer ranks did not finish: {pattern}")


def _generate(base, adapter=None, temperature=1.0):
    payload = {
        "text": PROMPT,
        "sampling_params": {"temperature": temperature, "max_new_tokens": 4},
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }
    if adapter is not None:
        payload["lora_path"] = adapter
    result = _request(base, "/generate", payload)
    meta = result["meta_info"]
    prompt_ids = [item[1] for item in meta["input_token_logprobs"]]
    output = meta["output_token_logprobs"]
    assert len(prompt_ids) == meta["prompt_tokens"], "Prompt token IDs must cover the entire prompt"
    assert len(output) == meta["completion_tokens"] and output
    match = re.search(r"\b([0-9])\b", result["text"])
    reward = int(match[1]) / 9 if match else 0.0
    return {
        "prompt_ids": prompt_ids,
        "response_ids": [item[1] for item in output],
        "rollout_log_probs": [item[0] for item in output],
        "text": result["text"],
        "reward": reward,
        "adapter": adapter,
        "raw_response": result,
    }


def _rollout(base, directory, cycle, adapter):
    # Preserve every attempt. A zero-variance group gives no learning signal;
    # resample the entire two-response group instead of inventing a reward.
    attempts = []
    for attempt in range(12):
        samples = [_generate(base, adapter) for _ in range(2)]
        attempts.append(samples)
        _write(directory / f"rollout-{cycle}-attempts.json", attempts)
        rewards = [sample["reward"] for sample in samples]
        if max(rewards) != min(rewards):
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
            for sample in samples:
                sample["advantage"] = (sample["reward"] - mean) / (std + 1e-6)
            return {"samples": samples, "attempts": attempt + 1}
    raise RuntimeError("All sampled groups had equal rewards; no RL update submitted")


def _memory_records(directory, operation, tags):
    records = {}
    for path in (directory / "tms").glob("rollout-tms-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # A writer may still be appending its final line.
            if item["operation"] == operation and item["tag"] in tags:
                key = (item["hostname"], item["gpu"], item["tag"])
                if key not in records or item["timestamp"] > records[key]["timestamp"]:
                    records[key] = item
    return records


def _memory_request(base, directory, operation, tags):
    before = _memory_records(directory, operation, tags)
    route = "/release_memory_occupation" if operation == "pause" else "/resume_memory_occupation"
    started = time.monotonic()
    response = _request(base, route, {"tags": tags})
    http_seconds = time.monotonic() - started
    # SGLang PP returns the HTTP response before all downstream PP stages have
    # processed the control message. Wait for every physical GPU/tag, not just
    # the head server's acknowledgement, before letting the other engine run.
    while time.monotonic() - started < 1800:
        records = _memory_records(directory, operation, tags)
        changed = {
            key: item
            for key, item in records.items()
            if key not in before or item["timestamp"] > before[key]["timestamp"]
        }
        if len(changed) == 24 * len(tags):
            return {
                "response": response,
                "_timing_details": {
                    "http_ack_seconds": http_seconds,
                    "verified_gpu_count": 24,
                    "tms_rank_records": list(changed.values()),
                },
            }
        time.sleep(0.5)
    raise TimeoutError(f"Not all 24 rollout GPUs completed {operation} for {tags}")


def _offload(base, directory):
    _request(base, "/flush_cache", {})
    return _memory_request(base, directory, "pause", ["kv_cache", "weights"])


def _cycle(args, directory, report, cycle, adapter):
    started = time.monotonic()
    rollout = _phase(report, directory, f"rollout_{cycle}", lambda: _rollout(args.server, directory, cycle, adapter))
    _phase(report, directory, f"rollout_offload_{cycle}", lambda: _offload(args.server, directory))
    cold_wait = 0.0
    if cycle == 1 and not all((directory / f"ready-rank{rank}.json").exists() for rank in range(24)):
        _write(directory / "initialize_trainer.json", {})
        cold_started = time.monotonic()
        ready = _phase(
            report, directory, "trainer_cold_start_wait", lambda: _wait_ranks(directory, "ready-rank{rank}.json")
        )
        cold_wait = time.monotonic() - cold_started
        assert len({item["gpu_uuid"] for item in ready}) == 24
        report["trainer_gpus"] = ready
    command = {"rollout_source": "sglang_generate", "base_revision": BASE_REVISION, **rollout}
    _write(directory / f"train-{cycle}.json", command)
    done = _phase(
        report, directory, f"trainer_total_{cycle}", lambda: _wait_ranks(directory, f"done-{cycle}-rank{{rank}}.json")
    )
    assert all(item["changed_adapters"] and item["grad_norm"] > 0 for item in done)
    _phase(
        report,
        directory,
        f"rollout_weights_onload_{cycle}",
        lambda: _memory_request(args.server, directory, "resume", ["weights"]),
    )
    if adapter is not None:
        _phase(
            report,
            directory,
            f"adapter_unload_{cycle}",
            lambda: _request(args.server, "/unload_lora_adapter", {"lora_name": adapter}),
        )
    new_adapter = f"rl-v{cycle}"
    response = _phase(
        report,
        directory,
        f"adapter_load_{cycle}",
        lambda: _request(
            args.server, "/load_lora_adapter", {"lora_name": new_adapter, "lora_path": done[0]["adapter_path"]}
        ),
    )
    assert response.get("success", True), response
    _phase(
        report,
        directory,
        f"rollout_kv_onload_{cycle}",
        lambda: _memory_request(args.server, directory, "resume", ["kv_cache"]),
    )
    post = _phase(report, directory, f"post_update_rollout_{cycle}", lambda: _generate(args.server, new_adapter))
    _write(directory / f"post-update-{cycle}.json", post)
    record = {
        "cycle": cycle,
        "seconds": time.monotonic() - started,
        "seconds_excluding_trainer_cold_start": time.monotonic() - started - cold_wait,
        "adapter": new_adapter,
        "rollout_attempts": rollout["attempts"],
        "rewards": [sample["reward"] for sample in rollout["samples"]],
        "post_update_response": post["text"],
    }
    report["cycles"].append(record)
    _write(directory / "controller-report.json", report)
    return new_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--resume-inference-on-start", action="store_true")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "RUNNING",
        "phases": [],
        "cycles": [],
        "base_revision": BASE_REVISION,
        "training_parallelism": "TP8 EP8 PP3",
        "rollout_parallelism": "TP8 EP1 PP3",
        "transport": "Native Miles HF LoRA export -> shared safetensors -> SGLang dynamic adapter load",
        "reward": "First generated standalone digit divided by 9; no digit receives zero",
        "policy_loss": "Miles clipped policy loss, response mask, group-normalized reward; KL coefficient zero",
        "orchestration": "Experimental HTTP/filesystem harness; not the Ray train.py driver",
    }
    try:
        report["server_info"] = _phase(report, directory, "wait_sglang_ready", lambda: _wait_engine(args.server))
        if args.resume_inference_on_start:
            ready = _phase(
                report, directory, "trainer_cold_start_wait", lambda: _wait_ranks(directory, "ready-rank{rank}.json")
            )
            assert len({item["gpu_uuid"] for item in ready}) == 24
            report["trainer_gpus"] = ready
            _phase(
                report,
                directory,
                "initial_rollout_weights_onload",
                lambda: _memory_request(args.server, directory, "resume", ["weights"]),
            )
            _phase(
                report,
                directory,
                "initial_rollout_kv_onload",
                lambda: _memory_request(args.server, directory, "resume", ["kv_cache"]),
            )
        adapter = None
        for cycle in range(1, args.cycles + 1):
            adapter = _cycle(args, directory, report, cycle, adapter)
        # The trainer restores its allocations to save a native resumable
        # checkpoint and exit, so first release the serving model again.
        _phase(report, directory, "rollout_final_offload", lambda: _offload(args.server, directory))
        _write(directory / "finish.json", {})
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = repr(error)
        raise
    finally:
        _write(directory / "controller-report.json", report)


if __name__ == "__main__":
    main()
