"""Regression for an early PP HTTP acknowledgement during GPU handoff."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "k3_rl_cycle", Path(__file__).resolve().parents[3] / "tools/kimi_k3_mxfp4_rl_cycle.py"
)
cycle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cycle)


@pytest.mark.parametrize("completed_gpus_at_http_ack", [8, 23])
def test_handoff_waits_for_every_gpu_after_early_http_ack(tmp_path, monkeypatch, completed_gpus_at_http_ack):
    trace = tmp_path / "tms" / "rollout-tms-test.jsonl"
    trace.parent.mkdir()
    tags = ["weights", "kv_cache"]

    def append_records(gpus, timestamp):
        with trace.open("a") as stream:
            for gpu in gpus:
                for tag in tags:
                    stream.write(
                        json.dumps(
                            {
                                "hostname": f"stage-{gpu // 8}",
                                "gpu": gpu % 8,
                                "tag": tag,
                                "operation": "pause",
                                "timestamp": timestamp,
                            }
                        )
                        + "\n"
                    )

    # A previous handoff already recorded all 24 GPUs. Those events must not
    # satisfy the next barrier, even though the HTTP request returns success.
    append_records(range(24), timestamp=1)

    def acknowledge_head_stage(*args, **kwargs):
        append_records(range(completed_gpus_at_http_ack), timestamp=2)
        return {"success": True}

    waits = []

    def finish_downstream_stages(seconds):
        waits.append(seconds)
        append_records(range(completed_gpus_at_http_ack, 24), timestamp=2)

    monkeypatch.setattr(cycle, "_request", acknowledge_head_stage)
    monkeypatch.setattr(cycle.time, "sleep", finish_downstream_stages)
    result = cycle._memory_request("http://unused", tmp_path, "pause", tags)
    assert waits, "The controller returned before the downstream GPUs completed"
    records = result["_timing_details"]["tms_rank_records"]
    assert len(records) == 48
    assert all(record["timestamp"] == 2 for record in records)
