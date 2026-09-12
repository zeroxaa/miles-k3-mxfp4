"""Persistent Megatron worker for a measured, colocated K3 RL smoke cycle.

SGLang produces every response consumed here. The controller hands this worker
token IDs, rollout log probabilities and group-normalized rewards through JSON.
Miles supplies the clipped policy loss, model, optimizer, gradient reductions,
native MXFP4 loader and HF LoRA export. This deliberately isolates GPU handoff
from the Ray scheduler; it is an experimental integration harness, not train.py.

Run through torchrun on the same 24 GPUs as the paused SGLang server. A TMS region
owns the persistent trainer allocations; temporary BF16 expert decodes are freed
after each backward. Only adapter tensors are exported, never the frozen base.
"""

import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import socket
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.utils import unwrap_model
from safetensors.torch import save_file
from torch_memory_saver import torch_memory_saver

from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.lora_utils import build_lora_sync_config, save_lora_checkpoint
from miles.backends.megatron_utils.model import finalize_model_grads_with_empty_cache, setup_model_and_optimizer
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.utils.arguments import parse_args
from miles.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_torch_dist,
    reload_process_groups,
)
from miles_plugins.models.kimi_k3.lora import export_kimi_k3_lora_hf_chunks
from miles_plugins.models.kimi_k3_mxfp4.linear import FrozenMXFP4GroupedLinear


def _arguments(parser):
    parser.add_argument("--rl-cycle-dir", required=True)
    parser.add_argument("--rl-cycle-count", type=int, default=2)
    parser.add_argument("--rl-adapter-local-dir", default=None)
    return parser


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "device_used_bytes": total - free,
    }


def _event(directory, rank, phase, started, **details):
    record = {"phase": phase, "seconds": time.monotonic() - started, "rank": rank, **details, **_memory()}
    with (directory / f"events-rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print("RL_PHASE " + json.dumps(record), flush=True)
    return record


def _wait_for(path, timeout=14400):
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"No controller command at {path}")
        time.sleep(1)
    return json.loads(path.read_text())


def _base_probes(raw_model):
    probes = []
    for chunk in raw_model:
        for name, module in chunk.named_modules():
            if isinstance(module, FrozenMXFP4GroupedLinear):
                for suffix in ("packed", "scales"):
                    tensor = getattr(module, suffix)
                    view = tensor.view(-1)
                    indices = torch.tensor([0, view.numel() // 2, view.numel() - 1], device=view.device)
                    probes.append((f"{name}.{suffix}", tensor, indices, view[indices].cpu()))
    return probes


def _verify_probes(probes, *, cpu_backup=False):
    for name, tensor, indices, expected in probes:
        if cpu_backup:
            backup = torch_memory_saver.get_cpu_backup(tensor, zero_copy=True)
            assert backup is not None, f"Untracked packed model buffer: {name}"
            actual = backup.view(-1)[indices.cpu()]
        else:
            actual = tensor.view(-1)[indices].cpu()
        assert torch.equal(actual, expected), f"Frozen MXFP4 buffer changed: {name}"


def _storage_inventory(raw_model):
    seen = set()
    inventory = {"base_bytes": 0, "adapter_bytes": 0, "packed_expert_bytes": 0}
    for chunk in raw_model:
        for name, tensor in itertools.chain(chunk.named_parameters(), chunk.named_buffers()):
            if not tensor.is_cuda or not tensor.numel():
                continue
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            adapter = ".lora_adapter." in name
            inventory["adapter_bytes" if adapter else "base_bytes"] += storage.nbytes()
            if name.endswith((".packed", ".scales")):
                inventory["packed_expert_bytes"] += storage.nbytes()
    return inventory


def _pause(directory, rank, phase, probes, control_group):
    dist.barrier(group=control_group)
    started = time.monotonic()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    destroy_process_groups()
    transfer_started = time.monotonic()
    torch_memory_saver.pause(tag="rl_trainer")
    torch.cuda.synchronize()
    transfer_seconds = time.monotonic() - transfer_started
    _verify_probes(probes, cpu_backup=True)
    dist.barrier(group=control_group)
    _event(directory, rank, phase, started, packed_cpu_backup_verified=True, tms_transfer_seconds=transfer_seconds)


def _resume(directory, rank, phase, probes, control_group):
    dist.barrier(group=control_group)
    started = time.monotonic()
    torch_memory_saver.resume(tag="rl_trainer")
    torch.cuda.synchronize()
    transfer_seconds = time.monotonic() - started
    reload_process_groups()
    _verify_probes(probes)
    dist.barrier(group=control_group)
    _event(directory, rank, phase, started, packed_roundtrip_verified=True, tms_transfer_seconds=transfer_seconds)


def _batch(sample, seq_length):
    tokens = sample["prompt_ids"] + sample["response_ids"]
    assert 1 < len(tokens) <= seq_length + 1
    padded = tokens + [0] * (seq_length + 1 - len(tokens))
    ids = torch.tensor(padded, device="cuda", dtype=torch.long).unsqueeze(0)
    start = len(sample["prompt_ids"]) - 1
    stop = start + len(sample["response_ids"])
    return {"input_ids": ids[:, :-1].contiguous(), "labels": ids[:, 1:].contiguous(), "span": (start, stop)}


def _policy_loss(output, *, batch, sample, group_size):
    start, stop = batch["span"]
    log_probs = -output.float().reshape(-1)[start:stop]
    old_log_probs = torch.tensor(sample["rollout_log_probs"], device=log_probs.device, dtype=torch.float32)
    assert log_probs.shape == old_log_probs.shape
    advantages = torch.full_like(log_probs, sample["advantage"])
    ppo_kl = old_log_probs - log_probs
    losses, clipped = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.28)
    loss = losses.mean() / group_size
    assert torch.isfinite(loss), "Nonfinite policy loss"
    return loss, {
        "policy_loss": loss.detach(),
        "clip_fraction": clipped.mean().detach(),
        "rollout_train_logprob_abs_diff": ppo_kl.abs().mean().detach(),
    }


def _forward(data_iterator, model_chunk, *, sample, group_size):
    batch = next(data_iterator)
    output = model_chunk(input_ids=batch["input_ids"], position_ids=None, attention_mask=None, labels=batch["labels"])
    return output, partial(_policy_loss, batch=batch, sample=sample, group_size=group_size)


def _train_group(args, model, optimizer, scheduler, samples):
    assert len(samples) >= 2 and any(sample["advantage"] != 0 for sample in samples)
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    # Run microbatches sequentially to retain only one sample's decoded weights.
    # Reduce the accumulated gradients once, after the complete rollout group.
    config.finalize_model_grads_func = None
    for chunk in model:
        chunk.train()
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    before = {
        name: param.detach().cpu().clone()
        for chunk in model
        for name, param in chunk.named_parameters()
        if param.requires_grad
    }
    metrics = []
    for sample in samples:
        batch = _batch(sample, args.seq_length)
        losses = get_forward_backward_func()(
            forward_step_func=partial(_forward, sample=sample, group_size=len(samples)),
            data_iterator=itertools.repeat(batch),
            model=model,
            num_microbatches=1,
            seq_length=args.seq_length,
            micro_batch_size=1,
            forward_only=False,
        )
        metrics.extend({key: float(value) for key, value in loss.items()} for loss in losses)
    finalize_model_grads_with_empty_cache(model, None)
    successful, grad_norm, _ = optimizer.step()
    assert successful and grad_norm is not None and math.isfinite(float(grad_norm)) and float(grad_norm) > 0
    scheduler.step(increment=len(samples))
    changed = [
        name
        for chunk in model
        for name, param in chunk.named_parameters()
        if param.requires_grad and not torch.equal(before[name], param.detach().cpu())
    ]
    assert changed, "RL update did not change an adapter on this rank"
    assert all(".lora_adapter." in name for name in changed)
    assert all(param.grad is None for chunk in model for param in chunk.parameters() if not param.requires_grad)
    return {"grad_norm": float(grad_norm), "changed_adapters": changed, "metrics": metrics}


def _export(args, model, directory, cycle, control_group):
    destination = directory / "adapters" / f"v{cycle}"
    destination.mkdir(parents=True, exist_ok=True)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    leader = parallel_state.get_tensor_model_parallel_rank() == 0
    tensors = {}
    # All ranks must join TP/EP gathers; only the stage leader writes the result.
    for group in export_kimi_k3_lora_hf_chunks(model):
        if leader:
            tensors.update({name: tensor.cpu().clone() for name, tensor in group})
    if leader:
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        filename = destination / f"adapter_model-{pp_rank + 1:05d}-of-00003.safetensors"
        save_file(tensors, filename, metadata={"format": "pt"})
        manifest = {
            "tensor_count": len(tensors),
            "payload_bytes": payload_bytes,
            "file_bytes": filename.stat().st_size,
            "sha256": {
                name: hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, tensor in tensors.items()
            },
        }
        _write_json(destination / f"manifest-pp{pp_rank}.json", manifest)
    if args.rank == 0:
        config = build_lora_sync_config(args) | {"experts_shared_outer_loras": True}
        _write_json(destination / "adapter_config.json", config)
    dist.barrier(group=control_group)
    return str(destination)


def _run_cycles(args, model, optimizer, scheduler, directory, probes, control_group):
    for cycle in range(1, args.rl_cycle_count + 1):
        command = _wait_for(directory / f"train-{cycle}.json")
        assert (
            command["rollout_source"] == "sglang_generate"
            and command["base_revision"] == "a590ce090cb049c93a33dfe8c208ec652aa20503"
        )
        _resume(directory, args.rank, f"trainer_onload_{cycle}", probes, control_group)
        started = time.monotonic()
        report = _train_group(args, model, optimizer, scheduler, command["samples"])
        torch.cuda.synchronize()
        _verify_probes(probes)
        _event(directory, args.rank, f"rl_train_{cycle}", started, **report)
        started = time.monotonic()
        adapter_path = _export(args, model, directory, cycle, control_group)
        _event(directory, args.rank, f"adapter_export_{cycle}", started, adapter_path=adapter_path)
        if args.rl_adapter_local_dir:
            started = time.monotonic()
            local_path = Path(args.rl_adapter_local_dir) / f"v{cycle}"
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                _copy_adapter_to_local(Path(adapter_path), local_path)
            dist.barrier(group=control_group)
            adapter_path = str(local_path)
            _event(directory, args.rank, f"adapter_stage_{cycle}", started, adapter_path=adapter_path)
        _pause(directory, args.rank, f"trainer_offload_{cycle}", probes, control_group)
        _write_json(directory / f"done-{cycle}-rank{args.rank}.json", {"adapter_path": adapter_path, **report})
    _wait_for(directory / "finish.json")


def _copy_adapter_to_local(source, destination):
    # The old SGLang file loader mmaps tensors. Sequentially staging all three
    # shards avoids many concurrent small page faults against the shared FS.
    # Each node's TP leader copies once; all serving nodes use the same path.
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copytree(source, temporary)
    temporary.rename(destination)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = parse_args(_arguments)
    if args.rl_adapter_local_dir:
        assert Path(args.rl_adapter_local_dir).is_absolute(), "All serving nodes need the same absolute local path"
    args.rank = int(os.environ["RANK"])
    assert args.world_size == int(os.environ["WORLD_SIZE"]) == 24
    assert args.num_layers == 93 and args.pipeline_model_parallel_size == 3
    assert args.tensor_model_parallel_size == args.expert_model_parallel_size == 8
    assert args.optimizer == "sgd" and args.sgd_momentum == 0
    assert not args.offload_train, "This harness owns its explicit TMS region"
    directory = Path(args.rl_cycle_dir)
    directory.mkdir(parents=True, exist_ok=True)
    monkey_patch_torch_dist()
    dist.init_process_group(backend="nccl")
    control_group = dist.new_group(backend="gloo")
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    init(args)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    started = time.monotonic()
    # This region includes frozen uint8 buffers, BF16 parameters and optimizer
    # storage. Activation and decode caches are allocated outside the region.
    with torch_memory_saver.region(tag="rl_trainer", enable_cpu_backup=True):
        with patch("megatron.core.optimizer.SGD", torch.optim.SGD):
            model, optimizer, scheduler = setup_model_and_optimizer(args)
        enforce_marked_param_dtypes(model)
        load_checkpoint(model, optimizer, scheduler, checkpointing_context=None, skip_load_to_model_and_opt=False)
    raw_model = unwrap_model(model)
    probes = _base_probes(raw_model)
    inventory = _storage_inventory(raw_model)
    _event(directory, args.rank, "trainer_cold_load", started, **inventory)
    _pause(directory, args.rank, "trainer_initial_offload", probes, control_group)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    _write_json(
        directory / f"ready-rank{args.rank}.json",
        {
            "rank": args.rank,
            "hostname": socket.gethostname(),
            "gpu_uuid": str(props.uuid),
            "gpu_name": props.name,
            **inventory,
        },
    )
    _run_cycles(args, model, optimizer, scheduler, directory, probes, control_group)
    # Restore before normal Python/CUDA tensor cleanup and release every worker.
    _resume(directory, args.rank, "trainer_final_cleanup_onload", probes, control_group)
    save_lora_checkpoint(
        model, args, args.save, optimizer=optimizer, opt_param_scheduler=scheduler, iteration=args.rl_cycle_count
    )
    dist.barrier(group=control_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
