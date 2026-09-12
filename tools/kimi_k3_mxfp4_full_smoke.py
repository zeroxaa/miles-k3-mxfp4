"""Train full K3 on 24 GPUs, updating only the configured layers' LoRA.

Launch with torchrun and the full kimi-k3-mxfp4 model arguments. This uses
Miles' model, checkpoint, optimizer and gradient-finalization paths plus the
Megatron pipeline schedule. The authored text fixture tests next-token training;
it is not an RL rollout or a model-quality evaluation.
"""

import itertools
import json
import math
import os
import socket
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training import get_tokenizer
from megatron.training.utils import unwrap_model

from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.lora_utils import save_lora_checkpoint
from miles.backends.megatron_utils.model import finalize_model_grads_with_empty_cache, setup_model_and_optimizer
from miles.utils.arguments import parse_args
from miles_plugins.models.kimi_k3_mxfp4 import linear as mxfp4_linear
from miles_plugins.models.kimi_k3_mxfp4.options import layer_indices

FIXTURE = "The capital of France is Paris. Water freezes at zero degrees Celsius. Two plus three equals five. "


def _arguments(parser):
    parser.add_argument("--full-smoke-report", required=True)
    parser.add_argument("--full-smoke-steps", type=int, default=2)
    return parser


def _observe_layers(raw_model, observed):
    handles = []
    for chunk in raw_model:
        for layer in chunk.decoder.layers:
            index = layer.layer_number - 1

            def forward_hook(module, inputs, output, index=index):
                observed["forward"].add(index)
                memory = {"layer": index, **_memory_stats()}
                observed["forward_memory"].append(memory)
                if int(os.environ["LOCAL_RANK"]) == 0:
                    print(f"FULL_LAYER_MEMORY rank={dist.get_rank()} {json.dumps(memory)}", flush=True)
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.requires_grad:
                    hidden.register_hook(lambda gradient, index=index: observed["backward"].add(index))

            handles.append(layer.register_forward_hook(forward_hook))
    return handles


def _memory_stats():
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _fixture_batch(args):
    tokens = get_tokenizer().tokenize(FIXTURE * (args.seq_length + 1))[: args.seq_length + 1]
    assert len(tokens) == args.seq_length + 1
    tensor = torch.tensor(tokens, device="cuda", dtype=torch.long).unsqueeze(0)
    return {"input_ids": tensor[:, :-1].contiguous(), "labels": tensor[:, 1:].contiguous()}, tokens


def _loss(output):
    loss = output.float().mean()
    assert torch.isfinite(loss), "Nonfinite next-token loss"
    return loss, {"lm_loss": loss.detach()}


def _forward(data_iterator, model_chunk):
    batch = next(data_iterator)
    output = model_chunk(input_ids=batch["input_ids"], position_ids=None, attention_mask=None, labels=batch["labels"])
    return output, _loss


def _verify_layout(raw_model, selected):
    layers = {}
    adapters = {}
    trainable = {}
    for chunk in raw_model:
        for layer in chunk.decoder.layers:
            index = layer.layer_number - 1
            layers[index] = layer
            for name, param in layer.named_parameters():
                key = f"layers.{index}.{name}"
                if ".lora_adapter." in key:
                    adapters[key] = param
                if param.requires_grad:
                    assert index in selected and ".lora_adapter." in key, key
                    trainable[key] = param
        layer_params = {id(param) for layer in chunk.decoder.layers for param in layer.parameters()}
        assert all(not param.requires_grad for param in chunk.parameters() if id(param) not in layer_params)
    assert len(layers) == 31
    assert {int(name.split(".")[1]) for name in trainable} == set(layers) & selected
    return layers, adapters, trainable


def _attention_lora_b_names(layers, selected):
    return {
        f"layers.{index}.self_attention.lora_adapter.{projection}_lora_B"
        for index, layer in layers.items()
        if index in selected
        for projection in (("o",) if layer.self_attention.is_kda else ("o", "q_a", "kv_a"))
    }


def _verify_retention(layers, retained):
    actual = []
    for index, layer in layers.items():
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        for linear in (experts.linear_fc1, experts.linear_fc2):
            assert isinstance(linear, mxfp4_linear.FrozenMXFP4GroupedLinear)
            assert linear.retain_bf16 == (index in retained)
        if experts.linear_fc1.retain_bf16:
            actual.append(index)
    return sorted(actual)


def _run_pipeline(args, model, batch, observed, step):
    decoded = {"forward_calls": 0, "backward_calls": 0, "forward_bytes": 0, "backward_bytes": 0}
    original_decode = mxfp4_linear.decode_weight

    def counted_decode(packed, scales):
        weight = original_decode(packed, scales)
        phase = "backward" if observed["backward"] else "forward"
        decoded[f"{phase}_calls"] += 1
        decoded[f"{phase}_bytes"] += weight.numel() * weight.element_size()
        return weight

    try:
        with patch.object(mxfp4_linear, "decode_weight", counted_decode):
            losses = get_forward_backward_func()(
                forward_step_func=_forward,
                data_iterator=itertools.repeat(batch),
                model=model,
                num_microbatches=1,
                seq_length=args.seq_length,
                micro_batch_size=1,
                forward_only=False,
            )
    except torch.cuda.OutOfMemoryError as exc:
        failure = {
            "status": "OOM",
            "rank": dist.get_rank(),
            "hostname": socket.gethostname(),
            "step": step,
            "error": str(exc),
            "forward_layers": sorted(observed["forward"]),
            "backward_layers": sorted(observed["backward"]),
            "forward_memory": observed["forward_memory"],
            "expert_decoding": decoded,
            **_memory_stats(),
        }
        path = Path(args.full_smoke_report).with_name(f"rank{dist.get_rank()}-oom.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failure, indent=2) + "\n")
        print(f"FULL_SMOKE_OOM report={path}", flush=True)
        raise
    all_retained = set(range(1, args.num_layers)) <= set(args.kimi_k3_mxfp4_retain_layers)
    if all_retained and args.recompute_granularity is None:
        assert decoded["backward_calls"] == 0, "Retained expert weights were decoded again in backward"
    return losses, decoded


def _train(args, model, raw_model, optimizer, scheduler):
    selected = layer_indices(args.kimi_k3_mxfp4_train_layers, num_layers=args.num_layers, option="train_layers")
    layers, adapters, trainable = _verify_layout(raw_model, selected)
    retained = _verify_retention(layers, set(args.kimi_k3_mxfp4_retain_layers))
    expected_attention_updates = _attention_lora_b_names(layers, selected)
    assert expected_attention_updates <= trainable.keys(), "Missing trainable attention adapters"
    before = {name: param.detach().cpu().clone() for name, param in adapters.items()}
    observed = {"forward": set(), "backward": set(), "forward_memory": []}
    handles = _observe_layers(raw_model, observed)
    batch, token_ids = _fixture_batch(args)
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache
    steps = []
    for step in range(args.full_smoke_steps):
        for chunk in model:
            chunk.train()
            chunk.zero_grad_buffer()
        optimizer.zero_grad()
        for value in observed.values():
            value.clear()
        started = time.monotonic()
        print(f"FULL_STEP_START rank={dist.get_rank()} step={step + 1}", flush=True)
        losses, decoded = _run_pipeline(args, model, batch, observed, step + 1)
        assert observed["forward"] == set(layers), observed
        assert observed["backward"] == set(layers), observed
        assert all(
            param.grad is None for chunk in raw_model for param in chunk.parameters() if not param.requires_grad
        )
        successful, grad_norm, _ = optimizer.step()
        assert successful, "Optimizer rejected update"
        assert grad_norm is None or math.isfinite(float(grad_norm)), "Nonfinite gradient norm"
        assert not trainable or (grad_norm is not None and float(grad_norm) > 0), "Missing trainable gradients"
        scheduler.step(increment=args.global_batch_size)
        torch.cuda.synchronize()
        record = {
            "step": step + 1,
            "losses": [{key: float(value) for key, value in loss.items()} for loss in losses],
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "forward_layers": sorted(observed["forward"]),
            "backward_layers": sorted(observed["backward"]),
            "seconds": time.monotonic() - started,
            "optimizer_update_successful": bool(successful),
            "forward_memory": list(observed["forward_memory"]),
            "expert_decoding": decoded,
            **_memory_stats(),
        }
        steps.append(record)
        print(f"FULL_STEP_DONE rank={dist.get_rank()} {json.dumps(record)}", flush=True)
    changed = [name for name, param in adapters.items() if not torch.equal(param.detach().cpu(), before[name])]
    assert all(name in trainable for name in changed), changed
    assert {int(name.split(".")[1]) for name in changed} == set(layers) & selected
    assert expected_attention_updates <= set(changed), "Some selected attention LoRA B tensors did not update"
    for handle in handles:
        handle.remove()
    return {
        "layers": sorted(layers),
        "verified_retained_moe_layers": retained,
        "attention_types": {index: "KDA" if layer.self_attention.is_kda else "MLA" for index, layer in layers.items()},
        "verified_attention_updates": sorted(expected_attention_updates),
        "steps": steps,
        "changed_adapters": changed,
        "trainable_adapter_names": sorted(trainable),
        "local_trainable_parameter_elements": sum(param.numel() for param in trainable.values()),
        "all_frozen_adapters_unchanged": True,
        "frozen_base_has_no_parameter_gradients": True,
        "token_ids": token_ids,
    }


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = parse_args(_arguments)
    args.rank = int(os.environ["RANK"])
    assert args.world_size == int(os.environ["WORLD_SIZE"]) == 24
    assert args.num_layers == 93 and args.pipeline_model_parallel_size == 3
    assert args.tensor_model_parallel_size == args.expert_model_parallel_size == 8
    assert args.expert_tensor_parallel_size == 1 and args.global_batch_size == args.micro_batch_size == 1
    assert args.full_smoke_steps >= 2
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(backend="nccl")
    init(args)
    # TE FusedSGD indexes the first parameter to choose a device and crashes on
    # the empty groups of fully frozen PP stages. Torch SGD supports those
    # groups; retain Megatron's mixed-precision wrapper and collective stats.
    assert args.optimizer == "sgd" and args.sgd_momentum == 0
    with patch("megatron.core.optimizer.SGD", torch.optim.SGD):
        model, optimizer, scheduler = setup_model_and_optimizer(args)
    # Preserve the KDA parameters explicitly marked FP32 across Float16Module.
    enforce_marked_param_dtypes(model)
    raw_model = unwrap_model(model)
    print(f"FULL_MODEL_BUILT rank={args.rank}", flush=True)
    load_checkpoint(model, optimizer, scheduler, checkpointing_context=None, skip_load_to_model_and_opt=False)
    print(f"FULL_CHECKPOINT_LOADED rank={args.rank}", flush=True)
    report = _train(args, model, raw_model, optimizer, scheduler)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report.update(
        rank=args.rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        hostname=socket.gethostname(),
        gpu_name=props.name,
        gpu_uuid=str(props.uuid),
        gpu_total_memory_bytes=props.total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        ep_rank=parallel_state.get_expert_model_parallel_rank(),
        pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
    )
    save_lora_checkpoint(
        model, args, args.save, optimizer=optimizer, opt_param_scheduler=scheduler, iteration=args.full_smoke_steps
    )
    report_path = Path(args.full_smoke_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.with_name(f"rank{args.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    reports = [None] * 24
    dist.all_gather_object(reports, report)
    if args.rank == 0:
        assert len({item["gpu_uuid"] for item in reports}) == 24
        assert {layer for item in reports for layer in item["layers"]} == set(range(93))
        result = {
            "status": "PASS",
            "scope": "Full 93-layer K3 next-token training on 24 physical H200 GPUs; configured LoRA layers only",
            "train_layers": sorted(args.kimi_k3_mxfp4_train_layers),
            "retain_bf16_layers": sorted(args.kimi_k3_mxfp4_retain_layers),
            "fixture": FIXTURE,
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "sequence_length": args.seq_length,
            "optimizer_steps": args.full_smoke_steps,
            "checkpoint": args.save,
            "ranks": reports,
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"FULL_SMOKE_PASS report={report_path}", flush=True)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
