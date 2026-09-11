"""Validate one real K3 expert with native TEGroupedMLP and Miles LoRA on CUDA.

This is an operator/integration smoke, not a full-model training result.
"""

import argparse
import gc
import json
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.utils.mxfp4 import dequantize_mxfp4
from miles_plugins.models.kimi_k3.lora import _apply_expert_lora
from miles_plugins.models.kimi_k3.ops import situ_and_mul
from miles_plugins.models.kimi_k3_mxfp4.checkpoint import NativeCheckpoint
from miles_plugins.models.kimi_k3_mxfp4.linear import FrozenMXFP4GroupedLinear


def _build_expert(reader, *, retain, layer, expert):
    config = TransformerConfig(
        num_layers=93,
        hidden_size=7168,
        num_attention_heads=96,
        kv_channels=256,
        ffn_hidden_size=33792,
        num_moe_experts=896,
        moe_ffn_hidden_size=3072,
        moe_latent_size=3584,
        moe_router_topk=16,
        gated_linear_unit=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        expert_tensor_parallel_size=1,
    )
    config.gated_activation_func = situ_and_mul
    config.init_method = lambda tensor: tensor
    config.output_layer_init_method = lambda tensor: tensor
    builder = partial(FrozenMXFP4GroupedLinear, retain_bf16=retain)
    groups = ProcessGroupCollection(tp=dist.group.WORLD, ep=dist.group.WORLD, expt_tp=dist.group.WORLD)
    module = TEGroupedMLP(
        1, config, GroupedMLPSubmodules(linear_fc1=builder, linear_fc2=builder), pg_collection=groups
    )
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    module.linear_fc1.load_expert(
        0,
        torch.cat([reader.read(f"{prefix}.{p}.weight_packed") for p in ("w1", "w3")]),
        torch.cat([reader.read(f"{prefix}.{p}.weight_scale") for p in ("w1", "w3")]),
    )
    module.linear_fc2.load_expert(
        0, reader.read(f"{prefix}.w2.weight_packed"), reader.read(f"{prefix}.w2.weight_scale")
    )
    return module, config


def _compare_base(module):
    inputs = torch.randn(4, 3584, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    counts = torch.tensor([4], device="cuda", dtype=torch.long)
    probs = torch.full((4,), 0.125, device="cuda", dtype=torch.float32)
    fc1, fc2 = module.linear_fc1, module.linear_fc2
    w1 = dequantize_mxfp4(fc1.packed[0], fc1.scales[0], 32)
    w2 = dequantize_mxfp4(fc2.packed[0], fc2.scales[0], 32)
    hidden = situ_and_mul(F.linear(ref_inputs, w1))
    reference = F.linear((hidden * probs.unsqueeze(-1)).to(hidden.dtype), w2)
    actual, bias = module(inputs, counts, probs)
    assert bias is None and torch.isfinite(actual).all()
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=0, atol=0)
    return {
        "output_max_abs_error": (actual - reference).abs().max().item(),
        "input_grad_max_abs_error": (inputs.grad - ref_inputs.grad).abs().max().item(),
    }


def _train_adapter(module, config):
    _apply_expert_lora(
        SimpleNamespace(experts=module, config=config), SimpleNamespace(lora_rank=4), 1, 2.0, 0.0, include_fc2=True
    )
    base_before = {name: value.clone() for name, value in module.named_buffers()}
    adapter_before = {name: value.clone() for name, value in module.named_parameters()}
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    inputs = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([8], device="cuda", dtype=torch.long)
    probs = torch.full((8,), 0.125, device="cuda", dtype=torch.float32)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        inputs.grad = None
        output, _ = module(inputs, counts, probs)
        loss = output.float().square().mean()
        loss.backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        optimizer.step()
        losses.append(loss.item())
    changed = [name for name, value in module.named_parameters() if not torch.equal(value, adapter_before[name])]
    assert any("lora_B" in name for name in changed)
    assert all(torch.equal(value, base_before[name]) for name, value in module.named_buffers())
    return {
        "optimizer_steps": 2,
        "losses": losses,
        "changed_adapter_parameters": changed,
        "packed_base_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rendezvous = Path(tempfile.mkdtemp()) / "rendezvous"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    reader = NativeCheckpoint(args.checkpoint)
    report = {
        "scope": "one original K3 expert, full matrix dimensions; not full-model training",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "modes": [],
    }
    try:
        for retain in (False, True):
            torch.manual_seed(29)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            module, config = _build_expert(reader, retain=retain, layer=args.layer, expert=args.expert)
            result = {"retain_bf16": retain, **_compare_base(module), **_train_adapter(module, config)}
            torch.cuda.synchronize()
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["modes"].append(result)
            del module
        report["status"] = "PASS"
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        reader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
