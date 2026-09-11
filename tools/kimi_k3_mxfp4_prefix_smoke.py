"""Validate the native spec/loader and layer freezing on K3's first three layers.

All 896 experts of each included MoE layer are loaded. This bounded integration
test is not a full 93-layer language-model or distributed training result.
"""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from scripts.run_kimi_k3 import _DEFAULT_TARGET_MODULES
from torch import nn

from miles_plugins.models.kimi_k3.lora import apply_kimi_k3_lora, export_kimi_k3_lora_hf_chunks
from miles_plugins.models.kimi_k3_mxfp4.checkpoint import load_native_checkpoint
from miles_plugins.models.kimi_k3_mxfp4.spec import get_kimi_k3_mxfp4_spec


def _build_model():
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
        moe_layer_freq=[0] + [1] * 92,
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=6144,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0,
        moe_token_dispatcher_type="alltoall",
        normalization="RMSNorm",
        hidden_dropout=0,
        attention_dropout=0,
        layernorm_epsilon=1e-5,
    )
    options = SimpleNamespace(
        kimi_k3_mxfp4=True,
        kimi_k3_mxfp4_train_layers=[0, 1],
        kimi_k3_mxfp4_retain_layers=[1],
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0,
        experts_shared_outer_loras=True,
        target_modules=_DEFAULT_TARGET_MODULES.split(","),
    )
    spec = get_kimi_k3_mxfp4_spec(options, config)
    assert len(spec.layer_specs) == 93
    groups = ProcessGroupCollection.use_mpu_process_groups()
    model = nn.Module()
    model.config = config
    model.pre_process = False
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList(
        [
            build_module(layer, config=config, layer_number=index + 1, pg_collection=groups)
            for index, layer in enumerate(spec.layer_specs[:3])
        ]
    )
    return apply_kimi_k3_lora(model, options)


def _forward(model, inputs):
    hidden, bank = inputs, None
    for layer in model.decoder.layers:
        hidden, bank = layer(hidden, context=bank)
    return hidden


def _verify_updates(model):
    all_parameters = dict(model.named_parameters())
    trainable = {name: value for name, value in all_parameters.items() if value.requires_grad}
    adapter_before = {
        name: value.detach().clone() for name, value in all_parameters.items() if ".lora_adapter." in name
    }
    assert trainable and all(".lora_adapter." in name for name in trainable)
    assert all(not parameter.requires_grad for parameter in model.decoder.layers[2].parameters())
    optimizer = torch.optim.SGD(trainable.values(), lr=1e-4)
    inputs = torch.randn(8, 1, 7168, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        inputs.grad = None
        output = _forward(model, inputs)
        loss = output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values()
        )
        assert all(parameter.grad is None for parameter in all_parameters.values() if not parameter.requires_grad)
        optimizer.step()
        losses.append(loss.item())
        print(f"PREFIX_STEP {step + 1}: loss={loss.item()}", flush=True)
    changed = [name for name, before in adapter_before.items() if not torch.equal(all_parameters[name], before)]
    assert changed and all(name in trainable for name in changed)
    assert any("decoder.layers.0." in name for name in changed)
    assert any("decoder.layers.1." in name for name in changed)
    exported = [name for chunk in export_kimi_k3_lora_hf_chunks([model]) for name, _tensor in chunk]
    assert any("layers.2." in name for name in exported)
    return {
        "losses": losses,
        "optimizer_steps": 2,
        "changed_parameters": changed,
        "exported_hf_tensors": len(exported),
        "frozen_layer_adapter_exported": True,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(
        "nccl", init_method=f"file://{Path(tempfile.mkdtemp()) / 'rendezvous'}", rank=0, world_size=1
    )
    parallel_state.initialize_model_parallel()
    tensor_parallel.model_parallel_cuda_manual_seed(41)
    try:
        model = _build_model()
        print("PREFIX_MODEL_BUILT", flush=True)
        counts = load_native_checkpoint([model], args.checkpoint, tp_rank=0, tp_size=1)
        print(f"PREFIX_CHECKPOINT_LOADED {counts}", flush=True)
        result = _verify_updates(model)
        torch.cuda.synchronize()
        result.update(
            status="PASS",
            scope="first 3 K3 layers, 896 experts per included MoE layer; not full-model training",
            loaded_counts=counts,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
