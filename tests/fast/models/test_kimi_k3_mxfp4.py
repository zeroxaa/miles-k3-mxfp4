import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from miles.utils.mxfp4 import dequantize_mxfp4, quantize_mxfp4
from miles_plugins.models.kimi_k3_mxfp4.checkpoint import NativeCheckpoint, _global_name, _load_dense, _load_experts
from miles_plugins.models.kimi_k3_mxfp4.linear import FrozenMXFP4GroupedLinear, decode_weight
from miles_plugins.models.kimi_k3_mxfp4.options import layer_indices, select_trainable_layers


def _linear(*, retain=False, experts=3, device="cpu", input_size=64, output_size=32):
    config = SimpleNamespace(expert_tensor_parallel_size=1, params_dtype=torch.bfloat16, use_cpu_initialization=True)
    module = FrozenMXFP4GroupedLinear(experts, input_size, output_size, config=config, retain_bf16=retain).to(device)
    for index in range(experts):
        weight = torch.randn(output_size, input_size, device=device) * 0.05
        module.load_expert(index, *quantize_mxfp4(weight, group_size=32))
    return module


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_forward_backward_matches_dequantized_bf16(retain, recompute):
    torch.manual_seed(17)
    module = _linear(retain=retain)
    inputs = torch.randn(5, 64, dtype=torch.bfloat16, requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_()
    splits = [2, 0, 3]
    reference = torch.cat(
        [
            F.linear(chunk, dequantize_mxfp4(module.packed[index], module.scales[index], 32))
            for index, chunk in enumerate(reference_input.split(splits))
        ]
    )

    def forward(tensor):
        return module(tensor, splits)[0]

    output = checkpoint(forward, inputs, use_reentrant=True) if recompute else forward(inputs)
    grad = torch.randn_like(output)
    output.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, reference_input.grad, rtol=0, atol=0)
    assert not list(module.parameters())


@pytest.mark.parametrize("retain", [False, True])
def test_saved_tensors_contain_only_requested_weight_representation(retain):
    module = _linear(retain=retain, experts=1)
    saved = []

    def capture(tensor):
        saved.append((tensor.dtype, tensor.numel()))
        return tensor

    inputs = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
        module(inputs, [2])[0].sum().backward()
    assert saved == ([(torch.bfloat16, 32 * 64)] if retain else [(torch.uint8, 32 * 32), (torch.uint8, 32 * 2)])


def test_empty_dispatch_keeps_input_gradient():
    module = _linear()
    inputs = torch.empty(0, 64, dtype=torch.bfloat16, requires_grad=True)
    output, bias = module(inputs, [0, 0, 0])
    output.sum().backward()
    assert output.shape == (0, 32) and bias is None
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape


def test_unloaded_experts_fail_before_computing():
    module = _linear()
    module._loaded_experts.clear()
    with pytest.raises(RuntimeError, match="must be loaded"):
        module(torch.zeros(1, 64, dtype=torch.bfloat16), [1, 0, 0])


def test_decoder_chunk_boundary_matches_reference():
    packed, scales = quantize_mxfp4(torch.randn(257, 64), 32)
    torch.testing.assert_close(decode_weight(packed, scales), dequantize_mxfp4(packed, scales, 32), rtol=0, atol=0)


def test_frozen_later_layer_propagates_gradient_to_earlier_lora():
    torch.manual_seed(23)
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList()
    for index in range(2):
        layer = nn.Module()
        layer.layer_number = index + 1
        layer.base = _linear(experts=1, input_size=64, output_size=64)
        layer.lora_adapter = nn.Module()
        layer.lora_adapter.A = nn.Parameter(torch.randn(4, 64, dtype=torch.bfloat16) * 0.1)
        layer.lora_adapter.B = nn.Parameter(torch.zeros(64, 4, dtype=torch.bfloat16))
        model.decoder.layers.append(layer)
    select_trainable_layers(model, {0})
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        hidden = torch.ones(2, 64, dtype=torch.bfloat16)
        for layer in model.decoder.layers:
            hidden = layer.base(hidden, [2])[0] + F.linear(
                F.linear(hidden, layer.lora_adapter.A), layer.lora_adapter.B
            )
        hidden.float().square().mean().backward()
        optimizer.step()
    changed = [name for name, value in model.state_dict().items() if not torch.equal(value, before[name])]
    assert set(changed) == {"decoder.layers.0.lora_adapter.A", "decoder.layers.0.lora_adapter.B"}
    assert all(parameter.grad is None for parameter in model.decoder.layers[1].parameters())


def _checkpoint(tmp_path, tensors):
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in tensors}})
    )
    return NativeCheckpoint(tmp_path)


def test_checkpoint_tp_slices_and_padding(tmp_path):
    full = torch.arange(128 * 4).reshape(128, 4)
    reader = _checkpoint(tmp_path, {"A_log": full})
    try:
        actual = reader.read("A_log", trim_rows=96, partition_dim=0, tp_rank=2, tp_size=8)
        torch.testing.assert_close(actual, full[24:36], rtol=0, atol=0)
        with pytest.raises(KeyError):
            reader.read("missing")
    finally:
        reader.close()


def test_dense_gated_fc1_shards_gate_and_up_independently(tmp_path, monkeypatch):
    # Exercise the real pure name converter without importing unrelated GPU backends.
    module_name = "miles.backends.megatron_utils.megatron_to_hf.kimi_k3"
    converter_path = Path(__file__).resolve().parents[3] / "miles/backends/megatron_utils/megatron_to_hf/kimi_k3.py"
    spec = importlib.util.spec_from_file_location(module_name, converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    monkeypatch.setitem(sys.modules, module_name, converter)
    model = nn.Module()
    model.config = SimpleNamespace(kimi_linear_num_heads=96)
    model.decoder = nn.Module()
    layer = nn.Module()
    layer.layer_number = 1
    layer.mlp = nn.Module()
    layer.mlp.linear_fc1 = nn.Linear(32, 8, bias=False)
    model.decoder.layers = nn.ModuleList([layer])
    weight = layer.mlp.linear_fc1.weight
    weight.tensor_model_parallel, weight.partition_dim, weight.partition_stride = True, 0, 2
    gate = torch.arange(32 * 32).float().reshape(32, 32)
    up = gate + 10000
    reader = _checkpoint(
        tmp_path,
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": gate,
            "language_model.model.layers.0.mlp.up_proj.weight": up,
        },
    )
    try:
        assert _load_dense(model, reader, tp_rank=3, tp_size=8) == 2
        torch.testing.assert_close(weight, torch.cat((gate[12:16], up[12:16])), rtol=0, atol=0)
    finally:
        reader.close()


def test_checkpoint_ep_load_uses_global_expert_and_layer_indices(tmp_path):
    tensors = {}
    expected = {}
    for index in (4, 5):
        for projection in ("w1", "w3", "w2"):
            key = f"language_model.model.layers.31.block_sparse_moe.experts.{index}.{projection}"
            packed, scales = quantize_mxfp4(torch.randn(32, 32), 32)
            tensors[key + ".weight_packed"] = packed
            tensors[key + ".weight_scale"] = scales
            expected[index, projection] = packed
    fc1 = _linear(experts=2, input_size=32, output_size=64)
    fc2 = _linear(experts=2, input_size=32, output_size=32)
    layer = SimpleNamespace(
        layer_number=32,
        mlp=SimpleNamespace(local_expert_indices=[4, 5], experts=SimpleNamespace(linear_fc1=fc1, linear_fc2=fc2)),
    )
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
    reader = _checkpoint(tmp_path, tensors)
    try:
        assert _load_experts(model, reader) == 2
        for local, index in enumerate((4, 5)):
            torch.testing.assert_close(fc1.packed[local], torch.cat((expected[index, "w1"], expected[index, "w3"])))
            torch.testing.assert_close(fc2.packed[local], expected[index, "w2"])
        assert (
            _global_name(model, "decoder.layers.0.self_attention.A_log")
            == "module.module.decoder.layers.31.self_attention.A_log"
        )
    finally:
        reader.close()


@pytest.mark.parametrize("indices", [[-1], [93], [1, 1], [True], [], "0,1"])
def test_invalid_layer_selection_fails(indices):
    with pytest.raises(ValueError):
        layer_indices(indices, num_layers=93, option="train_layers")
