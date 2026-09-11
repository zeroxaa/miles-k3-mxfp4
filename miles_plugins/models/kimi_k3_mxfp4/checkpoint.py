"""Stream native HF weights directly into the local PP/TP/EP model shard."""

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

from miles_plugins.models.kimi_k3_mxfp4.linear import FrozenMXFP4GroupedLinear

logger = logging.getLogger(__name__)


class NativeCheckpoint:
    """A small mmap handle cache; never materialize a full checkpoint shard."""

    def __init__(self, path):
        self.path = Path(path)
        with (self.path / "model.safetensors.index.json").open() as stream:
            self.weight_map = json.load(stream)["weight_map"]
        self.handles = OrderedDict()

    def read(self, name, *, partition_dim=None, tp_rank=0, tp_size=1, trim_rows=None):
        filename = self.weight_map[name]
        if filename not in self.handles:
            if len(self.handles) == 2:
                _, handle = self.handles.popitem(last=False)
                handle.__exit__(None, None, None)
            self.handles[filename] = safe_open(self.path / filename, framework="pt", device="cpu")
        self.handles.move_to_end(filename)
        tensor_slice = self.handles[filename].get_slice(name)
        shape = tensor_slice.get_shape()
        selection = [slice(None)] * len(shape)
        if trim_rows is not None:
            if shape[0] < trim_rows:
                raise ValueError(f"{name}: cannot trim {shape[0]} rows to {trim_rows}")
            shape[0] = trim_rows
            selection[0] = slice(0, trim_rows)
        if partition_dim is not None:
            width, remainder = divmod(shape[partition_dim], tp_size)
            if remainder or not 0 <= tp_rank < tp_size:
                raise ValueError(f"{name}: invalid TP{tp_size} partition of {shape}")
            selection[partition_dim] = slice(tp_rank * width, (tp_rank + 1) * width)
        return tensor_slice[tuple(selection)]

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()


def _global_name(model, name):
    match = re.fullmatch(r"decoder.layers.(\d+).(.+)", name)
    if match:
        local_index, rest = match.groups()
        index = model.decoder.layers[int(local_index)].layer_number - 1
        name = f"decoder.layers.{index}.{rest}"
    return f"module.module.{name}"


@torch.no_grad()
def _load_dense(model, reader, *, tp_rank, tp_size):
    # The existing converter imports Megatron; defer it for CPU checkpoint-reader tests.
    from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import convert_kimi_k3_to_hf

    tensors = list(model.named_parameters())
    tensors += [(name, buffer) for name, buffer in model.named_buffers() if "expert_bias" in name]
    count = 0
    for name, destination in tensors:
        if ".lora_adapter." in name:
            continue
        global_name = _global_name(model, name)
        parts = convert_kimi_k3_to_hf(None, global_name, destination)
        partition_dim = getattr(destination, "partition_dim", -1)
        partition_dim = partition_dim if getattr(destination, "tensor_model_parallel", False) else None
        stride = getattr(destination, "partition_stride", 1)
        # Gated FC1 has stride 2: shard gate and up independently, then concatenate.
        supported_stride = stride == 1 or (stride == 2 and len(parts) == 2 and partition_dim == 0)
        if partition_dim is not None and not supported_stride:
            raise ValueError(f"Unsupported strided TP partition: {name}")
        offset = 0
        for hf_name, part in parts:
            source = reader.read(
                hf_name,
                partition_dim=partition_dim,
                tp_rank=tp_rank,
                tp_size=tp_size,
                trim_rows=model.config.kimi_linear_num_heads if hf_name.endswith(".A_log") else None,
            )
            if source.shape != part.shape:
                raise ValueError(f"{name} <- {hf_name}: expected {part.shape}, got {source.shape}")
            target = destination if len(parts) == 1 else destination[offset : offset + part.shape[0]]
            target.copy_(source)
            offset += part.shape[0]
            count += 1
    return count


@torch.no_grad()
def _load_experts(model, reader):
    count = 0
    for layer in model.decoder.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        fc1, fc2 = experts.linear_fc1, experts.linear_fc2
        if not isinstance(fc1, FrozenMXFP4GroupedLinear) or not isinstance(fc2, FrozenMXFP4GroupedLinear):
            raise TypeError("MXFP4 checkpoint loader requires the MXFP4 model spec for every MoE layer")
        for local_index, global_index in enumerate(layer.mlp.local_expert_indices):
            prefix = f"language_model.model.layers.{layer.layer_number - 1}.block_sparse_moe.experts.{global_index}"
            gate = reader.read(f"{prefix}.w1.weight_packed")
            up = reader.read(f"{prefix}.w3.weight_packed")
            gate_scale = reader.read(f"{prefix}.w1.weight_scale")
            up_scale = reader.read(f"{prefix}.w3.weight_scale")
            fc1.load_expert(local_index, torch.cat((gate, up)), torch.cat((gate_scale, up_scale)))
            fc2.load_expert(
                local_index, reader.read(f"{prefix}.w2.weight_packed"), reader.read(f"{prefix}.w2.weight_scale")
            )
            count += 1
        logger.info("Loaded MXFP4 layer %d: %d local experts", layer.layer_number - 1, fc1.num_gemms)
    return count


def load_native_checkpoint(models, path, *, tp_rank, tp_size):
    reader = NativeCheckpoint(path)
    try:
        counts = [
            (_load_dense(model, reader, tp_rank=tp_rank, tp_size=tp_size), _load_experts(model, reader))
            for model in models
        ]
        logger.info("Loaded K3 native checkpoint; (dense tensors, local experts) per stage: %s", counts)
        return counts
    finally:
        reader.close()
