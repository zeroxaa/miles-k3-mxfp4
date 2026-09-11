"""Compose the standard K3 spec with a frozen packed expert linear backend."""

from functools import partial

from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles_plugins.models.kimi_k3.model import build_kimi_k3_spec, configure_kimi_k3
from miles_plugins.models.kimi_k3_mxfp4.linear import FrozenMXFP4GroupedLinear
from miles_plugins.models.kimi_k3_mxfp4.options import layer_indices


def get_kimi_k3_mxfp4_spec(args, config, vp_stage=None):
    if not getattr(args, "kimi_k3_mxfp4", False) or not getattr(args, "lora_rank", 0):
        raise ValueError("The MXFP4 spec requires kimi_k3_mxfp4: true and native K3 LoRA")
    if config.fp8 or config.fp4 or config.use_transformer_engine_op_fuser or config.fine_grained_activation_offloading:
        raise ValueError("The reference MXFP4 backend requires BF16 compute, no TE op fuser or fine-grained offload")
    if getattr(args, "offload_train", False):
        raise ValueError("MXFP4 trainer offload is not validated; use the offline experiment launcher first")
    if getattr(args, "use_megatron_fsdp", False) or getattr(args, "use_custom_fsdp", False):
        raise ValueError("The MXFP4 experiment currently supports Megatron DDP, not FSDP")
    selected = layer_indices(args.kimi_k3_mxfp4_train_layers, num_layers=config.num_layers, option="train_layers")
    retained = getattr(args, "kimi_k3_mxfp4_retain_layers", [])
    retained = (
        layer_indices(retained, num_layers=config.num_layers, option="retain_layers") if retained else frozenset()
    )
    config.kimi_k3_mxfp4_train_layers = selected
    configure_kimi_k3(config)
    block = build_kimi_k3_spec(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage)
    for local_index, layer_spec in enumerate(block.layer_specs):
        index = offset + local_index
        if config.moe_layer_freq[index]:
            experts = layer_spec.submodules.mlp.keywords["submodules"].experts
            submodules = experts.keywords["submodules"]
            builder = partial(FrozenMXFP4GroupedLinear, retain_bf16=index in retained)
            submodules.linear_fc1 = builder
            submodules.linear_fc2 = builder
    return block
