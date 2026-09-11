"""Reference MXFP4 linear: packed storage, BF16 compute, input gradients only.

Experts run sequentially. This deliberately favors bounded temporary memory and
an auditable backward over the throughput of a fused grouped GEMM kernel.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from miles.utils.mxfp4 import dequantize_mxfp4


def decode_weight(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Bound FP32 decoder intermediates to 128 rows; return one BF16 expert."""
    weight = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.bfloat16)
    for start in range(0, packed.shape[0], 128):
        stop = start + 128
        weight[start:stop].copy_(dequantize_mxfp4(packed[start:stop], scales[start:stop], group_size=32))
    return weight


class _FrozenMXFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, scales, retain_bf16):
        weight = decode_weight(packed, scales)
        if ctx.needs_input_grad[0]:
            ctx.retain_bf16 = retain_bf16
            if retain_bf16:
                ctx.save_for_backward(weight)
            else:
                ctx.save_for_backward(packed, scales)
        return F.linear(inputs, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        if ctx.retain_bf16:
            (weight,) = ctx.saved_tensors
        else:
            weight = decode_weight(*ctx.saved_tensors)
        return grad_output.matmul(weight), None, None, None


class FrozenMXFP4GroupedLinear(nn.Module):
    """The TEGroupedMLP linear interface, restricted to frozen weights and ETP1."""

    def __init__(
        self,
        num_gemms,
        input_size,
        output_size,
        *,
        config,
        init_method=None,
        bias=False,
        skip_bias_add=False,
        is_expert=True,
        tp_comm_buffer_name=None,
        pg_collection=None,
        name=None,
        retain_bf16=False,
    ):
        super().__init__()
        if bias or not is_expert or config.expert_tensor_parallel_size != 1:
            raise ValueError("K3 MXFP4 experts require bias=False, is_expert=True and expert TP=1")
        if config.params_dtype != torch.bfloat16 or input_size % 32:
            raise ValueError("K3 MXFP4 requires BF16 compute and input dimensions divisible by 32")
        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size
        self.retain_bf16 = retain_bf16
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.register_buffer(
            "packed", torch.empty(num_gemms, output_size, input_size // 2, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "scales", torch.empty(num_gemms, output_size, input_size // 32, dtype=torch.uint8, device=device)
        )
        # The native LoRA injector reads weight0 solely as a dtype/device reference.
        self.register_buffer("_lora_reference", torch.empty(0, dtype=torch.bfloat16, device=device), persistent=False)
        self._loaded_experts = set()

    @property
    def weight0(self):
        return self._lora_reference

    @torch.no_grad()
    def load_expert(self, local_index, packed, scales):
        if not 0 <= local_index < self.num_gemms:
            raise ValueError(f"Invalid local expert index {local_index}")
        for source, destination in ((packed, self.packed[local_index]), (scales, self.scales[local_index])):
            if source.dtype != torch.uint8 or source.shape != destination.shape:
                raise ValueError(
                    f"MXFP4 tensor mismatch: {source.dtype} {source.shape}, expected uint8 {destination.shape}"
                )
            destination.copy_(source)
        self._loaded_experts.add(local_index)

    def forward(self, inputs, m_splits):
        if len(self._loaded_experts) != self.num_gemms:
            raise RuntimeError("MXFP4 experts must be loaded from the native HF checkpoint before forward")
        if inputs.dtype != torch.bfloat16 or inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError("MXFP4 grouped linear requires a [tokens, input_size] BF16 input")
        if len(m_splits) != self.num_gemms or any(count < 0 for count in m_splits) or sum(m_splits) != inputs.shape[0]:
            raise ValueError("Expert token splits must match the input and local expert count")
        outputs = []
        for expert, chunk in enumerate(inputs.split(m_splits, dim=0)):
            if chunk.shape[0]:
                outputs.append(
                    _FrozenMXFP4Linear.apply(chunk, self.packed[expert], self.scales[expert], self.retain_bf16)
                )
        if outputs:
            return torch.cat(outputs, dim=0), None
        # Keep a zero input gradient on ranks that received no expert tokens.
        return inputs.sum(dim=-1, keepdim=True).expand(0, self.output_size), None

    def backward_dw(self):
        """Frozen base weights have no deferred weight-gradient work."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        raise RuntimeError("MXFP4 base torch_dist export is unsupported; save the native K3 LoRA adapter instead")
