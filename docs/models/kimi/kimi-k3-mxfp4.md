---
title: Kimi K3 MXFP4 LoRA experiment
description: Frozen native MXFP4 experts with BF16 computation and selectable trainable layers.
---

This experimental trainer reuses the complete K3 architecture and native Miles
LoRA adapters. Its model type is `kimi-k3-mxfp4`; the model still has 93 layers
and 896 routed experts per MoE layer. It does not require creating a BF16 copy
of the checkpoint.

## What changes

| Component | Experiment behavior |
|---|---|
| Routed expert base matrices | Frozen uint8 packed MXFP4 values and E8M0 scales, loaded directly from HF |
| Expert computation | Decode one active expert to BF16, then use ordinary matrix multiplication |
| Expert backward | Compute input gradients; base weights have no gradients or optimizer state |
| Other base weights | Existing K3 precision policy, predominantly BF16, with marked FP32 tensors preserved |
| Trainable parameters | Native LoRA adapters in the selected layers only |
| Frozen layers' adapters | Still present for the existing complete adapter export/resume layout |
| Rollout | Existing model architecture and adapter tensor names; online synchronization is not validated by this experiment |

The layer spec replaces only the routed expert linear builders. It keeps KDA,
MLA, SiTU, attention residuals, routing, shared experts and native adapter export
in their existing implementations. A small-rank fallback in native LoRA avoids
CUDA grouped GEMM's matrix-stride restriction for ranks such as 1, 2 and 4.

## Independent experiment choices

The default configuration is `examples/kimi_k3_mxfp4/two_layers.yaml`:

```yaml
kimi_k3_mxfp4: true
kimi_k3_mxfp4_train_layers: [0, 1]
kimi_k3_mxfp4_retain_layers: [1]
recompute_granularity: null
recompute_method: null
recompute_num_layers: null
offload_train: false
offload_rollout: false
```

Indices are **zero-based**. Layer 0 is dense; layer 1 is the first MoE layer.
Both layers' native LoRA adapters train. Their base matrices stay frozen.
This is additive `W_base + (alpha/r) BA` training, not a full update of those
layers' base matrices.

`train_layers` chooses trainability before DDP/optimizer construction. For the
first ten layers, use `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
`retain_layers` independently chooses which MoE layers save the expanded BF16
expert weights for backward; `[]` decodes them again during backward. Only
experts receiving tokens are decoded. Saved weights live in the autograd graph
and are released with it. A retained dense layer index has no effect because
its base is already BF16.

Activation checkpointing is separate. To enable it, set the three recompute
fields to `full`, `uniform`, `1`. The checkpointed original forward then does
not retain a graph; its recomputed forward decodes weights again, and the
retain option saves weights from that recomputation for its subsequent backward.

Freezing later layers does not eliminate their input-gradient work when earlier
layers train. Nor does lowering LoRA rank change the space occupied by the frozen
base or guarantee that a given sequence length fits.

## Offline launcher

Use the same pinned Megatron and SGLang sources as the native K3 recipe, with
the native checkpoint and previously saved Miles rollout samples. The samples
are inputs to training; this launcher does not start a rollout server. It
reuses the existing debug GRPO recipe and optimizer unless overridden.

The proposed full-model layout is TP8 / EP8 / ETP1 / PP3 on three eight-GPU
nodes. **This layout is a configuration to test, not a verified full-model
training result.** Run through the cluster's normal isolated container and Ray
setup; for an already joined Ray cluster:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k3_mxfp4.py \
  --hardware H200 --num-nodes 3 --num-gpus-per-node 8 \
  --hf-checkpoint /models/Kimi-K3 \
  --rollout-data /datasets/fixed-samples.pt \
  --data-dir /datasets --save-dir /checkpoints/k3-mxfp4 \
  --megatron-path /sources/megatron --sglang-path /sources/sglang/python \
  --experiment-config examples/kimi_k3_mxfp4/two_layers.yaml
```

The default LoRA rank is 4 and alpha is 8. `--hf-checkpoint` and `--ref-load`
both point at the native MXFP4 checkpoint unless explicitly overridden. Keep
the same base checkpoint, topology and experiment YAML when resuming with
`--extra-args '--lora-adapter-path /checkpoints/.../adapter'`.
Adapter checkpoints use Miles's existing per-rank format. Export of the packed
base as a Megatron `torch_dist` checkpoint is deliberately unsupported.

## Verification tools and limits

```bash
# CPU arithmetic, autograd retention, frozen-layer gradients and TP/EP loader slices
pytest tests/fast/models/test_kimi_k3_mxfp4.py

# Inside an appropriate CUDA container, on one H200:
python tools/kimi_k3_mxfp4_smoke.py \
  --checkpoint /models/Kimi-K3 --report /results/operator.json
python tools/kimi_k3_mxfp4_prefix_smoke.py \
  --checkpoint /models/Kimi-K3 --report /results/prefix.json
```

The first GPU tool compares one original expert's full-sized matrices against
explicit BF16 forward/backward, then runs two native LoRA optimizer steps in
each retention mode. The second constructs the native spec and loads the first
three layers, including all 896 experts in each included MoE layer, updates
only layers 0 and 1, and checks that the frozen layer's adapter still exports.
These are bounded integration tests. They do not establish 24-GPU model fit,
distributed optimizer correctness, end-to-end language-model loss quality or
online rollout compatibility.

The reference expert backend currently requires BF16 compute, ETP1 and
Megatron DDP. Trainer offload, TE op fusion, FP8/FP4 compute and FSDP are rejected.
It supports first-order gradients only and uses sequential expert GEMMs, so
throughput is expected to be substantially below a tuned fused implementation.

Validation on 2026-09-11 used PyTorch 2.11.0+cu129 on H200. Both GPU tools
passed: the expert reference had zero forward/input-gradient error in both
retention modes; the three-layer test completed two optimizer steps, changed
only the selected layers' adapters, and exported 36 HF tensors. Its PyTorch
peak allocated memory was 41.54 GiB. This is a three-layer measurement, not
a full-model memory estimate. Nineteen focused CPU tests and four new model/
launcher snapshot checks passed. The broader launcher suite retained the same
47 failing/error cases as the unmodified pinned source, with no new failure IDs.
