---
title: Kimi K3 MXFP4 LoRA experiment
description: Frozen native MXFP4 experts with BF16 computation and selectable trainable layers.
---

The separate [colocated RL integration instructions](../../../examples/kimi_k3_mxfp4/colocated_rl.md)
and [24-H200 results](../../../examples/kimi_k3_mxfp4/results/rl_cycle_h200_24.md)
describe two real rollout/training/handoff cycles completed on 2026-09-12.
They used the same 24 GPUs and full 93-layer model, with all-layer LoRA. The
observed cycles took 716.2 and 723.4 seconds, including shared-filesystem stalls
recovered by diagnostic prefetches. This validates the experimental handoff;
it does not establish steady-state throughput or native Ray/CUDA IPC integration.

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
| Rollout | Existing model architecture and adapter names; a separate colocated RL harness validates filesystem adapter handoff |

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

The full-model layout is TP8 / EP8 / ETP1 / PP3 on three eight-GPU nodes.
The direct training check below validates this layout; the Ray/debug-GRPO
launcher remains a separate integration path. Run it through the cluster's
normal isolated container and Ray setup; for an already joined Ray cluster:

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
Megatron DDP. Native trainer offload, TE op fusion, FP8/FP4 compute and FSDP are rejected.
The colocated RL harness below owns a separate, explicitly tested TMS region;
it does not enable the native Ray actor's offload route.
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

## Full-model training check with selected layers

`tools/kimi_k3_mxfp4_full_smoke.py` runs the complete language model with
trainable layers selected by `--custom-config-path`. Use
`examples/kimi_k3_mxfp4/three_layers.yaml` for layers 0–2, or
`examples/kimi_k3_mxfp4/four_layers.yaml` for layers 0–3.
`examples/kimi_k3_mxfp4/all_layers_retain_all.yaml` selects all 93 layers for
LoRA training and retains active experts' expanded BF16 weights in all 92 MoE
layers. All three configurations disable activation checkpointing.
Launch it with one torchrun process per GPU, three nodes and eight GPUs per
node. It requires the full model args from `kimi-k3-mxfp4`, TP8 / EP8 / ETP1 /
PP3, sequence parallelism, native K3 LoRA targets, rank 4 / alpha 8,
micro/global batch 1 and SGD with zero momentum.
Use `--full-smoke-report /results/full.json` and `--save /results/adapter`.
The checkpoint path must contain all tensors belonging to the local 31-layer
pipeline stage, plus the relevant embedding or output head and tokenizer files.

The check uses authored text with next-token cross entropy at the final output,
then performs two optimizer steps. It records forward and backward hooks for
every local layer, checks that changed adapter tensors belong exclusively to
the configured trainable layers, compares every frozen adapter against its
initial value, and saves native rank-sharded adapters and optimizer state.
Each selected layer must have actual updates, including every supported
attention projection's LoRA B tensor. The aggregate JSON requires 24 distinct
physical GPU UUIDs and coverage of all 93 layers, and records the selected
layers, attention types and verified attention updates.

The tool also verifies the actual expert modules' retention flags, records
allocated/reserved memory after each layer's forward, and counts expert weight
decodes during forward and backward. With all MoE layers retained and no
activation checkpointing, backward must perform zero expert weight decodes.
If the pipeline runs out of memory, `rank<N>-oom.json` records the exception,
completed layer coverage and memory measurements before the process fails.

The first four layers have the following native LoRA targets. All their base
weights, including norms and routers, remain frozen; both LoRA A and B train.

| Zero-based layer | Attention type | Attention LoRA projections | MLP LoRA targets |
|---|---|---|---|
| 0 | KDA linear attention | `o_proj` | Dense MLP gate/up and down |
| 1, 2 | KDA linear attention | `o_proj` | Routed and shared expert gate/up/down |
| 3 (fourth layer) | MLA full attention | `o_proj`, `q_a_proj`, `kv_a_proj_with_mqa` | Routed and shared expert gate/up/down |

The four-layer configuration selects `[0, 1, 2, 3]` for training and retains
expanded expert weights in MoE layers `[1, 2, 3]`. LoRA rank, alpha and the
attention implementations are unchanged. It can also be passed as
`--experiment-config examples/kimi_k3_mxfp4/four_layers.yaml` to the offline
launcher; that Ray/debug-GRPO integration path is still unvalidated.

The three stages contain layers 0–30, 31–61 and 62–92. In the three- and
four-layer configurations, only the first stage has trainable parameters;
the later stages still compute activation gradients and send them upstream.
The all-layer configuration trains LoRA in every stage. A single microbatch
leaves pipeline stages idle during parts of the step, so this is a memory and
correctness check rather than a throughput benchmark. The MXFP4 base remains packed; only active expert
matrices are temporarily decoded to BF16. Layers 1 and 2 retain those decoded
matrices for backward in the three-layer configuration; the four-layer
configuration also retains layer 3. Other MoE layers decode them again.
The all-layer/all-retained configuration keeps every active expert's decoded
weight until its backward use, then releases it. It does not expand inactive
experts or retain BF16 copies permanently between steps. Weight retention and
activation checkpointing are independent choices; enabling all-layer LoRA
does not require retaining all decoded weights.

Two compatibility details are explicit in the tool: marked KDA FP32 parameters
are restored after Megatron's BF16 module wrapper, and `torch.optim.SGD` replaces
TE FusedSGD during optimizer construction. TE FusedSGD in the pinned image
indexes the first parameter and fails on fully frozen pipeline stages. The
standard SGD fallback keeps Megatron's mixed-precision optimizer wrapper,
gradient synchronization and gradient statistics. No synthetic model or GPU
substitutes are used. This tool does not exercise Ray, rollout generation or
online adapter synchronization.

### Three-layer result

On 2026-09-11 the three-layer check **passed on 24 distinct H200 GPUs** using
PyTorch 2.11.0+cu129 and FLA 0.5.2. Both steps traversed all 93 layers in forward
and backward and completed optimizer updates. Sequence length was 32, global
batch 1, SGD learning rate `1e-4`, and activation recomputation was disabled.
Both losses were `1.5600831508636475`; the gradient norms were approximately
`0.06585174` and `0.06585191`. The equal losses on this tiny fixture do not
establish a quality improvement. The first step took about 615 seconds including
kernel compilation; the second took about 92 seconds.

| Pipeline stage | Zero-based layers | Trainable layers | PyTorch peak allocated per GPU |
|---|---|---|---|
| 0 | 0–30 | 0, 1, 2 | 66.77–68.00 GiB |
| 1 | 31–61 | None | 65.14–65.18 GiB |
| 2 | 62–92 | None | 65.41–65.44 GiB |

Each of the first eight ranks had 8,689,152 trainable adapter elements; this
is a local physical count and includes replicated tensors. Both LoRA A and B
were trainable. After two steps, 15 B tensors per first-stage rank had changed;
all other adapter values were unchanged. The changed modules were attention
output projections, the first layer's dense MLP, and the second/third layers'
routed and shared expert MLPs. The tool saved 24 native adapter shards and 24
optimizer/scheduler state files. A checkpoint reload and the online RL loop
were not exercised by this check.

### Four-layer result, including MLA full attention

On 2026-09-11 at 22:38:41 UTC, the same full-model check **passed on 24 distinct
H200 GPUs** with `four_layers.yaml`. Layers 0–3 were trainable and MoE layers
1–3 retained their expanded expert weights. The remaining 89 layers' adapters
and all base weights stayed frozen. Both steps covered all 93 layers in forward
and backward, with sequence length 32, global batch 1, rank 4 / alpha 8, SGD
learning rate `1e-4`, zero momentum and no activation recomputation.

The fourth layer's `o_proj`, `q_a_proj` and `kv_a_proj_with_mqa` LoRA B tensors
all changed on every first-stage rank. Its routed and shared expert MLP LoRA B
tensors also changed. Each of these eight ranks had 46 trainable adapter tensors
containing 13,028,096 local elements, including replicated tensors. After two
steps, 23 B tensors per rank had changed. A tensors were also trainable, but
their BF16 stored values were unchanged in this short check. Every frozen
adapter was compared against its initial value and remained unchanged.

| Pipeline stage | Zero-based layers | Trainable layers | PyTorch peak allocated per GPU |
|---|---|---|---|
| 0 | 0–30 | 0, 1, 2, 3 | 68.18–69.59 GiB |
| 1 | 31–61 | None | 65.14–65.19 GiB |
| 2 | 62–92 | None | 65.41–65.44 GiB |

The step losses were `1.5504195690` and `1.5723326206`; gradient norms were
`2.6606215464` and `0.0496122908`. Rank 0 took 228.42 and 88.81 seconds, using
existing kernel caches. The loss increased on this short fixture; this check
establishes execution and update scope, not training quality. The job exited
successfully after saving 24 native adapter shards and 24 optimizer/scheduler
state files. Nineteen focused CPU regression tests also passed. Checkpoint
reload, long sequences and online RL/rollout synchronization remain untested.

### All-layer LoRA with every active expert weight retained

On 2026-09-12 at 01:31:50 UTC, `all_layers_retain_all.yaml` **passed on 24
distinct H200 GPUs**. All 93 layers' native LoRA adapters were trainable and
all 92 MoE layers retained their active experts' expanded BF16 weights for
backward. The full native checkpoint remained loaded, with all base weights
frozen. Sequence length was 32, micro/global batch 1, rank 4 / alpha 8, SGD
learning rate `1e-4`, zero momentum and no activation checkpointing or offload.

Both steps covered all 93 layers in forward and backward. Every layer had
actual adapter updates, including all supported attention LoRA B tensors on
every rank. The tool verified both routed expert linears' retention flags in
each MoE layer. Across all 24 ranks and both steps it counted 64,716 expert
weight decodes in forward and **zero in backward**, confirming reuse of the
retained BF16 weights.

| Pipeline stage | Zero-based layers, all trainable | PyTorch peak allocated per GPU | Sampled device peak per GPU |
|---|---|---|---|
| 0 | 0–30 | 108.66–113.41 GiB | 116.39–121.40 GiB |
| 1 | 31–61 | 101.24–106.40 GiB | 109.20–114.38 GiB |
| 2 | 62–92 | 106.32–109.95 GiB | 114.04–117.65 GiB |

Device usage was sampled with `nvidia-smi` every three seconds and includes
allocations outside PyTorch. PyTorch allocator peaks are reported separately;
its largest reserved-memory peak was 114.05 GiB. The maximum device sample was
121.40 GiB out of 140.40 GiB reported by `nvidia-smi`. These measurements apply
to this short fixture: longer sequences or more diverse tokens may activate
more experts and retain more BF16 weights, in addition to larger activations.
This run does not establish that longer sequences or larger batches fit.

The local trainable element counts were 128,797,952, 132,994,048 and 133,059,840
per rank in stages 0, 1 and 2 respectively, including replicated parameters.
Both A and B were trainable; every LoRA B tensor changed, and 21 A tensors
across all ranks also changed their BF16 stored values. All base parameters
remained excluded from gradient updates.

Losses were `1.5508167744` and `1.5647609234`; gradient norms were `2.5502069128`
and `2.5645694285`. Rank 0 took 198.14 and 50.49 seconds using existing kernel
caches. The loss increased, so this remains an execution and memory check,
not evidence of quality improvement. The job exited successfully and saved
24 adapter shards (6,321,158,868 bytes) plus 24 optimizer/scheduler state files
(12,638,622,804 bytes). Nineteen focused CPU tests and formatting/lint checks
passed. Ray/RL integration, online adapter synchronization and checkpoint
reload were not exercised.
