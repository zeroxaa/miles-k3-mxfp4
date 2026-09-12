# Two full K3 RL handoffs on the same 24 H200 GPUs

On 2026-09-12, two real rollout → policy update → adapter handoff → updated
rollout cycles completed on the same three eight-H200 nodes. The observed
cycle durations were **716.2 seconds** and **723.4 seconds**. Both filesystem
adapter loads encountered shared-storage mmap stalls and were recovered with
diagnostic sequential prefetches. These are recovery-inclusive observations,
**not steady-state throughput measurements**.

The training step exited successfully at **13:52:24 UTC**, saving 24 native
adapter shards and 24 optimizer/scheduler state files. At **13:54:02 UTC**, a
subsequent cluster `NodeDown` event ended the allocation. It interrupted an
additional local-storage serving comparison, after the two RL cycles and
checkpoint save had finished.

See the [machine-readable measurements](rl_cycle_h200_24.json) and
[integration instructions](../colocated_rl.md). The raw run, per-rank logs,
rollout responses and adapter manifests remain separate from this public
summary; model checkpoints are not included in the repository.

## Configuration and scope

| Item | Value |
|---|---|
| Hardware | 24 distinct physical H200 GPUs, three nodes |
| Host memory | Approximately 2 TB per node, accommodating both engines' CPU backups |
| Model | Full 93-layer K3, all 896 routed experts per MoE layer |
| Checkpoint revision | `a590ce090cb049c93a33dfe8c208ec652aa20503` |
| Train parallelism | TP8 / EP8 / ETP1 / PP3 |
| Rollout parallelism | TP8 / EP1 / PP3, Marlin MXFP4 experts |
| Trainable tensors | Native LoRA in all 93 layers, rank 4, alpha 8 |
| Frozen base | Packed MXFP4 routed experts; native precision for other weights |
| Retention | Retain active experts' BF16 expansion through backward in all 92 MoE layers |
| Activation checkpointing | Disabled |
| Training sequence length | 32, including right padding |
| Samples per update | Two actual SGLang responses, at most four generated tokens each |
| Optimizer | SGD, learning rate `1e-4`, zero momentum |
| Loss | Miles clipped policy loss, response mask, group-normalized rewards, no KL term |

The reward is the first standalone generated digit divided by nine, or zero
when absent. Identical-reward groups are resampled. Cycle 1 needed one group;
cycle 2 needed two attempts. All attempts were recorded. Cycle 2 generated
its training samples with `rl-v1`; the final generation used `rl-v2`.

This uses a small HTTP/filesystem controller and a persistent torchrun worker.
Miles supplies the model, mixed-precision optimizer wrapper, gradient
reductions, policy loss and HF adapter exporter. **The Ray `train.py` driver
and native CUDA IPC weight-transfer protocol were not exercised.** The earlier
authored-text cross-entropy checks are separate from these RL cycles.

## Measured time

Seconds below are wall-clock phase durations. Trainer rows use the maximum
across 24 ranks. They approximately subdivide the controller's trainer wait;
small file, barrier and controller overheads account for the difference.

| Phase | Cycle 1 | Cycle 2 |
|---|---:|---:|
| Rollout sampling, including resampling | 2.75 | 5.96 |
| Flush/offload inference, all 24 GPUs complete | 24.80 | 24.29 |
| Restore trainer and communication groups | 18.26 | 16.88 |
| Two microbatches and one RL optimizer update | 272.08 | 197.05 |
| Gather, export and hash LoRA | 7.69 | 7.78 |
| Offload trainer, including cleanup and synchronization | 62.84 | 61.71 |
| Restore inference weights, all 24 GPUs complete | 26.61 | 26.79 |
| Unload previous adapter | — | 0.58 |
| Load adapter HTTP phase, including filesystem stall | 226.94 | 348.71 |
| KV restore and downstream control completion | 10.35 | 10.45 |
| First generation with the new adapter | 52.80 | 11.42 |
| **Complete cycle** | **716.22** | **723.37** |

These cycle totals exclude initial process/model startup and the final native
checkpoint save. Initial trainer startup/offload waiting took another 217.2
seconds. Initial serving startup and a 189.2-second Marlin compilation happened
during bring-up. First-use training and LoRA kernel costs within cycle 1 remain
included in its cycle duration.

The adapter load HTTP duration is **not** a pure network or GPU-copy timer.
The old serving loader reads and normalizes CPU tensors; later pipeline stages
process the control message downstream, and GPU installation is lazy on the
first generation. Thus adapter load, KV restore and first generation must be
read together: approximately **290.1 seconds** in cycle 1 and **370.6 seconds**
in cycle 2, including the storage stalls and diagnostic recovery.

Direct trainer TMS offload calls in cycle 2 took 30.35–35.74 seconds per rank;
the full offload phase took 61.71 seconds because it also includes cleanup,
communication-group destruction, verification and synchronization. Direct
trainer restore calls took 15.63–16.87 seconds, within the 16.88-second phase.

## What moves between the engines

The native exporter gathers TP/EP adapter pieces into three HF safetensors
files, one per pipeline stage. They contain **1,392 tensors**, with
**6,133,469,184 bytes of tensor payload** (6.13 GB; 5.71 GiB). All three files,
including headers, total 6,133,680,216 bytes. The serving engine receives their
directory through `load_lora_adapter`, normalizes and slices the tensors for
its layout, and installs them in its LoRA memory pool. The base is not merged
with LoRA, re-quantized or exported during an update.

The checkpoint size is large despite rank 4 because this configuration trains
LoRA throughout a model with many experts. It is not a measurement of only the
first three or four layers. Physical trainer adapter storage across ranks is
6.32 GB, including replicated pieces; the HF export removes that replication.

There is a separate residency cost. The trainer's frozen base occupies about
**1.64 TB across its 24 physical shards**. Each serving GPU holds roughly
74–76 GB of base-weight storage in its serving layout. Each engine keeps its
own CPU backup, and those unchanged weights still move between host RAM and
GPU memory when engines alternate. The four principal offload/restore phases
in cycle 2 total about **129.7 seconds**. Avoiding trainer-to-serving base
resynchronization does not eliminate these local transfers.

The measured file payload is not a measured total of network bytes. Multiple
serving ranks open the adapter files; caches and repeated reads affect actual
storage/network traffic.

## Verification and remaining limits

- Both updates had finite, nonzero gradient norms: **4.4990901** and
  **4.5635926**. Every rank changed adapter tensors, and frozen parameters had
  no gradients.
- Between exports v1 and v2, **703 HF tensors changed**, covering **all 93
  layers**. Both versions were dynamically installed and used for generation.
- Every local packed expert buffer and scale buffer had CPU-backup existence
  checks and first/middle/last-value checks after restoring GPU storage. This
  is sampled buffer verification, not a full checksum of the base checkpoint.
- The controller waited for completion records from all 24 GPUs for memory
  transitions. A regression test covers the early PP HTTP-acknowledgement bug.
- Peak PyTorch allocated memory across ranks was **113.64 GiB**; sampled
  physical device usage peaked at **126.10 GiB**, including the other engine's
  remaining context and allocations.
- Native checkpoint files total **18.96 GB**, including optimizer/master-weight
  state. Slurm recorded the training step as `COMPLETED`, exit code `0:0`.
  Reloading that native checkpoint was not exercised.
- The four sample-level mean absolute training/rollout log-probability
  differences were **0.0704, 0.0782, 0.2042 and 0.0542 nats**. Numerical parity
  is not established. The tiny reward, short raw prompt, TCP collectives and
  reference expert GEMMs do not establish convergence or production throughput.

During the stalls, a sequential copy of the 6.13 GB adapter to each node took
about **29.3–29.6 seconds** and unblocked shared-filesystem page reads. The
optional `--rl-adapter-local-dir` now stages completed adapters before handing
their local path to serving. Its copy helper ran on all three nodes in
**3.4–3.9 seconds with warmed source caches**. The subsequent serving comparison
could not run after the node failure; **a full cycle with this staging flag
has not yet been measured**.

The existing [Miles K3 recipe](https://github.com/radixark/miles/pull/1825)
provides colocation and LoRA synchronization, including a frozen-base CPU
backup option. Its colocated transport shares CUDA IPC bucket handles through
Ray, avoiding this filesystem adapter-loading path. Connecting that driver to
the experimental MXFP4 trainer and its explicit TMS ownership remains separate
work. The serving compatibility patch adapts the upstream
[K3 LoRA serving changes](https://github.com/sgl-project/sglang/pull/37704)
to the previously validated Torch 2.11 container.
