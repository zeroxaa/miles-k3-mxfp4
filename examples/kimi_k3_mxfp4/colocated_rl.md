# Full K3 MXFP4 LoRA: colocated RL integration check

This experiment alternates a real SGLang rollout engine and a persistent
Miles/Megatron training worker on the same three eight-H200 nodes. It keeps all
93 language-model layers and all 896 routed experts per MoE layer. The frozen
base is the native checkpoint at revision
`a590ce090cb049c93a33dfe8c208ec652aa20503`.

The controller is `tools/kimi_k3_mxfp4_rl_cycle.py`; the distributed worker is
`tools/kimi_k3_mxfp4_rl_worker.py`. This is an HTTP/filesystem integration
harness. It uses Miles's model, optimizer wrapper, clipped policy loss, gradient
reductions and native HF LoRA exporter. It does not invoke the Ray `train.py`
driver or the newer CUDA IPC adapter update protocol.

## What one cycle does

1. SGLang generates two responses and their token log probabilities. The reward
   is the first standalone generated digit divided by nine, or zero if absent.
   A group with identical rewards is resampled, up to twelve groups; every
   attempt is retained in the run directory.
2. Flush inference state and offload inference weights. Wait for memory-saver
   completion on **all 24 GPUs**, then wake the trainer.
3. Train on those actual generated tokens, using response-only clipped policy
   loss and group-normalized rewards. Each response runs as a separate
   microbatch; accumulated gradients receive one SGD update. KL coefficient is
   zero; there is no reference-model pass or learned reward model.
4. Gather the sharded LoRA tensors into three HF safetensors files, one per
   pipeline stage. Record tensor hashes and sizes. Offload the trainer.
5. Restore inference weights, unload the previous adapter, register the new
   adapter directory and restore KV storage. Generate with the new adapter.

The next cycle samples with the preceding cycle's adapter. The smoke reward,
short untemplated prompt and tiny batch test the handoff, not model quality.

## Runtime and source compatibility

The training environment used PyTorch `2.11.0+cu129`, Transformer Engine `2.17`,
FLA `0.5.2`, Megatron commit
`1750156a4bfa9d4464fb5da86c58b4caa381297c`, and the pinned Miles environment.
Training imports the native recipe's SGLang Python utilities at
`9574eb1b1daca8d795ed8cf824c4b7094ea4191e`.

Serving used the previously working SGLang `0.5.16` container, with
`sglang-kernel 0.4.5+cu129` and `flashinfer-python 0.6.15.post1`. Its source needs
the accompanying `compat/sglang-0.5.16-k3-lora.patch`. Check the original file
hashes in the adjacent JSON before applying it. The patch adapts the native
[K3 LoRA serving changes](https://github.com/sgl-project/sglang/pull/37704)
to that container; it is not a claim that arbitrary SGLang 0.5.16 builds match.
The newer pinned serving source requires a different Torch/kernel ABI and
cannot simply replace this container's Python package.

Run both CUDA processes inside their matching containers with GPU passthrough
(`apptainer exec --nv` in this experiment). Stage sources, checkpoint shards
and compiler caches on node-local storage. In particular, give each node its
own `TVM_FFI_CACHE_DIR`; a shared JIT cache caused a stale-lock stall during
bring-up. Do not delete locks belonging to other workloads.

## Serving configuration

Launch one SGLang process per node, setting the node rank to 0, 1 or 2. Supply
the checkpoint, rendezvous address and an identical shared run-directory path
on all nodes. `K3_RL_TRACE_DIR` must point at `<run-dir>/tms` in every serving
process; the controller uses these completion records for its GPU barrier.

```text
--trust-remote-code --model-path <node-local-checkpoint>
--tp-size 8 --pp-size 3 --ep-size 1 --nnodes 3 --node-rank <node-rank>
--dist-init-addr <first-node-address>:20801
--moe-runner-backend marlin --decode-attention-backend flashmla
--mem-fraction-static 0.72 --context-length 512
--max-running-requests 1 --max-total-tokens 1024
--chunked-prefill-size 256 --skip-server-warmup --disable-cuda-graph
--reasoning-parser kimi_k3 --host 0.0.0.0 --port 30801
--enable-lora --max-lora-rank 4 --max-loras-per-batch 1 --lora-backend triton
--lora-target-modules o_proj q_a_proj kv_a_proj_with_mqa gate_proj up_proj down_proj
--lora-strict-loading --experts-shared-outer-loras
--enable-memory-saver --enable-weights-cpu-backup
```

Pass `--experts-shared-outer-loras` explicitly even when no adapter is loaded
at startup. The adapter's routed-expert factors use shared outer matrices.
This compatibility path keeps ordinary CPU adapter copies and does not support
`--lora-no-cpu-backup`.

The measured run used `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=eth0`,
`GLOO_SOCKET_IFNAME=eth0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`,
`SGLANG_K3_AR_FUSION=0`, and disabled TP memory-imbalance checking. These are
cluster-specific choices, not general throughput recommendations.

## Training configuration and startup order

Start the serving engine first. On each of the same nodes, have the launcher
wait for `<run-dir>/initialize_trainer.json` **before starting torchrun**.
The controller writes that file only after inference has released all GPUs.
The worker itself expects its launcher to enforce this initial ordering.
Use a fresh run directory for each experiment; old command/completion files
are not a crash-recovery protocol.

The worker uses the model arguments returned by
`load_model_args("kimi-k3-mxfp4")` and the following additional arguments:

```text
--actor-num-nodes 3 --actor-num-gpus-per-node 8
--tensor-model-parallel-size 8 --pipeline-model-parallel-size 3
--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --sequence-parallel
--hf-checkpoint <node-local-checkpoint> --ref-load <node-local-checkpoint>
--megatron-to-hf-mode raw --model-name kimi_k3
--lora-rank 4 --lora-alpha 8 --lora-dropout 0 --experts-shared-outer-loras
--target-modules <scripts.run_kimi_k3._DEFAULT_TARGET_MODULES as one argument>
--no-gradient-accumulation-fusion
--custom-config-path examples/kimi_k3_mxfp4/all_layers_retain_all.yaml
--seq-length 32 --micro-batch-size 1 --global-batch-size 1
--num-rollout 2 --rollout-batch-size 1 --n-samples-per-prompt 1
--optimizer sgd --sgd-momentum 0 --lr 0.0001 --lr-decay-style constant
--weight-decay 0 --save-interval 2 --distributed-timeout-minutes 30
--rl-cycle-dir <run-dir> --rl-cycle-count 2 --save <run-dir>/adapter
--rl-adapter-local-dir <absolute-node-local-directory-for-this-run>
--no-offload-train --no-offload-rollout --train-memory-margin-bytes 0
```

Torchrun uses three nodes, eight processes per node and a separate rendezvous
port, e.g. 20802. Set `TMS_INIT_ENABLE=0` and preload the binary returned by
`torch_memory_saver.utils.get_binary_path_from_package(
"torch_memory_saver_hook_mode_preload")` before torchrun. The worker explicitly
owns the TMS region for persistent model, packed buffers and optimizer state.
The native `--offload-train` route remains rejected by the MXFP4 spec; these
flags avoid two independent owners of the same allocations.

The command-line rollout batch fields satisfy shared Miles initialization.
The harness itself constructs a two-response group, accumulates both
microbatches and divides their losses by two; it does not run the native
rollout manager. All 93 layers' native LoRA modules train. Expanded BF16 weights
of active experts are retained through backward in all 92 MoE layers, then
freed. Activation checkpointing is disabled.

With `--rl-adapter-local-dir`, each node's TP leader sequentially copies the
completed adapter to local storage before publishing the completion file.
SGLang receives the common absolute local path. Use a fresh destination for
each run. This avoids the shared-filesystem mmap stalls observed during the
original two cycles; those original timings used the shared path and required
diagnostic sequential prefetches. Local staging is timed as `adapter_stage_N`.
The copy helper was exercised on all three nodes, but the subsequent serving
comparison was interrupted by a cluster `NodeDown` event. A complete RL cycle
with this optional staging flag has not yet been measured.

Once both launchers are waiting/running, start the controller:

```bash
python tools/kimi_k3_mxfp4_rl_cycle.py \
  --server http://<first-node-address>:30801 --run-dir <run-dir> --cycles 2
```

The optional `--resume-inference-on-start` only covers the specific state where
all 24 trainer ready files exist, both models are offloaded, and the controller
must restore inference first. It is not general resume support.

## Evidence and timing interpretation

`controller-report.json` records the end-to-end cycles and HTTP phases.
`events-rank*.jsonl` splits trainer restore, RL update, adapter export and
trainer offload. TMS records contain direct per-GPU transfer timings and
physical device memory before/after; virtual PyTorch allocation counts alone
do not show memory released by TMS. The worker verifies CPU backup and restored
GPU samples from every local packed expert buffer and scale buffer.

Every rank must record nonzero finite gradient norm and changed adapter
tensors. Three adapter manifests record all exported tensor hashes. Retain
the actual rollout attempts, training commands and post-update generations.
After controller completion, also wait for torchrun to exit successfully and
check the native adapter/optimizer checkpoint; controller `PASS` precedes that
final checkpoint save.

Adapter HTTP load acknowledgement is not a pure GPU-copy timer. CPU file
loading/normalization, downstream pipeline handling and lazy GPU installation
are split across adapter load, KV restore and the first generation. Report
these together when describing adapter handoff. The same applies to pipeline
memory RPCs: the head stage can acknowledge before later stages finish.

Only LoRA tensors travel from the training representation to the serving
representation. Nevertheless, each engine's unchanged base still moves between
host RAM and GPU memory during colocation. The experiment keeps separate CPU
backups because training EP8 and serving EP1/Marlin have different layouts.
The native [Miles K3 recipe](https://github.com/radixark/miles/pull/1825) provides
colocation and adapter synchronization; its CUDA IPC streaming transport is a
separate integration step from the filesystem path measured here.
