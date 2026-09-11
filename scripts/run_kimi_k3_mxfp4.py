"""Offline K3 LoRA experiment with a native MXFP4 expert base on 24 H200s.

Requires the native HF checkpoint, saved Miles rollout samples and a Ray cluster
for multi-node runs. No BF16 checkpoint conversion is performed.

Args:
    experiment_config: YAML selecting trainable and retained-weight layers.
    rollout_data: Saved Miles samples used for the trainer-only experiment.
    Other fields reuse the standard K3 launcher's model, paths and parallelism.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k3_mxfp4.py \
        --hf-checkpoint /models/Kimi-K3 --rollout-data /datasets/fixed-samples.pt
"""

import shlex
from dataclasses import dataclass

import typer
from scripts.run_kimi_k3 import ScriptArgs as _K3Args
from scripts.run_kimi_k3 import _train

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(_K3Args):
    model_name: str = "Kimi-K3"
    num_nodes: int = 3
    num_gpus_per_node: int | None = 8
    pipeline_parallel_size: int = 3
    tp_size_override: int | None = 8
    ep_size_override: int | None = 8
    lora_rank: int = 4
    lora_alpha: int = 8
    max_tokens_per_gpu: int | None = 128
    experiment_config: str = str(U.repo_base_dir / "examples/kimi_k3_mxfp4/two_layers.yaml")
    rollout_data: str | None = None

    def __post_init__(self):
        if self.model_name != "Kimi-K3" or self.train_mode != "lora":
            raise ValueError("This experiment keeps the full K3 architecture and requires LoRA")
        if self.ref_load is None:
            self.ref_load = self.hf_checkpoint or f"{self.model_dir}/{self.model_name}"
        super().__post_init__()
        if self.rollout_data is None:
            self.rollout_data = f"{self.data_dir}/fixed-samples.pt"

    @property
    def megatron_model_type(self) -> str:
        return "kimi-k3-mxfp4"


@U.dataclass_cli
def execute(args: ScriptArgs) -> None:
    args.extra_args += (
        f" --custom-config-path {shlex.quote(args.experiment_config)}"
        f" --load-debug-rollout-data {shlex.quote(args.rollout_data)}"
        " --no-offload-train --no-offload-rollout"
        " --train-memory-margin-bytes 0"
    )
    _train(args)


if __name__ == "__main__":
    typer.run(execute)
