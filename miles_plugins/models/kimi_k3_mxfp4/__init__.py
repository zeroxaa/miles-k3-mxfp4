"""Experimental frozen MXFP4 routed experts for the existing Kimi K3 model."""


def get_kimi_k3_mxfp4_spec(*args, **kwargs):
    # Keep model-definition and CPU test discovery independent of Megatron/CUDA.
    from miles_plugins.models.kimi_k3_mxfp4.spec import get_kimi_k3_mxfp4_spec as get_spec

    return get_spec(*args, **kwargs)
