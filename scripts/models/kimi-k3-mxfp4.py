from model_args_utils import load_sibling_model_args


def model_args(nlayers=None) -> str:
    base = load_sibling_model_args(__file__, "kimi-k3", nlayers=nlayers)
    return base.replace(
        "--spec miles_plugins.models.kimi_k3 get_kimi_k3_spec",
        "--spec miles_plugins.models.kimi_k3_mxfp4 get_kimi_k3_mxfp4_spec",
    )
