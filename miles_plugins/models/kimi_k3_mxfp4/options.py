"""Experiment options, independent of Megatron imports."""


def layer_indices(value, *, num_layers, option):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{option} must be a nonempty list of zero-based layer indices")
    if any(type(index) is not int or not 0 <= index < num_layers for index in value):
        raise ValueError(f"{option} entries must be integers in [0, {num_layers})")
    if len(set(value)) != len(value):
        raise ValueError(f"{option} contains duplicate layer indices")
    return frozenset(value)


def select_trainable_layers(model, selected):
    """Keep the full native adapter layout for export, freezing excluded layers."""
    for layer in model.decoder.layers:
        if layer.layer_number - 1 not in selected:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
