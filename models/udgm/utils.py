from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def build_activation(name: str) -> nn.Module:
    activation_name = str(name).strip().lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if activation_name == "mish":
        return nn.Mish()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str = "leaky_relu",
    *,
    zero_init_last: bool = False,
) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(output_dim)]
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        linear = nn.Linear(in_dim, out_dim)
        is_last = index == len(dims) - 2
        if is_last and zero_init_last:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not is_last:
            layers.append(build_activation(activation))
    return nn.Sequential(*layers)
