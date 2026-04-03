from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from nflows.nn.nets.resnet import ResidualNet
from torch import nn


def _build_activation_module(name: str) -> nn.Module:
    activation_name = str(name).strip().lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported BasicMLP activation: {name}.")


def _build_activation_fn(name: str):
    activation_name = str(name).strip().lower()
    if activation_name == "relu":
        return F.relu
    if activation_name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    if activation_name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported BasicMLP activation: {name}.")


class BasicMLP(nn.Module):
    """Shared staged-task MLP for regression heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Iterable[int] | None = None,
        activation: str = "leaky_relu",
        network_type: str = "residual",
        residual_num_blocks: int = 2,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.mask = mask
        hidden_dims_list = [int(dim) for dim in (hidden_dims or [256, 256])]
        if not hidden_dims_list:
            raise ValueError("BasicMLP hidden_dims must contain at least one entry.")

        network_kind = str(network_type).strip().lower()
        if network_kind == "residual":
            num_blocks = int(residual_num_blocks)
            if num_blocks <= 0:
                raise ValueError(
                    "BasicMLP residual_num_blocks must be positive, "
                    f"got {residual_num_blocks}."
                )
            if len(hidden_dims_list) != num_blocks:
                raise ValueError(
                    "BasicMLP residual config requires len(hidden_dims) == residual_num_blocks, "
                    f"got hidden_dims={hidden_dims_list} and residual_num_blocks={num_blocks}."
                )
            hidden_features = int(hidden_dims_list[0])
            if any(int(dim) != hidden_features for dim in hidden_dims_list):
                raise ValueError(
                    "BasicMLP residual config requires all hidden_dims to be identical, "
                    f"got {hidden_dims_list}."
                )
            self.net: nn.Module = ResidualNet(
                int(input_dim),
                int(output_dim),
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                activation=_build_activation_fn(activation),
                dropout_probability=0.0,
                use_batch_norm=False,
            )
            return

        if network_kind not in {"plain", "mlp"}:
            raise ValueError(
                "Unsupported BasicMLP network_type: "
                f"{network_type}. Expected residual or plain."
            )

        dims = [int(input_dim), *hidden_dims_list, int(output_dim)]
        layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_index + 1 < len(dims) - 1:
                layers.append(_build_activation_module(activation))
        self.net = nn.Sequential(*layers)

    def apply_mask(self) -> None:
        """Compatibility hook for legacy call sites."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
