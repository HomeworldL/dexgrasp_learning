from __future__ import annotations

import torch
from torch import nn

from models.udgm.utils import build_mlp, build_residual_mlp


class ConditionedAffineCoupling(nn.Module):
    """简单条件 affine coupling block。"""

    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden_dim: int,
        num_blocks_per_layer: int,
        mask: torch.Tensor,
        scale_clamp: float = 2.0,
        activation: str = "leaky_relu",
        conditioner_type: str = "residual",
        residual_num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.scale_clamp = float(scale_clamp)
        if self.scale_clamp <= 0.0:
            raise ValueError(f"scale_clamp must be positive, got {scale_clamp}.")

        if mask.ndim != 1 or mask.shape[0] != self.target_dim:
            raise ValueError(
                f"mask must have shape [{self.target_dim}], got {tuple(mask.shape)}."
            )
        hidden_dims = [int(hidden_dim)] * int(num_blocks_per_layer)
        self.register_buffer("mask", mask.reshape(1, self.target_dim), persistent=False)
        self.register_buffer("inv_mask", (1.0 - mask).reshape(1, self.target_dim), persistent=False)
        conditioner_kind = str(conditioner_type).strip().lower()
        if conditioner_kind == "mlp":
            self.conditioner = build_mlp(
                input_dim=self.target_dim + self.condition_dim,
                hidden_dims=hidden_dims,
                output_dim=self.target_dim * 2,
                activation=activation,
                zero_init_last=True,
            )
        elif conditioner_kind == "residual":
            self.conditioner = build_residual_mlp(
                input_dim=self.target_dim + self.condition_dim,
                output_dim=self.target_dim * 2,
                hidden_features=int(hidden_dim),
                num_blocks=int(residual_num_blocks),
            )
            last_linear = self.conditioner.final_layer
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        else:
            raise ValueError(
                "Unsupported conditioner_type for ConditionedAffineCoupling: "
                f"{conditioner_type}."
            )

    def _condition(self, x_masked: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([x_masked, context], dim=-1)
        log_scale, shift = self.conditioner(features).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.scale_clamp
        log_scale = log_scale * self.inv_mask
        shift = shift * self.inv_mask
        return log_scale, shift

    def forward_transform(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_masked = z * self.mask
        log_scale, shift = self._condition(z_masked, context)
        x = z_masked + self.inv_mask * (z * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return x, log_det

    def inverse_transform(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_scale, shift = self._condition(x_masked, context)
        z = x_masked + self.inv_mask * ((x - shift) * torch.exp(-log_scale))
        log_det = -log_scale.sum(dim=-1)
        return z, log_det
