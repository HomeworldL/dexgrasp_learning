from __future__ import annotations

import math

import torch
from torch import nn

from models.udgm.coupling import ConditionedAffineCoupling
from models.udgm.layers import ActNorm1d, InvertibleLinear


class UDGMFlowStep(nn.Module):
    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden_dim: int,
        num_blocks_per_layer: int,
        mask: torch.Tensor,
        *,
        scale_clamp: float,
        activation: str,
        use_actnorm: bool,
        use_invertible_linear: bool,
        conditioner_type: str,
        residual_num_blocks: int,
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm1d(target_dim) if use_actnorm else None
        self.linear = InvertibleLinear(target_dim) if use_invertible_linear else None
        self.coupling = ConditionedAffineCoupling(
            target_dim=target_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_blocks_per_layer=num_blocks_per_layer,
            mask=mask,
            scale_clamp=scale_clamp,
            activation=activation,
            conditioner_type=conditioner_type,
            residual_num_blocks=residual_num_blocks,
        )

    def forward_transform(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        if self.actnorm is not None:
            x, delta = self.actnorm.forward_transform(x)
            log_det = log_det + delta
        if self.linear is not None:
            x, delta = self.linear.forward_transform(x)
            log_det = log_det + delta
        x, delta = self.coupling.forward_transform(x, context)
        log_det = log_det + delta
        return x, log_det

    def inverse_transform(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, log_det = self.coupling.inverse_transform(x, context)
        if self.linear is not None:
            z, delta = self.linear.inverse_transform(z)
            log_det = log_det + delta
        if self.actnorm is not None:
            z, delta = self.actnorm.inverse_transform(z)
            log_det = log_det + delta
        return z, log_det


class UDGMFlow(nn.Module):
    """面向统一 32D 抓取目标的条件 normalizing flow。"""

    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_blocks_per_layer: int,
        scale_clamp: float = 2.0,
        activation: str = "leaky_relu",
        use_actnorm: bool = True,
        use_invertible_linear: bool = True,
        conditioner_type: str = "residual",
        residual_num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_blocks_per_layer = int(num_blocks_per_layer)
        if self.target_dim <= 1:
            raise ValueError(f"target_dim must be > 1, got {target_dim}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        layers: list[UDGMFlowStep] = []
        for layer_index in range(self.num_layers):
            parity = layer_index % 2
            mask = ((torch.arange(self.target_dim) + parity) % 2 == 0).to(torch.float32)
            layers.append(
                UDGMFlowStep(
                    target_dim=self.target_dim,
                    condition_dim=self.condition_dim,
                    hidden_dim=self.hidden_dim,
                    num_blocks_per_layer=self.num_blocks_per_layer,
                    mask=mask,
                    scale_clamp=scale_clamp,
                    activation=activation,
                    use_actnorm=use_actnorm,
                    use_invertible_linear=use_invertible_linear,
                    conditioner_type=conditioner_type,
                    residual_num_blocks=residual_num_blocks,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _validate_inputs(self, x: torch.Tensor, context: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [B, D], got {tuple(x.shape)}.")
        if context.ndim != 2:
            raise ValueError(
                f"Expected context with shape [B, C], got {tuple(context.shape)}."
            )
        if x.shape[0] != context.shape[0]:
            raise ValueError(
                f"Batch mismatch between x {tuple(x.shape)} and context {tuple(context.shape)}."
            )
        if x.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected target dim {self.target_dim}, got {x.shape[-1]}."
            )
        if context.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {context.shape[-1]}."
            )

    def _base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_norm = math.log(2.0 * math.pi)
        return -0.5 * (z.pow(2) + log_norm).sum(dim=-1)

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(x, context)
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, delta = layer.inverse_transform(z, context)
            log_det = log_det + delta
        return self._base_log_prob(z) + log_det

    @torch.no_grad()
    def sample_and_log_prob(
        self,
        num_samples: int,
        context: torch.Tensor,
        *,
        sort_by_log_prob: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if context.ndim != 2:
            raise ValueError(
                f"Expected context with shape [B, C], got {tuple(context.shape)}."
            )
        if context.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition dim {self.condition_dim}, got {context.shape[-1]}."
            )

        batch_size = context.shape[0]
        repeated_context = (
            context[:, None, :]
            .expand(batch_size, num_samples, self.condition_dim)
            .reshape(batch_size * num_samples, self.condition_dim)
        )
        z = torch.randn(
            batch_size * num_samples,
            self.target_dim,
            device=context.device,
            dtype=context.dtype,
        )
        x = z
        log_det = torch.zeros(batch_size * num_samples, device=context.device, dtype=context.dtype)
        for layer in self.layers:
            x, delta = layer.forward_transform(x, repeated_context)
            log_det = log_det + delta
        log_prob = self._base_log_prob(z) - log_det

        x = x.reshape(batch_size, num_samples, self.target_dim)
        log_prob = log_prob.reshape(batch_size, num_samples)
        if sort_by_log_prob:
            order = torch.argsort(log_prob, dim=1, descending=True)
            gather_index = order.unsqueeze(-1).expand(-1, -1, self.target_dim)
            x = torch.gather(x, dim=1, index=gather_index)
            log_prob = torch.gather(log_prob, dim=1, index=order)
        return x, log_prob
