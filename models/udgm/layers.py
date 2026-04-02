from __future__ import annotations

import torch
from torch import nn


class ActNorm1d(nn.Module):
    """Glow-style ActNorm for vector features."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.log_scale = nn.Parameter(torch.zeros(self.dim))
        self.register_buffer("initialized", torch.tensor(False), persistent=True)

    def _maybe_initialize(self, x: torch.Tensor) -> None:
        if bool(self.initialized.item()):
            return
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0).clamp_min(self.eps)
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(-torch.log(std))
            self.initialized.fill_(True)

    def forward_transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_initialize(x)
        y = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse_transform(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = y * torch.exp(-self.log_scale) - self.bias
        log_det = (-self.log_scale.sum()).expand(y.shape[0])
        return x, log_det


class InvertibleLinear(nn.Module):
    """A small dense invertible linear transform with orthogonal initialization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        weight = torch.empty(self.dim, self.dim)
        nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)

    def _slogdet(self) -> torch.Tensor:
        sign, logabsdet = torch.linalg.slogdet(self.weight)
        if torch.any(sign == 0):
            raise RuntimeError("InvertibleLinear weight became singular.")
        return logabsdet

    def forward_transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x @ self.weight.t()
        log_det = self._slogdet().expand(x.shape[0])
        return y, log_det

    def inverse_transform(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_inv = torch.linalg.inv(self.weight)
        x = y @ weight_inv.t()
        log_det = (-self._slogdet()).expand(y.shape[0])
        return x, log_det
