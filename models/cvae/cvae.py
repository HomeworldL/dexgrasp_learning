from __future__ import annotations

import torch
from torch import nn


class VAE(nn.Module):
    """条件 VAE 的最小实现。"""

    def __init__(
        self,
        encoder_layer_sizes: list[int],
        latent_size: int,
        decoder_layer_sizes: list[int],
        conditional: bool = True,
        condition_size: int = 0,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size)
        self.encoder = Encoder(
            layer_sizes=list(encoder_layer_sizes),
            latent_size=self.latent_size,
            conditional=conditional,
            condition_size=int(condition_size),
        )
        self.decoder = Decoder(
            layer_sizes=list(decoder_layer_sizes),
            latent_size=self.latent_size,
            conditional=conditional,
            condition_size=int(condition_size),
        )

    def forward(
        self,
        target: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, log_var = self.encoder(target, c)
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        latent = means + noise * std
        reconstruction = self.decoder(latent, c)
        return reconstruction, means, log_var, latent

    def inference(self, n: int, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            raise ValueError("Conditional VAE inference requires condition tensor c.")
        latent = torch.randn((int(n), self.latent_size), device=c.device, dtype=c.dtype)
        return self.decoder(latent, c)


class Encoder(nn.Module):
    """VAE encoder。"""

    def __init__(
        self,
        layer_sizes: list[int],
        latent_size: int,
        conditional: bool,
        condition_size: int,
    ) -> None:
        super().__init__()
        if conditional:
            layer_sizes = list(layer_sizes)
            layer_sizes[0] += int(condition_size)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
            layers.append(nn.ReLU())
        self.conditional = conditional
        self.mlp = nn.Sequential(*layers)
        self.linear_means = nn.Linear(int(layer_sizes[-1]), int(latent_size))
        self.linear_log_var = nn.Linear(int(layer_sizes[-1]), int(latent_size))

    def forward(
        self,
        target: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = target
        if self.conditional:
            if c is None:
                raise ValueError("Conditional VAE encoder requires condition tensor c.")
            features = torch.cat([target, c], dim=-1)
        hidden = self.mlp(features)
        return self.linear_means(hidden), self.linear_log_var(hidden)


class Decoder(nn.Module):
    """VAE decoder。"""

    def __init__(
        self,
        layer_sizes: list[int],
        latent_size: int,
        conditional: bool,
        condition_size: int,
    ) -> None:
        super().__init__()
        input_size = int(latent_size) + int(condition_size) if conditional else int(latent_size)
        layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(
            zip([input_size] + list(layer_sizes[:-1]), layer_sizes)
        ):
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
            if layer_index + 1 < len(layer_sizes):
                layers.append(nn.ReLU())
        self.conditional = conditional
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        features = latent
        if self.conditional:
            if c is None:
                raise ValueError("Conditional VAE decoder requires condition tensor c.")
            features = torch.cat([latent, c], dim=-1)
        return self.mlp(features)
