from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """创建标准正弦 timestep embedding。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _pick_group_count(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def normalize_1d(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_pick_group_count(num_channels),
        num_channels=num_channels,
        eps=1e-6,
        affine=True,
    )


class ResBlock(nn.Module):
    """与参考实现同风格的 1D 残差块。"""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)

        self.in_layers = nn.Sequential(
            normalize_1d(self.in_channels),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(emb_channels), self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalize_1d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
        )
        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_value, gate = self.proj(x).chunk(2, dim=-1)
        return x_value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        glu: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else int(dim_out)
        project_in: nn.Module
        if glu:
            project_in = GEGLU(dim, inner_dim)
        else:
            project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim_head * heads)
        context_dim = query_dim if context_dim is None else int(context_dim)
        self.scale = float(dim_head) ** -0.5
        self.heads = int(heads)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        similarity = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attention = similarity.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attention, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int | None = None,
        mult_ff: int = 2,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim=dim, mult=mult_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """对 1D token 序列做 self/cross attention。"""

    def __init__(
        self,
        in_channels: int,
        n_heads: int = 8,
        d_head: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int | None = None,
        mult_ff: int = 2,
    ) -> None:
        super().__init__()
        inner_dim = int(n_heads * d_head)
        self.norm = normalize_1d(int(in_channels))
        self.proj_in = nn.Conv1d(int(in_channels), inner_dim, kernel_size=1)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    mult_ff=mult_ff,
                )
                for _ in range(int(depth))
            ]
        )
        self.proj_out = nn.Conv1d(inner_dim, int(in_channels), kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h = self.proj_in(self.norm(x))
        h = rearrange(h, "b c n -> b n c")
        for block in self.transformer_blocks:
            h = block(h, context=context)
        h = rearrange(h, "b n c -> b c n")
        return self.proj_out(h) + residual
