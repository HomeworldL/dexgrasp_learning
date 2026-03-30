from __future__ import annotations

import torch
from torch import nn

from models.dexdiffuser.utils import ResBlock, SpatialTransformer, timestep_embedding


class DexDiffuserUNet(nn.Module):
    """DexDiffuser 风格的 1D 条件去噪网络。"""

    def __init__(
        self,
        d_x: int,
        d_model: int,
        context_dim: int,
        time_embed_mult: int = 2,
        nblocks: int = 4,
        resblock_dropout: float = 0.0,
        transformer_num_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_dropout: float = 0.1,
        transformer_depth: int = 1,
        transformer_mult_ff: int = 2,
        use_position_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.d_x = int(d_x)
        self.d_model = int(d_model)
        self.context_dim = int(context_dim)
        self.nblocks = int(nblocks)
        self.use_position_embedding = bool(use_position_embedding)

        time_embed_dim = self.d_model * int(time_embed_mult)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.in_layers = nn.Sequential(nn.Conv1d(self.d_x, self.d_model, kernel_size=1))

        blocks: list[nn.Module] = []
        for _ in range(self.nblocks):
            blocks.append(
                ResBlock(
                    in_channels=self.d_model,
                    emb_channels=time_embed_dim,
                    dropout=float(resblock_dropout),
                    out_channels=self.d_model,
                )
            )
            blocks.append(
                SpatialTransformer(
                    in_channels=self.d_model,
                    n_heads=int(transformer_num_heads),
                    d_head=int(transformer_dim_head),
                    depth=int(transformer_depth),
                    dropout=float(transformer_dropout),
                    context_dim=self.context_dim,
                    mult_ff=int(transformer_mult_ff),
                )
            )
        self.layers = nn.ModuleList(blocks)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, kernel_size=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        input_ndim = x_t.ndim
        if input_ndim != 2:
            raise ValueError(
                f"DexDiffuserUNet expects x_t with shape [B, Dx], got {tuple(x_t.shape)}."
            )
        h = x_t.unsqueeze(1)
        t_emb = self.time_embed(timestep_embedding(timesteps, self.d_model))
        h = h.transpose(1, 2).contiguous()
        h = self.in_layers(h)

        if self.use_position_embedding:
            token_count = h.shape[-1]
            pos = torch.arange(token_count, device=h.device, dtype=timesteps.dtype)
            pos_embedding = timestep_embedding(pos, self.d_model).transpose(0, 1).unsqueeze(0)
            h = h + pos_embedding

        for block_index in range(self.nblocks):
            h = self.layers[block_index * 2](h, t_emb)
            h = self.layers[block_index * 2 + 1](h, context=context)

        h = self.out_layers(h)
        return h.transpose(1, 2).contiguous().squeeze(1)
