from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _sample_token_indices(
    num_available: int,
    num_target: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if num_target <= 0:
        raise ValueError(f"num_condition_tokens must be positive, got {num_target}.")
    if num_available <= num_target:
        return torch.arange(num_available, device=device, dtype=torch.long)
    sampling_mode = str(mode).strip().lower()
    if sampling_mode == "random":
        return torch.randperm(num_available, device=device)[:num_target]
    if sampling_mode == "uniform_stride":
        return torch.linspace(
            0,
            num_available - 1,
            steps=num_target,
            device=device,
        ).round().to(torch.long)
    raise ValueError(
        "Unsupported token_sampling="
        f"'{mode}'. Expected random or uniform_stride."
    )


class BPSConditionTokenizer(nn.Module):
    """为 DexDiffuser 条件分支单独构造 BPS token。"""

    def __init__(
        self,
        basis_path: str | None,
        feature_types: list[str],
        bps_type: str = "random_uniform",
        n_bps_points: int = 4096,
        radius: float = 1.0,
        n_dims: int = 3,
    ) -> None:
        super().__init__()
        if importlib.util.find_spec("bps_torch") is None:
            raise ModuleNotFoundError(
                "bps_torch is required for DexDiffuser BPS conditioning."
            )
        from bps_torch.bps import bps_torch

        feature_type_list = [str(name).strip().lower() for name in feature_types]
        if not feature_type_list:
            raise ValueError("DexDiffuser BPS tokenizer requires at least one feature type.")
        unsupported = sorted(set(feature_type_list) - {"dists", "deltas"})
        if unsupported:
            raise ValueError(f"Unsupported DexDiffuser BPS feature types: {unsupported}")
        self.feature_types = feature_type_list

        custom_basis = None
        if basis_path is not None and str(basis_path).strip():
            resolved_basis_path = Path(basis_path).expanduser().resolve()
            if not resolved_basis_path.exists():
                raise FileNotFoundError(f"BPS basis file not found: {resolved_basis_path}")
            custom_basis = np.load(resolved_basis_path).astype(np.float32)
            n_bps_points = int(custom_basis.shape[0])
        self.bps = bps_torch(
            bps_type=str(bps_type),
            n_bps_points=int(n_bps_points),
            radius=float(radius),
            n_dims=int(n_dims),
            custom_basis=custom_basis,
        )

    @property
    def token_feature_dim(self) -> int:
        dim = 0
        if "dists" in self.feature_types:
            dim += 1
        if "deltas" in self.feature_types:
            dim += 3
        return dim

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        caller_device = point_cloud.device
        encode_device = caller_device
        if encode_device.type == "cpu" and torch.cuda.is_available():
            encode_device = torch.device("cuda:0")
        encoded = self.bps.encode(
            point_cloud.to(encode_device),
            feature_type=self.feature_types,
        )
        token_parts: list[torch.Tensor] = []
        if "dists" in self.feature_types:
            token_parts.append(encoded["dists"].unsqueeze(-1).to(caller_device))
        if "deltas" in self.feature_types:
            token_parts.append(encoded["deltas"].to(caller_device))
        return torch.cat(token_parts, dim=-1)


class DexDiffuserConditionAdapter(nn.Module):
    """把 PointNet/BPS 条件特征统一适配为 context tokens。"""

    def __init__(
        self,
        point_feat_dim: int,
        input_encoder_name: str,
        input_encoder_config: dict[str, Any],
        condition_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.input_encoder_name = str(input_encoder_name).strip().lower()
        self.context_dim = int(condition_config.get("context_dim", 512))
        self.append_global_token = bool(condition_config.get("append_global_token", True))
        self.token_sampling = str(condition_config.get("token_sampling", "random"))
        self.global_projector = nn.Linear(int(point_feat_dim), self.context_dim)

        pointnet_config = dict(condition_config.get("pointnet", {}))
        pointnet_local_dim = int(
            list(input_encoder_config.get("local_conv_hidden_dims", [64, 128, 256]))[-1]
        )
        self.pointnet_num_tokens = int(pointnet_config.get("num_condition_tokens", 128))
        self.pointnet_projector = nn.Linear(pointnet_local_dim, self.context_dim)

        bps_config = dict(condition_config.get("bps", {}))
        feature_types = list(
            bps_config.get("feature_types", input_encoder_config.get("feature_types", ["dists"]))
        )
        self.bps_num_tokens = int(bps_config.get("num_condition_tokens", 128))
        bps_feature_dim = 0
        if "dists" in feature_types:
            bps_feature_dim += 1
        if "deltas" in feature_types:
            bps_feature_dim += 3
        self.bps_projector = nn.Linear(bps_feature_dim, self.context_dim)

    def from_pointnet(
        self,
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        local_tokens = local_feature.transpose(1, 2).contiguous()
        local_tokens = self._sample_tokens(
            local_tokens,
            num_target=self.pointnet_num_tokens,
        )
        local_tokens = self.pointnet_projector(local_tokens)
        return self._append_global_token(global_feature, local_tokens)

    def from_bps(
        self,
        global_feature: torch.Tensor,
        bps_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sampled_tokens = self._sample_tokens(
            bps_tokens,
            num_target=self.bps_num_tokens,
        )
        sampled_tokens = self.bps_projector(sampled_tokens)
        return self._append_global_token(global_feature, sampled_tokens)

    def _append_global_token(
        self,
        global_feature: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not self.append_global_token:
            return local_tokens
        global_token = self.global_projector(global_feature).unsqueeze(1)
        return torch.cat([global_token, local_tokens], dim=1)

    def _sample_tokens(self, tokens: torch.Tensor, num_target: int) -> torch.Tensor:
        batch_size, num_available, _ = tokens.shape
        indices = _sample_token_indices(
            num_available=num_available,
            num_target=num_target,
            mode=self.token_sampling,
            device=tokens.device,
        )
        sampled = tokens.index_select(dim=1, index=indices)
        if sampled.shape[1] == num_target:
            return sampled
        pad_count = num_target - sampled.shape[1]
        pad_indices = torch.randint(
            0,
            sampled.shape[1],
            size=(pad_count,),
            device=tokens.device,
        )
        padded = torch.cat([sampled, sampled.index_select(dim=1, index=pad_indices)], dim=1)
        return padded.reshape(batch_size, num_target, tokens.shape[-1])
