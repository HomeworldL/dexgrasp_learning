from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    activation_name = str(name).strip().lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    raise ValueError(f"Unsupported BPS activation: {name}")


class BPSEncoder(nn.Module):
    """基于 bps_torch 的 Basis Point Set 点云编码器。"""

    def __init__(
        self,
        output_dim: int,
        basis_path: str | None,
        feature_types: Iterable[str],
        mlp_hidden_dims: Iterable[int],
        activation: str = "leaky_relu",
        bps_type: str = "random_uniform",
        n_bps_points: int = 4096,
        radius: float = 1.0,
        n_dims: int = 3,
    ) -> None:
        super().__init__()
        if importlib.util.find_spec("bps_torch") is None:
            raise ModuleNotFoundError(
                "bps_torch is required for model.input_encoder.name=bps. "
                "Install chamfer_distance and bps_torch first."
            )
        from bps_torch.bps import bps_torch

        feature_type_list = [str(name).strip().lower() for name in feature_types]
        if not feature_type_list:
            raise ValueError("BPS feature_types must contain at least one entry.")
        unsupported = sorted(set(feature_type_list) - {"dists", "deltas"})
        if unsupported:
            raise ValueError(f"Unsupported BPS feature types: {unsupported}")

        custom_basis = None
        resolved_basis_path: Path | None = None
        if basis_path is not None and str(basis_path).strip():
            resolved_basis_path = Path(basis_path).expanduser().resolve()
            if not resolved_basis_path.exists():
                raise FileNotFoundError(f"BPS basis file not found: {resolved_basis_path}")
            custom_basis = np.load(resolved_basis_path).astype(np.float32)
            if custom_basis.ndim != 2 or custom_basis.shape[1] != int(n_dims):
                raise ValueError(
                    "BPS basis must have shape [K, n_dims], got "
                    f"{custom_basis.shape} with n_dims={n_dims}."
                )
            n_bps_points = int(custom_basis.shape[0])

        self.feature_types = feature_type_list
        self.n_bps_points = int(n_bps_points)
        self.n_dims = int(n_dims)
        self.output_dim = int(output_dim)
        self.basis_path = None if resolved_basis_path is None else str(resolved_basis_path)
        self.bps = bps_torch(
            bps_type=str(bps_type),
            n_bps_points=self.n_bps_points,
            radius=float(radius),
            n_dims=self.n_dims,
            custom_basis=custom_basis,
        )

        input_dim = 0
        if "dists" in self.feature_types:
            input_dim += self.n_bps_points
        if "deltas" in self.feature_types:
            input_dim += self.n_bps_points * self.n_dims

        mlp_dims = [int(input_dim), *[int(dim) for dim in mlp_hidden_dims], self.output_dim]
        layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_index + 1 < len(mlp_dims) - 1:
                layers.append(_build_activation(activation))
        self.projection = nn.Sequential(*layers)

    def forward(self, point_cloud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把点云编码成固定长度全局特征。"""
        if point_cloud.ndim != 3:
            raise ValueError(
                f"BPSEncoder expects point_cloud with shape [B, N, C], got {point_cloud.shape}."
            )
        caller_device = point_cloud.device
        encode_device = caller_device
        # bps_torch 会把 CPU 输入自动迁到 cuda:0；这里显式接管设备，避免和调用方设备不一致。
        if encode_device.type == "cpu" and torch.cuda.is_available():
            encode_device = torch.device("cuda:0")
        encoded = self.bps.encode(
            point_cloud.to(encode_device),
            feature_type=self.feature_types,
        )
        feature_parts: list[torch.Tensor] = []
        if "dists" in self.feature_types:
            feature_parts.append(encoded["dists"].to(caller_device))
        if "deltas" in self.feature_types:
            deltas = encoded["deltas"].reshape(point_cloud.shape[0], -1).to(caller_device)
            feature_parts.append(deltas)
        bps_feature = torch.cat(feature_parts, dim=-1)
        global_feature = self.projection(bps_feature)
        return global_feature, bps_feature.unsqueeze(1)
