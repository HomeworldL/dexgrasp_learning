from __future__ import annotations

from typing import Any

from models.cvae_sc import CVAESingleConditionModel


def list_algorithms() -> list[str]:
    return ["cvae", "diffusion", "flow"]


def build_model(config: dict[str, Any]) -> CVAESingleConditionModel:
    """按配置构建当前主线模型。"""
    algorithm = str(config.get("model", {}).get("algorithm", "cvae")).strip().lower()
    if algorithm != "cvae":
        raise NotImplementedError(
            f"model.algorithm={algorithm} is reserved for future work. "
            "The current mainline only implements cvae."
        )
    input_encoder_name = str(
        config.get("model", {}).get("input_encoder", {}).get("name", "pointnet")
    ).strip().lower()
    if input_encoder_name not in {"pointnet", "bps"}:
        raise NotImplementedError(
            f"model.input_encoder.name={input_encoder_name} is reserved for future work. "
            "The current mainline implements pointnet and bps."
        )
    return CVAESingleConditionModel(config)
