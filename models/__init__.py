from __future__ import annotations

from typing import Any

from models.base_sc import BaseSCModel
from models.cvae_sc import CVAESingleConditionModel
from models.dp_rt_sc import DPRTScModel
from models.dp_sc import DPScModel
from models.dexdiffuser_rt_sc import DexDiffuserRTSCModel
from models.dexdiffuser_sc import DexDiffuserSCModel
from models.udgm_rt_sc import UDGMRTScModel
from models.udgm_sc import UDGMScModel


def list_algorithms() -> list[str]:
    return ["cvae", "dexdiffuser", "dexdiffuser_rt", "udgm", "udgm_rt", "dp", "dp_rt"]


def build_model(config: dict[str, Any]) -> BaseSCModel:
    """按配置构建当前主线模型。"""
    algorithm = str(config.get("model", {}).get("algorithm", "cvae")).strip().lower()
    input_encoder_name = str(
        config.get("model", {}).get("input_encoder", {}).get("name", "pointnet")
    ).strip().lower()
    if input_encoder_name not in {"pointnet", "bps"}:
        raise NotImplementedError(
            f"model.input_encoder.name={input_encoder_name} is reserved for future work. "
            "The current mainline implements pointnet and bps."
        )
    if algorithm == "cvae":
        return CVAESingleConditionModel(config)
    if algorithm == "dexdiffuser":
        return DexDiffuserSCModel(config)
    if algorithm == "dexdiffuser_rt":
        return DexDiffuserRTSCModel(config)
    if algorithm == "udgm":
        return UDGMScModel(config)
    if algorithm == "udgm_rt":
        return UDGMRTScModel(config)
    if algorithm == "dp":
        return DPScModel(config)
    if algorithm == "dp_rt":
        return DPRTScModel(config)
    raise NotImplementedError(
        f"model.algorithm={algorithm} is reserved for future work. "
        "The current mainline implements cvae, dexdiffuser, dexdiffuser_rt, udgm, udgm_rt, dp, and dp_rt."
    )
