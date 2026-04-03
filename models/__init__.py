from __future__ import annotations

from models.cvae_model import CVAEModel
from models.dp_model import DPModel
from models.dexdiffuser_model import DexDiffuserModel
from models.udgm_model import UDGMModel
from models.base_model import BaseModel, get_model_required, normalize_algorithm_name

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "cvae": CVAEModel,
    "dp": DPModel,
    "dexdiffuser": DexDiffuserModel,
    "udgm": UDGMModel,
}


def list_algorithms() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_model(config: dict[str, object]) -> BaseModel:
    """按配置构建当前主线模型。"""
    algorithm = normalize_algorithm_name(get_model_required(config, "model.algorithm"))
    model_cls = MODEL_REGISTRY.get(algorithm)
    if model_cls is None:
        raise NotImplementedError(
            f"model.algorithm={algorithm} is reserved for future work. "
            "The current mainline implements " + ", ".join(list_algorithms()) + "."
        )
    return model_cls(config)
