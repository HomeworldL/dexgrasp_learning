from __future__ import annotations

from models.base_model import BaseModel
from models.registry import (
    ALGORITHM_REGISTRY,
    SUPPORTED_INPUT_ENCODERS,
    normalize_algorithm_name,
    normalize_input_encoder_name,
)


def list_algorithms() -> list[str]:
    return list(ALGORITHM_REGISTRY.keys())


def build_model(config: dict[str, object]) -> BaseModel:
    """按配置构建当前主线模型。"""
    model_config = dict(config.get("model", {}))
    algorithm = normalize_algorithm_name(model_config.get("algorithm", "cvae"))
    input_encoder_name = normalize_input_encoder_name(
        dict(model_config.get("input_encoder", {})).get("name", "pointnet")
    )
    if input_encoder_name not in SUPPORTED_INPUT_ENCODERS:
        raise NotImplementedError(
            f"model.input_encoder.name={input_encoder_name} is reserved for future work. "
            "The current mainline implements "
            f"{', '.join(sorted(SUPPORTED_INPUT_ENCODERS))}."
        )
    spec = ALGORITHM_REGISTRY.get(algorithm)
    if spec is not None:
        return spec.model_cls(config)
    raise NotImplementedError(
        f"model.algorithm={algorithm} is reserved for future work. "
        "The current mainline implements " + ", ".join(list_algorithms()) + "."
    )
