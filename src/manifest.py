from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestItem:
    """统一后的 object-scale 条目。"""

    object_scale_key: str
    object_name: str
    mjcf_path: str
    grasp_h5_path: str
    grasp_h5_fail_path: str | None
    grasp_fail_npy_path: str | None
    partial_pc_path: tuple[str, ...]
    partial_pc_cam_path: tuple[str, ...]
    cam_ex_path: tuple[str, ...]
    global_pc_path: str | None


def resolve_manifest_path(manifest_path: str, split: str | None = None) -> Path:
    """根据 split 解析 train/test manifest 路径。"""
    path = Path(manifest_path).expanduser().resolve()
    if split is None:
        return path
    split_name = "test.json" if split in {"eval", "test"} else "train.json"
    if path.name in {"train.json", "test.json"}:
        return path.with_name(split_name)
    return path


def load_manifest(
    manifest_path: str,
    split: str | None = None,
) -> tuple[Path, list[ManifestItem]]:
    """读取 manifest，并归一化成内部使用的条目结构。"""
    manifest_file = resolve_manifest_path(manifest_path, split=split)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {manifest_file}")
    items = [_normalize_manifest_item(item) for item in payload]
    return manifest_file.parent.resolve(), items


def _normalize_manifest_item(raw_item: Any) -> ManifestItem:
    if not isinstance(raw_item, dict):
        raise ValueError("Each manifest item must be a mapping.")
    object_scale_key = _require_string(raw_item, "object_scale_key")
    object_name = _require_string(raw_item, "object_name")
    mjcf_path = _require_string(raw_item, "mjcf_path")
    grasp_h5_path = _require_string(raw_item, "grasp_h5_path")
    grasp_h5_fail_path_raw = raw_item.get("grasp_h5_fail_path")
    grasp_h5_fail_path = (
        None if grasp_h5_fail_path_raw is None else str(grasp_h5_fail_path_raw)
    )
    grasp_fail_npy_path_raw = raw_item.get("grasp_fail_npy_path")
    grasp_fail_npy_path = (
        None if grasp_fail_npy_path_raw is None else str(grasp_fail_npy_path_raw)
    )
    partial_pc_path = _require_string_list(raw_item, "partial_pc_path", allow_missing=True)
    partial_pc_cam_path = _require_string_list(
        raw_item, "partial_pc_cam_path", allow_missing=True
    )
    cam_ex_path = _require_string_list(raw_item, "cam_ex_path", allow_missing=True)
    global_pc_path_raw = raw_item.get("global_pc_path")
    global_pc_path = None if global_pc_path_raw is None else str(global_pc_path_raw)
    return ManifestItem(
        object_scale_key=object_scale_key,
        object_name=object_name,
        mjcf_path=mjcf_path,
        grasp_h5_path=grasp_h5_path,
        grasp_h5_fail_path=grasp_h5_fail_path,
        grasp_fail_npy_path=grasp_fail_npy_path,
        partial_pc_path=tuple(partial_pc_path),
        partial_pc_cam_path=tuple(partial_pc_cam_path),
        cam_ex_path=tuple(cam_ex_path),
        global_pc_path=global_pc_path,
    )


def _require_string(raw_item: dict[str, Any], key: str) -> str:
    value = raw_item.get(key)
    if value is None:
        raise KeyError(f"Manifest item missing required key: {key}")
    return str(value)


def _require_string_list(
    raw_item: dict[str, Any],
    key: str,
    allow_missing: bool = False,
) -> list[str]:
    value = raw_item.get(key)
    if value is None:
        if allow_missing:
            return []
        raise KeyError(f"Manifest item missing required key: {key}")
    if not isinstance(value, list):
        raise ValueError(f"Manifest key '{key}' must be a list.")
    return [str(item) for item in value]
