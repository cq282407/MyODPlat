#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : dataset_path.py
# @Project   : ODPlatform
# @Function  : Dataset yaml resolution helpers.
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from od_platform.common.paths import META_DIR
from od_platform.common.refs import resolve_yaml

logger = logging.getLogger(__name__)
RUNTIME_DATASET_DIR = META_DIR / "runtime_datasets"


def resolve_dataset_path(data: str | Path | None) -> Path:
    """Resolve a dataset yaml path using the shared refs resolver."""

    if data is None:
        raise ValueError("训练数据集 data 不能为空; 请在 train.yaml 或 --data 中指定数据集 yaml")
    resolved = resolve_yaml(data)
    if resolved.exists():
        logger.info("数据集配置文件已找到: %s", resolved)
    else:
        logger.warning("数据集配置文件未找到: %s", resolved)
    return resolved


def prepare_ultralytics_dataset_yaml(data: str | Path | None) -> Path:
    """Write a runtime dataset yaml with an absolute dataset root for Ultralytics."""

    yaml_path = resolve_dataset_path(data).resolve()
    yaml_dict = _load_dataset_yaml(yaml_path)
    runtime_dict = dict(yaml_dict)
    runtime_dict["path"] = _resolve_dataset_root(yaml_path, yaml_dict).as_posix()

    output_path = _runtime_dataset_copy_path(yaml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(runtime_dict, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Ultralytics 运行时数据集配置: %s", output_path)
    return output_path


def _load_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    try:
        content = yaml_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 解码失败, 尝试系统默认编码: %s", yaml_path)
        content = yaml_path.read_text()

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"数据集 YAML 格式错误: {yaml_path}\n原始错误: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"数据集 YAML 顶层必须是字典: {yaml_path}")
    return data


def _resolve_dataset_root(yaml_path: Path, yaml_dict: dict[str, Any]) -> Path:
    raw_root = yaml_dict.get("path")
    if not raw_root:
        return yaml_path.parent.resolve()

    root = Path(str(raw_root))
    if root.is_absolute():
        return root.resolve()
    return (yaml_path.parent / root).resolve()


def _runtime_dataset_copy_path(yaml_path: Path) -> Path:
    digest = hashlib.sha1(str(yaml_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return RUNTIME_DATASET_DIR / f"{yaml_path.stem}-{digest}.ultra.yaml"
