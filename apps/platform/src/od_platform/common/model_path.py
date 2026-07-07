#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : model_path.py
# @Project   : ODPlatform
# @Function  : Model path resolution helper for train/val/infer services.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from od_platform.common.paths import PRETRAINED_MODELS_DIR
from od_platform.common.refs import resolve_pretrained_model

logger = logging.getLogger(__name__)


def resolve_model_path(model: str | Path | None, *, search_dirs: Sequence[Path] | None = None) -> Path:
    """Resolve a model path, preferring local project weights when present.

    Missing bare model names are returned unchanged so ultralytics can download
    official weights such as ``yolo11n.pt``.
    """

    model = model or "yolo11n.pt"
    model_path = Path(model)
    if model_path.is_absolute() or len(model_path.parts) > 1:
        return model_path.resolve()

    dirs = list(search_dirs) if search_dirs is not None else [PRETRAINED_MODELS_DIR]
    for directory in dirs:
        candidate = directory / model_path.name
        if candidate.exists():
            logger.info("模型文件已找到: %s", candidate)
            return candidate

    resolved = resolve_pretrained_model(model_path.name)
    logger.warning(
        "模型文件未在本地命中: %s; 搜索目录: %s; 将交给 ultralytics 继续处理",
        model_path.name,
        [str(directory) for directory in dirs],
    )
    return Path(resolved.name) if not resolved.exists() else resolved
