#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : archive.py
# @Project   : ODPlatform
# @Function  : Archive ultralytics best/last weights.
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from od_platform.common.paths import TRAINED_MODELS_DIR

logger = logging.getLogger(__name__)


def archive_checkpoints(
    train_dir: Path,
    model_filename: str | Path,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Path]:
    """Copy best.pt and last.pt from an ultralytics train dir to models/trained."""

    checkpoint_dir = checkpoint_dir or TRAINED_MODELS_DIR
    results: dict[str, Path] = {}
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        logger.warning("训练目录不存在或不是目录, 跳过归档: %s", train_dir)
        return results

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("创建归档目录失败, 跳过归档: %s", exc)
        return results

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_stem = Path(model_filename).stem
    train_suffix = train_dir.name

    for model_type in ("best", "last"):
        src = train_dir / "weights" / f"{model_type}.pt"
        if not src.exists():
            logger.warning("未找到权重文件, 跳过: %s", src)
            continue
        dest = checkpoint_dir / f"{train_suffix}-{timestamp}-{model_stem}-{model_type}.pt"
        try:
            shutil.copy2(src, dest)
        except (OSError, shutil.Error) as exc:
            logger.warning("归档 %s.pt 失败: %s", model_type, exc)
            continue
        logger.info("权重已归档: %s", dest)
        results[model_type] = dest
    return results
