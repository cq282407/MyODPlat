#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : dataset_path.py
# @Project   : ODPlatform
# @Function  : Dataset yaml path resolution helper.
from __future__ import annotations

import logging
from pathlib import Path

from od_platform.common.refs import resolve_yaml

logger = logging.getLogger(__name__)


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
