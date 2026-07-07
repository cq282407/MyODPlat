#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : log_rename.py
# @Project   : ODPlatform
# @Function  : Rename active D2 file log to match ultralytics save_dir.
from __future__ import annotations

import logging
import re
from pathlib import Path

from od_platform.common.logging_utils import ROOT_LOGGER_NAME

logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6}(?:-\d+)?)")


def rename_log_to_save_dir(save_dir: Path, model_stem: str, *, logger_name: str = ROOT_LOGGER_NAME) -> Path | None:
    """Rename the active named-root FileHandler log after ultralytics creates save_dir."""

    root = logging.getLogger(logger_name)
    file_handler = next((handler for handler in root.handlers if isinstance(handler, logging.FileHandler)), None)
    if file_handler is None:
        logger.warning("%s 根 logger 没有 FileHandler, 跳过日志改名", logger_name)
        return None

    old_path = Path(file_handler.baseFilename)
    match = _TIMESTAMP_RE.search(old_path.stem)
    timestamp = match.group(1) if match else "unknown-time"
    if match is None:
        logger.warning("原日志 %s 没有时间戳, 使用 unknown-time", old_path)

    new_path = old_path.parent / f"{Path(save_dir).name}_{timestamp}_{_safe_name(model_stem)}.log"
    if old_path == new_path:
        return old_path

    formatter = file_handler.formatter
    level = file_handler.level
    encoding = getattr(file_handler, "encoding", None) or "utf-8"
    file_handler.flush()
    file_handler.close()
    root.removeHandler(file_handler)

    if not old_path.exists():
        logger.warning("原日志 %s 不存在, 跳过改名", old_path)
        _restore_handler(root, old_path, formatter, level, encoding)
        return None

    try:
        old_path.rename(new_path)
    except OSError as exc:
        logger.warning("日志改名失败, 恢复旧 handler: %s", exc)
        _restore_handler(root, old_path, formatter, level, encoding)
        return None

    try:
        new_handler = logging.FileHandler(new_path, encoding=encoding)
        if formatter is not None:
            new_handler.setFormatter(formatter)
        new_handler.setLevel(level)
        root.addHandler(new_handler)
    except OSError as exc:
        logger.error("创建新日志 handler 失败: %s", exc)
        return new_path

    logger.info("日志文件已对齐训练目录: %s", new_path)
    return new_path


def _restore_handler(
    root: logging.Logger,
    path: Path,
    formatter: logging.Formatter | None,
    level: int,
    encoding: str,
) -> None:
    try:
        restored = logging.FileHandler(path, encoding=encoding)
        if formatter is not None:
            restored.setFormatter(formatter)
        restored.setLevel(level)
        root.addHandler(restored)
    except OSError as exc:
        logger.error("恢复日志 handler 失败: %s", exc)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)
