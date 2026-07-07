#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : config_log.py
# @Project   : ODPlatform
# @Function  : Field-level runtime config logging.
from __future__ import annotations

import logging
from typing import Any

from od_platform.common.string_utils import pad_to_width


def log_effective_config(
    config: Any,
    merger: Any,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Log each effective config field with its current source."""

    log = logger or logging.getLogger(__name__)
    log.info("=" * section_width)
    log.info(pad_to_width("配置参数信息", section_width, align="center"))
    log.info("-" * section_width)
    for field_name in config.__class__.model_fields:
        value = getattr(config, field_name, None)
        meta = _safe_get_metadata(merger, field_name)
        source_label = meta.source_label if meta is not None else "未知"
        log.info("%s: %s  (来源: %s)", pad_to_width(field_name, key_width), value, source_label)


def log_override_chains(
    config: Any,
    merger: Any,
    *,
    logger: logging.Logger | None = None,
    key_width: int = 20,
    section_width: int = 60,
) -> None:
    """Log full DEFAULT -> YAML -> CLI override chains for each field."""

    log = logger or logging.getLogger(__name__)
    log.info("-" * section_width)
    log.info(pad_to_width("配置覆盖情况", section_width, align="center"))
    log.info("-" * section_width)
    for field_name in config.__class__.model_fields:
        meta = _safe_get_metadata(merger, field_name)
        if meta is None:
            log.info("%s: %s", pad_to_width(field_name, key_width), getattr(config, field_name, None))
            continue
        chain = list(reversed(meta.chain()))
        chain_str = " <- ".join(f"{item.value}({item.source_label})" for item in chain)
        log.info("%s: %s", pad_to_width(field_name, key_width), chain_str)


def _safe_get_metadata(merger: Any, field_name: str) -> Any:
    if not hasattr(merger, "get_metadata"):
        return None
    try:
        return merger.get_metadata(field_name)
    except Exception:
        return None
