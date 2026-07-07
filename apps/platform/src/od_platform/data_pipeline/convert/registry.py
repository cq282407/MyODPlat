#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Project   : ODPlatform
# @Function  : Function registry for data converters.

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    """Common option bag shared by all low-level converters."""

    task: str = "detect"
    classes: Optional[List[str]] = field(default=None)
    coco_cls91to80: bool = False
    clip_boxes: bool = True
    write_empty: bool = True


ConverterFunc = Callable[[Path, Path, ConvertOptions], List[str]]


@dataclass(frozen=True)
class ConverterEntry:
    """Registry entry: implementation function plus supported tasks."""

    func: ConverterFunc
    supported_tasks: Tuple[str, ...]

    def supports(self, task: str) -> bool:
        return task in self.supported_tasks


_REGISTRY: Dict[str, ConverterEntry] = {}
_LAZY_INITIALIZED = False


def register(
    format_name: str,
    *,
    supported_tasks: Tuple[str, ...],
) -> Callable[[ConverterFunc], ConverterFunc]:
    """Register a converter function."""

    def decorator(func: ConverterFunc) -> ConverterFunc:
        normalized = format_name.lower()
        if normalized in _REGISTRY:
            logger.warning("Converter %s is registered more than once.", normalized)
        _REGISTRY[normalized] = ConverterEntry(
            func=func,
            supported_tasks=tuple(supported_tasks),
        )
        logger.debug("Registered converter: format=%s tasks=%s", normalized, supported_tasks)
        return func

    return decorator


def get_converter(format_name: str) -> ConverterEntry:
    """Return the registered converter entry by annotation format name."""

    _lazy_init()
    normalized = format_name.lower()
    if normalized not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Format {format_name!r} is not registered. Available: {available}")
    return _REGISTRY[normalized]


def list_capabilities() -> Dict[str, Tuple[str, ...]]:
    """Return currently registered formats and their supported tasks."""

    _lazy_init()
    return {fmt: entry.supported_tasks for fmt, entry in _REGISTRY.items()}


def _lazy_init() -> None:
    """Import built-in converter modules on first registry access."""

    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    _LAZY_INITIALIZED = True

    from od_platform.data_pipeline.convert import converters

    for module_info in pkgutil.iter_modules(converters.__path__):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{converters.__name__}.{module_info.name}")
