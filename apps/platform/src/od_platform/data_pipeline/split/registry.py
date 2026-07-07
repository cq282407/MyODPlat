#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Project   : ODPlatform
# @Function  : Function registry for dataset split strategies.

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetItem:
    """One image and its matching YOLO label file."""

    stem: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class SplitOptions:
    """Common options shared by split strategies."""

    train_rate: float = 0.8
    val_rate: float = 0.1
    test_rate: float = 0.1
    random_state: int = 42


SplitResult = Dict[str, List[DatasetItem]]
SplitFunc = Callable[[List[DatasetItem], SplitOptions], SplitResult]


@dataclass(frozen=True)
class SplitEntry:
    """Registry entry for a split strategy."""

    func: SplitFunc


_REGISTRY: Dict[str, SplitEntry] = {}
_LAZY_INITIALIZED = False


def register(strategy_name: str) -> Callable[[SplitFunc], SplitFunc]:
    """Register a split strategy function."""

    def decorator(func: SplitFunc) -> SplitFunc:
        normalized = strategy_name.lower()
        if normalized in _REGISTRY:
            logger.warning("Split strategy %s is registered more than once.", normalized)
        _REGISTRY[normalized] = SplitEntry(func=func)
        return func

    return decorator


def get_splitter(strategy_name: str) -> SplitEntry:
    """Return split strategy by name."""

    _lazy_init()
    normalized = strategy_name.lower()
    if normalized not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Split strategy {strategy_name!r} is not registered. Available: {available}")
    return _REGISTRY[normalized]


def list_split_strategies() -> List[str]:
    """Return registered split strategy names."""

    _lazy_init()
    return sorted(_REGISTRY)


def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    _LAZY_INITIALIZED = True

    from od_platform.data_pipeline.split import strategies

    for module_info in pkgutil.iter_modules(strategies.__path__):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{strategies.__name__}.{module_info.name}")
