#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Project   : ODPlatform
# @Function  : Registry contracts for data validation checks.

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckSeverity:
    """Severity names shared by checks, reports and the CLI exit-code mapping."""

    PASS = "PASS"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    _ORDER = {PASS: 0, INFO: 1, WARNING: 2, ERROR: 3}

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return cls.PASS, cls.INFO, cls.WARNING, cls.ERROR

    @classmethod
    def score(cls, severity: str) -> int:
        return cls._ORDER[severity]


@dataclass(frozen=True)
class ValidationOptions:
    """Option bag kept stable so new checks can grow without changing signatures."""

    run_id: Optional[str] = None
    output_dir: Optional[Path] = None
    operator: Optional[str] = None
    operator_role: Optional[str] = None
    device_tag: Optional[str] = None
    operation: str = "data_validation"
    notes: Optional[str] = None
    details_preview_limit: int = 20
    check_phash: bool = False
    phash_threshold: int = 6


@dataclass(frozen=True)
class CheckResult:
    """One check result. `passed` is derived from severity, not stored."""

    name: str
    severity: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity in {CheckSeverity.PASS, CheckSeverity.INFO}


@dataclass(frozen=True)
class CheckContext:
    """Arguments shared by every validation check."""

    yaml_path: Path
    snapshot: "DatasetSnapshot"
    options: ValidationOptions


CheckFunc = Callable[[CheckContext], CheckResult]


@dataclass(frozen=True)
class CheckEntry:
    """Registered validation check metadata."""

    name: str
    func: CheckFunc
    order: int


_REGISTRY: Dict[str, CheckEntry] = {}
_LAZY_INITIALIZED = False


def check(name: str, *, order: int = 100) -> Callable[[CheckFunc], CheckFunc]:
    """Register a validation check function."""

    def decorator(func: CheckFunc) -> CheckFunc:
        normalized = name.lower()
        if normalized in _REGISTRY:
            logger.warning("Validation check %s is registered more than once.", normalized)
        _REGISTRY[normalized] = CheckEntry(name=normalized, func=func, order=order)
        return func

    return decorator


def list_checks() -> List[CheckEntry]:
    """Return registered checks in execution order."""

    _lazy_init()
    return sorted(_REGISTRY.values(), key=lambda entry: (entry.order, entry.name))


def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    _LAZY_INITIALIZED = True

    from od_platform.data_validation import checks

    for module_info in sorted(pkgutil.iter_modules(checks.__path__), key=lambda item: item.name):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{checks.__name__}.{module_info.name}")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from od_platform.data_validation.snapshot import DatasetSnapshot
