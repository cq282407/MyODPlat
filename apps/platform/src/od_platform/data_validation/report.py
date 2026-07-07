#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : report.py
# @Project   : ODPlatform
# @Function  : Structured validation report data model.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_platform import __version__
from od_platform.data_validation.registry import CheckResult, CheckSeverity
from od_platform.data_validation.snapshot import DatasetSnapshot


@dataclass(frozen=True)
class ValidationReport:
    """One validation run result. Rendering lives outside this class."""

    run_id: str
    yaml_path: Path
    snapshot: DatasetSnapshot
    results: List[CheckResult]
    run_dir: Path
    operator: Optional[str] = None
    duration_seconds: float = 0.0
    started_at_iso: str = ""
    tool_version: str = __version__

    @property
    def overall_severity(self) -> str:
        if not self.results:
            return CheckSeverity.ERROR
        return max(self.results, key=lambda result: CheckSeverity.score(result.severity)).severity

    @property
    def has_errors(self) -> bool:
        return any(result.severity == CheckSeverity.ERROR for result in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(result.severity == CheckSeverity.WARNING for result in self.results)

    @property
    def exit_code(self) -> int:
        if self.has_errors:
            return 2
        if self.has_warnings:
            return 1
        return 0

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def markdown_path(self) -> Path:
        return self.run_dir / "report.md"

    @property
    def data_dictionary_path(self) -> Path:
        return self.run_dir / "data_dictionary.json"

    @property
    def failed_results(self) -> List[CheckResult]:
        return [result for result in self.results if result.severity in {CheckSeverity.WARNING, CheckSeverity.ERROR}]

    def severity_counts(self) -> Dict[str, int]:
        counts = {severity: 0 for severity in CheckSeverity.all()}
        for result in self.results:
            counts[result.severity] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "operator": self.operator,
            "tool_version": self.tool_version,
            "started_at": self.started_at_iso,
            "duration_seconds": round(self.duration_seconds, 3),
            "yaml_path": str(self.yaml_path),
            "overall_severity": self.overall_severity,
            "exit_code": self.exit_code,
            "severity_counts": self.severity_counts(),
            "data_dictionary": self.snapshot.as_data_dictionary(),
            "results": [
                {
                    "name": result.name,
                    "severity": result.severity,
                    "summary": result.summary,
                    "details": _json_safe(result.details),
                }
                for result in self.results
            ],
        }

    def write_json(self) -> None:
        self.report_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    return value
