#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : Data validation orchestration service.

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from od_platform.common.paths import validation_run_dir
from od_platform.data_validation.artifacts import (
    artifact_manifest,
    build_recommendations,
    write_extra_artifacts,
)
from od_platform.data_validation.audit import build_audit_context
from od_platform.data_validation.data_dictionary import write_data_dictionary
from od_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    ValidationOptions,
    list_checks,
)
from od_platform.data_validation.render import render_markdown, render_to_logger
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.snapshot import build_snapshot

logger = logging.getLogger(__name__)


def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    """Run all registered checks. One crashing check must not stop the rest."""

    results: List[CheckResult] = []
    for entry in list_checks():
        result = _safe_run_one(entry, ctx)
        results.append(result)
    return results


def validate_dataset(yaml_path: Path, options: ValidationOptions | None = None) -> ValidationReport:
    """Build a snapshot, run all checks, and write validation artifacts."""

    options = options or ValidationOptions()
    run_id = options.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = options.output_dir or validation_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at_iso = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    snapshot = build_snapshot(yaml_path)
    ctx = CheckContext(yaml_path=Path(yaml_path).resolve(), snapshot=snapshot, options=options)
    results = run_all_checks(ctx)
    duration_seconds = time.perf_counter() - started
    audit = build_audit_context(
        yaml_path=Path(yaml_path).resolve(),
        run_id=run_id,
        started_at_iso=started_at_iso,
        options=options,
    )
    report = ValidationReport(
        run_id=run_id,
        yaml_path=Path(yaml_path).resolve(),
        snapshot=snapshot,
        results=results,
        run_dir=run_dir,
        operator=options.operator,
        operator_role=options.operator_role,
        device_tag=options.device_tag,
        operation=options.operation,
        notes=options.notes,
        audit=audit,
        duration_seconds=duration_seconds,
        started_at_iso=started_at_iso,
    )
    object.__setattr__(report, "recommendations", build_recommendations(report))
    object.__setattr__(report, "artifacts", artifact_manifest(report))

    write_data_dictionary(snapshot, report.data_dictionary_path)
    report.write_json()
    report.markdown_path.write_text(render_markdown(report), encoding="utf-8")
    write_extra_artifacts(report)
    render_to_logger(report, logger)
    return report


def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as exc:  # Keep aggregation alive and expose the failed check.
        logger.exception("Validation check failed unexpectedly: %s", entry.name)
        return CheckResult(
            name=entry.name,
            severity=CheckSeverity.ERROR,
            summary=f"check crashed: {exc}",
            details={"reason": "check_exception", "exception": repr(exc)},
        )
