#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : annotation_coverage.py
# @Project   : ODPlatform
# @Function  : Validate non-empty annotation coverage.

from __future__ import annotations

from od_platform.common.constants import ANNOTATION_COVERAGE_WARN_THRESHOLD
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("annotation_coverage", order=60)
def validate_annotation_coverage(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    total = snap.total_images
    annotated = sum(stats.annotated_count for stats in snap.stats_per_split.values())
    ratio = annotated / total if total else 0.0

    details = {"total_images": total, "annotated_images": annotated, "coverage": ratio}
    if total == 0:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.ERROR,
            summary="数据集中没有图像",
            details=details,
        )
    if annotated == 0:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.ERROR,
            summary="数据集中没有任何有标注图像",
            details=details,
        )
    if ratio < ANNOTATION_COVERAGE_WARN_THRESHOLD:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.WARNING,
            summary=f"标注覆盖率偏低: {annotated}/{total} ({ratio:.1%})",
            details=details,
        )
    return CheckResult(
        name="annotation_coverage",
        severity=CheckSeverity.PASS,
        summary=f"标注覆盖率 {annotated}/{total} ({ratio:.1%})",
        details=details,
    )
