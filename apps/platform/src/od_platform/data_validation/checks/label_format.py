#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : label_format.py
# @Project   : ODPlatform
# @Function  : Validate YOLO txt line format and scalar ranges.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("label_format", order=30)
def validate_label_format(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    limit = ctx.options.details_preview_limit
    problems: list[dict[str, object]] = [
        {
            "split": problem.split,
            "path": str(problem.label_path),
            "line": problem.line_number,
            "reason": problem.reason,
            "raw": problem.raw,
        }
        for problem in snap.label_problems
    ]

    for record in snap.records:
        if snap.nc is not None and not 0 <= record.class_id < snap.nc:
            problems.append(_record_problem(record, f"class_id 越界: {record.class_id}"))
        values = (record.x_center, record.y_center, record.width, record.height)
        if any(value < 0 or value > 1 for value in values):
            problems.append(_record_problem(record, "YOLO 坐标必须在 [0, 1] 内"))
        if record.width <= 0 or record.height <= 0:
            problems.append(_record_problem(record, "bbox 宽高必须为正数"))

    if problems:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.ERROR,
            summary=f"标签格式存在 {len(problems)} 个问题",
            details={"problems_count": len(problems), "problems_preview": problems[:limit]},
        )

    return CheckResult(
        name="label_format",
        severity=CheckSeverity.PASS,
        summary=f"全部 {snap.total_instances} 行标签格式正确 (task={snap.task_type})",
        details={"instances": snap.total_instances},
    )


def _record_problem(record, reason: str) -> dict[str, object]:
    return {
        "split": record.split,
        "path": str(record.label_path),
        "line": record.line_number,
        "reason": reason,
    }
