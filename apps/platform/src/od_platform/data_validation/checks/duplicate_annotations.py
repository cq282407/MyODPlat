#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : duplicate_annotations.py
# @Project   : ODPlatform
# @Function  : Detect duplicated YOLO annotation rows inside one label file.

from __future__ import annotations

from collections import defaultdict

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("duplicate_annotations", order=55)
def validate_duplicate_annotations(ctx: CheckContext) -> CheckResult:
    by_file: dict[str, list[tuple[int, tuple[object, ...]]]] = defaultdict(list)
    for record in ctx.snapshot.records:
        key = (
            record.class_id,
            round(record.x_center, 6),
            round(record.y_center, 6),
            round(record.width, 6),
            round(record.height, 6),
        )
        by_file[str(record.label_path)].append((record.line_number, key))

    duplicates: list[dict[str, object]] = []
    limit = ctx.options.details_preview_limit
    for label_path, rows in by_file.items():
        seen: dict[tuple[object, ...], int] = {}
        for line_number, key in rows:
            if key in seen:
                duplicates.append(
                    {
                        "path": label_path,
                        "first_line": seen[key],
                        "duplicate_line": line_number,
                        "annotation": list(key),
                    }
                )
            else:
                seen[key] = line_number

    if duplicates:
        return CheckResult(
            name="duplicate_annotations",
            severity=CheckSeverity.WARNING,
            summary=f"{len(duplicates)} duplicated annotation rows need review",
            details={
                "duplicate_count": len(duplicates),
                "duplicates_preview": duplicates[:limit],
            },
        )

    return CheckResult(
        name="duplicate_annotations",
        severity=CheckSeverity.PASS,
        summary="no duplicated annotation rows found",
        details={"checked_instances": ctx.snapshot.total_instances},
    )
