#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : empty_labels.py
# @Project   : ODPlatform
# @Function  : Profile empty YOLO label files as background samples.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("empty_labels", order=35)
def validate_empty_labels(ctx: CheckContext) -> CheckResult:
    empty_by_split: dict[str, int] = {}
    empty_preview: list[str] = []
    total_labels = 0
    total_empty = 0
    limit = ctx.options.details_preview_limit

    for split in ctx.snapshot.splits:
        empty = 0
        for label_path in ctx.snapshot.labels_per_split.get(split, tuple()):
            total_labels += 1
            if not label_path.read_text(encoding="utf-8").strip():
                empty += 1
                total_empty += 1
                if len(empty_preview) < limit:
                    empty_preview.append(f"{split}/{label_path.stem}")
        empty_by_split[split] = empty

    ratio = total_empty / total_labels if total_labels else 0.0
    details = {
        "total_labels": total_labels,
        "empty_labels": total_empty,
        "empty_ratio": ratio,
        "empty_by_split": empty_by_split,
        "empty_preview": empty_preview,
    }

    if total_labels == 0:
        return CheckResult(
            name="empty_labels",
            severity=CheckSeverity.INFO,
            summary="no label files were found for empty-label profiling",
            details=details,
        )

    return CheckResult(
        name="empty_labels",
        severity=CheckSeverity.PASS,
        summary=f"background label files: {total_empty}/{total_labels} ({ratio:.1%})",
        details=details,
    )
