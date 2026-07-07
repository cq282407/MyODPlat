#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : bbox_within_image.py
# @Project   : ODPlatform
# @Function  : Validate YOLO boxes stay inside image boundaries.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("bbox_within_image", order=50)
def validate_bbox_within_image(ctx: CheckContext) -> CheckResult:
    outside = [
        {
            "split": record.split,
            "path": str(record.label_path),
            "line": record.line_number,
            "class_id": record.class_id,
            "x_center": record.x_center,
            "y_center": record.y_center,
            "width": record.width,
            "height": record.height,
        }
        for record in ctx.snapshot.records
        if record.extends_outside_image
    ]
    if outside:
        return CheckResult(
            name="bbox_within_image",
            severity=CheckSeverity.ERROR,
            summary=f"{len(outside)} 个 bbox 伸出图像边界",
            details={
                "outside_count": len(outside),
                "outside_preview": outside[: ctx.options.details_preview_limit],
            },
        )

    return CheckResult(
        name="bbox_within_image",
        severity=CheckSeverity.PASS,
        summary="全部 bbox 均在图像边界内",
        details={"instances": ctx.snapshot.total_instances},
    )
