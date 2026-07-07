#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : bbox_area_profile.py
# @Project   : ODPlatform
# @Function  : Profile bbox area distribution and flag extreme boxes.

from __future__ import annotations

from statistics import mean, median

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("bbox_area_profile", order=80)
def validate_bbox_area_profile(ctx: CheckContext) -> CheckResult:
    areas = [record.width * record.height for record in ctx.snapshot.records]
    if not areas:
        return CheckResult(
            name="bbox_area_profile",
            severity=CheckSeverity.INFO,
            summary="no bbox instances available for area profiling",
            details={"instances": 0},
        )

    tiny_threshold = 1e-4
    huge_threshold = 0.9
    tiny = [
        _record_item(record, area)
        for record, area in zip(ctx.snapshot.records, areas)
        if area < tiny_threshold
    ]
    huge = [
        _record_item(record, area)
        for record, area in zip(ctx.snapshot.records, areas)
        if area > huge_threshold
    ]
    sorted_areas = sorted(areas)
    details = {
        "instances": len(areas),
        "min_area": sorted_areas[0],
        "max_area": sorted_areas[-1],
        "mean_area": mean(areas),
        "median_area": median(areas),
        "tiny_threshold": tiny_threshold,
        "huge_threshold": huge_threshold,
        "tiny_count": len(tiny),
        "huge_count": len(huge),
        "tiny_preview": tiny[: ctx.options.details_preview_limit],
        "huge_preview": huge[: ctx.options.details_preview_limit],
    }

    if tiny or huge:
        return CheckResult(
            name="bbox_area_profile",
            severity=CheckSeverity.WARNING,
            summary=f"bbox area outliers detected: tiny={len(tiny)}, huge={len(huge)}",
            details=details,
        )

    return CheckResult(
        name="bbox_area_profile",
        severity=CheckSeverity.PASS,
        summary=(
            f"bbox area profile is normal: min={sorted_areas[0]:.6f}, "
            f"median={median(areas):.6f}, max={sorted_areas[-1]:.6f}"
        ),
        details=details,
    )


def _record_item(record, area: float) -> dict[str, object]:
    return {
        "split": record.split,
        "path": str(record.label_path),
        "line": record.line_number,
        "class_id": record.class_id,
        "area": area,
        "width": record.width,
        "height": record.height,
    }
