#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : class_presence.py
# @Project   : ODPlatform
# @Function  : Validate class occurrence across the dataset and train split.

from __future__ import annotations

from collections import Counter, defaultdict

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("class_presence", order=70)
def validate_class_presence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    total_counter: Counter[int] = Counter()
    split_counter: dict[str, Counter[int]] = defaultdict(Counter)
    for record in snap.records:
        total_counter[record.class_id] += 1
        split_counter[record.split][record.class_id] += 1

    errors: list[str] = []
    warnings: list[str] = []
    for class_id, class_name in enumerate(snap.class_names):
        if total_counter[class_id] == 0:
            errors.append(f"类别 {class_id} ({class_name}) 在标签中从未出现")
        elif split_counter["train"][class_id] == 0:
            errors.append(f"类别 {class_id} ({class_name}) 未出现在 train split")
        if split_counter["val"][class_id] == 0:
            warnings.append(f"类别 {class_id} ({class_name}) 未出现在 val split")
        if "test" in snap.splits and split_counter["test"][class_id] == 0:
            warnings.append(f"类别 {class_id} ({class_name}) 未出现在 test split")

    details = {
        "class_instance_counts": snap.class_instance_counts,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.ERROR,
            summary=f"类别出现性存在 {len(errors)} 个阻断问题",
            details=details,
        )
    if warnings:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.WARNING,
            summary=f"类别出现性存在 {len(warnings)} 个警告",
            details=details,
        )
    return CheckResult(
        name="class_presence",
        severity=CheckSeverity.PASS,
        summary="yaml 中全部类别均出现在 train/val/test",
        details=details,
    )
