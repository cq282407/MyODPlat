#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : class_balance.py
# @Project   : ODPlatform
# @Function  : Profile class imbalance for training risk assessment.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("class_balance", order=75)
def validate_class_balance(ctx: CheckContext) -> CheckResult:
    counts = ctx.snapshot.class_instance_counts
    non_zero = {name: count for name, count in counts.items() if count > 0}
    total = sum(counts.values())
    if not counts or not total:
        return CheckResult(
            name="class_balance",
            severity=CheckSeverity.INFO,
            summary="no class instances available for balance profiling",
            details={"class_instance_counts": counts},
        )

    max_name, max_count = max(non_zero.items(), key=lambda item: item[1])
    min_name, min_count = min(non_zero.items(), key=lambda item: item[1])
    imbalance_ratio = max_count / min_count if min_count else float("inf")
    shares = {name: count / total for name, count in counts.items()}
    rare_classes = [
        {"class_name": name, "count": count, "share": shares[name]}
        for name, count in counts.items()
        if count > 0 and shares[name] < 0.02
    ]

    details = {
        "class_instance_counts": counts,
        "class_shares": shares,
        "max_class": {"name": max_name, "count": max_count},
        "min_non_zero_class": {"name": min_name, "count": min_count},
        "imbalance_ratio": imbalance_ratio,
        "rare_classes": rare_classes,
    }

    if imbalance_ratio >= 20 or rare_classes:
        return CheckResult(
            name="class_balance",
            severity=CheckSeverity.WARNING,
            summary=(
                f"class imbalance detected: max/min={imbalance_ratio:.1f}, "
                f"rare_classes={len(rare_classes)}"
            ),
            details=details,
        )

    return CheckResult(
        name="class_balance",
        severity=CheckSeverity.PASS,
        summary=f"class balance is acceptable: max/min={imbalance_ratio:.1f}",
        details=details,
    )
