#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : yaml_schema.py
# @Project   : ODPlatform
# @Function  : Validate dataset yaml schema consistency.

from __future__ import annotations

from od_platform.common.constants import DATASET_SPLITS
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("yaml_schema", order=10)
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    if snap.yaml_load_error is not None:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary=snap.yaml_load_error,
            details={"yaml_path": str(snap.yaml_path), "reason": "yaml_load_error"},
        )

    problems: list[str] = []
    if snap.nc is None or snap.nc <= 0:
        problems.append(f"nc 字段不存在或不是正整数: {snap.yaml_data.get('nc')!r}")
    if not snap.class_names:
        problems.append("names 字段缺失，或不是非空列表/字典")
    if snap.nc is not None and snap.class_names and len(snap.class_names) != snap.nc:
        problems.append(f"nc 和 names 长度不一致: nc={snap.nc}, names={len(snap.class_names)}")

    for split in ("train", "val"):
        if split not in snap.yaml_data:
            problems.append(f"缺少必要 split 字段: {split}")
    missing_known = [split for split in DATASET_SPLITS if split in snap.yaml_data and split not in snap.split_paths]
    problems.extend(f"split 路径为空或非法: {split}" for split in missing_known)

    if problems:
        return CheckResult(
            name="yaml_schema",
            severity=CheckSeverity.ERROR,
            summary=f"yaml 字段不一致: 共有 {len(problems)} 个问题",
            details={"problems": problems, "yaml_path": str(snap.yaml_path)},
        )

    return CheckResult(
        name="yaml_schema",
        severity=CheckSeverity.PASS,
        summary=f"yaml 字段一致 (nc={snap.nc}, names_count={len(snap.class_names)})",
        details={"nc": snap.nc, "names_count": len(snap.class_names), "splits": list(snap.splits)},
    )
