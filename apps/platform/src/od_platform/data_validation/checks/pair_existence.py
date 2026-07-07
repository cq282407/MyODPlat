#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pair_existence.py
# @Project   : ODPlatform
# @Function  : Validate image-label pairing for each split.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("pair_existence", order=20)
def validate_pair_existence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    limit = ctx.options.details_preview_limit

    for split in snap.splits:
        image_stems = {path.stem for path in snap.images_per_split.get(split, tuple())}
        label_stems = {path.stem for path in snap.labels_per_split.get(split, tuple())}
        missing_labels.extend(f"{split}/{stem}" for stem in sorted(image_stems - label_stems))
        orphan_labels.extend(f"{split}/{stem}" for stem in sorted(label_stems - image_stems))

    if missing_labels or orphan_labels:
        return CheckResult(
            name="pair_existence",
            severity=CheckSeverity.ERROR,
            summary=(
                f"图像/标签配对失败: 缺标签={len(missing_labels)}, "
                f"孤儿标签={len(orphan_labels)}"
            ),
            details={
                "missing_labels_count": len(missing_labels),
                "orphan_labels_count": len(orphan_labels),
                "missing_labels_preview": missing_labels[:limit],
                "orphan_labels_preview": orphan_labels[:limit],
            },
        )

    return CheckResult(
        name="pair_existence",
        severity=CheckSeverity.PASS,
        summary=f"全部 {snap.total_images} 张图像都有对应标签文件",
        details={"total_images": snap.total_images, "total_labels": snap.total_labels},
    )
