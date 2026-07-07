#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : split_uniqueness.py
# @Project   : ODPlatform
# @Function  : Validate that samples do not leak across splits.

from __future__ import annotations

from collections import defaultdict

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("split_uniqueness", order=40)
def validate_split_uniqueness(ctx: CheckContext) -> CheckResult:
    stem_splits: dict[str, set[str]] = defaultdict(set)
    for split in ctx.snapshot.splits:
        for image_path in ctx.snapshot.images_per_split.get(split, tuple()):
            stem_splits[image_path.stem].add(split)

    leaked = {stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1}
    if leaked:
        preview = dict(list(sorted(leaked.items()))[: ctx.options.details_preview_limit])
        return CheckResult(
            name="split_uniqueness",
            severity=CheckSeverity.ERROR,
            summary=f"split 间有 {len(leaked)} 个图像名重复，存在数据泄露风险",
            details={"duplicate_stems_count": len(leaked), "duplicate_stems_preview": preview},
        )

    return CheckResult(
        name="split_uniqueness",
        severity=CheckSeverity.PASS,
        summary=f"{len(ctx.snapshot.splits)} 个 split ({' / '.join(ctx.snapshot.splits)}) 之间无图像名重复",
        details={"checked_images": ctx.snapshot.total_images},
    )
