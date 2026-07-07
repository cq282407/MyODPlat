#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : image_integrity.py
# @Project   : ODPlatform
# @Function  : Validate that image files are readable and non-empty.

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("image_integrity", order=25)
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    problems: list[dict[str, object]] = []
    checked = 0
    limit = ctx.options.details_preview_limit

    try:
        from PIL import Image
    except Exception:
        return CheckResult(
            name="image_integrity",
            severity=CheckSeverity.INFO,
            summary="Pillow is unavailable; skipped image decoding check",
            details={"checked_images": 0},
        )

    for split in ctx.snapshot.splits:
        for image_path in ctx.snapshot.images_per_split.get(split, tuple()):
            checked += 1
            try:
                if image_path.stat().st_size <= 0:
                    problems.append(
                        {
                            "split": split,
                            "path": str(image_path),
                            "reason": "zero_byte_image",
                        }
                    )
                    continue
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                problems.append(
                    {
                        "split": split,
                        "path": str(image_path),
                        "reason": "cannot_decode_image",
                        "error": str(exc),
                    }
                )

    if problems:
        return CheckResult(
            name="image_integrity",
            severity=CheckSeverity.ERROR,
            summary=f"{len(problems)} image files are unreadable or empty",
            details={
                "checked_images": checked,
                "problem_count": len(problems),
                "problems_preview": problems[:limit],
            },
        )

    return CheckResult(
        name="image_integrity",
        severity=CheckSeverity.PASS,
        summary=f"all {checked} image files can be decoded",
        details={"checked_images": checked},
    )
