#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : phash_duplicates.py
# @Project   : ODPlatform
# @Function  : Heavy perceptual-hash near-duplicate image check.

from __future__ import annotations

import math
import statistics
from pathlib import Path

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("phash_duplicates", order=26)
def validate_phash_duplicates(ctx: CheckContext) -> CheckResult:
    """Find visually near-duplicate images with a pHash Hamming threshold."""

    threshold = max(0, int(ctx.options.phash_threshold))
    limit = ctx.options.details_preview_limit

    if not ctx.options.check_phash:
        return CheckResult(
            name="phash_duplicates",
            severity=CheckSeverity.INFO,
            summary="pHash 近重复图片检测未开启; 使用 --check-phash 启用重型检查",
            details={"enabled": False, "threshold": threshold},
        )

    try:
        from PIL import Image
    except Exception:
        return CheckResult(
            name="phash_duplicates",
            severity=CheckSeverity.INFO,
            summary="Pillow is unavailable; skipped pHash duplicate check",
            details={"enabled": True, "threshold": threshold, "checked_images": 0},
        )

    hashes: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for split in ctx.snapshot.splits:
        for image_path in ctx.snapshot.images_per_split.get(split, tuple()):
            try:
                hash_value = _phash(image_path, Image)
            except Exception as exc:
                skipped.append(
                    {
                        "split": split,
                        "path": str(image_path),
                        "reason": "phash_failed",
                        "error": str(exc),
                    }
                )
                continue
            hashes.append({"split": split, "path": image_path, "hash": hash_value})

    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(hashes):
        for right in hashes[left_index + 1:]:
            distance = _hamming_distance(str(left["hash"]), str(right["hash"]))
            if distance <= threshold:
                pairs.append(
                    {
                        "split": left["split"],
                        "path": str(left["path"]),
                        "paired_split": right["split"],
                        "paired_path": str(right["path"]),
                        "distance": distance,
                        "phash": left["hash"],
                        "paired_phash": right["hash"],
                    }
                )

    if pairs:
        return CheckResult(
            name="phash_duplicates",
            severity=CheckSeverity.WARNING,
            summary=f"发现 {len(pairs)} 组 pHash 近重复图片 (阈值 <= {threshold})",
            details={
                "enabled": True,
                "threshold": threshold,
                "checked_images": len(hashes),
                "skipped_images": len(skipped),
                "pairs_count": len(pairs),
                "pairs_preview": pairs[:limit],
                "skipped_preview": skipped[:limit],
            },
        )

    return CheckResult(
        name="phash_duplicates",
        severity=CheckSeverity.PASS,
        summary=f"pHash 近重复检测通过: {len(hashes)} 张图片, 阈值 <= {threshold}",
        details={
            "enabled": True,
            "threshold": threshold,
            "checked_images": len(hashes),
            "skipped_images": len(skipped),
            "skipped_preview": skipped[:limit],
        },
    )


def _phash(image_path: Path, image_module) -> str:
    with image_module.open(image_path) as image:
        pixels = list(image.convert("L").resize((32, 32)).tobytes())

    coefficients: list[float] = []
    for u in range(8):
        for v in range(8):
            total = 0.0
            for x in range(32):
                cos_x = math.cos(((2 * x + 1) * u * math.pi) / 64)
                row_offset = x * 32
                for y in range(32):
                    cos_y = math.cos(((2 * y + 1) * v * math.pi) / 64)
                    total += pixels[row_offset + y] * cos_x * cos_y
            coefficients.append(total)

    median = statistics.median(coefficients[1:])
    bits = ["1" if value > median else "0" for value in coefficients]
    return f"{int(''.join(bits), 2):016x}"


def _hamming_distance(left_hash: str, right_hash: str) -> int:
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()
