#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : stratified_split.py
# @Project   : ODPlatform
# @Function  : Simple primary-class stratified train/val/test split.

from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import DefaultDict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.registry import (
    DatasetItem,
    SplitOptions,
    SplitResult,
    register,
)
from od_platform.data_pipeline.split.strategies.random_split import _validate_rates


@register(SplitStrategy.STRATIFIED)
def stratified_split(items: List[DatasetItem], options: SplitOptions) -> SplitResult:
    """Split by each image's primary class while keeping a fixed seed."""

    _validate_rates(options)
    grouped: DefaultDict[int, List[DatasetItem]] = defaultdict(list)
    for item in items:
        grouped[_primary_class_id(item)].append(item)

    result: SplitResult = {"train": [], "val": [], "test": []}
    rng = random.Random(options.random_state)
    for _, group_items in sorted(grouped.items(), key=lambda kv: kv[0]):
        group_items = sorted(group_items, key=lambda item: item.stem)
        rng.shuffle(group_items)
        total = len(group_items)
        train_count = int(total * options.train_rate)
        val_count = int(total * options.val_rate)
        result["train"].extend(group_items[:train_count])
        result["val"].extend(group_items[train_count:train_count + val_count])
        result["test"].extend(group_items[train_count + val_count:])

    for split_items in result.values():
        split_items.sort(key=lambda item: item.stem)
    return result


def _primary_class_id(item: DatasetItem) -> int:
    """Return the most frequent class id in a label file, or -1 for background."""

    counter: Counter[int] = Counter()
    for line in item.label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 5:
            counter[int(parts[0])] += 1
    if not counter:
        return -1
    return counter.most_common(1)[0][0]
