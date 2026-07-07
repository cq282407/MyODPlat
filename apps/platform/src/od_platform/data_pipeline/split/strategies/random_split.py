#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : random_split.py
# @Project   : ODPlatform
# @Function  : Reproducible random train/val/test split.

from __future__ import annotations

import random
from typing import List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.registry import (
    DatasetItem,
    SplitOptions,
    SplitResult,
    register,
)


@register(SplitStrategy.RANDOM)
def random_split(items: List[DatasetItem], options: SplitOptions) -> SplitResult:
    """Split items into train/val/test with a fixed random seed."""

    _validate_rates(options)
    shuffled = sorted(items, key=lambda item: item.stem)
    random.Random(options.random_state).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * options.train_rate)
    val_count = int(total * options.val_rate)

    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]

    return {"train": train, "val": val, "test": test}


def _validate_rates(options: SplitOptions) -> None:
    rates = (options.train_rate, options.val_rate, options.test_rate)
    if any(rate < 0 for rate in rates):
        raise ValueError("Split rates must be non-negative.")
    if abs(sum(rates) - 1.0) > 1e-6:
        raise ValueError(
            "Split rates must sum to 1.0: "
            f"train={options.train_rate}, val={options.val_rate}, test={options.test_rate}"
        )
