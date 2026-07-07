#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : Unified service entrance for dataset splitting.

from __future__ import annotations

from typing import List

from od_platform.data_pipeline.split.registry import (
    DatasetItem,
    SplitOptions,
    SplitResult,
    get_splitter,
)


def split_dataset(
    items: List[DatasetItem],
    strategy: str,
    options: SplitOptions,
) -> SplitResult:
    """Split dataset items by strategy name."""

    entry = get_splitter(strategy)
    return entry.func(items, options)
