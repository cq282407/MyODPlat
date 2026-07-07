#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_reset_project.py
# @Project   : ODPlatform
# @Function  : reset_project safety smoke tests

from od_platform.common.paths import (
    APP_DIR,
    CONFIGS_DIR,
    DOCS_DIR,
    LOGGING_DIR,
    META_LOGGING_DIR,
    PRETRAINED_MODELS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    RUNS_DIR,
    get_dirs_to_reset,
    is_protected,
)


def test_reset_targets_are_runtime_dirs() -> None:
    targets = get_dirs_to_reset()
    assert RUNS_DIR in targets
    assert LOGGING_DIR in targets
    assert CONFIGS_DIR not in targets
    assert RAW_DATA_DIR not in targets
    assert PRETRAINED_MODELS_DIR not in targets
    assert META_LOGGING_DIR not in targets


def test_core_dirs_are_protected() -> None:
    assert is_protected(ROOT_DIR)
    assert is_protected(APP_DIR)
    assert is_protected(DOCS_DIR)
    assert is_protected(CONFIGS_DIR)
    assert is_protected(RAW_DATA_DIR)
    assert is_protected(PRETRAINED_MODELS_DIR)
    assert is_protected(META_LOGGING_DIR)


def test_runtime_dirs_are_not_protected() -> None:
    assert not is_protected(RUNS_DIR)
    assert not is_protected(LOGGING_DIR)
