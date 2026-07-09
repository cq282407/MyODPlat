#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_paths.py
# @Project   : ODPlatform
# @Function  : paths.py basic smoke tests

from pathlib import Path

from od_platform.common.paths import (
    APP_DIR,
    DATA_DIR,
    META_BACKUPS_DIR,
    ROOT_DIR,
    get_dirs_to_initialize,
    is_protected,
)


def test_root_dir_has_workspace_marker() -> None:
    assert (ROOT_DIR / ".odp-workspace").exists()


def test_app_dir_points_to_platform() -> None:
    assert APP_DIR == ROOT_DIR / "apps" / "platform"


def test_data_dir_is_shared_root_asset() -> None:
    assert DATA_DIR == ROOT_DIR / "data"


def test_dirs_to_initialize_are_paths() -> None:
    assert all(isinstance(path, Path) for path in get_dirs_to_initialize())


def test_meta_backups_dir_is_initialized_and_protected() -> None:
    assert META_BACKUPS_DIR in get_dirs_to_initialize()
    assert is_protected(META_BACKUPS_DIR)
