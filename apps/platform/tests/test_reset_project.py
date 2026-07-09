#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_reset_project.py
# @Project   : ODPlatform
# @Function  : reset_project safety smoke tests

import json
import zipfile
from pathlib import Path

from od_platform.cli.reset_project import BACKUP_MANIFEST_NAME, create_core_backup
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


def test_create_core_backup_writes_zip_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "core"
    source.mkdir()
    (source / "module.py").write_text("print('hello')\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "module.pyc").write_bytes(b"cache")

    output_dir = tmp_path / "backups"
    backup_path = create_core_backup(
        output_dir=output_dir,
        sources=(source, tmp_path / "missing"),
        context={"user": "pytest"},
    )

    manifest_path = backup_path.with_suffix(".manifest.json")
    assert backup_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["file_count"] == 1
    assert manifest["context"]["user"] == "pytest"
    assert manifest["missing_sources"] == ["missing"]

    with zipfile.ZipFile(backup_path) as archive:
        names = set(archive.namelist())
        assert "core/module.py" in names
        assert BACKUP_MANIFEST_NAME in names
        assert "core/__pycache__/module.pyc" not in names
