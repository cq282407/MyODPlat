#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : data_dictionary.py
# @Project   : ODPlatform
# @Function  : JSON-friendly dataset dictionary helpers.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from od_platform.data_validation.snapshot import DatasetSnapshot, build_snapshot


def build_data_dictionary(yaml_path: Path) -> Dict[str, Any]:
    """Build a data dictionary from a dataset yaml path."""

    return build_snapshot(yaml_path).as_data_dictionary()


def write_data_dictionary(snapshot: DatasetSnapshot, output_path: Path) -> None:
    """Write the snapshot dictionary as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot.as_data_dictionary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
