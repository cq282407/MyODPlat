#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : refs.py
# @Project   : ODPlatform
# @Function  : Pure reference resolvers for dataset/config/model names.
"""Resolve user-facing resource names into project paths.

The rule is intentionally syntactic: absolute paths or values containing path
separators are treated as paths; bare names are resolved under the corresponding
project directory. This keeps path semantics predictable across D3-D8.
"""
from __future__ import annotations

from pathlib import Path

from od_platform.common.paths import (
    DATASET_CONFIGS_DIR,
    PRETRAINED_MODELS_DIR,
    RAW_DATA_DIR,
    TRAINED_MODELS_DIR,
)


def resolve_ref(ref: str | Path, *, base_dir: Path, default_suffix: str | None = None) -> Path:
    """Resolve a bare name or explicit path to a Path without probing existence."""

    p = Path(ref)
    if p.is_absolute() or len(p.parts) > 1:
        return p.resolve()
    name = str(ref)
    if default_suffix and not name.endswith(default_suffix):
        name += default_suffix
    return (base_dir / name).resolve()


def resolve_dataset(ref: str | Path) -> Path:
    """Resolve a raw dataset reference: bare name -> data/raw/<name>."""

    return resolve_ref(ref, base_dir=RAW_DATA_DIR)


def resolve_yaml(ref: str | Path) -> Path:
    """Resolve a dataset yaml reference: bare name -> configs/datasets/<name>.yaml."""

    return resolve_ref(ref, base_dir=DATASET_CONFIGS_DIR, default_suffix=".yaml")


def resolve_pretrained_model(ref: str | Path) -> Path:
    """Resolve a training model reference: bare name -> models/pretrained/<name>.pt."""

    return resolve_ref(ref, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")


def resolve_trained_model(ref: str | Path) -> Path:
    """Resolve a trained weight reference: bare name -> models/trained/<name>.pt."""

    return resolve_ref(ref, base_dir=TRAINED_MODELS_DIR, default_suffix=".pt")
