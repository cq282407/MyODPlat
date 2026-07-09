#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : constants.py
# @Project   : ODPlatform
# @Function  : Shared constants for CLI, registries and pipelines.

from __future__ import annotations

from typing import Tuple

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
DATASET_SPLITS: Tuple[str, ...] = ("train", "val", "test")

ANNOTATION_COVERAGE_WARN_THRESHOLD = 0.5
BBOX_BOUNDARY_EPSILON = 1e-6


class AnnotationFormat:
    """Annotation format names used by the conversion registry."""

    PASCAL_VOC = "pascal_voc"
    COCO = "coco"
    YOLO = "yolo"
    NWPU_VHR10 = "nwpu_vhr10"

    # Backward-compatible short name used by our earlier CLI.
    VOC = "voc"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.PASCAL_VOC, cls.COCO, cls.YOLO, cls.NWPU_VHR10

    @classmethod
    def cli_choices(cls) -> Tuple[str, ...]:
        return cls.PASCAL_VOC, cls.VOC, cls.COCO, cls.YOLO, cls.NWPU_VHR10

    @classmethod
    def normalize(cls, value: str) -> str:
        normalized = value.lower()
        if normalized in {cls.PASCAL_VOC, cls.VOC}:
            return cls.PASCAL_VOC
        return normalized


class Task:
    """Computer vision task names."""

    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.DETECT, cls.SEGMENT


class SplitStrategy:
    """Dataset split strategy names reserved for the full pipeline."""

    NONE = "none"
    RANDOM = "random"
    STRATIFIED = "stratified"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.NONE, cls.RANDOM, cls.STRATIFIED


class MaterializeMode:
    """Dataset materialization modes for split outputs."""

    HARDLINK = "hardlink"
    COPY = "copy"
    TXT = "txt"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.HARDLINK, cls.COPY, cls.TXT
