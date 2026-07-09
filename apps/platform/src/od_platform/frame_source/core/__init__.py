#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Core frame source contracts."""

from __future__ import annotations

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.config import CameraBackend, CameraCodec, CameraConfig
from od_platform.frame_source.core.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)

__all__ = [
    "CameraBackend",
    "CameraCodec",
    "CameraConfig",
    "Frame",
    "FrameInfo",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "SourceType",
    "VIDEO_EXTENSIONS",
]

