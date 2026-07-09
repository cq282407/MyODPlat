#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Frame source data types and local extension constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    """Supported input source types."""

    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_FOLDER = "image_folder"


IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
)


@dataclass(frozen=True)
class FrameInfo:
    """Metadata carried with every emitted frame."""

    width: int
    height: int
    source_type: SourceType
    source_path: str
    frame_index: int = 0
    total_frames: Optional[int] = None
    timestamp: float = 0.0
    fps: Optional[float] = None
    filename: Optional[str] = None


@dataclass
class Frame:
    """One BGR image plus source metadata."""

    image: np.ndarray
    info: FrameInfo

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height

