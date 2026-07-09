#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Single-image and image-folder frame sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.types import (
    IMAGE_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)

logger = logging.getLogger(__name__)


class ImageSource(FrameSource):
    """FrameSource for one image file."""

    def __init__(self, image_path: str, *, stride: int = 1) -> None:
        super().__init__(image_path, stride=stride)
        self._image: Optional[np.ndarray] = None
        self._read_count = 0
        self._file_name = Path(image_path).name

    def open(self) -> bool:
        self._read_count = 0
        self._image = cv2.imread(self.source_path)
        if self._image is None:
            logger.error("failed to read image: %s", self.source_path)
            return False
        return True

    def read(self) -> Optional[Frame]:
        if self._image is None or self._read_count > 0:
            return None

        height, width = self._image.shape[:2]
        self._read_count += 1
        return Frame(
            image=self._image.copy(),
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.IMAGE,
                source_path=self.source_path,
                frame_index=0,
                total_frames=1,
                filename=self._file_name,
            ),
        )

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE


class ImageFolderSource(FrameSource):
    """FrameSource for a folder of image files."""

    def __init__(self, folder_path: str, *, stride: int = 1) -> None:
        super().__init__(folder_path, stride=stride)
        self._image_files: List[Path] = []
        self._current_index = 0

    def open(self) -> bool:
        self._current_index = 0
        folder = Path(self.source_path)
        if not folder.is_dir():
            logger.error("not a valid image folder: %s", self.source_path)
            return False

        self._image_files = sorted(
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._image_files:
            logger.error("no supported image files found in %s", self.source_path)
            return False
        return True

    def read(self) -> Optional[Frame]:
        while self._current_index < len(self._image_files):
            source_index = self._current_index
            image_path = self._image_files[source_index]
            self._current_index += self._stride

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("skipping unreadable image: %s", image_path)
                continue

            height, width = image.shape[:2]
            return Frame(
                image=image,
                info=FrameInfo(
                    width=width,
                    height=height,
                    source_type=SourceType.IMAGE_FOLDER,
                    source_path=self.source_path,
                    frame_index=source_index,
                    total_frames=len(self._image_files),
                    filename=image_path.name,
                ),
            )
        return None

    def close(self) -> None:
        self._image_files = []
        self._current_index = 0

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER

    @property
    def seekable(self) -> bool:
        return True

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if time_sec is not None:
            logger.warning("image folders do not support time-based seek")
            return False
        if frame is None:
            logger.error("frame index is required for image folder seek")
            return False

        total = len(self._image_files)
        self._current_index = max(0, min(int(frame), total - 1)) if total else 0
        return True

