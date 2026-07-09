#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Abstract frame source protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from od_platform.frame_source.core.types import Frame, SourceType


class FrameSource(ABC, Iterator[Frame]):
    """Iterator and context manager that emits BGR frames."""

    def __init__(self, source_path: str, *, stride: int = 1) -> None:
        self.source_path = str(source_path)
        self._stride = max(1, int(stride))
        self._opened = False

    @abstractmethod
    def open(self) -> bool:
        """Open the underlying source and reset iteration state."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read one frame, or return None when exhausted/unavailable."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the source type."""

    def __enter__(self) -> "FrameSource":
        if not self.open():
            raise RuntimeError(f"failed to open frame source: {self.source_path}")
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        self._opened = False

    def __iter__(self) -> "FrameSource":
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def stride(self) -> int:
        return self._stride

    def set_stride(self, stride: int) -> None:
        self._stride = max(1, int(stride))

    @property
    def seekable(self) -> bool:
        return False

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        return False

