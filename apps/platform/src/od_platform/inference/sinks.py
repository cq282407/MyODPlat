#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : sinks.py
# @Project   : ODPlatform
# @Function  : Inference output sinks.
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from od_platform.frame_source import SourceType

logger = logging.getLogger(__name__)


class OutputSink(ABC):
    """Output adapter for annotated frames."""

    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        pass

    @abstractmethod
    def write(self, frame, annotated: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class LocalFileSink(OutputSink):
    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self._is_stream: bool = False
        self._video = None
        self.write_count: int = 0

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.output_dir = output_dir
        self._is_stream = source_type in (SourceType.VIDEO, SourceType.CAMERA)

    def write(self, frame, annotated: np.ndarray) -> None:
        import cv2

        if self.output_dir is None:
            return
        try:
            if self._is_stream:
                if self._video is None:
                    height, width = annotated.shape[:2]
                    fps = float(frame.info.fps) if getattr(frame.info, "fps", None) else 30.0
                    output_path = self.output_dir / "output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                self._video.write(annotated)
            else:
                filename = frame.info.filename or f"frame_{frame.info.frame_index:06d}.jpg"
                output_path = self.output_dir / f"{Path(filename).stem}.jpg"
                cv2.imwrite(str(output_path), annotated)
            self.write_count += 1
        except Exception as exc:
            logger.warning("LocalFileSink.write 失败, 跳过: %s", exc)

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.release()
            except Exception as exc:
                logger.warning("LocalFileSink.close release 失败 (已吞): %s", exc)
            finally:
                self._video = None


class NullSink(OutputSink):
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        pass

    def write(self, frame, annotated: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass
