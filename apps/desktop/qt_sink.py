#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from od_platform.frame_source import SourceType
from od_platform.inference import OutputSink

logger = logging.getLogger(__name__)


class QtSignalSink(QObject, OutputSink):
    """Bridge OutputSink frames to Qt signals."""

    frame_ready = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        QObject.__init__(self, parent)
        self.write_count = 0

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        logger.info("QtSignalSink.open: dir=%s source_type=%s", output_dir, source_type)

    def write(self, frame, annotated: np.ndarray) -> None:
        try:
            if not annotated.flags["C_CONTIGUOUS"]:
                annotated = np.ascontiguousarray(annotated)
            self.frame_ready.emit(annotated)
            self.write_count += 1
        except Exception as exc:
            logger.warning("QtSignalSink.write 失败 (已吞): %s", exc)

    def close(self) -> None:
        logger.info("QtSignalSink.close: 共发了 %s 帧", self.write_count)
