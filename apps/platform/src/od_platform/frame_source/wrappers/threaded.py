#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Background-collection wrapper for FrameSource."""

from __future__ import annotations

import queue
import threading
from typing import Optional

from od_platform.frame_source.core.base import FrameSource
from od_platform.frame_source.core.types import Frame, SourceType


class ThreadedSource(FrameSource):
    """Read frames on a background thread and expose them through read()."""

    def __init__(
        self,
        source: FrameSource,
        *,
        buffer_size: int = 1,
        drop_oldest: bool = True,
    ) -> None:
        super().__init__(source.source_path, stride=source.stride)
        self.source = source
        self.buffer_size = max(1, int(buffer_size))
        self.drop_oldest = drop_oldest
        self._queue: queue.Queue[Optional[Frame]] = queue.Queue(maxsize=self.buffer_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        if not self.source.open():
            return False
        self._stop.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._thread = threading.Thread(target=self._collect, name="frame-source", daemon=True)
        self._thread.start()
        return True

    def read(self) -> Optional[Frame]:
        item = self._queue.get()
        return item

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.source.close()

    def get_source_type(self) -> SourceType:
        return self.source.get_source_type()

    def _collect(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    self._put(None)
                    break
                self._put(frame)
        finally:
            self._put(None)

    def _put(self, frame: Optional[Frame]) -> None:
        if self._stop.is_set():
            return
        while True:
            try:
                self._queue.put(frame, timeout=0.05)
                return
            except queue.Full:
                if not self.drop_oldest:
                    continue
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

