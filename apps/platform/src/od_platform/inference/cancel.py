#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : cancel.py
# @Project   : ODPlatform
# @Function  : Inference cancel token.
from __future__ import annotations

import threading


class CancelToken:
    """Thread-safe cancellation signal."""

    def __init__(self) -> None:
        self._evt = threading.Event()

    def cancel(self) -> None:
        self._evt.set()

    def is_cancelled(self) -> bool:
        return self._evt.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._evt.wait(timeout)


class InferenceCancelled(Exception):
    """Raised when inference is explicitly cancelled."""
