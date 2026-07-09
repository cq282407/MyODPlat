#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""帧源注册表 — 取代 factory.py 中的硬编码 if/elif 分支.

每个帧源通过 @register_source(priority=N) 自动注册.
priority 越小越优先匹配; 默认 100.
"""
from __future__ import annotations

from typing import Callable, List, Tuple, Type

from od_platform.frame_source.core.base import FrameSource

# (matcher_fn, source_cls, priority)
_SOURCE_REGISTRY: List[Tuple[Callable[[str], bool], Type[FrameSource], int]] = []


def register_source(matcher: Callable[[str], bool], priority: int = 100):
    """装饰器: 将帧源类注册到全局注册表.

    Args:
        matcher: 判断函数, 输入 source_text (str), 返回 True 表示"我能处理"
        priority: 优先级 (越小越优先), 默认 100.
                  内置: camera=10, network=20, dir=30, image=40, video=50
    """

    def decorator(cls: Type[FrameSource]) -> Type[FrameSource]:
        _SOURCE_REGISTRY.append((matcher, cls, priority))
        _SOURCE_REGISTRY.sort(key=lambda x: x[2])
        return cls

    return decorator


def list_sources() -> List[Type[FrameSource]]:
    """返回所有已注册的帧源类 (按优先级排序)."""
    return [cls for _, cls, _ in _SOURCE_REGISTRY]


def find_matching_source(source_text: str) -> Type[FrameSource] | None:
    """遍历注册表, 返回第一个匹配的帧源类. 无匹配返回 None."""
    for matcher, cls, _ in _SOURCE_REGISTRY:
        try:
            if matcher(source_text):
                return cls
        except Exception:
            continue
    return None