#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : plot_style.py
# @Project   : ODPlatform
# @Function  : Explicit matplotlib academic style.
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ACADEMIC_RCPARAMS: dict[str, object] = {
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_academic_style(*, style: str = "seaborn-v0_8-whitegrid") -> bool:
    """Apply matplotlib style only when explicitly called."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装, 跳过学术风格应用")
        return False

    try:
        if style:
            plt.style.use(style)
    except OSError:
        logger.warning("matplotlib style 不存在: %s, 仅应用 rcParams", style)
    plt.rcParams.update(_ACADEMIC_RCPARAMS)
    return True
