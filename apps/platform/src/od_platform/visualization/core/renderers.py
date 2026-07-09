#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Pillow text renderer for BGR images."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from od_platform.visualization.core.data_types import DrawStyle
from od_platform.visualization.core.text_cache import TextSizeCache, _resolve_font_path


class PillowTextRenderer:
    """Render text onto an image with cached fonts."""

    def __init__(self, size_cache: Optional[TextSizeCache] = None) -> None:
        self._size_cache = size_cache

    def render_batch(
        self,
        img: np.ndarray,
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
        style: DrawStyle,
    ) -> np.ndarray:
        if not texts:
            return img

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(style)
        for text, position, color in texts:
            draw.text(position, text, font=font, fill=color)
        return np.array(pil_img)

    def get_text_size(self, text: str, style: DrawStyle) -> Tuple[int, int]:
        font = self._get_font(style)
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _get_font(self, style: DrawStyle):
        if self._size_cache is not None:
            return self._size_cache.get_font(style.font_size)
        from PIL import ImageFont

        try:
            return ImageFont.truetype(_resolve_font_path(style.font_path), style.font_size)
        except OSError:
            return ImageFont.load_default()

