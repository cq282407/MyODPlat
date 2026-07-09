#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Visualization data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass
class Detection:
    """One detection box ready for drawing."""

    box: Tuple[int, int, int, int]
    confidence: float
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)


class LabelPosition(Enum):
    ABOVE = auto()
    INSIDE_TOP = auto()
    BELOW = auto()


@dataclass
class LabelLayout:
    box: Tuple[int, int, int, int]
    text_pos: Tuple[int, int]
    position: LabelPosition
    align_right: bool = False
    label_wider: bool = False


class DrawStyle(BaseModel):
    """Style for rounded boxes and labels."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    font_path: Optional[str] = None
    font_size: int = Field(default=26, gt=0, le=500)
    line_width: int = Field(default=2, gt=0, le=50)
    padding_x: int = Field(default=10, ge=0, le=500)
    padding_y: int = Field(default=10, ge=0, le=500)
    radius: int = Field(default=8, ge=0, le=500)
    text_color: Tuple[int, int, int] = (0, 0, 0)

    @field_validator("text_color")
    @classmethod
    def _validate_color(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        for channel in value:
            if not isinstance(channel, int) or not 0 <= channel <= 255:
                raise ValueError(f"text_color values must be integers in 0..255, got {value}")
        return value

    @classmethod
    def from_image_size(
        cls,
        height: int,
        width: int,
        *,
        ref_dim: int = 720,
        base_font_size: int = 26,
        base_line_width: int = 2,
        base_padding_x: int = 10,
        base_padding_y: int = 10,
        base_radius: int = 8,
        font_scale: float = 1.0,
        **kwargs,
    ) -> "DrawStyle":
        scale = min(height, width) / max(ref_dim, 1)
        params = {
            "font_size": max(10, int(base_font_size * scale * font_scale)),
            "line_width": max(1, int(base_line_width * scale)),
            "padding_x": max(5, int(base_padding_x * scale)),
            "padding_y": max(5, int(base_padding_y * scale)),
            "radius": max(3, int(base_radius * scale)),
        }
        params.update(kwargs)
        return cls(**params)

