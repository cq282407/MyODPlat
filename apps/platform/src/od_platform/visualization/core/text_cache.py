#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Text size and font cache for visualization labels."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DEFAULT_FONT_NAME = "LXGWWenKai-Bold"
_FALLBACK_FONT_NAMES = (
    _DEFAULT_FONT_NAME,
    "msyh",
    "simhei",
    "simsun",
    "Microsoft YaHei",
)
_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def _iter_system_font_dirs() -> List[Path]:
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs.extend([Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library" / "Fonts"])
    else:
        dirs.extend([Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"])
    return [d for d in dirs if d.is_dir()]


def _match_font_in_dir(directory: Path, name: str, recursive: bool) -> Optional[str]:
    has_ext = Path(name).suffix.lower() in _FONT_EXTENSIONS
    if not recursive:
        candidates = [directory / name] if has_ext else [directory / f"{name}{ext}" for ext in _FONT_EXTENSIONS]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    wanted = name.lower()
    try:
        for file in directory.rglob("*"):
            if file.suffix.lower() not in _FONT_EXTENSIONS:
                continue
            if file.name.lower() == wanted or file.stem.lower() == wanted:
                return str(file)
    except (OSError, PermissionError):
        return None
    return None


def _resolve_font_path(font: Optional[str]) -> str:
    name = font or _DEFAULT_FONT_NAME
    if Path(name).is_file():
        return str(name)

    names = [name] if font else list(_FALLBACK_FONT_NAMES)
    for candidate_name in names:
        hit = _match_font_in_dir(_ASSETS_DIR, candidate_name, recursive=False)
        if hit:
            return hit

        for directory in _iter_system_font_dirs():
            hit = _match_font_in_dir(directory, candidate_name, recursive=True)
            if hit:
                return hit
    return name


class TextSizeCache:
    """Pre-compute label text sizes for common font sizes."""

    def __init__(
        self,
        labels: List[str],
        label_mapping: Optional[Dict[str, str]] = None,
        font_path: Optional[str] = None,
        font_sizes: Optional[Tuple[int, ...]] = None,
        confidence_template: str = "99.0%",
    ) -> None:
        self.font_path = _resolve_font_path(font_path)
        self.label_mapping = label_mapping or {}
        self.font_sizes = font_sizes or tuple(range(10, 32))
        self.confidence_template = confidence_template
        self._fallback_warned = False
        self._size_cache: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._font_cache: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._precompute(labels)

    def get_size(self, display_label: str, font_size: int) -> Tuple[int, int]:
        key = (display_label, font_size)
        if key in self._size_cache:
            return self._size_cache[key]

        nearest = min(self.font_sizes, key=lambda size: abs(size - font_size))
        fallback = (display_label, nearest)
        if fallback in self._size_cache:
            width, height = self._size_cache[fallback]
            scale = font_size / nearest
            return int(width * scale), int(height * scale)
        return 100, 30

    def get_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if font_size not in self._font_cache:
            self._font_cache[font_size] = self._load_font(font_size)
        return self._font_cache[font_size]

    def _precompute(self, labels: List[str]) -> None:
        image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(image)
        display_labels = set(labels)
        display_labels.update(self.label_mapping.get(label, label) for label in labels)

        for size in self.font_sizes:
            font = self.get_font(size)
            for label in display_labels:
                text = f"{label} {self.confidence_template}"
                bbox = draw.textbbox((0, 0), text, font=font)
                self._size_cache[(label, size)] = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(self.font_path, size)
        except OSError as exc:
            if not self._fallback_warned:
                logger.warning("font '%s' failed to load (%s); using PIL default font", self.font_path, exc)
                self._fallback_warned = True
            return ImageFont.load_default()
