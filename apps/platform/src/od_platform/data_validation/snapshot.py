#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : snapshot.py
# @Project   : ODPlatform
# @Function  : Dataset dictionary and one-pass scan for validation checks.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from od_platform.common.constants import BBOX_BOUNDARY_EPSILON, DATASET_SPLITS, IMAGE_EXTENSIONS


@dataclass(frozen=True)
class SplitStats:
    """Lightweight per-split data dictionary stats."""

    image_count: int
    label_count: int
    annotated_count: int
    total_instances: int


@dataclass(frozen=True)
class LabelRecord:
    """One parsed YOLO bbox line."""

    split: str
    label_path: Path
    line_number: int
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def extends_outside_image(self) -> bool:
        eps = BBOX_BOUNDARY_EPSILON
        return (
            self.x_center - self.width / 2 < -eps
            or self.x_center + self.width / 2 > 1 + eps
            or self.y_center - self.height / 2 < -eps
            or self.y_center + self.height / 2 > 1 + eps
        )


@dataclass(frozen=True)
class LabelProblem:
    """One label-line parsing problem."""

    split: str
    label_path: Path
    line_number: int
    raw: str
    reason: str


@dataclass(frozen=True)
class DatasetSnapshot:
    """Shared dataset dictionary consumed by all checks."""

    yaml_path: Path
    yaml_data: Dict[str, Any]
    yaml_load_error: Optional[str]
    data_root: Path
    nc: Optional[int]
    class_names: Tuple[str, ...]
    task_type: str
    split_paths: Dict[str, Path]
    images_per_split: Dict[str, Tuple[Path, ...]]
    labels_per_split: Dict[str, Tuple[Path, ...]]
    records: Tuple[LabelRecord, ...]
    label_problems: Tuple[LabelProblem, ...]
    stats_per_split: Dict[str, SplitStats]
    scan_warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def splits(self) -> Tuple[str, ...]:
        return tuple(split for split in DATASET_SPLITS if split in self.split_paths)

    @property
    def total_images(self) -> int:
        return sum(stats.image_count for stats in self.stats_per_split.values())

    @property
    def total_labels(self) -> int:
        return sum(stats.label_count for stats in self.stats_per_split.values())

    @property
    def total_instances(self) -> int:
        return len(self.records)

    @property
    def class_instance_counts(self) -> Dict[str, int]:
        counter: Counter[int] = Counter(record.class_id for record in self.records)
        return {
            name: counter.get(class_id, 0)
            for class_id, name in enumerate(self.class_names)
        }

    def as_data_dictionary(self) -> Dict[str, Any]:
        """Return a JSON-friendly dataset dictionary for reports."""

        return {
            "yaml_path": str(self.yaml_path),
            "data_root": str(self.data_root),
            "task_type": self.task_type,
            "nc": self.nc,
            "class_names": list(self.class_names),
            "totals": {
                "images": self.total_images,
                "labels": self.total_labels,
                "instances": self.total_instances,
            },
            "splits": {
                split: {
                    "path": str(self.split_paths[split]),
                    "images": self.stats_per_split[split].image_count,
                    "labels": self.stats_per_split[split].label_count,
                    "annotated_images": self.stats_per_split[split].annotated_count,
                    "instances": self.stats_per_split[split].total_instances,
                }
                for split in self.splits
            },
            "class_instance_counts": self.class_instance_counts,
            "scan_warnings": list(self.scan_warnings),
        }


def build_snapshot(yaml_path: Path) -> DatasetSnapshot:
    """Load the dataset yaml and scan image/label files once."""

    yaml_path = yaml_path.resolve()
    yaml_data, yaml_error = _load_yaml(yaml_path)
    data_root = _resolve_data_root(yaml_path, yaml_data)
    class_names = _extract_class_names(yaml_data)
    nc = _extract_nc(yaml_data)
    task_type = str(yaml_data.get("task", "detect")) if isinstance(yaml_data, dict) else "detect"

    split_paths = _resolve_split_paths(data_root, yaml_data)
    images_per_split: Dict[str, Tuple[Path, ...]] = {}
    labels_per_split: Dict[str, Tuple[Path, ...]] = {}
    records: list[LabelRecord] = []
    problems: list[LabelProblem] = []
    stats_per_split: Dict[str, SplitStats] = {}
    warnings: list[str] = []

    for split, images_dir in split_paths.items():
        labels_dir = _labels_dir_from_images_dir(images_dir)
        images = tuple(_iter_images(images_dir)) if images_dir.exists() else tuple()
        labels = tuple(sorted(labels_dir.glob("*.txt"))) if labels_dir.exists() else tuple()
        images_per_split[split] = images
        labels_per_split[split] = labels

        annotated_count = 0
        total_instances = 0
        for label_path in labels:
            parsed_records, parsed_problems = _parse_label_file(split, label_path)
            records.extend(parsed_records)
            problems.extend(parsed_problems)
            if parsed_records:
                annotated_count += 1
                total_instances += len(parsed_records)

        if not images_dir.exists():
            warnings.append(f"{split} image directory does not exist: {images_dir}")
        if not labels_dir.exists():
            warnings.append(f"{split} label directory does not exist: {labels_dir}")

        stats_per_split[split] = SplitStats(
            image_count=len(images),
            label_count=len(labels),
            annotated_count=annotated_count,
            total_instances=total_instances,
        )

    return DatasetSnapshot(
        yaml_path=yaml_path,
        yaml_data=yaml_data,
        yaml_load_error=yaml_error,
        data_root=data_root,
        nc=nc,
        class_names=class_names,
        task_type=task_type,
        split_paths=split_paths,
        images_per_split=images_per_split,
        labels_per_split=labels_per_split,
        records=tuple(records),
        label_problems=tuple(problems),
        stats_per_split=stats_per_split,
        scan_warnings=tuple(warnings),
    )


def _load_yaml(yaml_path: Path) -> tuple[Dict[str, Any], Optional[str]]:
    if not yaml_path.exists():
        return {}, f"YAML file does not exist: {yaml_path}"

    try:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            data = _load_simple_yaml(yaml_path)
        else:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}, f"YAML top-level must be a mapping, got {type(data).__name__}"
        return data, None
    except Exception as exc:  # YAML/parser errors should become validation results.
        return {}, f"Error loading YAML file: {exc}"


def _load_simple_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Tiny fallback parser for the dataset yaml produced by this project."""

    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line[:1].isspace():
            if current_key is None:
                continue
            child = line.strip()
            if child.startswith("- "):
                data.setdefault(current_key, []).append(_parse_scalar(child[2:].strip()))
                continue
            if ":" in child:
                key, value = child.split(":", 1)
                container = data.setdefault(current_key, {})
                if isinstance(container, dict):
                    container[_parse_scalar(key.strip())] = _parse_scalar(value.strip())
            continue

        key, value = line.split(":", 1) if ":" in line else (line, "")
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = _parse_scalar(value)
            current_key = None
        else:
            data[key] = {}
            current_key = key
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _resolve_data_root(yaml_path: Path, yaml_data: Dict[str, Any]) -> Path:
    raw_root = yaml_data.get("path")
    if not raw_root:
        return yaml_path.parent
    root = Path(str(raw_root))
    if root.is_absolute():
        return root
    yaml_relative = (yaml_path.parent / root).resolve()
    if yaml_relative.exists():
        return yaml_relative
    cwd_relative = root.resolve()
    if cwd_relative.exists():
        return cwd_relative
    return yaml_relative


def _resolve_split_paths(data_root: Path, yaml_data: Dict[str, Any]) -> Dict[str, Path]:
    split_paths: Dict[str, Path] = {}
    for split in DATASET_SPLITS:
        raw_value = yaml_data.get(split)
        if not raw_value:
            continue
        path = Path(str(raw_value))
        split_paths[split] = path if path.is_absolute() else data_root / path
    return split_paths


def _extract_nc(yaml_data: Dict[str, Any]) -> Optional[int]:
    value = yaml_data.get("nc")
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_class_names(yaml_data: Dict[str, Any]) -> Tuple[str, ...]:
    raw_names = yaml_data.get("names")
    if isinstance(raw_names, dict):
        try:
            return tuple(str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item)))
        except (TypeError, ValueError):
            return tuple(str(value) for _, value in sorted(raw_names.items(), key=lambda item: str(item[0])))
    if isinstance(raw_names, list):
        return tuple(str(value) for value in raw_names)
    return tuple()


def _labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _parse_label_file(split: str, label_path: Path) -> tuple[list[LabelRecord], list[LabelProblem]]:
    records: list[LabelRecord] = []
    problems: list[LabelProblem] = []
    for index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            problems.append(LabelProblem(split, label_path, index, raw, "expected 5 fields"))
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            problems.append(LabelProblem(split, label_path, index, raw, "non-numeric field"))
            continue
        records.append(
            LabelRecord(
                split=split,
                label_path=label_path,
                line_number=index,
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return records, problems
