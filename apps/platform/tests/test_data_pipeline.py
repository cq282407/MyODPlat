#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_data_pipeline.py
# @Project   : ODPlatform
# @Function  : data_pipeline converter and registry smoke tests.

from pathlib import Path

from PIL import Image

from od_platform.common.constants import AnnotationFormat
from od_platform.data_pipeline.convert.registry import ConvertOptions, list_capabilities
from od_platform.data_pipeline.convert.service import convert_data_to_yolo


def test_nwpu_vhr10_converter_is_registered() -> None:
    assert AnnotationFormat.NWPU_VHR10 in list_capabilities()


def test_nwpu_vhr10_converter_writes_yolo_labels_and_backgrounds(tmp_path: Path) -> None:
    raw_root = tmp_path / "nwpu"
    positive_dir = raw_root / "positive image set"
    negative_dir = raw_root / "negative image set"
    gt_dir = raw_root / "ground truth"
    positive_dir.mkdir(parents=True)
    negative_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    Image.new("RGB", (100, 200)).save(positive_dir / "001.jpg")
    Image.new("RGB", (100, 200)).save(negative_dir / "001.jpg")
    (gt_dir / "001.txt").write_text("(10,20),(50,100),1\n", encoding="utf-8")

    out_labels = tmp_path / "out" / "labels"
    classes = convert_data_to_yolo(
        input_dir=raw_root,
        output_labels_dir=out_labels,
        annotation_format=AnnotationFormat.NWPU_VHR10,
        options=ConvertOptions(),
    )

    assert classes[0] == "airplane"
    assert (out_labels / "001.txt").read_text(encoding="utf-8").strip() == "0 0.300000 0.300000 0.400000 0.400000"
    assert (out_labels / "neg_001.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "out" / "_source_images" / "001.jpg").exists()
    assert (tmp_path / "out" / "_source_images" / "neg_001.jpg").exists()
