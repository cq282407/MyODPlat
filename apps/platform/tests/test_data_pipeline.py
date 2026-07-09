#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_data_pipeline.py
# @Project   : ODPlatform
# @Function  : data_pipeline converter and registry smoke tests.

from pathlib import Path
import json

from PIL import Image

from od_platform.common.constants import AnnotationFormat, MaterializeMode
from od_platform.data_pipeline.convert.registry import ConvertOptions, list_capabilities
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.orchestrator import DatasetPipeline


def test_nwpu_vhr10_converter_is_registered() -> None:
    assert AnnotationFormat.NWPU_VHR10 in list_capabilities()


def test_coco_converter_is_registered() -> None:
    assert AnnotationFormat.COCO in list_capabilities()


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


def test_coco_converter_writes_yolo_labels_and_metadata(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 200},
            {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 5, "name": "person"},
            {"id": 9, "name": "helmet"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 9, "bbox": [10, 20, 40, 80]},
            {"id": 2, "image_id": 1, "category_id": 999, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 5, "bbox": [90, 190, 20, 20]},
        ],
    }
    (annotations / "instances_train.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    out_labels = tmp_path / "out" / "labels"
    classes = convert_data_to_yolo(
        input_dir=annotations,
        output_labels_dir=out_labels,
        annotation_format=AnnotationFormat.COCO,
        options=ConvertOptions(),
    )

    assert classes == ["person", "helmet"]
    assert (out_labels / "a.txt").read_text(encoding="utf-8").splitlines() == [
        "1 0.300000 0.300000 0.400000 0.400000",
        "0 0.950000 0.975000 0.100000 0.050000",
    ]
    assert (out_labels / "empty.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "out" / "classes.txt").read_text(encoding="utf-8").splitlines() == [
        "0 person",
        "1 helmet",
    ]
    report = (tmp_path / "out" / "conversion_report.txt").read_text(encoding="utf-8")
    assert "annotation_files: 1" in report
    assert "image_records: 2" in report
    assert "label_files: 2" in report
    assert "objects: 2" in report
    assert "skipped_unknown_classes: 1" in report
    assert "clipped_boxes: 1" in report


def test_pipeline_txt_mode_writes_fingerprint_and_avoids_existing_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    configs_root = tmp_path / "configs"
    dataset_root = raw_root / "demo"
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    processed_root.mkdir()
    configs_root.mkdir()
    existing_output = processed_root / "demo_yolo"
    existing_output.mkdir()
    sentinel = existing_output / "keep.txt"
    sentinel.write_text("do not overwrite\n", encoding="utf-8")
    existing_config = configs_root / "demo.yaml"
    existing_config.write_text("existing: true\n", encoding="utf-8")

    for index in range(4):
        image_name = f"img_{index}.jpg"
        Image.new("RGB", (100, 100)).save(images_dir / image_name)
        _write_voc_xml(
            annotations_dir / f"img_{index}.xml",
            image_name=image_name,
            class_name="helmet",
        )

    monkeypatch.setattr("od_platform.data_pipeline.orchestrator.RAW_DATA_DIR", raw_root)
    monkeypatch.setattr("od_platform.data_pipeline.orchestrator.PROCESSED_DATA_DIR", processed_root)
    monkeypatch.setattr("od_platform.data_pipeline.orchestrator.DATASET_CONFIGS_DIR", configs_root)

    result = DatasetPipeline(
        dataset="demo",
        annotation_format=AnnotationFormat.PASCAL_VOC,
        split_strategy="random",
        output_name="demo_yolo",
        train_rate=0.5,
        val_rate=0.25,
        test_rate=0.25,
        random_state=7,
        materialize_mode=MaterializeMode.TXT,
    ).run()

    output_dir = Path(result["output"])
    assert output_dir == processed_root / "demo_yolo_2"
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite\n"
    assert existing_config.read_text(encoding="utf-8") == "existing: true\n"
    assert Path(result["config_yaml"]) == configs_root / "demo_yolo_2.yaml"

    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: train.txt" in dataset_yaml
    assert "val: val.txt" in dataset_yaml
    assert "test: test.txt" in dataset_yaml
    assert (output_dir / "train.txt").exists()
    assert (output_dir / "val.txt").exists()
    assert (output_dir / "test.txt").exists()
    assert (output_dir / "images" / "train").exists()
    assert (output_dir / "labels" / "train").exists()

    fingerprint_csv = output_dir / "dataset_fingerprint.csv"
    fingerprint_json = output_dir / "dataset_fingerprint.json"
    assert Path(result["fingerprint_csv"]) == fingerprint_csv
    assert Path(result["fingerprint_json"]) == fingerprint_json
    assert fingerprint_csv.exists()
    payload = json.loads(fingerprint_json.read_text(encoding="utf-8"))
    assert payload["materialize_mode"] == MaterializeMode.TXT
    assert payload["item_count"] == 4
    assert payload["fingerprint"] == result["fingerprint"]
    assert "sha256" in fingerprint_csv.read_text(encoding="utf-8").splitlines()[0]


def _write_voc_xml(path: Path, image_name: str, class_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "<annotation>",
                f"  <filename>{image_name}</filename>",
                "  <size><width>100</width><height>100</height><depth>3</depth></size>",
                "  <object>",
                f"    <name>{class_name}</name>",
                "    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>80</ymax></bndbox>",
                "  </object>",
                "</annotation>",
                "",
            ]
        ),
        encoding="utf-8",
    )
