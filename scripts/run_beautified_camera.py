#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Run camera inference with ODPlatform beautified Chinese labels.

This script is a manual smoke test for:
  camera -> YOLO -> BeautifyVisualizer -> OpenCV window
Press q or Esc to exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from od_platform.frame_source import CameraConfig, create_frame_source
from od_platform.visualization import BeautifyVisualizer, Detection


DEFAULT_MODEL = Path("models/trained/train-8-20260707-200344-yolo11s-best.pt")

NWPU_LABEL_MAPPING = {
    "airplane": "飞机",
    "ship": "船只",
    "storage_tank": "储油罐",
    "baseball_diamond": "棒球场",
    "tennis_court": "网球场",
    "basketball_court": "篮球场",
    "ground_track_field": "田径场",
    "harbor": "港口",
    "bridge": "桥梁",
    "vehicle": "车辆",
}

NWPU_COLOR_MAPPING = {
    "airplane": (0, 220, 255),
    "ship": (255, 160, 0),
    "storage_tank": (0, 180, 0),
    "baseball_diamond": (220, 0, 220),
    "tennis_court": (90, 220, 90),
    "basketball_court": (0, 128, 255),
    "ground_track_field": (180, 120, 0),
    "harbor": (255, 80, 80),
    "bridge": (120, 120, 255),
    "vehicle": (0, 0, 255),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Beautified camera inference")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained .pt model")
    parser.add_argument("--camera", default="0", help="Camera id, usually 0")
    parser.add_argument("--backend", default="msmf", choices=["auto", "msmf", "dshow", "v4l2", "avfoundation"])
    parser.add_argument("--codec", default="MJPG", choices=["MJPG", "YUY2", "H264"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--font", default=None, help="Optional Chinese-capable .ttf/.ttc font path")
    args = parser.parse_args()

    model = YOLO(args.model)
    names = model.names or {}
    labels = [names[index] for index in sorted(names)]
    visualizer = BeautifyVisualizer(
        labels=labels,
        label_mapping=NWPU_LABEL_MAPPING,
        color_mapping=NWPU_COLOR_MAPPING,
        default_color=(200, 200, 200),
        font_path=args.font,
    )

    config = CameraConfig(
        camera_id=int(args.camera),
        backend=args.backend,
        codec=args.codec,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    with create_frame_source(args.camera, camera_config=config) as source:
        print("Camera opened. Press q or Esc to exit.")
        for frame in source:
            result = model(frame.image, conf=args.conf, verbose=False)[0]
            boxes = result.boxes
            detections: list[Detection] = []
            if boxes is not None and len(boxes):
                data = boxes.data.cpu().numpy()
                for row in data:
                    x1, y1, x2, y2, conf, cls_id = row[:6]
                    label = names.get(int(cls_id), f"cls_{int(cls_id)}")
                    detections.append(
                        Detection(
                            box=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            label=label,
                        )
                    )

            annotated = visualizer.draw(frame.image, detections, use_label_mapping=True)
            cv2.imshow("ODPlatform Beautified Camera", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
