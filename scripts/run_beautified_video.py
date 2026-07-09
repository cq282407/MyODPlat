#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Run video inference with ODPlatform beautified Chinese labels.

This is a manual D8 demo:
  video -> frame_source stride sampling -> YOLO -> BeautifyVisualizer -> window/output video
Press q or Esc to exit the preview window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from od_platform.frame_source import create_frame_source
from od_platform.visualization import BeautifyVisualizer, Detection


DEFAULT_MODEL = Path("models/trained/train-8-20260707-200344-yolo11s-best.pt")
DEFAULT_VIDEO = Path("runs/d8_video_demo/nwpu_val_demo.mp4")
DEFAULT_OUTPUT = Path("runs/d8_video_demo/beautified_stride5/nwpu_val_demo_beautified.mp4")

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
    parser = argparse.ArgumentParser(description="Beautified video inference preview")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained .pt model")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Annotated output video path")
    parser.add_argument("--stride", type=int, default=5, help="Process one frame every N frames")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--font", default=None, help="Optional Chinese-capable .ttf/.ttc font path")
    parser.add_argument("--no-show", action="store_true", help="Save output without opening preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 means all")
    parser.add_argument(
        "--no-preserve-duration",
        dest="preserve_duration",
        action="store_false",
        default=True,
        help="Do not duplicate sampled frames to keep the output video duration close to the input",
    )
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    processed = 0
    preview_name = "ODPlatform Beautified Video"

    if not args.no_show:
        cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)

    try:
        with create_frame_source(args.video, stride=args.stride) as source:
            for frame in source:
                result = model(frame.image, conf=args.conf, verbose=False)[0]
                detections = _detections_from_result(result, names)
                annotated = visualizer.draw(frame.image, detections, use_label_mapping=True)
                annotated = _draw_frame_status(annotated, frame.info.frame_index, len(detections), args.stride)

                if writer is None:
                    height, width = annotated.shape[:2]
                    source_fps = frame.info.fps or 25.0
                    fps = source_fps if args.preserve_duration else max(0.1, source_fps / max(args.stride, 1))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open video writer: {output_path}")

                repeat_count = max(args.stride, 1) if args.preserve_duration else 1
                for _ in range(repeat_count):
                    writer.write(annotated)
                processed += 1

                print(
                    f"frame={frame.info.frame_index} detections={len(detections)} "
                    f"labels={','.join(det.label for det in detections) or '-'}"
                )

                if not args.no_show:
                    cv2.imshow(preview_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"processed_frames={processed}")
    print(f"saved={output_path.resolve()}")
    return 0


def _detections_from_result(result, names: dict[int, str]) -> list[Detection]:
    boxes = result.boxes
    detections: list[Detection] = []
    if boxes is None or not len(boxes):
        return detections

    for row in boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = row[:6]
        label = names.get(int(cls_id), f"cls_{int(cls_id)}")
        detections.append(
            Detection(
                box=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(conf),
                label=label,
                color=NWPU_COLOR_MAPPING.get(label, (200, 200, 200)),
            )
        )
    return detections


def _draw_frame_status(image, frame_index: int, count: int, stride: int):
    text = f"frame {frame_index} | stride {stride} | detections {count}"
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    raise SystemExit(main())
