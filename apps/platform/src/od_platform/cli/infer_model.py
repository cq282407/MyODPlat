#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : infer_model.py
# @Project   : ODPlatform
# @Function  : odp-infer CLI.
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.inference import infer_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-infer",
        description="ODPlatform YOLO 推理 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, default=None, help="输入源: 摄像头号 / 视频文件 / 图片文件或目录 / RTSP")
    parser.add_argument("--model", type=str, default=None, help="模型文件名或路径 (.pt / .engine)")
    parser.add_argument("--yaml", type=str, default=None, help="D5 infer.yaml 路径")
    parser.add_argument("--pipeline-yaml", type=str, default=None, help="infer_pipeline.yaml 路径")
    parser.add_argument("--name", type=str, default=None, dest="experiment_name", help="输出子目录名")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸")
    parser.add_argument("--max-det", type=int, default=None, dest="max_det", help="单图最大检测数")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="只保留这些类别 ID")
    parser.add_argument("--device", type=str, default=None, help="cpu / 0 / 0,1 / mps")
    parser.add_argument("--batch", type=int, default=None, help="批大小 (视频/图片目录)")
    parser.add_argument("--vid-stride", type=int, default=None, dest="vid_stride", help="视频抽帧间隔")
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", action="store_true", default=None, help="弹窗显示画面")
    show_group.add_argument("--no-show", dest="show", action="store_false", default=None, help="关闭弹窗显示")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save", action="store_true", default=None, help="保存结果")
    save_group.add_argument("--no-save", dest="save", action="store_false", default=None, help="不保存结果")
    parser.add_argument("--save-txt", action="store_true", default=None, dest="save_txt", help="保存 txt 预测")
    parser.add_argument("--save-conf", action="store_true", default=None, dest="save_conf", help="txt 中附带置信度")
    parser.add_argument("--save-crop", action="store_true", default=None, dest="save_crop", help="保存裁剪目标")
    parser.add_argument("--save-frames", action="store_true", default=None, dest="save_frames", help="视频逐帧保存")
    parser.add_argument("--line-width", type=int, default=None, dest="line_width", help="绘制线宽")
    parser.add_argument("--embed", type=int, nargs="+", default=None, help="提取 embedding 层索引")
    parser.add_argument("--stream", action="store_true", default=None, help="启用流式推理")
    parser.add_argument("--stream-buffer", action="store_true", default=None, dest="stream_buffer", help="缓冲流帧")
    parser.add_argument("--visualize", action="store_true", default=None, help="保存特征图")
    parser.add_argument("--no-viz", action="store_true", help="关闭美化绘制，退回 YOLO 原生 plot()")
    parser.add_argument("--no-hud", action="store_true", help="画面不叠加 HUD FPS 面板")
    parser.add_argument("--warmup", type=int, default=0, help="启动丢弃前 N 帧")
    parser.add_argument("--project", type=str, default=None, help="Ultralytics project/output root")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--no-rename-log", dest="rename_log", action="store_false", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    get_logger(base_path=LOGGING_DIR, log_type="infer", log_level=getattr(logging, args.log_level))
    log = logging.getLogger(__name__)

    cli_args = _ns_to_cli_args(args)
    try:
        result = infer_yolo(
            yaml_path=args.yaml,
            pipeline_yaml=args.pipeline_yaml,
            cli_args=cli_args,
            beautify=(not args.no_viz),
            rename_log=args.rename_log,
            warmup_frames=args.warmup,
            show_info=(not args.no_hud),
        )
    except KeyboardInterrupt:
        log.warning("用户中断 (Ctrl+C)")
        return 130
    if result.success:
        return 0
    sys.stderr.write(f"\n推理失败: {result.error}\n")
    if result.log_path:
        sys.stderr.write(f"详细日志见: {result.log_path}\n")
    return 1


def _ns_to_cli_args(ns: argparse.Namespace) -> dict[str, object]:
    keys = (
        "source",
        "model",
        "experiment_name",
        "conf",
        "iou",
        "imgsz",
        "max_det",
        "classes",
        "device",
        "batch",
        "project",
        "show",
        "save",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "line_width",
        "embed",
        "stream",
        "stream_buffer",
        "visualize",
        "vid_stride",
    )
    return {key: value for key in keys if (value := getattr(ns, key, None)) is not None}


if __name__ == "__main__":
    sys.exit(main())



