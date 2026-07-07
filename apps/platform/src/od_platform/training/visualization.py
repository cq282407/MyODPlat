#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : visualization.py
# @Project   : ODPlatform
# @Function  : Render training results.csv visualization.
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def render_training_results_chart(
    train_dir: Path,
    *,
    output_name: str = "training_results.png",
    logger_: logging.Logger | None = None,
) -> Path | None:
    """Render a 3x2 training-process chart from ultralytics results.csv."""

    log = logger_ or logger
    csv_path = Path(train_dir) / "results.csv"
    if not csv_path.exists():
        log.warning("未找到 results.csv, 跳过训练曲线可视化: %s", csv_path)
        return None

    rows = _read_results_csv(csv_path)
    if not rows:
        log.warning("results.csv 为空, 跳过训练曲线可视化: %s", csv_path)
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib 未安装, 跳过训练曲线可视化")
        return None

    text = _chart_text(plt)
    epochs = _series(rows, "epoch")
    if not epochs:
        epochs = [float(index) for index in range(len(rows))]
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(text["suptitle"], fontsize=16, y=0.995)

    _plot_lines(
        axes[0, 0],
        epochs,
        rows,
        [
            ("train/box_loss", "Train Box Loss"),
            ("val/box_loss", "Val Box Loss"),
            ("train/cls_loss", "Train CLS Loss"),
            ("val/cls_loss", "Val CLS Loss"),
            ("train/dfl_loss", "Train DFL Loss"),
            ("val/dfl_loss", "Val DFL Loss"),
        ],
        text["loss_title"],
        "Loss",
    )
    _plot_lines(
        axes[0, 1],
        epochs,
        rows,
        [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP@50"),
            ("metrics/mAP50-95(B)", "mAP@50-95"),
        ],
        text["metric_title"],
        "Score",
    )
    _plot_lines(
        axes[1, 0],
        epochs,
        rows,
        [("lr/pg0", "lr/pg0"), ("lr/pg1", "lr/pg1"), ("lr/pg2", "lr/pg2")],
        text["lr_title"],
        "Learning Rate",
    )

    ax4 = axes[1, 1]
    train_total = _sum_columns(rows, ("train/box_loss", "train/cls_loss", "train/dfl_loss"))
    val_total = _sum_columns(rows, ("val/box_loss", "val/cls_loss", "val/dfl_loss"))
    if train_total:
        ax4.plot(epochs, train_total, label="Train Total Loss")
    if val_total:
        ax4.plot(epochs, val_total, label="Val Total Loss")
    _finish_axis(ax4, text["total_loss_title"], "Total Loss")

    ax5 = axes[2, 0]
    time_series = _series(rows, "time")
    if time_series:
        ax5.plot(epochs, time_series, marker=".", markersize=2)
    _finish_axis(ax5, text["time_title"], "Time (s)")

    ax6 = axes[2, 1]
    map50 = _series(rows, "metrics/mAP50(B)")
    map50_95 = _series(rows, "metrics/mAP50-95(B)")
    if map50:
        ax6.plot(epochs, map50, label="mAP@50")
        ax6.axhline(_mean(map50), linestyle="--", alpha=0.7, label=f"mAP@50 Avg: {_mean(map50):.4f}")
    if map50_95:
        ax6.plot(epochs, map50_95, label="mAP@50-95")
        ax6.axhline(
            _mean(map50_95),
            linestyle="--",
            alpha=0.7,
            label=f"mAP@50-95 Avg: {_mean(map50_95):.4f}",
        )
    _finish_axis(ax6, text["map_title"], "mAP Score")

    fig.tight_layout()
    output_path = Path(train_dir) / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("训练结果可视化已写入: %s", output_path)
    return output_path


def _read_results_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                normalized_key = key.strip() if key is not None else ""
                if not normalized_key:
                    continue
                try:
                    row[normalized_key] = float(str(value).strip())
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def _series(rows: list[dict[str, float]], column: str) -> list[float]:
    return [row[column] for row in rows if column in row]


def _sum_columns(rows: list[dict[str, float]], columns: Iterable[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if all(column in row for column in columns):
            values.append(sum(row[column] for column in columns))
    return values


def _plot_lines(ax, epochs: list[float], rows: list[dict[str, float]], columns: list[tuple[str, str]], title: str, ylabel: str) -> None:
    for column, label in columns:
        values = _series(rows, column)
        if values:
            ax.plot(epochs[: len(values)], values, label=label)
    _finish_axis(ax, title, ylabel)


def _finish_axis(ax, title: str, ylabel: str) -> None:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()
    ax.grid(True)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _chart_text(plt) -> dict[str, str]:
    if _configure_cjk_font(plt):
        return {
            "suptitle": "目标检测模型训练过程可视化",
            "loss_title": "训练和验证损失曲线",
            "metric_title": "评估指标变化趋势",
            "lr_title": "学习率调度曲线",
            "total_loss_title": "总损失对比",
            "time_title": "每个 epoch 训练时间",
            "map_title": "mAP指标变化（含平均值）",
        }
    return {
        "suptitle": "Object Detection Training Curves",
        "loss_title": "Train and Validation Loss",
        "metric_title": "Evaluation Metrics",
        "lr_title": "Learning Rate Schedule",
        "total_loss_title": "Total Loss",
        "time_title": "Epoch Time",
        "map_title": "mAP Curves with Mean",
    }


def _configure_cjk_font(plt) -> bool:
    try:
        from matplotlib import font_manager
    except Exception:
        return False
    preferred = (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False
