#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : render.py
# @Project   : ODPlatform
# @Function  : Render validation reports for humans.

from __future__ import annotations

import logging
from pathlib import Path

from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.report import ValidationReport

logger = logging.getLogger(__name__)

H1_LINE = "=" * 72
H2_LINE = "-" * 72


def render_to_logger(report: ValidationReport, log: logging.Logger | None = None) -> None:
    """Render a teacher-style five-section terminal report."""

    log = log or logger
    _render_header(report, log)
    _render_dataset_summary(report, log)
    _render_check_overview(report, log)
    if report.failed_results:
        _render_failure_details(report, log)
    _render_footer(report, log)


def _render_header(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H1_LINE)
    log.info("                       YOLO 数据集验证报告")
    log.info(H1_LINE)
    log.info("  run_id   %s", report.run_id)
    log.info("  yaml     %s", report.yaml_path)
    log.info(
        "  task     %-8s  耗时  %.2fs  severity  %s  exit_code  %s",
        report.snapshot.task_type,
        report.duration_seconds,
        report.overall_severity,
        report.exit_code,
    )


def _render_dataset_summary(report: ValidationReport, log: logging.Logger) -> None:
    snapshot = report.snapshot
    log.info(H2_LINE)
    log.info("  数据集摘要")

    if snapshot.class_names:
        names = ", ".join(snapshot.class_names)
        log.info("    类别:  %s  (nc=%s)", names, snapshot.nc)
    else:
        log.info("    类别:  (未读取到，yaml_schema 应已报告)")

    if not snapshot.stats_per_split:
        log.info("    (无任何 split 可统计)")
        return

    for split in snapshot.splits:
        stat = snapshot.stats_per_split[split]
        log.info(
            "    %-6s:  %6s 张图  /  %6s 标注  /  %6s 实例",
            split,
            f"{stat.image_count:,}",
            f"{stat.annotated_count:,}",
            f"{stat.total_instances:,}",
        )


def _render_check_overview(report: ValidationReport, log: logging.Logger) -> None:
    counts = report.severity_counts()
    log.info(H2_LINE)
    log.info("  检查项一览")
    log.info(
        "    汇总: %s PASS / %s INFO / %s WARNING / %s ERROR",
        counts[CheckSeverity.PASS],
        counts[CheckSeverity.INFO],
        counts[CheckSeverity.WARNING],
        counts[CheckSeverity.ERROR],
    )
    for result in report.results:
        log.info("    [%-7s]  %-20s  %s", result.severity, result.name, result.summary)


def _render_failure_details(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H2_LINE)
    log.info("  失败详情")
    for result in report.failed_results:
        _render_one_check_details(result, log)


def _render_one_check_details(result, log: logging.Logger) -> None:
    log.info("")
    log.info("    >> %s  [%s]", result.name, result.severity)
    details = result.details

    if result.name == "yaml_schema":
        for problem in details.get("problems", []):
            log.info("        - %s", problem)
        if "reason" in details:
            log.info("        reason: %s", details["reason"])

    elif result.name == "pair_existence":
        _render_path_preview(log, "缺失标签", details.get("missing_labels_preview", []))
        _render_path_preview(log, "孤儿标签", details.get("orphan_labels_preview", []))
        missing = details.get("missing_per_split", {})
        if missing:
            parts = ", ".join(f"{split}={count}" for split, count in missing.items())
            log.info("        各 split 缺失: %s", parts)

    elif result.name == "label_format":
        if "error_kinds" in details:
            parts = ", ".join(f"{kind}={count}" for kind, count in details["error_kinds"].items())
            log.info("        错误类型: %s", parts)
        problems = details.get("problems_preview") or details.get("errors_preview") or []
        for item in problems[:5]:
            path = item.get("path") or item.get("label") or ""
            line = item.get("line") or item.get("line_no") or "?"
            reason = item.get("reason") or item.get("kind") or "unknown"
            raw = item.get("raw") or item.get("detail") or ""
            log.info("        - %s:%s  %s  %s", Path(str(path)).name, line, reason, raw)

    elif result.name == "split_uniqueness":
        overlaps = details.get("overlaps", [])
        if overlaps:
            for overlap in overlaps:
                log.info(
                    "        %s -> %s: %s 张重复",
                    overlap.get("split_a"),
                    overlap.get("split_b"),
                    overlap.get("count"),
                )
                for stem in overlap.get("preview", [])[:5]:
                    log.info("          %s", stem)
        else:
            preview = details.get("duplicate_stems_preview", {})
            for stem, splits in list(preview.items())[:5]:
                log.info("        %s: %s", stem, ", ".join(splits))

    elif result.name == "bbox_within_image":
        for item in details.get("outside_preview", [])[:5]:
            log.info(
                "        - %s:%s  cls=%s  xywh=(%.6f, %.6f, %.6f, %.6f)",
                Path(str(item["path"])).name,
                item["line"],
                item["class_id"],
                item["x_center"],
                item["y_center"],
                item["width"],
                item["height"],
            )

    elif result.name == "annotation_coverage":
        log.info(
            "        annotated=%s / total=%s, coverage=%.2f%%",
            details.get("annotated_images"),
            details.get("total_images"),
            float(details.get("coverage", 0.0)) * 100,
        )

    elif result.name == "class_presence":
        for problem in details.get("errors", [])[:5]:
            log.info("        - %s", problem)
        for warning in details.get("warnings", [])[:5]:
            log.info("        - %s", warning)

    elif "reason" in details:
        log.info("        reason: %s", details["reason"])


def _render_path_preview(log: logging.Logger, title: str, paths: list) -> None:
    if not paths:
        return
    log.info("        %s示例 (前 %s 条):", title, min(5, len(paths)))
    for path in paths[:5]:
        log.info("          %s", path)


def _render_footer(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H2_LINE)
    log.info("  详细报告:      %s", report.report_path)
    log.info("  数据字典:      %s", report.data_dictionary_path)
    log.info("  Markdown报告:  %s", report.markdown_path)
    log.info(H1_LINE)


def render_markdown(report: ValidationReport) -> str:
    """Render a compact Markdown report."""

    lines = [
        f"# ODPlatform Data Validation Report",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- YAML: `{report.yaml_path}`",
        f"- Overall: **{report.overall_severity}**",
        f"- Exit code: `{report.exit_code}`",
        f"- Operator: `{report.operator or 'N/A'}`",
        "",
        "## Data Dictionary",
        "",
        "| Split | Images | Labels | Annotated Images | Instances |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in report.snapshot.splits:
        stats = report.snapshot.stats_per_split[split]
        lines.append(
            f"| {split} | {stats.image_count} | {stats.label_count} | "
            f"{stats.annotated_count} | {stats.total_instances} |"
        )
    lines.extend(
        [
            "",
            "## Classes",
            "",
            "| ID | Name | Instances |",
            "| ---: | --- | ---: |",
        ]
    )
    class_counts = report.snapshot.class_instance_counts
    for class_id, class_name in enumerate(report.snapshot.class_names):
        lines.append(f"| {class_id} | {class_name} | {class_counts.get(class_name, 0)} |")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Severity | Summary |",
            "| --- | --- | --- |",
        ]
    )
    for result in report.results:
        lines.append(f"| `{result.name}` | {result.severity} | {result.summary} |")

    blocking = [
        result
        for result in report.results
        if result.severity in {CheckSeverity.ERROR, CheckSeverity.WARNING}
    ]
    if blocking:
        lines.extend(["", "## Attention", ""])
        for result in blocking:
            lines.append(f"- **{result.name}** [{result.severity}]: {result.summary}")

    return "\n".join(lines) + "\n"
