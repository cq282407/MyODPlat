#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : render.py
# @Project   : ODPlatform
# @Function  : Render validation reports for humans.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.report import ValidationReport

logger = logging.getLogger(__name__)

H1_LINE = "=" * 72
H2_LINE = "-" * 72


def render_to_logger(report: ValidationReport, log: logging.Logger | None = None) -> None:
    """Render a concise terminal report."""

    log = log or logger
    _render_header(report, log)
    _render_dataset_summary(report, log)
    _render_check_overview(report, log)
    _render_recommendations(report, log)
    if report.failed_results:
        _render_failure_details(report, log)
    _render_footer(report, log)


def _render_header(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H1_LINE)
    log.info("                    ODPlatform Data Validation Report")
    log.info(H1_LINE)
    log.info("  run_id    %s", report.run_id)
    log.info("  yaml      %s", report.yaml_path)
    log.info("  operator  %s", report.operator or "N/A")
    log.info("  role      %s", report.operator_role or "N/A")
    log.info("  device    %s", report.device_tag or "N/A")
    log.info(
        "  task      %-8s  duration %.2fs  severity %s  exit_code %s",
        report.snapshot.task_type,
        report.duration_seconds,
        report.overall_severity,
        report.exit_code,
    )


def _render_dataset_summary(report: ValidationReport, log: logging.Logger) -> None:
    snapshot = report.snapshot
    log.info(H2_LINE)
    log.info("  Dataset Summary")

    if snapshot.class_names:
        names = ", ".join(snapshot.class_names)
        log.info("    classes: %s (nc=%s)", names, snapshot.nc)
    else:
        log.info("    classes: unavailable")

    if not snapshot.stats_per_split:
        log.info("    no split statistics available")
        return

    for split in snapshot.splits:
        stat = snapshot.stats_per_split[split]
        log.info(
            "    %-6s: %6s images / %6s labels / %6s annotated / %6s instances",
            split,
            f"{stat.image_count:,}",
            f"{stat.label_count:,}",
            f"{stat.annotated_count:,}",
            f"{stat.total_instances:,}",
        )


def _render_check_overview(report: ValidationReport, log: logging.Logger) -> None:
    counts = report.severity_counts()
    log.info(H2_LINE)
    log.info("  Check Overview")
    log.info(
        "    summary: %s PASS / %s INFO / %s WARNING / %s ERROR",
        counts[CheckSeverity.PASS],
        counts[CheckSeverity.INFO],
        counts[CheckSeverity.WARNING],
        counts[CheckSeverity.ERROR],
    )
    for result in report.results:
        log.info("    [%-7s]  %-24s  %s", result.severity, result.name, result.summary)


def _render_recommendations(report: ValidationReport, log: logging.Logger) -> None:
    recommendations = report.recommendations or []
    if not recommendations:
        return
    log.info(H2_LINE)
    log.info("  Recommendations")
    for item in recommendations[:10]:
        log.info(
            "    [%-7s] %-24s %s",
            item.get("severity", ""),
            item.get("check", ""),
            item.get("suggestion", ""),
        )


def _render_failure_details(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H2_LINE)
    log.info("  Failed / Warning Details")
    for result in report.failed_results:
        log.info("")
        log.info("    >> %s [%s]", result.name, result.severity)
        _render_details_preview(result.details, log)


def _render_details_preview(details: dict[str, Any], log: logging.Logger) -> None:
    for key in (
        "problems",
        "problems_preview",
        "missing_labels_preview",
        "orphan_labels_preview",
        "outside_preview",
        "duplicates_preview",
        "tiny_preview",
        "huge_preview",
        "errors",
        "warnings",
    ):
        values = details.get(key)
        if not values:
            continue
        log.info("        %s:", key)
        if isinstance(values, dict):
            values = list(values.items())
        for item in list(values)[:5]:
            log.info("          - %s", _short_item(item))
    if "reason" in details:
        log.info("        reason: %s", details["reason"])


def _short_item(item: Any) -> str:
    if isinstance(item, dict):
        path = item.get("path") or item.get("file") or ""
        line = item.get("line") or item.get("duplicate_line") or ""
        reason = item.get("reason") or item.get("issue") or item.get("error") or ""
        if path or line or reason:
            return f"{Path(str(path)).name}:{line} {reason}".strip()
    return str(item)


def _render_footer(report: ValidationReport, log: logging.Logger) -> None:
    log.info(H2_LINE)
    log.info("  JSON report:      %s", report.report_path)
    log.info("  Data dictionary:  %s", report.data_dictionary_path)
    log.info("  Markdown report:  %s", report.markdown_path)
    log.info("  HTML report:      %s", report.html_path)
    log.info("  Word report:      %s", report.word_path)
    log.info("  Repair CSV:       %s", report.repair_csv_path)
    log.info("  Repair Excel:     %s", report.repair_excel_path)
    log.info("  Audit JSON:       %s", report.audit_path)
    log.info("  Recommendations:  %s", report.recommendations_path)
    log.info("  Charts:           %s", report.charts_dir)
    log.info(H1_LINE)


def render_markdown(report: ValidationReport) -> str:
    """Render a compact Markdown report."""

    lines = [
        "# ODPlatform Data Validation Report",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- YAML: `{report.yaml_path}`",
        f"- Overall: **{report.overall_severity}**",
        f"- Exit code: `{report.exit_code}`",
        f"- Operator: `{report.operator or 'N/A'}`",
        f"- Operator role: `{report.operator_role or 'N/A'}`",
        f"- Device tag: `{report.device_tag or 'N/A'}`",
        f"- Operation: `{report.operation}`",
        f"- Notes: `{report.notes or 'N/A'}`",
        "",
        "## Audit",
        "",
        f"- Login user: `{(report.audit or {}).get('user', {}).get('login', 'N/A')}`",
        f"- Hostname: `{(report.audit or {}).get('device', {}).get('hostname', 'N/A')}`",
        f"- Python: `{(report.audit or {}).get('process', {}).get('python_executable', 'N/A')}`",
        f"- CWD: `{(report.audit or {}).get('process', {}).get('cwd', 'N/A')}`",
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

    lines.extend(["", "## Recommendations", ""])
    for item in report.recommendations or []:
        lines.append(
            f"- **{item.get('check')}** [{item.get('severity')}]: "
            f"{item.get('suggestion')}"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON report: `{report.report_path}`",
            f"- HTML report: `{report.html_path}`",
            f"- Word report: `{report.word_path}`",
            f"- Data dictionary: `{report.data_dictionary_path}`",
            f"- Repair CSV: `{report.repair_csv_path}`",
            f"- Repair Excel: `{report.repair_excel_path}`",
            f"- Audit JSON: `{report.audit_path}`",
            f"- Recommendations JSON: `{report.recommendations_path}`",
            f"- Charts directory: `{report.charts_dir}`",
        ]
    )

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
