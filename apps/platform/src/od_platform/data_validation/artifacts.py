#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : artifacts.py
# @Project   : ODPlatform
# @Function  : Extra data validation artifacts: advice, repair tables, HTML and SVG charts.

from __future__ import annotations

import csv
import html
import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.report import ValidationReport


def artifact_manifest(report: ValidationReport) -> dict[str, str]:
    return {
        "json_report": str(report.report_path),
        "markdown_report": str(report.markdown_path),
        "html_report": str(report.html_path),
        "data_dictionary": str(report.data_dictionary_path),
        "recommendations": str(report.recommendations_path),
        "audit": str(report.audit_path),
        "repair_csv": str(report.repair_csv_path),
        "repair_excel": str(report.repair_excel_path),
        "charts_dir": str(report.charts_dir),
    }


def build_recommendations(report: ValidationReport) -> list[dict[str, Any]]:
    """Build actionable quality advice from check results."""

    advice: list[dict[str, Any]] = []
    for result in report.results:
        severity = result.severity
        if severity == CheckSeverity.PASS:
            continue
        advice.append(
            {
                "check": result.name,
                "severity": severity,
                "summary": result.summary,
                "suggestion": _suggestion_for(result.name, result.details),
            }
        )
    if not advice:
        advice.append(
            {
                "check": "overall",
                "severity": CheckSeverity.PASS,
                "summary": "All quality gates passed.",
                "suggestion": "The dataset can be handed to training. Keep this report with the run record.",
            }
        )
    return advice


def write_extra_artifacts(report: ValidationReport) -> None:
    report.recommendations_path.write_text(
        json.dumps(report.recommendations or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report.audit_path.write_text(
        json.dumps(report.audit or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    repair_rows = build_repair_rows(report)
    write_repair_csv(report.repair_csv_path, repair_rows)
    write_repair_xlsx(report.repair_excel_path, repair_rows)
    write_charts(report)
    report.html_path.write_text(render_to_html(report), encoding="utf-8")


def build_repair_rows(report: ValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.results:
        if result.severity not in {CheckSeverity.WARNING, CheckSeverity.ERROR}:
            continue
        rows.extend(_rows_for_result(result.name, result.severity, result.summary, result.details))
    return rows


def write_repair_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "severity",
        "check",
        "split",
        "file",
        "line",
        "class_id",
        "issue",
        "suggestion",
        "raw",
        "details",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def write_repair_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a tiny XLSX workbook using only the standard library."""

    headers = [
        "severity",
        "check",
        "split",
        "file",
        "line",
        "class_id",
        "issue",
        "suggestion",
        "raw",
        "details",
    ]
    table = [headers] + [[str(row.get(key, "")) for key in headers] for row in rows]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", _xlsx_content_types())
        workbook.writestr("_rels/.rels", _xlsx_root_rels())
        workbook.writestr("xl/workbook.xml", _xlsx_workbook())
        workbook.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_rels())
        workbook.writestr("xl/worksheets/sheet1.xml", _xlsx_sheet(table))


def write_charts(report: ValidationReport) -> None:
    report.charts_dir.mkdir(parents=True, exist_ok=True)
    (report.charts_dir / "split_summary.svg").write_text(_split_chart_svg(report), encoding="utf-8")
    (report.charts_dir / "class_distribution.svg").write_text(_class_chart_svg(report), encoding="utf-8")
    (report.charts_dir / "check_severity.svg").write_text(_severity_chart_svg(report), encoding="utf-8")
    (report.charts_dir / "bbox_area_profile.svg").write_text(_bbox_chart_svg(report), encoding="utf-8")


def render_to_html(report: ValidationReport) -> str:
    counts = report.severity_counts()
    recommendations = report.recommendations or []
    charts = [
        "charts/split_summary.svg",
        "charts/class_distribution.svg",
        "charts/check_severity.svg",
        "charts/bbox_area_profile.svg",
    ]
    check_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(result.name)}</td>"
        f"<td><span class='sev {result.severity.lower()}'>{result.severity}</span></td>"
        f"<td>{html.escape(result.summary)}</td>"
        "</tr>"
        for result in report.results
    )
    rec_items = "\n".join(
        "<li>"
        f"<strong>{html.escape(item['check'])}</strong> "
        f"[{html.escape(item['severity'])}]: "
        f"{html.escape(item['suggestion'])}"
        "</li>"
        for item in recommendations
    )
    chart_imgs = "\n".join(
        f"<figure><img src='{chart}' alt='{chart}'><figcaption>{html.escape(chart)}</figcaption></figure>"
        for chart in charts
    )
    operator = html.escape(report.operator or "N/A")
    role = html.escape(report.operator_role or "N/A")
    device = html.escape(report.device_tag or "N/A")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ODPlatform Data Validation Report - {html.escape(report.run_id)}</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .meta {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 8px 24px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #eef2ff; }}
    .sev.pass {{ color: #047857; font-weight: 700; }}
    .sev.info {{ color: #0369a1; font-weight: 700; }}
    .sev.warning {{ color: #b45309; font-weight: 700; }}
    .sev.error {{ color: #b91c1c; font-weight: 700; }}
    figure {{ margin: 16px 0; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>ODPlatform Data Validation Report</h1>
  <p><span class="badge">{html.escape(report.overall_severity)}</span> exit_code={report.exit_code}</p>
  <section class="meta">
    <div><strong>Run ID:</strong> {html.escape(report.run_id)}</div>
    <div><strong>Started:</strong> {html.escape(report.started_at_iso)}</div>
    <div><strong>Operator:</strong> {operator}</div>
    <div><strong>Role:</strong> {role}</div>
    <div><strong>Device:</strong> {device}</div>
    <div><strong>YAML:</strong> <code>{html.escape(str(report.yaml_path))}</code></div>
    <div><strong>Counts:</strong> PASS={counts[CheckSeverity.PASS]},
      INFO={counts[CheckSeverity.INFO]},
      WARNING={counts[CheckSeverity.WARNING]},
      ERROR={counts[CheckSeverity.ERROR]}</div>
  </section>

  <h2>Recommendations</h2>
  <ul>{rec_items}</ul>

  <h2>Charts</h2>
  {chart_imgs}

  <h2>Checks</h2>
  <table>
    <thead><tr><th>Check</th><th>Severity</th><th>Summary</th></tr></thead>
    <tbody>{check_rows}</tbody>
  </table>

  <h2>Artifacts</h2>
  <ul>
    <li><code>report.json</code></li>
    <li><code>report.md</code></li>
    <li><code>repair_items.csv</code></li>
    <li><code>repair_items.xlsx</code></li>
    <li><code>audit.json</code></li>
    <li><code>recommendations.json</code></li>
  </ul>
</body>
</html>
"""


def _suggestion_for(check_name: str, details: dict[str, Any]) -> str:
    suggestions = {
        "yaml_schema": "Fix dataset.yaml fields first: path, train, val, nc and names must be consistent.",
        "pair_existence": "Ask the data owner to add missing label files or remove orphan labels before training.",
        "image_integrity": "Replace unreadable or zero-byte images and rerun validation.",
        "label_format": "Return listed label files to the annotation team and enforce YOLO class x y w h format.",
        "split_uniqueness": "Move duplicated samples so one image stem appears in only one split.",
        "bbox_within_image": "Clip or relabel boxes that extend outside image boundaries.",
        "duplicate_annotations": "Review duplicated rows; keep only one annotation for the same object.",
        "annotation_coverage": "Check whether empty labels are true background samples or missed annotations.",
        "class_presence": "Add or rebalance samples so every class appears in train and preferably val/test.",
        "class_balance": "Consider targeted collection, class-aware split, sampling weights or augmentation for rare classes.",
        "bbox_area_profile": "Review tiny or huge boxes; they may be labeling mistakes or require special training settings.",
    }
    return suggestions.get(check_name, f"Review details for {check_name}: {json.dumps(details, ensure_ascii=False)[:160]}")


def _rows_for_result(
    check_name: str,
    severity: str,
    summary: str,
    details: dict[str, Any],
) -> list[dict[str, Any]]:
    suggestion = _suggestion_for(check_name, details)
    rows: list[dict[str, Any]] = []

    def add(item: dict[str, Any], issue: str | None = None) -> None:
        rows.append(
            {
                "severity": severity,
                "check": check_name,
                "split": item.get("split", ""),
                "file": item.get("path") or item.get("label") or item.get("file") or "",
                "line": item.get("line") or item.get("line_no") or item.get("duplicate_line") or "",
                "class_id": item.get("class_id", ""),
                "issue": issue or item.get("reason") or summary,
                "suggestion": suggestion,
                "raw": item.get("raw", ""),
                "details": json.dumps(item, ensure_ascii=False),
            }
        )

    for key in (
        "problems_preview",
        "outside_preview",
        "duplicates_preview",
        "tiny_preview",
        "huge_preview",
    ):
        for item in details.get(key, []) or []:
            add(item)

    for key, issue in (
        ("missing_labels_preview", "missing label file"),
        ("orphan_labels_preview", "orphan label file"),
    ):
        for value in details.get(key, []) or []:
            split, _, stem = str(value).partition("/")
            add({"split": split, "path": stem}, issue)

    for problem in details.get("problems", []) or []:
        add({"path": details.get("yaml_path", ""), "reason": str(problem)}, str(problem))
    for warning in details.get("warnings", []) or []:
        add({"reason": str(warning)}, str(warning))
    for error in details.get("errors", []) or []:
        add({"reason": str(error)}, str(error))

    if not rows and severity in {CheckSeverity.WARNING, CheckSeverity.ERROR}:
        add({"details": details}, summary)
    return rows


def _split_chart_svg(report: ValidationReport) -> str:
    values = [
        (split, report.snapshot.stats_per_split[split].image_count)
        for split in report.snapshot.splits
    ]
    return _bar_chart_svg("Images per split", values, color="#2563eb")


def _class_chart_svg(report: ValidationReport) -> str:
    values = list(report.snapshot.class_instance_counts.items())
    return _bar_chart_svg("Instances per class", values, color="#059669", width=900)


def _severity_chart_svg(report: ValidationReport) -> str:
    counts = report.severity_counts()
    values = [(severity, counts[severity]) for severity in CheckSeverity.all()]
    return _bar_chart_svg("Validation severity counts", values, color="#7c3aed")


def _bbox_chart_svg(report: ValidationReport) -> str:
    result = next((item for item in report.results if item.name == "bbox_area_profile"), None)
    if result is None or "instances" not in result.details:
        return _empty_svg("BBox area profile", "No bbox profile is available.")
    values = [
        ("tiny", int(result.details.get("tiny_count", 0))),
        ("normal", max(0, int(result.details.get("instances", 0)) - int(result.details.get("tiny_count", 0)) - int(result.details.get("huge_count", 0)))),
        ("huge", int(result.details.get("huge_count", 0))),
    ]
    return _bar_chart_svg("BBox area profile", values, color="#dc2626")


def _bar_chart_svg(
    title: str,
    values: list[tuple[str, int]],
    *,
    color: str,
    width: int = 760,
    row_height: int = 30,
) -> str:
    if not values:
        return _empty_svg(title, "No data.")
    left = 190
    right = 36
    top = 52
    height = top + row_height * len(values) + 24
    max_value = max(value for _, value in values) or 1
    bar_max = width - left - right - 80
    rows = []
    for index, (label, value) in enumerate(values):
        y = top + index * row_height
        bar_width = int(bar_max * value / max_value)
        rows.append(
            f"<text x='12' y='{y + 18}' font-size='13'>{escape(str(label))}</text>"
            f"<rect x='{left}' y='{y + 5}' width='{bar_width}' height='18' fill='{color}'/>"
            f"<text x='{left + bar_width + 8}' y='{y + 19}' font-size='13'>{value}</text>"
        )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text x='12' y='30' font-size='18' font-weight='700'>{escape(title)}</text>"
        + "".join(rows)
        + "</svg>"
    )


def _empty_svg(title: str, message: str) -> str:
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='760' height='120'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<text x='12' y='32' font-size='18' font-weight='700'>{escape(title)}</text>"
        f"<text x='12' y='70' font-size='14'>{escape(message)}</text>"
        "</svg>"
    )


def _xlsx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""


def _xlsx_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _xlsx_workbook() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="repair_items" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def _xlsx_workbook_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""


def _xlsx_sheet(table: list[list[str]]) -> str:
    rows = []
    for row_index, row in enumerate(table, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            ref = f"{_excel_column(column_index)}{row_index}"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
            )
        rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(rows)
        + "</sheetData></worksheet>"
    )


def _excel_column(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result
