import base64
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from docx import Document
from docx.shared import Inches, Pt

from app.tools.data_tools import (
    get_dataset_profile,
    get_dataset_statistics
)
from app.tools.ml_tools import train_model
from app.tools.anomaly_tools import detect_anomalies
from app.tools.analysis_tools import rank_categories
from app.tools.trend_tools import analyze_trends, compare_periods
from app.tools.visualization_tools import create_visualization


# ---------------------------------------------------------------------
# Reporting Skill
# ---------------------------------------------------------------------
# Ties Dataset Profiling, Statistics, and (optionally) the ML Skill
# together into one "Generate full analysis report" tool call.
#
# Design choice: this tool does all the orchestration itself in plain
# Python (profile -> stats -> optional model training -> write files),
# rather than asking the LLM to plan and call four separate tools in
# sequence. That keeps "analyze this dataset completely" to a single
# tool call, which matters a lot for latency on CPU-only local models.
# The LLM still decides *when* to call this tool and still writes the
# final natural-language explanation - the agentic decision-making
# stays with the LLM, only the internal report assembly is scripted.
#
# The dict returned to the LLM is intentionally compact (headline
# numbers + a few key insights + file paths), not the full report
# text, so the second LLM call doesn't have to chew through a large
# tool result.

def _build_markdown(profile, stats, ml_result, target_column):
    lines = []

    lines.append("# Automatic Data Analysis Report")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- Rows: {profile['rows']}")
    lines.append(f"- Columns: {profile['columns']}")
    lines.append(
        f"- Numeric columns: {', '.join(profile['numeric_columns']) or 'none'}"
    )
    lines.append(
        f"- Categorical columns: "
        f"{', '.join(profile['categorical_columns']) or 'none'}"
    )
    lines.append(f"- Duplicate rows: {profile['duplicate_rows']}")
    lines.append("")

    lines.append("## Data Quality")
    lines.append("")
    missing = {
        column: percent
        for column, percent in profile["missing_percent_by_column"].items()
        if percent > 0
    }
    if missing:
        lines.append("Missing values by column:")
        lines.append("")
        for column, percent in missing.items():
            lines.append(f"- {column}: {percent}%")
    else:
        lines.append("No missing values detected.")
    lines.append("")

    if profile["outlier_counts_by_column"]:
        lines.append("Outliers detected (IQR method):")
        lines.append("")
        for column, count in profile["outlier_counts_by_column"].items():
            lines.append(f"- {column}: {count} potential outlier(s)")
    else:
        lines.append("No significant outliers detected.")
    lines.append("")

    lines.append("## Key Statistics")
    lines.append("")
    if isinstance(stats, dict) and "message" in stats:
        lines.append(stats["message"])
    else:
        for column, values in stats.items():
            mean = values.get("mean")
            std = values.get("std")
            if mean is not None:
                lines.append(
                    f"- {column}: mean={round(mean, 3)}, "
                    f"std={round(std, 3) if std is not None else 'n/a'}"
                )
    lines.append("")

    lines.append("## Correlations")
    lines.append("")
    if profile["strongest_correlations"]:
        for pair in profile["strongest_correlations"]:
            col_a, col_b = pair["columns"]
            lines.append(
                f"- {col_a} vs {col_b}: r = {pair['correlation']}"
            )
    else:
        lines.append("No strong correlations found.")
    lines.append("")

    if profile["candidate_target_columns"]:
        lines.append("## Possible Target Columns")
        lines.append("")
        for column in profile["candidate_target_columns"]:
            lines.append(f"- {column}")
        lines.append("")

    if ml_result:
        lines.append("## Machine Learning Results")
        lines.append("")
        if "error" in ml_result:
            lines.append(f"Model training was skipped: {ml_result['error']}")
        else:
            lines.append(f"- Target column: {target_column}")
            lines.append(f"- Task type: {ml_result['task_type']}")
            lines.append(
                f"- Train / test rows: "
                f"{ml_result['train_rows']} / {ml_result['test_rows']}"
            )
            for metric_name, metric_value in ml_result["metrics"].items():
                if metric_name == "confusion_matrix":
                    continue
                lines.append(f"- {metric_name}: {metric_value}")
            lines.append("")
            lines.append("Top features:")
            for feature in ml_result["top_features"][:5]:
                lines.append(
                    f"- {feature['feature']}: {feature['importance']}"
                )
            lines.append("")
            lines.append(f"_Note: {ml_result['note']}_")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- This report is generated automatically from the uploaded "
        "CSV only; it does not account for context outside the data."
    )
    lines.append(
        "- Any feature importance or correlation above reflects "
        "association within this dataset, not causation."
    )
    if ml_result and "error" not in ml_result:
        lines.append(
            "- The model is a baseline Random Forest with default "
            "settings; it is meant to establish a reference point, "
            "not a production-ready model."
        )

    return "\n".join(lines)


_IMAGE_LINE_PATTERN = re.compile(r"^!\[(.*?)\]\((.*?)\)$")


def _image_to_data_uri(path: str) -> str:
    """Base64-embed a chart PNG so the HTML report is self-contained
    and displays correctly whether opened from outputs/, emailed as
    an attachment, or served from a different working directory -
    a relative <img src="outputs/x.png"> would silently break in all
    of those cases."""
    try:
        with open(path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _markdown_to_html(markdown_text: str) -> str:
    """Minimal, dependency-free Markdown -> HTML conversion, just
    enough for headers, bullets, paragraphs, and chart images so the
    report is readable in a browser without adding a markdown
    library."""

    html_lines = ["<html><head><meta charset='utf-8'>",
                  "<title>Data Analysis Report</title></head><body>"]

    in_list = False

    for raw_line in markdown_text.split("\n"):
        line = raw_line.strip()

        if not line:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        image_match = _IMAGE_LINE_PATTERN.match(line)
        if image_match:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            alt_text, image_path = image_match.groups()
            data_uri = _image_to_data_uri(image_path)
            if data_uri:
                html_lines.append(
                    f'<img src="{data_uri}" alt="{alt_text}" '
                    f'style="max-width:100%;margin:0.75rem 0;'
                    f'border:1px solid #e2e8f0;border-radius:8px;">'
                )
            continue

        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("_") and line.endswith("_"):
            html_lines.append(f"<p><em>{line[1:-1]}</em></p>")
        else:
            html_lines.append(f"<p>{line}</p>")

    if in_list:
        html_lines.append("</ul>")

    html_lines.append("</body></html>")

    return "\n".join(html_lines)


_PDF_INK = (30, 41, 59)          # body text
_PDF_HEADING = (27, 58, 75)      # #1B3A4B - matches the app's accent color
_PDF_MUTED = (100, 116, 139)     # italic notes / footer
_PDF_RULE = (226, 232, 240)      # light divider under ## headings


class _ReportPDF(FPDF):
    """FPDF subclass so every page automatically gets a thin footer
    rule and page number - the missing polish that made the plain
    version feel like a text dump rather than a document."""

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*_PDF_RULE)
        self.line(18, self.get_y(), self.w - 18, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_PDF_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _markdown_to_pdf(markdown_text: str, path: str) -> None:
    """Render the same simple markdown subset used for the HTML
    report (#, ##, - bullets, _italic_ lines, plain paragraphs) into
    a real, readably-formatted PDF: colored section headings with a
    divider rule, breathing room between sections, and an indented
    block for bullets - instead of every line packed at the same
    size and spacing, which is what made the first version feel like
    everything was "stuck together" with nothing standing out.

    Uses fpdf2 - pure Python, no system-level dependencies (unlike
    e.g. WeasyPrint, which needs Cairo/Pango installed separately) -
    so this works the same on any machine that can `pip install`.
    """
    BODY_X = 18
    BULLET_X = 24  # indented block so bullets read as a distinct group

    pdf = _ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    pdf.set_text_color(*_PDF_INK)

    first_heading = True

    for raw_line in markdown_text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        image_match = _IMAGE_LINE_PATTERN.match(line)
        if image_match:
            _, image_path = image_match.groups()
            if os.path.exists(image_path):
                # Fit within the body width; fpdf2 preserves aspect
                # ratio when only one of w/h is given.
                available_width = pdf.w - BODY_X - 18
                pdf.image(image_path, x=BODY_X, w=available_width)
                pdf.ln(4)
            continue

        pdf.set_x(BODY_X)

        if line.startswith("# "):
            # Title band: large, colored, with a rule underneath and
            # generous space below before the first section starts.
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(*_PDF_HEADING)
            pdf.multi_cell(0, 11, line[2:])
            pdf.set_draw_color(*_PDF_HEADING)
            pdf.set_line_width(0.6)
            pdf.line(BODY_X, pdf.get_y() + 1, pdf.w - 18, pdf.get_y() + 1)
            pdf.set_line_width(0.2)
            pdf.ln(6)
            pdf.set_text_color(*_PDF_INK)

        elif line.startswith("## "):
            # Real section break: extra space above (except right
            # after the title), bold colored label, thin rule below
            # so each section is visually distinct at a glance.
            if not first_heading:
                pdf.ln(4)
            first_heading = False
            pdf.set_x(BODY_X)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(*_PDF_HEADING)
            pdf.multi_cell(0, 8, line[3:])
            pdf.set_draw_color(*_PDF_RULE)
            pdf.set_line_width(0.3)
            pdf.line(BODY_X, pdf.get_y() + 1, pdf.w - 18, pdf.get_y() + 1)
            pdf.ln(4)
            pdf.set_text_color(*_PDF_INK)

        elif line.startswith("- "):
            # Hanging block: dash + text both sit at BULLET_X, with
            # a small gap after each item instead of lines touching.
            pdf.set_x(BULLET_X)
            pdf.set_left_margin(BULLET_X)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(pdf.w - BULLET_X - 18, 6.5, f"-  {line[2:]}")
            pdf.set_left_margin(BODY_X)
            pdf.ln(1.5)

        elif line.startswith("_") and line.endswith("_") and len(line) > 1:
            pdf.set_font("Helvetica", "I", 9.5)
            pdf.set_text_color(*_PDF_MUTED)
            pdf.multi_cell(0, 6, line[1:-1])
            pdf.set_text_color(*_PDF_INK)
            pdf.ln(2)

        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6.5, line)
            pdf.ln(1.5)

    pdf.output(path)


def _markdown_to_docx(markdown_text: str, path: str) -> None:
    """Render the same simple markdown subset into a real .docx file
    using python-docx (pure Python, no system dependencies)."""
    document = Document()

    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for raw_line in markdown_text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        image_match = _IMAGE_LINE_PATTERN.match(line)
        if image_match:
            _, image_path = image_match.groups()
            if os.path.exists(image_path):
                document.add_picture(image_path, width=Inches(6))
            continue

        if line.startswith("# "):
            document.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=2)
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        elif line.startswith("_") and line.endswith("_") and len(line) > 1:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(line[1:-1])
            run.italic = True
        else:
            document.add_paragraph(line)

    document.save(path)


def generate_report(file_path: str, target_column: str = None) -> dict:
    """
    Generate a full automatic analysis report for the dataset:
    overview, data quality, statistics, correlations, and (if a
    target_column is given or a candidate is confidently detected)
    ML results. Saves both a .md and a .html file to outputs/ and
    returns a compact summary plus the file paths.
    """

    profile = get_dataset_profile(file_path)
    if "error" in profile:
        return profile

    stats = get_dataset_statistics(file_path)

    ml_result = None
    resolved_target = target_column

    if not resolved_target and len(profile["candidate_target_columns"]) == 1:
        # Only auto-pick a target when there's exactly one confident
        # candidate; otherwise leave ML out rather than guessing.
        resolved_target = profile["candidate_target_columns"][0]

    if resolved_target:
        ml_result = train_model(file_path, resolved_target)

    markdown_report = _build_markdown(
        profile, stats, ml_result, resolved_target
    )
    html_report = _markdown_to_html(markdown_report)

    os.makedirs("outputs", exist_ok=True)

    md_path = "outputs/analysis_report.md"
    html_path = "outputs/analysis_report.html"
    pdf_path = "outputs/analysis_report.pdf"
    docx_path = "outputs/analysis_report.docx"

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(markdown_report)

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_report)

    _markdown_to_pdf(markdown_report, pdf_path)
    _markdown_to_docx(markdown_report, docx_path)

    # Compact key insights for the LLM to relay - not the full report.
    key_insights = []

    if profile["duplicate_rows"] > 0:
        key_insights.append(
            f"{profile['duplicate_rows']} duplicate row(s) found."
        )

    high_missing = {
        column: percent
        for column, percent in profile["missing_percent_by_column"].items()
        if percent > 20
    }
    if high_missing:
        worst_column = max(high_missing, key=high_missing.get)
        key_insights.append(
            f"'{worst_column}' has {high_missing[worst_column]}% missing "
            f"values."
        )

    if profile["strongest_correlations"]:
        top_pair = profile["strongest_correlations"][0]
        key_insights.append(
            f"Strongest correlation: {top_pair['columns'][0]} vs "
            f"{top_pair['columns'][1]} (r={top_pair['correlation']})."
        )

    if ml_result and "error" not in ml_result:
        if ml_result["task_type"] == "classification":
            key_insights.append(
                f"Baseline model accuracy for '{resolved_target}': "
                f"{ml_result['metrics']['accuracy']}."
            )
        else:
            key_insights.append(
                f"Baseline model R^2 for '{resolved_target}': "
                f"{ml_result['metrics']['r2']}."
            )

    return {
        "status": "success",
        "report_path_markdown": md_path,
        "report_path_html": html_path,
        "report_path_pdf": pdf_path,
        "report_path_docx": docx_path,
        "headline": {
            "rows": profile["rows"],
            "columns": profile["columns"],
            "duplicate_rows": profile["duplicate_rows"],
            "target_used": resolved_target
        },
        "key_insights": key_insights
    }


# ---------------------------------------------------------------------
# Business Report Skill
# ---------------------------------------------------------------------
# The flagship "create a business report" tool. Unlike generate_report
# above (a generic technical data report), this one is business-framed:
# it auto-detects a date column, a revenue/sales-like metric column,
# and a grouping column (region/product/segment) when not given, then
# ties together profiling, trends, period comparison, anomaly
# detection, and top/bottom performers into one document with an
# Executive Summary and actionable Recommendations.
#
# Every number in the report comes from a real tool call above - the
# "Insights" and "Recommendations" sections are built with plain
# Python string templates around those real numbers, never generated
# freeform by the LLM, so nothing here can be hallucinated.

_DATE_NAME_HINTS = ("date", "time", "day", "month", "period", "timestamp")
_METRIC_NAME_HINTS = (
    "revenue", "sales", "amount", "total", "price", "profit",
    "income", "cost", "value"
)
_GROUP_NAME_HINTS = (
    "region", "product", "category", "segment", "department",
    "channel", "country", "store"
)


def _detect_date_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if any(hint in column.lower() for hint in _DATE_NAME_HINTS):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() > 0.7:
                return column
    for column in df.columns:
        if df[column].dtype == object:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return column
    return None


def _detect_metric_column(df: pd.DataFrame) -> str:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return None
    for column in numeric_columns:
        if any(hint in column.lower() for hint in _METRIC_NAME_HINTS):
            return column
    # Fall back to the numeric column with the highest variance,
    # since a flat/constant column rarely makes an interesting metric.
    variances = df[numeric_columns].var(numeric_only=True)
    return variances.idxmax() if not variances.empty else numeric_columns[0]


def _detect_group_column(df: pd.DataFrame) -> str:
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for column in categorical_columns:
        if any(hint in column.lower() for hint in _GROUP_NAME_HINTS):
            if 2 <= df[column].nunique() <= 50:
                return column
    for column in categorical_columns:
        if 2 <= df[column].nunique() <= 50:
            return column
    return None


_CHART_INK = "#1B3A4B"
_CHART_DECLINE = "#DC2626"

# Above this many periods, showing every x-axis label produces the
# crammed, overlapping diagonal text that made the trend chart
# unreadable - space out the labels shown instead of printing all of
# them.
_MAX_TREND_XTICKS = 12


def _format_number(value) -> str:
    """4.0 -> '4', 61.2134 -> '61.21' - avoids trailing '.0' on whole
    numbers while keeping useful precision on the rest."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def _format_anomaly(anomaly) -> str:
    """
    Turn one anomaly record into a readable sentence instead of
    printing the raw Python dict (str(anomaly) was producing lines
    like "{'row_index': 22, 'value': 4.0, 'date': '...'}" in the
    report).

    detect_anomalies returns two different record shapes depending on
    whether it ran univariate (row_index, value, date) or
    multivariate (row_index, one entry per numeric column, date)
    detection - this handles both without needing to know which one
    produced the record.
    """
    if not isinstance(anomaly, dict):
        return str(anomaly)

    row_index = anomaly.get("row_index")
    date = anomaly.get("date")
    values = [
        f"{key} = {_format_number(value)}"
        for key, value in anomaly.items()
        if key not in ("row_index", "date")
    ]

    pieces = []
    if row_index is not None:
        pieces.append(f"Row {row_index}")
    if date:
        pieces.append(f"({date})")

    header = " ".join(pieces)
    return f"{header}: {', '.join(values)}" if values else (header or str(anomaly))


def _plot_trend_chart(trends: dict, value_column: str) -> str:
    """Line chart of the period-by-period series already computed by
    analyze_trends - plotted directly from that aggregate rather than
    re-reading the raw CSV, since the aggregation is the whole point."""
    if not trends or "error" in trends:
        return None

    periods = trends.get("periods") or []
    if len(periods) < 2:
        return None

    os.makedirs("outputs", exist_ok=True)

    labels = [p["period"] for p in periods]
    values = [p["value"] for p in periods]

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(labels)), values, marker="o", color=_CHART_INK, linewidth=2, markersize=4)
    plt.title(f"{value_column} Trend Over Time")
    plt.xlabel("Period")
    plt.ylabel(value_column)

    # Show at most _MAX_TREND_XTICKS labels, evenly spaced, instead
    # of one per period - a report spanning 60+ months was printing
    # all 60 rotated labels on top of each other and unreadable.
    step = max(1, len(labels) // _MAX_TREND_XTICKS)
    tick_positions = list(range(0, len(labels), step))
    last = len(labels) - 1
    if tick_positions[-1] != last:
        # Don't just append the last tick - if it would land right
        # next to the previous one, replace it instead so the two
        # labels don't crowd together.
        if last - tick_positions[-1] < step / 2:
            tick_positions[-1] = last
        else:
            tick_positions.append(last)
    plt.xticks(tick_positions, [labels[i] for i in tick_positions], rotation=45, ha="right")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = "outputs/report_trend_chart.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_top_bottom_chart(top_bottom: dict, group_column: str) -> str:
    """Horizontal bar chart of the top/bottom performers already
    computed by rank_categories - top in the report's accent color,
    bottom in red, so the contrast reads at a glance."""
    if not top_bottom or "error" in top_bottom:
        return None

    top = top_bottom.get("top_performers") or []
    bottom = top_bottom.get("bottom_performers") or []
    if not top and not bottom:
        return None

    os.makedirs("outputs", exist_ok=True)

    # When a dataset has few distinct categories, "top N" and
    # "bottom N" can be the exact same rows in reverse order (e.g. 5
    # order statuses, asked for top 5 and bottom 5) - drawing both
    # would silently plot every bar twice. Each category is kept
    # only once, classified as "top" if it appears there.
    seen = {entry["category"] for entry in top}
    bottom_unique = [
        entry for entry in reversed(bottom)
        if entry["category"] not in seen
    ]

    categories = (
        [entry["category"] for entry in top]
        + [entry["category"] for entry in bottom_unique]
    )
    values = (
        [entry["value"] for entry in top]
        + [entry["value"] for entry in bottom_unique]
    )
    colors = [_CHART_INK] * len(top) + [_CHART_DECLINE] * len(bottom_unique)

    plt.figure(figsize=(8, max(3.5, 0.45 * len(categories))))
    bars = plt.barh(categories, values, color=colors)
    plt.xlabel(top_bottom.get("metric_column", ""))
    plt.title(f"Top & Bottom {group_column} Performers")
    plt.bar_label(
        bars,
        labels=[_format_number(v) for v in values],
        padding=3, fontsize=9,
    )
    plt.margins(x=0.12)
    plt.tight_layout()

    path = "outputs/report_top_bottom_chart.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _generate_report_charts(
    file_path: str,
    resolved_date: str,
    resolved_value: str,
    resolved_group: str,
    trends: dict,
    top_bottom: dict
) -> dict:
    """
    Real matplotlib charts for the business report - the same kind
    shown in the live dashboard, not just tables of numbers. Every
    chart is optional: if the underlying data isn't there (no date
    column, not enough numeric columns, etc), that entry is simply
    None and the report section it would have gone in is skipped.
    """
    charts = {
        "trend": None,
        "top_bottom": None,
        "distribution": None,
        "correlation": None,
    }

    if trends:
        charts["trend"] = _plot_trend_chart(trends, resolved_value)

    if top_bottom:
        charts["top_bottom"] = _plot_top_bottom_chart(top_bottom, resolved_group)

    if resolved_value:
        histogram = create_visualization(
            file_path, "histogram", column=resolved_value
        )
        if "error" not in histogram:
            charts["distribution"] = histogram["file_path"]

    correlation = create_visualization(file_path, "correlation_heatmap")
    if "error" not in correlation:
        charts["correlation"] = correlation["file_path"]

    return charts


def _build_business_markdown(
    profile, trends, period_comparison, anomalies,
    top_bottom, ml_result, target_column,
    date_column, value_column, group_column,
    key_findings, recommendations, charts=None
):
    charts = charts or {}
    lines = []
    lines.append("# Business Analysis Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    if key_findings:
        for finding in key_findings:
            lines.append(f"- {finding}")
    else:
        lines.append(
            "- Dataset profiled successfully; no date/metric/grouping "
            "columns were confidently detected for deeper business "
            "analysis, so this report covers data quality and "
            "statistics only."
        )
    lines.append("")

    lines.append("## Key Metrics")
    lines.append("")
    lines.append(f"- Rows: {profile['rows']}")
    lines.append(f"- Columns: {profile['columns']}")
    lines.append(f"- Duplicate rows: {profile['duplicate_rows']}")
    if value_column:
        lines.append(f"- Metric analyzed: {value_column}")
    if date_column:
        lines.append(f"- Date column used: {date_column}")
    if group_column:
        lines.append(f"- Grouping dimension: {group_column}")
    lines.append("")

    if charts.get("distribution") or charts.get("correlation"):
        lines.append("## Visual Overview")
        lines.append("")
        if charts.get("distribution"):
            lines.append(f"![Distribution of {value_column}]({charts['distribution']})")
            lines.append("")
        if charts.get("correlation"):
            lines.append("![Correlation heatmap](" + charts["correlation"] + ")")
            lines.append("")

    lines.append("## Trends")
    lines.append("")
    if trends and "error" not in trends:
        lines.append(
            f"- Overall direction for **{trends['value_column']}**: "
            f"{trends['overall_direction']}"
        )
        if trends["total_change_percent"] is not None:
            lines.append(
                f"- Total change over the period: "
                f"{trends['total_change_percent']}%"
            )
        if trends["largest_period_over_period_drop"]:
            drop = trends["largest_period_over_period_drop"]
            lines.append(
                f"- Largest single-period drop: {drop['change_percent']}% "
                f"in {drop['period']}"
            )
        if trends["largest_period_over_period_rise"]:
            rise = trends["largest_period_over_period_rise"]
            lines.append(
                f"- Largest single-period rise: {rise['change_percent']}% "
                f"in {rise['period']}"
            )
    else:
        lines.append(
            "- No usable date/metric combination was found for "
            "trend analysis."
        )
    lines.append("")

    if charts.get("trend"):
        lines.append(f"![{value_column} trend over time]({charts['trend']})")
        lines.append("")

    if period_comparison and "error" not in period_comparison:
        lines.append("## Period Comparison")
        lines.append("")
        lines.append(
            f"- {period_comparison['period_a']}: "
            f"{period_comparison['value_a']}"
        )
        lines.append(
            f"- {period_comparison['period_b']}: "
            f"{period_comparison['value_b']}"
        )
        lines.append(
            f"- Change: {period_comparison['absolute_change']} "
            f"({period_comparison['percent_change']}%, "
            f"{period_comparison['direction']})"
        )
        if period_comparison.get("top_growth"):
            growing = [
                entry for entry in period_comparison["top_growth"]
                if entry.get("percent_change") is not None
                and entry["percent_change"] > 0
            ]
            lines.append("")
            if growing:
                lines.append(f"Fastest-growing {group_column}:")
                lines.append("")
                for entry in growing[:3]:
                    lines.append(
                        f"- {entry['group']}: {entry['percent_change']}%"
                    )
            else:
                # Every group moved in the same (negative) direction -
                # calling the least-bad decliner "fastest-growing"
                # would be actively misleading, so say what actually
                # happened instead.
                least_bad = sorted(
                    (
                        entry for entry in period_comparison["top_growth"]
                        if entry.get("percent_change") is not None
                    ),
                    key=lambda entry: entry["percent_change"],
                    reverse=True,
                )[:3]
                if least_bad:
                    lines.append(
                        f"No {group_column} grew between the two "
                        f"periods; least-severe decline:"
                    )
                    lines.append("")
                    for entry in least_bad:
                        lines.append(
                            f"- {entry['group']}: {entry['percent_change']}%"
                        )
        if period_comparison.get("top_decline"):
            declining = [
                entry for entry in period_comparison["top_decline"]
                if entry.get("percent_change") is not None
                and entry["percent_change"] < 0
            ]
            lines.append("")
            if declining:
                lines.append(f"Fastest-declining {group_column}:")
                lines.append("")
                for entry in declining[:3]:
                    lines.append(
                        f"- {entry['group']}: {entry['percent_change']}%"
                    )
            else:
                least_good = sorted(
                    (
                        entry for entry in period_comparison["top_decline"]
                        if entry.get("percent_change") is not None
                    ),
                    key=lambda entry: entry["percent_change"],
                )[:3]
                if least_good:
                    lines.append(
                        f"No {group_column} declined between the two "
                        f"periods; slowest growth:"
                    )
                    lines.append("")
                    for entry in least_good:
                        lines.append(
                            f"- {entry['group']}: {entry['percent_change']}%"
                        )
        lines.append("")

    lines.append("## Anomalies")
    lines.append("")
    if anomalies and "error" not in anomalies:
        count = anomalies.get("anomaly_count", 0)
        if count:
            lines.append(
                f"- {count} anomal{'y' if count == 1 else 'ies'} "
                f"detected using {anomalies['method']}."
            )
            for anomaly in anomalies.get("anomalies", [])[:5]:
                lines.append(f"- {_format_anomaly(anomaly)}")
        else:
            lines.append("- No significant anomalies detected.")
    else:
        lines.append("- Anomaly detection was not run or found no numeric data.")
    lines.append("")

    if top_bottom and "error" not in top_bottom:
        lines.append("## Top / Bottom Performers")
        lines.append("")
        lines.append(
            f"Top {group_column} by {top_bottom['metric_column']} "
            f"({top_bottom['aggregation']}):"
        )
        lines.append("")
        for entry in top_bottom["top_performers"]:
            lines.append(f"- {entry['category']}: {entry['value']}")
        lines.append("")
        lines.append(f"Bottom {group_column} by {top_bottom['metric_column']}:")
        lines.append("")
        for entry in top_bottom["bottom_performers"]:
            lines.append(f"- {entry['category']}: {entry['value']}")
        lines.append("")
        if charts.get("top_bottom"):
            lines.append(f"![Top and bottom {group_column} performers]({charts['top_bottom']})")
            lines.append("")

    if ml_result and "error" not in ml_result:
        lines.append("## Predictive Signal")
        lines.append("")
        lines.append(f"- Target column: {target_column}")
        lines.append(f"- Task type: {ml_result['task_type']}")
        for metric_name, metric_value in ml_result["metrics"].items():
            if metric_name == "confusion_matrix":
                continue
            lines.append(f"- {metric_name}: {metric_value}")
        lines.append("")
        lines.append("Top features:")
        for feature in ml_result["top_features"][:5]:
            lines.append(f"- {feature['feature']}: {feature['importance']}")
        lines.append("")

    lines.append("## Insights")
    lines.append("")
    if key_findings:
        for finding in key_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- No additional insights beyond the sections above.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    if recommendations:
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
    else:
        lines.append(
            "- No specific recommendations were generated; the "
            "detected signals were not strong enough to act on."
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Generated automatically from the uploaded CSV only; it "
        "does not incorporate context outside the data."
    )
    lines.append(
        "- Date, metric, and grouping columns were auto-detected "
        "when not specified and may not match true business intent."
    )
    lines.append(
        "- Correlations, trends, and feature importance reflect "
        "association within this dataset, not proven causation."
    )

    return "\n".join(lines)


def generate_business_report(
    file_path: str,
    date_column: str = None,
    value_column: str = None,
    category_column: str = None,
    target_column: str = None
) -> dict:
    """
    Generate a full business analysis report: Executive Summary, Key
    Metrics, Trends, Anomalies, Top/Bottom Performers, Insights, and
    Recommendations, saved as .md and .html in outputs/.

    Auto-detects a date column, a revenue/sales-like metric column,
    and a category/region-like grouping column when not given.
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    profile = get_dataset_profile(file_path)
    if "error" in profile:
        return profile

    resolved_date = date_column or _detect_date_column(df)
    resolved_value = value_column or _detect_metric_column(df)
    resolved_group = category_column or _detect_group_column(df)

    trends = None
    period_comparison = None
    if resolved_date and resolved_value:
        trends = analyze_trends(file_path, resolved_date, resolved_value)
        period_comparison = compare_periods(
            file_path, resolved_date, resolved_value,
            group_column=resolved_group
        )

    anomalies = detect_anomalies(
        file_path,
        column=resolved_value,
        date_column=resolved_date
    )

    top_bottom = None
    if resolved_group and resolved_value:
        top_bottom = rank_categories(
            file_path, resolved_group, resolved_value
        )

    charts = _generate_report_charts(
        file_path, resolved_date, resolved_value, resolved_group,
        trends, top_bottom
    )

    ml_result = None
    resolved_target = target_column
    if not resolved_target and len(profile["candidate_target_columns"]) == 1:
        resolved_target = profile["candidate_target_columns"][0]
    if resolved_target:
        ml_result = train_model(file_path, resolved_target)

    # --- Build key findings and recommendations from real numbers ---
    key_findings = []
    recommendations = []

    if profile["duplicate_rows"] > 0:
        key_findings.append(
            f"{profile['duplicate_rows']} duplicate row(s) found."
        )
        recommendations.append(
            "Deduplicate the source data before relying on totals "
            "computed from this dataset."
        )

    high_missing = {
        column: percent
        for column, percent in profile["missing_percent_by_column"].items()
        if percent > 20
    }
    if high_missing:
        worst_column = max(high_missing, key=high_missing.get)
        key_findings.append(
            f"'{worst_column}' has {high_missing[worst_column]}% missing "
            f"values, which may bias any analysis using it."
        )
        recommendations.append(
            f"Investigate why '{worst_column}' has a high missing rate "
            f"before using it in reporting or modeling."
        )

    if trends and "error" not in trends:
        key_findings.append(
            f"{resolved_value} is {trends['overall_direction']} overall "
            f"({trends['total_change_percent']}% total change)."
        )
        if trends["largest_period_over_period_drop"]:
            drop = trends["largest_period_over_period_drop"]
            key_findings.append(
                f"The sharpest drop in {resolved_value} was "
                f"{drop['change_percent']}% in {drop['period']}."
            )
            recommendations.append(
                f"Review what happened around {drop['period']} - it "
                f"had the largest period-over-period drop in "
                f"{resolved_value}."
            )

    if period_comparison and "error" not in period_comparison:
        key_findings.append(
            f"{resolved_value} moved from "
            f"{period_comparison['value_a']} in "
            f"{period_comparison['period_a']} to "
            f"{period_comparison['value_b']} in "
            f"{period_comparison['period_b']} "
            f"({period_comparison['percent_change']}%)."
        )
        if period_comparison.get("top_decline"):
            worst = period_comparison["top_decline"][0]
            if worst["percent_change"] is not None and worst["percent_change"] < 0:
                key_findings.append(
                    f"'{worst['group']}' had the steepest decline "
                    f"({worst['percent_change']}%) between the two "
                    f"periods."
                )
                recommendations.append(
                    f"Prioritize investigating '{worst['group']}' - it "
                    f"declined {worst['percent_change']}% between the "
                    f"two most recent periods."
                )

    if anomalies and "error" not in anomalies and anomalies.get("anomaly_count"):
        key_findings.append(
            f"{anomalies['anomaly_count']} anomal"
            f"{'y' if anomalies['anomaly_count'] == 1 else 'ies'} "
            f"detected via {anomalies['method']}."
        )
        recommendations.append(
            "Manually review the flagged anomalous rows for data-entry "
            "errors or genuinely unusual business events."
        )

    if top_bottom and "error" not in top_bottom:
        worst = top_bottom["bottom_performers"][0]
        best = top_bottom["top_performers"][0]
        key_findings.append(
            f"By {resolved_value}, '{best['category']}' leads "
            f"({best['value']}) and '{worst['category']}' trails "
            f"({worst['value']}) among {resolved_group}."
        )
        recommendations.append(
            f"Examine why '{worst['category']}' underperforms other "
            f"{resolved_group} on {resolved_value}, and whether "
            f"'{best['category']}' has practices worth replicating."
        )

    if ml_result and "error" not in ml_result:
        if ml_result["task_type"] == "classification":
            key_findings.append(
                f"A baseline model predicts '{resolved_target}' with "
                f"{ml_result['metrics']['accuracy']} accuracy; the top "
                f"driver is "
                f"'{ml_result['top_features'][0]['feature']}'."
            )
        else:
            key_findings.append(
                f"A baseline model explains "
                f"{ml_result['metrics']['r2']} of the variance in "
                f"'{resolved_target}'; the top driver is "
                f"'{ml_result['top_features'][0]['feature']}'."
            )

    markdown_report = _build_business_markdown(
        profile, trends, period_comparison, anomalies, top_bottom,
        ml_result, resolved_target, resolved_date, resolved_value,
        resolved_group, key_findings, recommendations, charts
    )
    html_report = _markdown_to_html(markdown_report)

    os.makedirs("outputs", exist_ok=True)
    md_path = "outputs/business_report.md"
    html_path = "outputs/business_report.html"
    pdf_path = "outputs/business_report.pdf"
    docx_path = "outputs/business_report.docx"

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(markdown_report)
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_report)

    _markdown_to_pdf(markdown_report, pdf_path)
    _markdown_to_docx(markdown_report, docx_path)

    return {
        "status": "success",
        "report_path_markdown": md_path,
        "report_path_html": html_path,
        "report_path_pdf": pdf_path,
        "report_path_docx": docx_path,
        "detected_columns": {
            "date_column": resolved_date,
            "value_column": resolved_value,
            "category_column": resolved_group,
            "target_column": resolved_target
        },
        "key_insights": key_findings,
        "recommendations": recommendations,
        "charts": {name: path for name, path in charts.items() if path}
    }