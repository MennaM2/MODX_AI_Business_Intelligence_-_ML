import os
import re

import matplotlib

# Force a non-interactive backend. Streamlit runs matplotlib off the
# main thread in some environments, and the default backend can try to
# open a GUI window, which either errors or silently hangs. This must
# happen before pyplot is imported anywhere in the process.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Shared chart styling
# ---------------------------------------------------------------------
# Applied once at import time so every chart this module (and
# report_tools.py, which imports create_visualization from here) draws
# looks like one coherent product instead of raw matplotlib defaults -
# no top/right border, a light grid, and one consistent accent color
# instead of matplotlib's default blue everywhere.

CHART_INK = "#1B3A4B"
CHART_ACCENT = "#0E7490"
CHART_DECLINE = "#DC2626"
CHART_GRID = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#94A3B8",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": CHART_GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": CHART_INK,
    "axes.labelsize": 10,
    "figure.autolayout": False,
})


def _identifier_columns(df: pd.DataFrame, numeric_columns) -> set:
    """
    Columns that are numeric but are really identifiers (order_id,
    user_id, ...): sequential/unique integers with no real magnitude
    meaning. Including them in a correlation heatmap or a histogram
    doesn't describe anything about the business - it just shows that
    an ID column correlates with another ID column because both were
    assigned sequentially, which is a coincidence of how the data was
    generated, not a finding.
    """
    id_name_pattern = re.compile(
        r"(^|_)(id|uuid|guid|key|code|no|num|number|pk)($|_)",
        re.IGNORECASE,
    )
    rows = len(df)
    identifiers = set()
    for column in numeric_columns:
        looks_like_id = bool(id_name_pattern.search(str(column)))
        nearly_unique = (
            rows > 0 and df[column].nunique(dropna=True) >= 0.98 * rows
        )
        if looks_like_id and nearly_unique:
            identifiers.add(column)
    return identifiers


def _integer_aligned_bins(series: pd.Series):
    """
    Histogram bin edges for a column that is really discrete (a small
    number of whole-number values, like a 1-4 item count or a 1-5
    rating) rather than continuous.

    Blindly using a fixed bin count (e.g. 20) on data like [1, 2, 3, 4]
    produces a chart with 16 empty gaps and 4 oddly-thin bars - exactly
    the "distribution" chart nobody can read. Detecting discreteness
    and aligning one bin per integer value fixes that; anything else
    keeps its bin count as given by the caller.
    """
    values = series.dropna()
    if values.empty:
        return None
    distinct = values.unique()
    is_whole_numbers = np.allclose(distinct, np.round(distinct))
    if is_whole_numbers and len(distinct) <= 20:
        low, high = int(values.min()), int(values.max())
        return np.arange(low - 0.5, high + 1.5, 1)
    return None


def create_churn_plot(file_path: str) -> dict:

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    target_column = next(
        (
            column
            for column in df.columns
            if column.lower() == "churn"
        ),
        None
    )

    if target_column is None:
        return {
            "error": "No Churn column was found."
        }

    os.makedirs("outputs", exist_ok=True)

    counts = df[target_column].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    output_path = "outputs/churn_distribution.png"

    plt.savefig(output_path)
    plt.close()

    return {
        "status": "success",
        "file_path": output_path,
        "distribution": counts.to_dict()
    }


# ---------------------------------------------------------------------
# Visualization Skill
# ---------------------------------------------------------------------
# A single, generic visualization tool the agent can call for any chart
# type, instead of one hardcoded function per chart. This is what lets
# the "Visualization Skill" scale to new chart types without adding a
# new tool (and a new schema, and new latency) every time.

SUPPORTED_CHART_TYPES = {
    "histogram",
    "bar_chart",
    "box_plot",
    "correlation_heatmap",
    "scatter_plot",
    "target_distribution",
    "category_comparison"
}


def _safe_filename(*parts: str) -> str:
    """Turn chart-type + column names into a filesystem-safe filename."""
    raw = "_".join(part for part in parts if part)
    cleaned = "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in raw
    )
    return cleaned or "chart"


def create_visualization(
    file_path: str,
    chart_type: str,
    column: str = None,
    column_x: str = None,
    column_y: str = None,
    category_column: str = None,
    target_column: str = None
) -> dict:
    """
    Create a chart from the dataset and save it as a PNG.

    chart_type must be one of:
      histogram            - needs `column` (numeric)
      bar_chart             - needs `column` (categorical)
      box_plot               - needs `column` (numeric), optional `category_column` to group by
      correlation_heatmap    - no columns needed, uses all numeric columns
      scatter_plot            - needs `column_x` and `column_y` (both numeric)
      target_distribution     - needs `target_column` (categorical/binary)
      category_comparison      - needs `category_column` and `target_column`

    Returns the saved file path plus compact metadata describing what
    was plotted, so the agent can explain the chart without needing to
    "see" the image.
    """

    if chart_type not in SUPPORTED_CHART_TYPES:
        return {
            "error": (
                f"Unsupported chart_type '{chart_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHART_TYPES)}"
            )
        }

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    os.makedirs("outputs", exist_ok=True)

    def missing_column_error(name, value):
        return {"error": f"Column '{value}' ({name}) was not found."}

    plt.figure(figsize=(7, 5))
    metadata = {}

    try:
        if chart_type == "histogram":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            bins = _integer_aligned_bins(df[column])
            if bins is None:
                bins = 20
            df[column].dropna().plot(kind="hist", bins=bins, color=CHART_ACCENT, edgecolor="white")
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            metadata = {
                "column": column,
                "mean": round(float(df[column].mean()), 4),
                "std": round(float(df[column].std()), 4)
            }

        elif chart_type == "bar_chart":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            counts = df[column].value_counts().head(15)
            counts.plot(kind="bar", color=CHART_ACCENT)
            plt.title(f"Counts by {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")

            metadata = {
                "column": column,
                "top_values": counts.to_dict()
            }

        elif chart_type == "box_plot":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            if category_column and category_column in df.columns:
                df.boxplot(column=column, by=category_column)
                plt.title(f"{column} by {category_column}")
                plt.suptitle("")
                plt.xlabel(category_column)
            else:
                df.boxplot(column=column)
                plt.title(f"Distribution of {column}")

            plt.ylabel(column)

            metadata = {
                "column": column,
                "category_column": category_column,
                "median": round(float(df[column].median()), 4),
                "q1": round(float(df[column].quantile(0.25)), 4),
                "q3": round(float(df[column].quantile(0.75)), 4)
            }

        elif chart_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include="number")

            # Identifier columns (order_id, user_id, ...) are numbers
            # but not measurements - they correlate with each other
            # only because both happen to be sequential integers, and
            # including them produces exactly the "giant solid block
            # of meaningless red" look that makes a heatmap useless.
            identifiers = _identifier_columns(df, numeric_df.columns)
            if identifiers:
                numeric_df = numeric_df.drop(columns=list(identifiers))

            if numeric_df.shape[1] < 2:
                plt.close()
                return {
                    "error": (
                        "Need at least two non-identifier numerical "
                        "columns for a correlation heatmap."
                    )
                }

            corr = numeric_df.corr()
            n = len(corr.columns)

            plt.close()
            fig_size = max(5, min(10, 1.1 * n + 2))
            plt.figure(figsize=(fig_size, fig_size * 0.85))

            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)

            # A heatmap with no numbers on it is barely readable,
            # especially for a small 2-4 column matrix like this one -
            # annotate every cell with its actual value.
            for i in range(n):
                for j in range(n):
                    value = corr.iloc[i, j]
                    text_color = "white" if abs(value) > 0.6 else "#1e293b"
                    plt.text(
                        j, i, f"{value:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=9,
                    )

            plt.xticks(range(n), corr.columns, rotation=45, ha="right")
            plt.yticks(range(n), corr.columns)
            plt.grid(False)
            plt.title("Correlation Heatmap")

            # Compact metadata: only the strongest pairs, not the
            # whole matrix, to keep the tool result small.
            pairs = (
                corr.where(
                    ~corr.abs().eq(1.0)
                )
                .unstack()
                .dropna()
                .abs()
                .sort_values(ascending=False)
            )
            seen = set()
            top_pairs = []
            for (col_a, col_b), value in pairs.items():
                key = frozenset((col_a, col_b))
                if key in seen:
                    continue
                seen.add(key)
                top_pairs.append({
                    "columns": [col_a, col_b],
                    "correlation": round(float(corr.loc[col_a, col_b]), 4)
                })
                if len(top_pairs) >= 5:
                    break

            metadata = {
                "strongest_correlations": top_pairs,
                "excluded_identifier_columns": sorted(identifiers),
            }

        elif chart_type == "scatter_plot":
            if not column_x or column_x not in df.columns:
                plt.close()
                return missing_column_error("column_x", column_x)
            if not column_y or column_y not in df.columns:
                plt.close()
                return missing_column_error("column_y", column_y)

            plt.scatter(df[column_x], df[column_y], alpha=0.6, color=CHART_ACCENT, edgecolor="white", linewidth=0.3)
            plt.title(f"{column_y} vs {column_x}")
            plt.xlabel(column_x)
            plt.ylabel(column_y)

            metadata = {
                "column_x": column_x,
                "column_y": column_y,
                "correlation": round(
                    float(df[column_x].corr(df[column_y])), 4
                )
            }

        elif chart_type == "target_distribution":
            if not target_column or target_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "target_column", target_column
                )

            counts = df[target_column].value_counts()
            counts.plot(kind="bar", color=CHART_ACCENT)
            plt.title(f"Distribution of {target_column}")
            plt.xlabel(target_column)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")

            metadata = {
                "target_column": target_column,
                "distribution": counts.to_dict()
            }

        elif chart_type == "category_comparison":
            if not category_column or category_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "category_column", category_column
                )
            if not target_column or target_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "target_column", target_column
                )

            grouped = df.groupby(category_column)[target_column].mean()
            grouped.plot(kind="bar", color=CHART_ACCENT)
            plt.title(f"Average {target_column} by {category_column}")
            plt.xlabel(category_column)
            plt.ylabel(f"Average {target_column}")
            plt.xticks(rotation=45, ha="right")

            metadata = {
                "category_column": category_column,
                "target_column": target_column,
                "averages": grouped.round(4).to_dict()
            }

        plt.tight_layout()

        filename = _safe_filename(
            chart_type, column, column_x, column_y,
            category_column, target_column
        )
        output_path = f"outputs/{filename}.png"

        plt.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close()

        return {
            "status": "success",
            "chart_type": chart_type,
            "file_path": output_path,
            "metadata": metadata
        }

    except Exception as exc:
        plt.close()
        return {"error": f"Failed to create visualization: {exc}"}