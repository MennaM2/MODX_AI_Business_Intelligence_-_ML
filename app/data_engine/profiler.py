"""
Structural profiling.

This answers "what is actually in this file?" before anything is
changed. It runs on the raw, as-loaded frame (everything still
strings) and again on the cleaned frame, so the report can show a
genuine before/after.

Deliberately distinct from ``app.tools.data_tools.get_dataset_profile``:
that one is an agent-facing tool the LLM calls mid-conversation to
answer a question about one dataset. This one is a pipeline stage
whose output drives the cleaner's decisions. No Gemini involved.
"""

import re
from typing import Optional

import pandas as pd


# Above this many distinct values relative to row count, a column is
# treated as effectively unique - the main signal for an ID column.
_ID_UNIQUENESS_THRESHOLD = 0.98

# A text column with few distinct values relative to its length is
# categorical; beyond this ratio it is free text and should not be
# treated as a category.
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.5
_CATEGORICAL_MAX_DISTINCT = 100

# Tokens that commonly mean "missing" but arrive as literal text.
NULL_LIKE_TOKENS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--",
    "?", "unknown", "undefined", "missing", "not available", "#n/a",
    "#null!", "\\n",
}

_ID_NAME_PATTERN = re.compile(
    r"(^|_)(id|uuid|guid|key|code|no|num|number|pk)($|_)", re.IGNORECASE
)

_DATE_NAME_PATTERN = re.compile(
    r"(date|time|timestamp|_at$|_on$|dob|birth|created|updated|"
    r"signup|joined|expiry|expires)",
    re.IGNORECASE,
)

# Strings that are numeric once currency symbols, thousands
# separators, percent signs, or accounting parentheses are removed.
_NUMERIC_LIKE_PATTERN = re.compile(
    r"^\s*[\(\-\+]?\s*[$€£¥₹]?\s*"
    r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
    r"\s*%?\s*\)?\s*$"
)

_BOOLEAN_TOKENS = {
    "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"
}


def _sample(series: pd.Series, limit: int = 5000) -> pd.Series:
    """Profiling a 5-million-row file exhaustively is wasteful; a
    deterministic head sample is enough to infer type and pattern.
    Counts that must be exact (nulls, duplicates) use the full
    column, not this."""
    return series if len(series) <= limit else series.head(limit)


def _as_text(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip()


def looks_numeric(series: pd.Series) -> bool:
    values = _as_text(_sample(series))
    values = values[~values.str.lower().isin(NULL_LIKE_TOKENS)]
    if values.empty:
        return False
    matches = values.str.match(_NUMERIC_LIKE_PATTERN, na=False)
    return bool(matches.mean() >= 0.9)


def looks_boolean(series: pd.Series) -> bool:
    values = _as_text(_sample(series)).str.lower()
    values = values[~values.isin(NULL_LIKE_TOKENS)]
    if values.empty:
        return False
    distinct = set(values.unique())
    return distinct.issubset(_BOOLEAN_TOKENS) and 1 <= len(distinct) <= 4


def looks_datetime(series: pd.Series, column_name: str = "") -> bool:
    """
    A column is a date column if most of its values parse as dates.

    The name is used only as a tie-breaker for the ambiguous case of
    all-numeric strings: '20240115' is a date, but so is a plain
    order quantity of 20240115 in principle, and the column name is
    the only thing that distinguishes them.
    """
    values = _as_text(_sample(series, 2000))
    values = values[~values.str.lower().isin(NULL_LIKE_TOKENS)]
    if values.empty:
        return False

    all_digits = values.str.fullmatch(r"\d+").fillna(False).mean() > 0.9
    if all_digits and not _DATE_NAME_PATTERN.search(str(column_name)):
        return False

    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.85)


def detect_suspicious_values(series: pd.Series, column_name: str) -> list:
    """
    Values that are not missing but are almost certainly wrong.

    This only *reports*. Nothing here is deleted or overwritten by
    the profiler - the cleaner decides what, if anything, to do.
    """
    findings = []
    text = _as_text(_sample(series))
    if text.empty:
        return findings

    lowered = text.str.lower()

    null_like = int(lowered.isin(NULL_LIKE_TOKENS).sum())
    if null_like:
        findings.append({
            "issue": "null_like_text",
            "count": null_like,
            "detail": (
                "Values such as 'N/A' or 'unknown' are stored as text "
                "rather than as real missing values."
            ),
        })

    whitespace = int((text != text.str.strip()).sum())
    padded = int(
        (series.dropna().astype(str) != series.dropna().astype(str).str.strip())
        .sum()
    )
    if padded:
        findings.append({
            "issue": "surrounding_whitespace",
            "count": padded,
            "detail": "Values have leading or trailing whitespace.",
        })

    # Same category written several ways: 'USA' / 'usa' / ' Usa '.
    normalized = text.str.lower().str.strip()
    if 0 < normalized.nunique() < text.nunique():
        findings.append({
            "issue": "inconsistent_casing",
            "count": int(text.nunique() - normalized.nunique()),
            "detail": (
                "The same value appears with different casing or "
                "spacing and will be counted as separate categories."
            ),
        })

    if looks_numeric(series):
        numeric = pd.to_numeric(
            text.str.replace(r"[,$€£¥₹%\s]", "", regex=True),
            errors="coerce",
        )
        negatives = int((numeric < 0).sum())
        name = str(column_name).lower()
        if negatives and re.search(
            r"(qty|quantity|count|amount|price|cost|revenue|sales|"
            r"age|income|salary|total)", name
        ):
            findings.append({
                "issue": "negative_values",
                "count": negatives,
                "detail": (
                    f"Column '{column_name}' contains negative values "
                    "where the name implies they should not occur."
                ),
            })

    return findings


def profile_column(series: pd.Series, column_name: str) -> dict:
    total = int(len(series))
    missing = int(series.isna().sum())

    text = _as_text(series)
    lowered = text.str.lower()
    null_like = int(lowered.isin(NULL_LIKE_TOKENS).sum())

    # Effective missingness counts text placeholders too - that is
    # the number a user actually cares about.
    effective_missing = missing + null_like

    distinct = int(series.nunique(dropna=True))
    non_null = max(total - missing, 1)
    unique_ratio = distinct / non_null

    if pd.api.types.is_numeric_dtype(series):
        inferred = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        inferred = "datetime"
    elif pd.api.types.is_bool_dtype(series):
        inferred = "boolean"
    elif looks_boolean(series):
        inferred = "boolean"
    elif looks_datetime(series, column_name):
        inferred = "datetime"
    elif looks_numeric(series):
        inferred = "numeric"
    elif (
        distinct <= _CATEGORICAL_MAX_DISTINCT
        and unique_ratio <= _CATEGORICAL_MAX_UNIQUE_RATIO
    ):
        inferred = "categorical"
    else:
        inferred = "text"

    is_id_like = (
        unique_ratio >= _ID_UNIQUENESS_THRESHOLD
        and missing == 0
        and distinct > 1
    )
    name_suggests_id = bool(_ID_NAME_PATTERN.search(str(column_name)))

    profile = {
        "name": column_name,
        "pandas_dtype": str(series.dtype),
        "inferred_type": inferred,
        "missing": missing,
        "null_like_text": null_like,
        "effective_missing": effective_missing,
        "missing_percent": round(effective_missing / total * 100, 2) if total else 0.0,
        "distinct": distinct,
        "unique_ratio": round(unique_ratio, 4),
        "is_constant": distinct <= 1,
        "id_candidate": bool(is_id_like and (name_suggests_id or unique_ratio == 1.0)),
        "name_suggests_id": name_suggests_id,
        "suspicious": detect_suspicious_values(series, column_name),
    }

    if inferred == "categorical" and distinct <= 30:
        counts = series.dropna().astype(str).value_counts().head(10)
        profile["top_values"] = {
            str(key): int(value) for key, value in counts.items()
        }

    sample_values = text.head(3).tolist()
    profile["sample_values"] = sample_values

    return profile


def profile_dataframe(df: pd.DataFrame, table_name: str) -> dict:
    """
    Full structural profile of one dataset.

    Shape: rows/columns, per-column detail, duplicate count, and the
    grouped column lists (numeric / categorical / datetime / id) the
    rest of the pipeline and the report both rely on.
    """
    rows = int(len(df))
    columns = [profile_column(df[column], str(column)) for column in df.columns]

    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        # Unhashable cell values (lists from nested JSON) break
        # duplicated(); stringify as a fallback rather than crash.
        duplicate_rows = int(df.astype(str).duplicated().sum())

    total_cells = rows * max(len(df.columns), 1)
    total_missing = sum(column["effective_missing"] for column in columns)

    by_type = {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "text": []}
    for column in columns:
        by_type.setdefault(column["inferred_type"], []).append(column["name"])

    return {
        "table_name": table_name,
        "rows": rows,
        "columns": len(df.columns),
        "column_names": [str(column) for column in df.columns],
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": (
            round(duplicate_rows / rows * 100, 2) if rows else 0.0
        ),
        "missing_cells": total_missing,
        "missing_percent": (
            round(total_missing / total_cells * 100, 2) if total_cells else 0.0
        ),
        "id_candidates": [
            column["name"] for column in columns if column["id_candidate"]
        ],
        "numeric_columns": by_type.get("numeric", []),
        "categorical_columns": by_type.get("categorical", []),
        "datetime_columns": by_type.get("datetime", []),
        "boolean_columns": by_type.get("boolean", []),
        "text_columns": by_type.get("text", []),
        "constant_columns": [
            column["name"] for column in columns if column["is_constant"]
        ],
        "columns_detail": columns,
    }
