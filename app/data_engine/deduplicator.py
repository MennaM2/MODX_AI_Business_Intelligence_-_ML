"""
Duplicate detection and removal.

Two very different cases, handled differently on purpose:

Exact duplicate rows - every column identical - are removed. A fully
identical row carries no information that the first copy does not,
and leaving it in inflates every count, sum, and average downstream.

Duplicate *keys* - two rows sharing a customer_id but differing
elsewhere - are only reported. Those are usually either a legitimate
one-to-many relationship or a genuine conflict that needs a human
decision; dropping one side silently would lose real data and
quietly change join results.
"""

import pandas as pd


def remove_exact_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Drop fully identical rows, keeping the first occurrence."""
    try:
        mask = df.duplicated(keep="first")
    except TypeError:
        # Unhashable values (lists from nested JSON) - compare on a
        # stringified view instead of failing the whole pipeline.
        mask = df.astype(str).duplicated(keep="first")

    removed = int(mask.sum())
    if not removed:
        return df

    result = df[~mask].reset_index(drop=True)

    log.append({
        "stage": "deduplicate",
        "action": "removed_exact_duplicates",
        "detail": (
            f"{removed} fully identical row(s) removed "
            f"({len(df)} -> {len(result)} rows)."
        ),
        "rows_removed": removed,
    })

    return result


def find_duplicate_keys(df: pd.DataFrame, profile: dict) -> list:
    """
    Report ID columns that are not actually unique.

    This is the signal that decides join direction later: a key that
    is unique is a valid join target ("one" side), a key that repeats
    is the "many" side. It also catches the real error case - a
    primary key that should be unique and isn't.
    """
    findings = []

    candidates = list(profile.get("id_candidates", []))
    for column in profile.get("columns_detail", []):
        if column.get("name_suggests_id") and column["name"] not in candidates:
            candidates.append(column["name"])

    for column_name in candidates:
        if column_name not in df.columns:
            continue

        series = df[column_name].dropna()
        if series.empty:
            continue

        distinct = int(series.nunique())
        duplicated = int(len(series) - distinct)

        if duplicated > 0:
            findings.append({
                "column": column_name,
                "duplicate_values": duplicated,
                "distinct_values": distinct,
                "detail": (
                    f"'{column_name}' repeats across rows - it is not a "
                    f"unique identifier in this dataset."
                ),
            })

    return findings


def deduplicate(df: pd.DataFrame, profile: dict, log: list) -> tuple:
    """Run the deduplication stage. Returns (df, duplicate_key_findings)."""
    rows_before = len(df)
    df = remove_exact_duplicates(df, log)
    duplicate_keys = find_duplicate_keys(df, profile)

    if duplicate_keys:
        log.append({
            "stage": "deduplicate",
            "action": "flagged_duplicate_keys",
            "detail": (
                f"{len(duplicate_keys)} identifier column(s) contain "
                f"repeated values (reported, not modified)."
            ),
            "findings": duplicate_keys,
        })

    return df, {
        "rows_before": rows_before,
        "rows_after": len(df),
        "exact_duplicates_removed": rows_before - len(df),
        "duplicate_key_columns": duplicate_keys,
    }
