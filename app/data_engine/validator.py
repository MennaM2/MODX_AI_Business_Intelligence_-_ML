"""
Post-preparation validation.

Runs after cleaning and integration and answers one question: is this
data now safe to analyse, and what should the user be told before
they trust a number that comes out of it?

Every check returns one of three outcomes:

  passed   the check ran and the data is fine
  warning  something worth knowing that does not invalidate analysis
  failed   something that will produce wrong answers if ignored

Nothing here modifies data. Validation that silently fixes things is
how bad data becomes invisible.
"""

import pandas as pd


# Above this share of missing values a column is too sparse for any
# conclusion drawn from it to be reliable.
HIGH_MISSING_WARNING = 0.20

# A categorical column with more distinct values than this is
# probably free text or an un-normalized category set.
HIGH_CARDINALITY_WARNING = 50


def _check(status: str, message: str, **extra) -> dict:
    result = {"status": status, "message": message}
    result.update(extra)
    return result


def validate_dataset(
    df: pd.DataFrame,
    profile: dict,
    table_name: str,
    invalid_findings: list = None,
    duplicate_info: dict = None,
) -> dict:
    """Validate one prepared dataset. Returns a per-table report."""
    checks = []
    rows = len(df)

    if rows == 0:
        checks.append(_check("failed", "The dataset has no rows."))
        return _summarize(table_name, checks)

    # --- identifiers -------------------------------------------------
    id_columns = profile.get("id_candidates", [])
    for column in id_columns:
        if column not in df.columns:
            continue
        series = df[column]
        duplicated = int(series.dropna().duplicated().sum())
        nulls = int(series.isna().sum())

        if duplicated == 0 and nulls == 0:
            checks.append(
                _check("passed", f"'{column}' is unique and complete.",
                       column=column)
            )
        elif nulls:
            checks.append(
                _check(
                    "failed",
                    f"'{column}' is an identifier but has {nulls:,} "
                    f"missing value(s) - joins on it will drop rows.",
                    column=column,
                )
            )
        else:
            checks.append(
                _check(
                    "warning",
                    f"'{column}' repeats {duplicated:,} time(s); it is "
                    f"not a unique key in this dataset.",
                    column=column,
                )
            )

    # --- missing values ----------------------------------------------
    high_missing = []
    for column in profile.get("columns_detail", []):
        if column["missing_percent"] / 100 >= HIGH_MISSING_WARNING:
            high_missing.append(
                (column["name"], column["missing_percent"])
            )

    if high_missing:
        for name, percent in high_missing:
            checks.append(
                _check(
                    "warning",
                    f"{percent:.1f}% of '{name}' is missing.",
                    column=name,
                )
            )
    else:
        checks.append(
            _check("passed", "No column is missing an unusual share of values.")
        )

    # --- dates --------------------------------------------------------
    date_columns = profile.get("datetime_columns", [])
    for column in date_columns:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        non_null = int(df[column].notna().sum())
        if not non_null:
            continue
        unparseable = non_null - int(parsed.notna().sum())
        if unparseable:
            checks.append(
                _check(
                    "warning",
                    f"{unparseable:,} value(s) in '{column}' could not be "
                    f"read as a date.",
                    column=column,
                )
            )
        else:
            checks.append(
                _check("passed", f"All dates in '{column}' are valid.",
                       column=column)
            )

    # --- invalid / impossible values ----------------------------------
    for finding in (invalid_findings or []):
        checks.append(
            _check(
                "warning",
                f"{finding['count']:,} row(s): {finding['detail']}",
                column=finding.get("column"),
            )
        )

    # --- duplicates ----------------------------------------------------
    removed = (duplicate_info or {}).get("exact_duplicates_removed", 0)
    if removed:
        checks.append(
            _check(
                "passed",
                f"{removed:,} exact duplicate row(s) were removed.",
            )
        )
    else:
        checks.append(_check("passed", "No exact duplicate rows remain."))

    # --- category consistency -------------------------------------------
    for column in profile.get("categorical_columns", []):
        if column not in df.columns:
            continue
        distinct = int(df[column].nunique(dropna=True))
        if distinct > HIGH_CARDINALITY_WARNING:
            checks.append(
                _check(
                    "warning",
                    f"'{column}' has {distinct:,} distinct values - it may "
                    f"contain inconsistent or un-normalized categories.",
                    column=column,
                )
            )

    # --- constants -------------------------------------------------------
    constants = profile.get("constant_columns", [])
    if constants:
        checks.append(
            _check(
                "warning",
                f"{len(constants)} column(s) hold a single value and carry "
                f"no information: {', '.join(constants[:5])}"
                + ("..." if len(constants) > 5 else ""),
            )
        )

    return _summarize(table_name, checks)


def validate_integration(integrated: list) -> list:
    """
    Check the results of any materialized join.

    Unmatched foreign keys are the thing that most often invalidates
    a joined analysis - a revenue total that silently excludes 30% of
    orders looks perfectly reasonable on a chart.
    """
    checks = []

    for dataset in integrated:
        name = dataset["table_name"]
        unmatched = dataset["unmatched_rows"]
        percent = dataset["unmatched_percent"]

        if unmatched == 0:
            checks.append(
                _check(
                    "passed",
                    f"Every row in '{dataset['left_table']}' matched a row "
                    f"in '{dataset['right_table']}'.",
                    table=name,
                )
            )
        elif percent >= 10:
            checks.append(
                _check(
                    "failed",
                    f"{unmatched:,} row(s) ({percent:.1f}%) of "
                    f"'{dataset['left_table']}' have no match in "
                    f"'{dataset['right_table']}' - aggregates over "
                    f"'{name}' will have gaps.",
                    table=name,
                )
            )
        else:
            checks.append(
                _check(
                    "warning",
                    f"{unmatched:,} row(s) ({percent:.1f}%) of "
                    f"'{dataset['left_table']}' did not match.",
                    table=name,
                )
            )

        if dataset["rows"] > dataset.get("expected_rows", dataset["rows"]):
            checks.append(
                _check(
                    "failed",
                    f"'{name}' has more rows than its source table - the "
                    f"join key was not unique.",
                    table=name,
                )
            )

    return checks


def _summarize(table_name: str, checks: list) -> dict:
    return {
        "table_name": table_name,
        "passed": [c for c in checks if c["status"] == "passed"],
        "warnings": [c for c in checks if c["status"] == "warning"],
        "failed": [c for c in checks if c["status"] == "failed"],
        "checks": checks,
    }
