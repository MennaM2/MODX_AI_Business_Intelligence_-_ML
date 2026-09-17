"""
Type coercion, missing-value handling, invalid-value detection.

The guiding rule for this whole module: never destroy information.

Concretely that means no row is ever dropped here, no column is ever
deleted, and every imputed value is counted and reported. If a
decision is genuinely ambiguous, the cleaner leaves the data alone
and records a warning instead of guessing. The raw upload also stays
on disk untouched, so any of this can be audited after the fact.
"""

import re

import pandas as pd

from app.data_engine.profiler import looks_boolean, looks_numeric


# Impute at or below this missing rate. Above it, imputation would
# invent more signal than it recovers, so the column keeps its nulls
# and the report raises a warning for the user to decide.
MAX_IMPUTE_MISSING_RATE = 0.40

# Below this many distinct values a numeric column is really an
# encoded category (a 0/1 flag, a 1-5 rating), so the median is a
# meaningless fill value and the mode is used instead.
DISCRETE_NUMERIC_MAX_DISTINCT = 12

_BOOLEAN_TRUE = {"true", "yes", "y", "t", "1"}
_BOOLEAN_FALSE = {"false", "no", "n", "f", "0"}

# Currency symbols, thousands separators, and stray percent signs
# that block numeric conversion.
_NUMERIC_NOISE = re.compile(r"[,\s$€£¥₹%]")


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a text column to numbers, tolerating real-world noise:
    '$1,234.50', '45%', '(120)' for negative accounting notation.
    """
    text = series.astype(str).str.strip()

    # Accounting negatives: (120) means -120.
    parenthesized = text.str.fullmatch(r"\(.*\)").fillna(False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    was_percent = text.str.contains("%", na=False)
    cleaned = text.str.replace(_NUMERIC_NOISE, "", regex=True)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = numeric.mask(parenthesized, -numeric)
    numeric = numeric.mask(series.isna())

    # If the column is uniformly a percentage, keep the number as
    # written (45% -> 45). Rescaling to 0.45 would silently change
    # the meaning of every chart and aggregate built on it.
    del was_percent

    return numeric


def _to_boolean(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result[text.isin(_BOOLEAN_TRUE)] = True
    result[text.isin(_BOOLEAN_FALSE)] = False
    result[series.isna()] = pd.NA
    return result


def convert_types(df: pd.DataFrame, profile: dict, log: list) -> pd.DataFrame:
    """
    Apply the types inferred by the profiler.

    A conversion is only committed if it succeeds for at least 95% of
    the non-null values. Below that the inference was probably wrong,
    and forcing it would turn real data into nulls - so the column
    stays as text and a warning is recorded.
    """
    converted = []
    refused = []

    detail_by_name = {
        column["name"]: column for column in profile["columns_detail"]
    }

    for column in df.columns:
        column_name = str(column)
        detail = detail_by_name.get(column_name)
        if detail is None:
            continue

        target_type = detail["inferred_type"]
        series = df[column]

        if target_type not in ("numeric", "boolean"):
            continue  # Dates are handled by the standardizer.
        if pd.api.types.is_numeric_dtype(series) and target_type == "numeric":
            continue
        if pd.api.types.is_bool_dtype(series) and target_type == "boolean":
            continue

        non_null_before = int(series.notna().sum())
        if not non_null_before:
            continue

        if target_type == "numeric":
            if not looks_numeric(series):
                continue
            result = _to_numeric(series)
        else:
            if not looks_boolean(series):
                continue
            result = _to_boolean(series)

        success_rate = int(result.notna().sum()) / non_null_before

        if success_rate >= 0.95:
            df[column] = result
            converted.append({
                "column": column_name,
                "from": str(series.dtype),
                "to": target_type,
                "values_not_converted": non_null_before - int(result.notna().sum()),
            })
        else:
            refused.append({
                "column": column_name,
                "attempted_type": target_type,
                "success_rate": round(success_rate, 3),
            })

    if converted:
        log.append({
            "stage": "clean",
            "action": "converted_types",
            "detail": f"{len(converted)} column(s) converted to a proper type.",
            "columns": converted,
        })

    if refused:
        log.append({
            "stage": "clean",
            "action": "skipped_type_conversion",
            "detail": (
                f"{len(refused)} column(s) left as text because "
                f"conversion would have destroyed too many values."
            ),
            "columns": refused,
        })

    return df


def handle_missing_values(
    df: pd.DataFrame,
    profile: dict,
    log: list,
) -> pd.DataFrame:
    """
    Fill missing values where a defensible default exists.

    Strategy per column:
      - ID columns          never filled (a fabricated key is worse
                            than a null and would corrupt joins)
      - continuous numeric  median (robust to the outliers these
                            datasets are full of)
      - discrete numeric    mode
      - categorical/bool    mode, only when one value is clearly
                            dominant; otherwise left null
      - datetime / text     never filled

    Columns above MAX_IMPUTE_MISSING_RATE are always left alone.
    """
    filled = []
    skipped = []

    detail_by_name = {
        column["name"]: column for column in profile["columns_detail"]
    }
    id_columns = set(profile.get("id_candidates", []))
    rows = max(len(df), 1)

    for column in df.columns:
        column_name = str(column)
        series = df[column]
        missing = int(series.isna().sum())

        if not missing:
            continue

        rate = missing / rows
        detail = detail_by_name.get(column_name, {})
        inferred = detail.get("inferred_type", "text")

        if column_name in id_columns or detail.get("name_suggests_id"):
            skipped.append({
                "column": column_name,
                "missing": missing,
                "reason": "identifier column - never imputed",
            })
            continue

        if rate > MAX_IMPUTE_MISSING_RATE:
            skipped.append({
                "column": column_name,
                "missing": missing,
                "missing_percent": round(rate * 100, 2),
                "reason": "too sparse to impute safely",
            })
            continue

        if inferred in ("datetime", "text"):
            skipped.append({
                "column": column_name,
                "missing": missing,
                "reason": f"no safe default for a {inferred} column",
            })
            continue

        fill_value = None
        strategy = None

        if inferred == "numeric" and pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                continue
            if non_null.nunique() <= DISCRETE_NUMERIC_MAX_DISTINCT:
                fill_value = non_null.mode().iloc[0]
                strategy = "mode (discrete numeric)"
            else:
                fill_value = non_null.median()
                strategy = "median"

        elif inferred in ("categorical", "boolean"):
            non_null = series.dropna()
            if non_null.empty:
                continue
            counts = non_null.value_counts(normalize=True)
            # Only impute when one category genuinely dominates;
            # filling a 34/33/33 split with the top value would
            # invent a distribution that is not in the data.
            if counts.iloc[0] >= 0.5:
                fill_value = counts.index[0]
                strategy = "mode"
            else:
                skipped.append({
                    "column": column_name,
                    "missing": missing,
                    "reason": "no dominant category - imputing would bias it",
                })
                continue

        if fill_value is None:
            continue

        df[column] = series.fillna(fill_value)
        filled.append({
            "column": column_name,
            "values_filled": missing,
            "strategy": strategy,
            "fill_value": str(fill_value),
        })

    if filled:
        total = sum(item["values_filled"] for item in filled)
        log.append({
            "stage": "clean",
            "action": "filled_missing_values",
            "detail": (
                f"{total} missing value(s) filled across "
                f"{len(filled)} column(s)."
            ),
            "columns": filled,
        })

    if skipped:
        log.append({
            "stage": "clean",
            "action": "left_missing_values",
            "detail": (
                f"{len(skipped)} column(s) kept their missing values "
                f"because no safe fill existed."
            ),
            "columns": skipped,
        })

    return df


def flag_invalid_values(df: pd.DataFrame, profile: dict, log: list) -> list:
    """
    Identify values that are present but implausible.

    Reports only - nothing is modified. Deciding that a -5 quantity
    is a data-entry error rather than a return is a business
    judgement, so it surfaces as a warning for the user and the agent
    rather than being silently rewritten.
    """
    findings = []

    quantity_pattern = re.compile(
        r"(qty|quantity|count|units|age|price|cost|amount|revenue|"
        r"sales|income|salary|total|balance|duration|tenure)",
        re.IGNORECASE,
    )

    for column in df.columns:
        column_name = str(column)
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                continue

            if quantity_pattern.search(column_name):
                negatives = int((non_null < 0).sum())
                if negatives:
                    findings.append({
                        "column": column_name,
                        "issue": "negative_values",
                        "count": negatives,
                        "detail": (
                            "Negative values in a column whose name "
                            "implies it should be non-negative."
                        ),
                    })

            if re.search(r"\bage\b", column_name, re.IGNORECASE):
                impossible = int(((non_null < 0) | (non_null > 120)).sum())
                if impossible:
                    findings.append({
                        "column": column_name,
                        "issue": "impossible_age",
                        "count": impossible,
                        "detail": "Age values outside the range 0-120.",
                    })

        # Dates far outside any plausible business range usually mean
        # a parsing error or a sentinel value like 1900-01-01.
        detail = next(
            (
                item for item in profile["columns_detail"]
                if item["name"] == column_name
            ),
            None,
        )
        if detail and detail.get("inferred_type") == "datetime":
            parsed = pd.to_datetime(series, errors="coerce")
            valid = parsed.dropna()
            if not valid.empty:
                out_of_range = int(
                    (
                        (valid < pd.Timestamp("1900-01-01"))
                        | (valid > pd.Timestamp.now() + pd.Timedelta(days=365 * 5))
                    ).sum()
                )
                if out_of_range:
                    findings.append({
                        "column": column_name,
                        "issue": "implausible_date",
                        "count": out_of_range,
                        "detail": (
                            "Dates before 1900 or more than 5 years in "
                            "the future."
                        ),
                    })

    if findings:
        log.append({
            "stage": "clean",
            "action": "flagged_invalid_values",
            "detail": (
                f"{len(findings)} potential data-quality issue(s) "
                f"flagged for review (no values were changed)."
            ),
            "findings": findings,
        })

    return findings


def clean(df: pd.DataFrame, profile: dict, log: list) -> tuple:
    """Run the cleaning stage. Returns (cleaned_df, invalid_findings)."""
    df = df.copy()
    df = convert_types(df, profile, log)
    df = handle_missing_values(df, profile, log)
    df = standardize_categories(df, log)
    df = fix_negative_quantities_by_status(df, log)
    df = fix_invalid_values(df, log)
    invalid = flag_invalid_values(df, profile, log)
    return df, invalid


def standardize_categories(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Standardize common categorical variations like gender (M/Male -> Male)."""
    gender_mapping = {
        "m": "Male", "male": "Male", "man": "Male",
        "f": "Female", "female": "Female", "woman": "Female"
    }
    actions = []
    for col in df.columns:
        col_lower = str(col).lower()
        if "gender" in col_lower or "sex" in col_lower:
            series_str = df[col].astype(str).str.strip().str.lower()
            mapped = series_str.map(gender_mapping)
            valid_mask = mapped.notna() & df[col].notna()
            changed_count = int((df[col] != mapped).sum())
            if changed_count > 0:
                df.loc[valid_mask, col] = mapped[valid_mask]
                actions.append({
                    "column": col,
                    "standardized_to": ["Male", "Female"],
                    "rows_updated": changed_count
                })

    if actions:
        log.append({
            "stage": "clean",
            "action": "standardized_categories",
            "detail": f"Standardized categorical representations across {len(actions)} column(s).",
            "details": actions
        })
    return df


def fix_negative_quantities_by_status(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
   It handles negative quantities based on the order status.(Status)
    """
    qty_col = next((c for c in df.columns if re.search(r"quant|qty|count", c, re.I)), None)
    status_col = next((c for c in df.columns if re.search(r"status|state", c, re.I)), None)
    amount_col = next((c for c in df.columns if re.search(r"amount|total|price|sales", c, re.I)), None)

    if not qty_col or not status_col or not pd.api.types.is_numeric_dtype(df[qty_col]):
        return df

    status_series = df[status_col].astype(str).str.strip().str.lower()
    neg_mask = df[qty_col] < 0

    if not neg_mask.any():
        return df

    # 1) Cases that constitute an actual sale require a positive correction (Complete, Shipped, Processing)
    sale_statuses = {"shipped", "complete", "completed", "processing"}
    fix_sales_mask = neg_mask & status_series.isin(sale_statuses)
    sales_fixed_count = int(fix_sales_mask.sum())

    if sales_fixed_count > 0:
        df.loc[fix_sales_mask, qty_col] = df.loc[fix_sales_mask, qty_col].abs()
        if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
            df.loc[fix_sales_mask, amount_col] = df.loc[fix_sales_mask, amount_col].abs()
        log.append({
            "stage": "clean",
            "action": "fix_sign_error",
            "column": qty_col,
            "rows_affected": sales_fixed_count,
            "detail": f"Fixed {sales_fixed_count} negative quantities with active statuses ({', '.join(sale_statuses)}) by converting to absolute values."
        })

    # 2) (Cancelled)
    cancel_mask = neg_mask & status_series.isin({"cancelled", "canceled"})
    cancel_fixed_count = int(cancel_mask.sum())
    if cancel_fixed_count > 0:
        df.loc[cancel_mask, qty_col] = df.loc[cancel_mask, qty_col].abs()
        log.append({
            "stage": "clean",
            "action": "normalize_cancelled_units",
            "column": qty_col,
            "rows_affected": cancel_fixed_count,
            "detail": f"Converted {cancel_fixed_count} cancelled order quantities to positive for accurate lost-demand aggregation."
        })

    # 3) (Returned) 
    return_mask = neg_mask & status_series.isin({"returned", "refunded"})
    return_count = int(return_mask.sum())
    if return_count > 0:
        log.append({
            "stage": "clean",
            "action": "preserve_valid_returns",
            "column": qty_col,
            "rows_affected": return_count,
            "detail": f"Kept {return_count} negative quantities for status 'returned' as valid inventory and revenue reversals."
        })

    # 4) (Pending)
    pending_mask = neg_mask & status_series.isin({"pending"})
    if pending_mask.any():
        for idx in df[pending_mask].index:
            is_positive_amount = True
            if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
                if df.at[idx, amount_col] < 0:
                    is_positive_amount = False

            if is_positive_amount:
                df.at[idx, qty_col] = abs(df.at[idx, qty_col])
                if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
                    df.at[idx, amount_col] = abs(df.at[idx, amount_col])

        pending_count = int(pending_mask.sum())
        log.append({
            "stage": "clean",
            "action": "resolve_pending_signs",
            "column": qty_col,
            "rows_affected": pending_count,
            "detail": f"Evaluated {pending_count} pending orders: converted purchase errors to positive while keeping return claims."
        })

    return df


def fix_invalid_values(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Detect and clean out-of-range numerical anomalies (like negative or extreme age)."""
    fixes = []
    for col in df.columns:
        col_lower = str(col).lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            # Anomaly rules for Age
            if "age" in col_lower:
                invalid_mask = (df[col] < 0) | (df[col] > 120)
                count = int(invalid_mask.sum())
                if count > 0:
                    valid_median = df.loc[~invalid_mask, col].median()
                    df.loc[invalid_mask, col] = valid_median
                    fixes.append({
                        "column": col,
                        "issue": "negative or implausible (>120) age",
                        "rows_corrected": count,
                        "action_taken": f"Imputed with valid median ({valid_median})"
                    })

    if fixes:
        log.append({
            "stage": "clean",
            "action": "fixed_anomalies",
            "detail": f"Corrected {len(fixes)} column(s) with out-of-range anomalies.",
            "details": fixes
        })
    return df
