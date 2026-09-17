"""
Anomaly Detection Skill.

Method selection is data-driven rather than hardcoded to one
algorithm:

- A specific `column` given -> univariate detection combining IQR
  (robust to non-normal distributions) and Z-score (catches extreme
  single points), which works even on small datasets.
- No column given, and enough numeric data is available -> Isolation
  Forest across all numeric columns, which catches unusual
  *combinations* of values that no single-column check would flag
  (e.g. high revenue with very low transaction count).
- No column given, but too little data/columns for a multivariate
  model -> falls back to a per-column IQR summary instead of forcing
  Isolation Forest onto a handful of rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


MAX_LISTED_ANOMALIES = 20
MIN_ROWS_FOR_ISOLATION_FOREST = 20


def _iqr_bounds(series: pd.Series):
    clean = series.dropna()
    if len(clean) < 4:
        return None, None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None, None
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _zscore_flags(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    clean = series.dropna()
    if len(clean) < 4 or clean.std() == 0:
        return pd.Series(False, index=clean.index)
    z_scores = (clean - clean.mean()) / clean.std()
    return z_scores.abs() > threshold


def _detect_single_column(
    df: pd.DataFrame,
    column: str,
    date_column: str = None,
) -> dict:
    series = df[column]
    lower, upper = _iqr_bounds(series)
    z_flags = _zscore_flags(series)

    if lower is not None:
        iqr_flagged = set(series.dropna()[
            (series.dropna() < lower) | (series.dropna() > upper)
        ].index)
    else:
        iqr_flagged = set()

    flagged_index = sorted(iqr_flagged | set(z_flags[z_flags].index))

    anomalies = []
    for idx in flagged_index[:MAX_LISTED_ANOMALIES]:
        record = {"row_index": int(idx), "value": float(series.loc[idx])}
        if date_column and date_column in df.columns:
            record["date"] = str(df.loc[idx, date_column])
        anomalies.append(record)

    total_valid = int(series.dropna().shape[0])

    return {
        "method": "IQR + Z-score (univariate)",
        "column": column,
        "total_rows": total_valid,
        "anomaly_count": len(flagged_index),
        "anomaly_rate_percent": (
            round(len(flagged_index) / total_valid * 100, 2)
            if total_valid else 0.0
        ),
        "bounds": (
            {"lower": round(lower, 4), "upper": round(upper, 4)}
            if lower is not None else None
        ),
        "anomalies": anomalies,
        "truncated": len(flagged_index) > MAX_LISTED_ANOMALIES,
    }


def _detect_multivariate(df: pd.DataFrame, date_column: str = None) -> dict:
    numeric_df = df.select_dtypes(include="number").dropna()

    if numeric_df.shape[1] < 2 or len(numeric_df) < MIN_ROWS_FOR_ISOLATION_FOREST:
        # Not enough columns/rows for a multivariate model - fall
        # back to a per-column IQR summary instead of forcing it.
        summary = {}
        for column in df.select_dtypes(include="number").columns:
            lower, upper = _iqr_bounds(df[column])
            if lower is None:
                continue
            clean = df[column].dropna()
            count = int(((clean < lower) | (clean > upper)).sum())
            if count > 0:
                summary[column] = count

        return {
            "method": (
                "IQR per-column (fallback - not enough rows/columns "
                "for multivariate detection)"
            ),
            "anomalies_by_column": summary,
        }

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    predictions = model.fit_predict(numeric_df)

    anomaly_positions = np.where(predictions == -1)[0]
    anomaly_index = numeric_df.index[anomaly_positions]

    anomalies = []
    for idx in list(anomaly_index)[:MAX_LISTED_ANOMALIES]:
        record = {"row_index": int(idx)}
        for column in numeric_df.columns:
            record[column] = float(df.loc[idx, column])
        if date_column and date_column in df.columns:
            record["date"] = str(df.loc[idx, date_column])
        anomalies.append(record)

    total_rows = int(len(numeric_df))

    return {
        "method": "Isolation Forest (multivariate)",
        "columns_used": numeric_df.columns.tolist(),
        "total_rows": total_rows,
        "anomaly_count": int(len(anomaly_index)),
        "anomaly_rate_percent": round(len(anomaly_index) / total_rows * 100, 2),
        "anomalies": anomalies,
        "truncated": len(anomaly_index) > MAX_LISTED_ANOMALIES,
    }


def detect_anomalies(
    file_path: str,
    column: str = None,
    date_column: str = None,
) -> dict:
    """
    Detect unusual values.

    - `column` given: univariate anomaly detection on that numeric
      column (e.g. revenue, transaction_amount).
    - `column` omitted: multivariate anomaly detection across all
      numeric columns, to catch unusual combinations of values.
    - `date_column` (optional): included on each flagged row so the
      caller can explain *when* anomalies happened.
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if date_column and date_column not in df.columns:
        return {"error": f"Column '{date_column}' was not found."}

    if column:
        if column not in df.columns:
            return {"error": f"Column '{column}' was not found."}
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric."}
        return _detect_single_column(df, column, date_column)

    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        return {"error": "No numeric columns found to check for anomalies."}

    return _detect_multivariate(df, date_column)
