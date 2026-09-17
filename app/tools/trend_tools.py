"""
Trend Analysis Skill.

Two tools:
- analyze_trends: how a metric moved over time (direction, % change,
  biggest single-period swings).
- compare_periods: a direct "period A vs period B" comparison, with
  an optional `group_column` breakdown (e.g. per-region) so the agent
  can answer "which region grew the most?" from one tool call.
"""

import pandas as pd


_FREQ_MAP = {
    "day": "D",
    "week": "W",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}

# pandas 2.2+ / 3.x deprecated the bare "M"/"Q"/"Y" resample aliases in
# favor of "ME"/"QE"/"YE" (month/quarter/year *end*). Period() still
# uses the short codes, so we keep both mappings.
_RESAMPLE_FREQ_MAP = {
    "day": "D",
    "week": "W",
    "month": "ME",
    "quarter": "QE",
    "year": "YE",
}


def _resample_code(freq: str) -> str:
    return _RESAMPLE_FREQ_MAP.get(freq, "ME")


def _period_code(freq: str) -> str:
    return _FREQ_MAP.get(freq, "M")


def analyze_trends(
    file_path: str,
    date_column: str,
    value_column: str,
    freq: str = "month",
) -> dict:
    """
    Resample `value_column` by `freq` over `date_column` and describe
    the trend: period-by-period totals, overall direction, total %
    change, and the single largest rise/drop between two consecutive
    periods.
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if date_column not in df.columns:
        return {"error": f"Column '{date_column}' was not found."}
    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' was not found."}
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"error": f"Column '{value_column}' is not numeric."}

    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.isna().all():
        return {"error": f"Column '{date_column}' could not be parsed as dates."}

    working = pd.DataFrame({"date": dates, "value": df[value_column]}).dropna()
    if len(working) < 2:
        return {
            "error": (
                "Not enough valid rows with both a date and a value "
                "to compute a trend."
            )
        }

    series = working.set_index("date")["value"].resample(_resample_code(freq)).sum()

    if len(series) < 2:
        return {"error": "Not enough distinct time periods to compute a trend."}

    pct_changes = series.pct_change() * 100
    start_value, end_value = float(series.iloc[0]), float(series.iloc[-1])

    total_change_percent = (
        round((end_value - start_value) / start_value * 100, 2)
        if start_value != 0 else None
    )

    if end_value > start_value:
        direction = "increasing"
    elif end_value < start_value:
        direction = "decreasing"
    else:
        direction = "flat"

    periods = [
        {"period": str(idx.date()), "value": round(float(val), 2)}
        for idx, val in series.items()
    ]

    valid_changes = pct_changes.dropna()
    largest_drop = None
    largest_rise = None
    if not valid_changes.empty:
        drop_idx = valid_changes.idxmin()
        rise_idx = valid_changes.idxmax()
        if valid_changes.loc[drop_idx] < 0:
            largest_drop = {
                "period": str(drop_idx.date()),
                "change_percent": round(float(valid_changes.loc[drop_idx]), 2),
            }
        if valid_changes.loc[rise_idx] > 0:
            largest_rise = {
                "period": str(rise_idx.date()),
                "change_percent": round(float(valid_changes.loc[rise_idx]), 2),
            }

    return {
        "date_column": date_column,
        "value_column": value_column,
        "frequency": freq,
        "periods": periods,
        "overall_direction": direction,
        "total_change_percent": total_change_percent,
        "largest_period_over_period_drop": largest_drop,
        "largest_period_over_period_rise": largest_rise,
    }


def compare_periods(
    file_path: str,
    date_column: str,
    value_column: str,
    freq: str = "month",
    period_a: str = None,
    period_b: str = None,
    group_column: str = None,
) -> dict:
    """
    Compare `value_column` between two periods.

    If `period_a`/`period_b` are omitted, the two most recent periods
    in the data are compared automatically. Periods are given in a
    format matching `freq`, e.g. freq="month" -> "2024-05".

    If `group_column` is given (e.g. region, product), also returns a
    per-group breakdown ranked by percent change, so the caller can
    answer "which group grew/declined the most between these periods".
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    required_columns = [date_column, value_column]
    if group_column:
        required_columns.append(group_column)
    for column in required_columns:
        if column not in df.columns:
            return {"error": f"Column '{column}' was not found."}

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"error": f"Column '{value_column}' is not numeric."}

    dates = pd.to_datetime(df[date_column], errors="coerce")

    frame = pd.DataFrame({"date": dates, "value": df[value_column]})
    if group_column:
        frame["group"] = df[group_column]
    frame = frame.dropna(subset=["date", "value"])

    if frame.empty:
        return {
            "error": (
                "No valid rows with both a parseable date and a "
                "numeric value."
            )
        }

    period_code = _period_code(freq)
    frame["period"] = frame["date"].dt.to_period(period_code)

    overall = frame.groupby("period")["value"].sum().sort_index()

    if period_a and period_b:
        try:
            period_a_key = pd.Period(period_a, freq=period_code)
            period_b_key = pd.Period(period_b, freq=period_code)
        except Exception:
            return {
                "error": (
                    f"Could not parse period_a/period_b for frequency "
                    f"'{freq}'. Use a format like '2024-05' for month."
                )
            }
    else:
        if len(overall) < 2:
            return {"error": "Not enough distinct time periods to compare."}
        period_a_key, period_b_key = overall.index[-2], overall.index[-1]

    if period_a_key not in overall.index or period_b_key not in overall.index:
        return {"error": "Could not find the requested periods in the data."}

    value_a = float(overall.loc[period_a_key])
    value_b = float(overall.loc[period_b_key])
    change = value_b - value_a
    percent_change = round(change / value_a * 100, 2) if value_a != 0 else None

    if change > 0:
        direction = "increase"
    elif change < 0:
        direction = "decrease"
    else:
        direction = "no change"

    result = {
        "period_a": str(period_a_key),
        "period_b": str(period_b_key),
        "value_a": round(value_a, 2),
        "value_b": round(value_b, 2),
        "absolute_change": round(change, 2),
        "percent_change": percent_change,
        "direction": direction,
    }

    if group_column:
        pivot = (
            frame.groupby(["group", "period"])["value"]
            .sum()
            .unstack("period")
        )

        if period_a_key in pivot.columns and period_b_key in pivot.columns:
            group_a = pivot[period_a_key].fillna(0)
            group_b = pivot[period_b_key].fillna(0)

            breakdown = []
            for group_name in pivot.index:
                a_value = float(group_a.get(group_name, 0.0))
                b_value = float(group_b.get(group_name, 0.0))
                group_change = b_value - a_value
                group_pct = (
                    round(group_change / a_value * 100, 2)
                    if a_value != 0 else None
                )
                breakdown.append({
                    "group": str(group_name),
                    "value_a": round(a_value, 2),
                    "value_b": round(b_value, 2),
                    "absolute_change": round(group_change, 2),
                    "percent_change": group_pct,
                })

            ranked = sorted(
                (b for b in breakdown if b["percent_change"] is not None),
                key=lambda b: b["percent_change"],
                reverse=True,
            )

            result["group_column"] = group_column
            result["top_growth"] = ranked[:5]
            result["top_decline"] = list(reversed(ranked[-5:])) if ranked else []

    return result
