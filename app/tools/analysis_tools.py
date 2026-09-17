import pandas as pd


def analyze_column(
    file_path: str,
    column_name: str
) -> dict:

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if column_name not in df.columns:
        return {
            "error": f"Column '{column_name}' was not found."
        }

    series = df[column_name]

    result = {
        "column": column_name,
        "dtype": str(series.dtype),
        "unique_values": int(series.nunique()),
        "missing_values": int(series.isnull().sum())
    }

    if pd.api.types.is_numeric_dtype(series):
        result["mean"] = float(series.mean())
        result["median"] = float(series.median())
        result["min"] = float(series.min())
        result["max"] = float(series.max())

    else:
        result["top_values"] = (
            series.value_counts()
            .head(10)
            .to_dict()
        )

    return result


def compare_categories(
    file_path: str,
    category_column: str,
    target_column: str
) -> dict:

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if category_column not in df.columns:
        return {
            "error": f"Column '{category_column}' was not found."
        }

    if target_column not in df.columns:
        return {
            "error": f"Column '{target_column}' was not found."
        }

    grouped = (
        df.groupby(category_column)[target_column]
        .agg(["count", "mean"])
        .reset_index()
    )

    return grouped.round(4).to_dict(
        orient="records"
    )


# ---------------------------------------------------------------------
# Top/Bottom Performers Skill
# ---------------------------------------------------------------------
# compare_categories (above) reports count+mean per category. This
# tool ranks categories by an aggregated metric to directly answer
# "which products/regions are underperforming" or "top N by revenue"
# without the caller having to sort compare_categories output itself.

VALID_AGGREGATIONS = {"sum", "mean", "count", "median"}


def rank_categories(
    file_path: str,
    category_column: str,
    metric_column: str,
    aggregation: str = "sum",
    top_n: int = 5
) -> dict:
    """
    Rank categories (e.g. products, regions, customers) by an
    aggregated numeric metric, returning both the top and bottom
    performers.
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if category_column not in df.columns:
        return {
            "error": f"Column '{category_column}' was not found."
        }

    if metric_column not in df.columns:
        return {
            "error": f"Column '{metric_column}' was not found."
        }

    if not pd.api.types.is_numeric_dtype(df[metric_column]):
        return {
            "error": f"Column '{metric_column}' is not numeric."
        }

    if aggregation not in VALID_AGGREGATIONS:
        return {
            "error": (
                f"aggregation must be one of {sorted(VALID_AGGREGATIONS)}."
            )
        }

    top_n = max(1, min(int(top_n), 25))

    grouped = (
        df.groupby(category_column)[metric_column]
        .agg(aggregation)
        .round(4)
        .sort_values(ascending=False)
    )

    top = grouped.head(top_n)
    bottom = grouped.tail(top_n).sort_values()

    return {
        "category_column": category_column,
        "metric_column": metric_column,
        "aggregation": aggregation,
        "category_count": int(grouped.shape[0]),
        "top_performers": [
            {"category": str(key), "value": float(value)}
            for key, value in top.items()
        ],
        "bottom_performers": [
            {"category": str(key), "value": float(value)}
            for key, value in bottom.items()
        ]
    }