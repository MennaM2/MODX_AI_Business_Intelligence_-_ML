import pandas as pd


def get_dataset_info(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }


def get_dataset_statistics(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "message": "No numerical columns found."
        }

    return numeric_df.describe().round(3).to_dict()


# ---------------------------------------------------------------------
# Dataset Profiling Skill
# ---------------------------------------------------------------------
# One level up from get_dataset_info/get_dataset_statistics: automatic
# column typing, outlier detection, top correlations, and candidate
# target-column suggestions. Used directly by the agent, and also by
# the report tool to build the "Automatic Data Report".

def get_dataset_profile(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if df.empty:
        return {"error": "The dataset is empty."}

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing_percent = (
        df.isnull().mean() * 100
    ).round(2).to_dict()

    # Outlier detection via the IQR method, capped to the first 15
    # numeric columns so the payload stays small on wide datasets.
    outliers = {}
    for column in numeric_columns[:15]:
        series = df[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(
            ((series < lower_bound) | (series > upper_bound)).sum()
        )
        if outlier_count > 0:
            outliers[column] = outlier_count

    # Strongest pairwise correlations, top 5 only.
    top_correlations = []
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        pairs = (
            corr.where(~corr.abs().eq(1.0))
            .unstack()
            .dropna()
            .abs()
            .sort_values(ascending=False)
        )
        seen = set()
        for (col_a, col_b), _value in pairs.items():
            key = frozenset((col_a, col_b))
            if key in seen:
                continue
            seen.add(key)
            top_correlations.append({
                "columns": [col_a, col_b],
                "correlation": round(float(corr.loc[col_a, col_b]), 4)
            })
            if len(top_correlations) >= 5:
                break

    # Heuristic candidate target columns: name-based hints, or
    # low-cardinality columns that look like a label/class/outcome.
    name_hints = {
        "target", "label", "churn", "class", "y",
        "outcome", "result", "default", "fraud"
    }
    candidate_targets = [
        column for column in df.columns
        if column.lower() in name_hints
    ]
    for column in categorical_columns:
        if column in candidate_targets:
            continue
        if 2 <= df[column].nunique() <= 10:
            candidate_targets.append(column)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_percent_by_column": missing_percent,
        "duplicate_rows": int(df.duplicated().sum()),
        "outlier_counts_by_column": outliers,
        "strongest_correlations": top_correlations,
        "candidate_target_columns": candidate_targets[:5]
    }