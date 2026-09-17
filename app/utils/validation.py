from pathlib import Path

import pandas as pd


def validate_csv(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        return {
            "valid": False,
            "error": "File does not exist."
        }

    if path.suffix.lower() != ".csv":
        return {
            "valid": False,
            "error": "Only CSV files are supported."
        }

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {
            "valid": False,
            "error": f"Unable to read CSV: {exc}"
        }

    if df.empty:
        return {
            "valid": False,
            "error": "The dataset is empty."
        }

    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist()
    }