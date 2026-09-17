import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CAUSATION_NOTE = (
    "Feature importance reflects predictive association within this "
    "dataset, not a causal effect. Higher importance does not mean a "
    "feature causes the outcome."
)


ID_LIKE_NAMES = {"id", "customerid", "customer_id", "index"}


def analyze_churn(file_path: str) -> dict:

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
            "error": "No Churn target column was found."
        }

    target = df[target_column]

    if not pd.api.types.is_numeric_dtype(target):
        target = (
            target.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    valid = target.notna()

    df = df.loc[valid].copy()
    target = target.loc[valid]

    if target.nunique() < 2:
        return {
            "error": "The target must contain at least two classes."
        }

    X = df.drop(columns=[target_column])

    id_columns = [
        column
        for column in X.columns
        if column.lower()
        in {"id", "customerid", "customer_id"}
    ]

    X = X.drop(
        columns=id_columns,
        errors="ignore"
    )

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, target)

    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        pipeline
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(10)
    )

    return {
        "rows_used": len(df),
        "churn_rate_percent": round(
            float(target.mean() * 100),
            2
        ),
        "top_features": (
            importance_df
            .round(4)
            .to_dict(orient="records")
        )
    }


# ---------------------------------------------------------------------
# Machine Learning Skill
# ---------------------------------------------------------------------
# A generic train/evaluate tool that works for any target column, not
# just churn. It auto-detects classification vs. regression, trains a
# baseline Random Forest with a held-out test split, and returns the
# metrics appropriate to the task. `analyze_churn` above is left
# untouched as the dedicated churn tool.

def _detect_task_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 15:
        return "regression"
    return "classification"


def train_model(
    file_path: str,
    target_column: str,
    task_type: str = None,
    feature_columns: list = None
) -> dict:
    """
    Train a baseline model to predict `target_column`.

    task_type: "classification", "regression", or omitted to
    auto-detect from the target column's data type and cardinality.

    feature_columns: optional list of column names to use as the
    only features. When omitted, every remaining column (minus
    obvious id columns) is used - which is dangerous for datasets
    with high-cardinality text columns like timestamps, since
    OneHotEncoder will explode those into one column per unique
    value (seen in production: a few timestamp columns alone
    produced 10,000+ one-hot columns from a 5,000-row sample),
    making training take far longer than a typical request timeout.
    Pass feature_columns whenever the user names specific columns to
    predict from.
    """

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    if target_column not in df.columns:
        return {
            "error": f"Column '{target_column}' was not found."
        }

    df = df.dropna(subset=[target_column]).copy()

    if len(df) < 10:
        return {
            "error": (
                "Not enough non-missing rows in the target column "
                "to train a model (need at least 10)."
            )
        }

    target = df[target_column]

    resolved_task_type = task_type or _detect_task_type(target)

    if resolved_task_type not in ("classification", "regression"):
        return {
            "error": (
                "task_type must be 'classification' or 'regression'."
            )
        }

    label_mapping = None

    if resolved_task_type == "classification":
        if not pd.api.types.is_numeric_dtype(target):
            categories = sorted(target.astype(str).unique())
            label_mapping = {
                category: index
                for index, category in enumerate(categories)
            }
            target = target.astype(str).map(label_mapping)

        if target.nunique() < 2:
            return {
                "error": (
                    "The target must contain at least two classes "
                    "for classification."
                )
            }

    X = df.drop(columns=[target_column])

    if feature_columns:
        missing = [
            column for column in feature_columns
            if column not in X.columns
        ]
        if missing:
            return {
                "error": (
                    f"Unknown feature_columns {missing}. Available "
                    f"columns: {X.columns.tolist()}"
                )
            }
        X = X[feature_columns].copy()
    else:
        id_columns = [
            column
            for column in X.columns
            if column.lower() in ID_LIKE_NAMES
        ]

        X = X.drop(columns=id_columns, errors="ignore")

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    dropped_high_cardinality = []
    if not feature_columns:
        # Defense in depth: even if the caller forgot to pass
        # feature_columns, never let a column with too many unique
        # values (e.g. a timestamp) go into OneHotEncoder - that's
        # what caused a production request to hang for 300+ seconds
        # until the API timed out. A handful of extra categories is
        # normal; tens of thousands is always a text/id/timestamp
        # column that was never meant to be a feature.
        MAX_CATEGORIES = 50
        safe_categorical_columns = []
        for column in categorical_columns:
            if X[column].nunique(dropna=True) > MAX_CATEGORIES:
                dropped_high_cardinality.append(column)
            else:
                safe_categorical_columns.append(column)
        categorical_columns = safe_categorical_columns
        X = X.drop(columns=dropped_high_cardinality, errors="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    if resolved_task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    try:
        if resolved_task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, target,
                test_size=0.2,
                random_state=42,
                stratify=target
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, target,
                test_size=0.2,
                random_state=42
            )
    except ValueError:
        # Stratification can fail if a class has too few members;
        # fall back to a plain split rather than erroring out.
        X_train, X_test, y_train, y_test = train_test_split(
            X, target,
            test_size=0.2,
            random_state=42
        )

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except Exception as exc:
        return {"error": f"Model training failed: {exc}"}

    feature_names = (
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(10)

    result = {
        "task_type": resolved_task_type,
        "target_column": target_column,
        "rows_used": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "top_features": importance_df.round(4).to_dict(orient="records"),
        "note": CAUSATION_NOTE
    }

    if dropped_high_cardinality:
        result["dropped_high_cardinality_columns"] = dropped_high_cardinality

    if resolved_task_type == "classification":
        average_mode = "binary" if target.nunique() == 2 else "weighted"

        result["metrics"] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(
                float(precision_score(
                    y_test, predictions,
                    average=average_mode,
                    zero_division=0
                )), 4
            ),
            "recall": round(
                float(recall_score(
                    y_test, predictions,
                    average=average_mode,
                    zero_division=0
                )), 4
            ),
            "f1_score": round(
                float(f1_score(
                    y_test, predictions,
                    average=average_mode,
                    zero_division=0
                )), 4
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist()
        }

        if label_mapping:
            result["label_mapping"] = label_mapping

    else:
        mse = mean_squared_error(y_test, predictions)

        result["metrics"] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(mse ** 0.5), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4)
        }

    return result