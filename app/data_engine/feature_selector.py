"""
Feature relevance assessment.

Two entry points, for two different moments:

``assess_structural_relevance`` runs during preparation, with no
knowledge of what the user wants. It can only judge a column on its
own merits - a constant column is useless for anything, a row-number
column is never a real feature - so that is all it claims to do.

``assess_feature_relevance`` runs later, as an agent tool, once the
user has said what they actually want ("predict churn"). Gemini's
only job there is to pass along the goal and the target column it
parsed from natural language; the ranking itself is mutual
information and statistical structure, computed here in sklearn.

Nothing in this module deletes a column. It produces a ranking and a
rationale; the existing ML pipeline in app/tools/ml_tools.py decides
what to train on.
"""

import re

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)


# Columns whose names match this are bookkeeping artifacts of the
# export process, not business features.
_ARTIFACT_PATTERN = re.compile(
    r"(^|_)(index|unnamed|row_?num(ber)?|record_?num(ber)?|"
    r"import|ingest|etl|load|extract|batch|source_?file|"
    r"internal|_raw$|sys_|tmp_)",
    re.IGNORECASE,
)

# Sampled for mutual information - the estimate is stable well below
# full data and this keeps a large dataset responsive.
_MI_SAMPLE_ROWS = 20000

# Categorical features beyond this cardinality are one-hot hostile
# and usually identifiers in disguise.
_MAX_USEFUL_CARDINALITY = 50


def _relevance_band(score: float) -> str:
    if score >= 0.6:
        return "highly_relevant"
    if score >= 0.3:
        return "potentially_relevant"
    return "low_relevance"


def assess_structural_relevance(df: pd.DataFrame, profile: dict) -> dict:
    """
    Goal-agnostic screening, run during preparation.

    Flags columns that cannot be useful for *any* analysis: constants,
    near-unique identifiers, export artifacts, and columns that are
    mostly missing. Deliberately says nothing about which columns are
    important, because without a goal that question has no answer.
    """
    findings = []

    for column in profile.get("columns_detail", []):
        name = column["name"]
        reasons = []
        band = "potentially_relevant"

        if column["is_constant"]:
            reasons.append("holds a single value for every row")
            band = "low_relevance"

        if column["id_candidate"]:
            reasons.append("appears to be a unique identifier")
            band = "low_relevance"

        if _ARTIFACT_PATTERN.search(name):
            reasons.append("name suggests an import or bookkeeping artifact")
            band = "low_relevance"

        if column["missing_percent"] >= 60:
            reasons.append(
                f"{column['missing_percent']:.0f}% of values are missing"
            )
            band = "low_relevance"

        if (
            column["inferred_type"] == "text"
            and column["unique_ratio"] > 0.8
        ):
            reasons.append("free text with almost no repeated values")
            band = "low_relevance"

        if band == "low_relevance":
            findings.append({
                "column": name,
                "relevance": band,
                "reasons": reasons,
            })

    return {
        "method": "structural screening (no analytical goal provided)",
        "low_relevance": findings,
        "note": (
            "These columns are unlikely to be useful as features. "
            "Nothing was removed - the analysis and ML tools decide "
            "what to use."
        ),
    }


def _goal_keyword_bonus(column_name: str, goal: str) -> float:
    """
    Small nudge when a column name echoes the stated goal.

    Intentionally small (capped at 0.15). Name-matching is a weak
    signal that should break ties between statistically similar
    features, never override what the data actually shows.
    """
    if not goal:
        return 0.0

    goal_tokens = {
        token for token in re.split(r"[^a-z0-9]+", goal.lower())
        if len(token) > 3
    }
    if not goal_tokens:
        return 0.0

    column_tokens = {
        token for token in re.split(r"[^a-z0-9]+", column_name.lower())
        if token
    }

    overlap = goal_tokens & column_tokens
    return min(0.15, 0.08 * len(overlap))


def _prepare_feature_matrix(df: pd.DataFrame, target_column: str):
    """Encode features numerically for mutual information, keeping
    track of which original column each encoded column came from."""
    features = df.drop(columns=[target_column])
    encoded = pd.DataFrame(index=features.index)
    usable_columns = []

    for column in features.columns:
        series = features[column]

        if pd.api.types.is_numeric_dtype(series):
            filled = series.fillna(series.median())
            if filled.nunique() <= 1:
                continue
            encoded[column] = filled
            usable_columns.append(column)

        elif pd.api.types.is_bool_dtype(series):
            encoded[column] = series.fillna(False).astype(int)
            usable_columns.append(column)

        else:
            as_text = series.astype(str)
            if as_text.nunique() > _MAX_USEFUL_CARDINALITY:
                continue
            if as_text.nunique() <= 1:
                continue
            # Ordinal codes are enough for a mutual-information
            # estimate with discrete_features=True; one-hot would
            # fragment the score across dummy columns.
            encoded[column] = pd.Categorical(as_text).codes
            usable_columns.append(column)

    return encoded, usable_columns


def assess_feature_relevance(
    file_path: str,
    goal: str = None,
    target_column: str = None,
) -> dict:
    """
    Rank a dataset's columns by relevance to an analytical goal.

    Exposed to the agent as a tool. The LLM supplies `goal` (the
    user's own words) and, where it can identify one, `target_column`.
    All scoring below is deterministic.

    With a target column, ranking is driven by mutual information -
    which captures non-linear relationships that a correlation matrix
    misses. Without one, columns are ranked by usable signal
    (variance, cardinality, completeness) and the response says so
    rather than implying a relationship that was never measured.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read dataset: {exc}"}

    if df.empty:
        return {"error": "The dataset is empty."}

    if target_column and target_column not in df.columns:
        return {
            "error": (
                f"Target column '{target_column}' is not in this dataset. "
                f"Available columns: {list(df.columns)}"
            )
        }

    structural_penalty = {}
    structural_reasons = {}

    for column in df.columns:
        name = str(column)
        penalty = 0.0
        reasons = []

        series = df[column]
        non_null = int(series.notna().sum())
        missing_rate = 1 - (non_null / max(len(df), 1))
        unique_ratio = (
            series.nunique(dropna=True) / max(non_null, 1)
        )

        if series.nunique(dropna=True) <= 1:
            penalty += 1.0
            reasons.append("constant value")
        if unique_ratio >= 0.98 and non_null == len(df):
            penalty += 0.8
            reasons.append("unique identifier")
        if _ARTIFACT_PATTERN.search(name):
            penalty += 0.8
            reasons.append("import or bookkeeping artifact")
        if missing_rate >= 0.5:
            penalty += 0.5
            reasons.append(f"{missing_rate:.0%} missing")

        structural_penalty[name] = penalty
        structural_reasons[name] = reasons

    ranked = []

    if target_column:
        working = df.dropna(subset=[target_column])
        if len(working) > _MI_SAMPLE_ROWS:
            working = working.sample(_MI_SAMPLE_ROWS, random_state=42)

        if len(working) < 10:
            return {
                "error": (
                    f"Only {len(working)} row(s) have a value for "
                    f"'{target_column}' - too few to assess relevance."
                )
            }

        target = working[target_column]
        is_classification = (
            not pd.api.types.is_numeric_dtype(target)
            or target.nunique() <= 20
        )

        encoded, usable = _prepare_feature_matrix(working, target_column)

        if encoded.empty:
            return {
                "error": "No usable feature columns remain after encoding."
            }

        if is_classification:
            y = pd.Categorical(target.astype(str)).codes
            scores = mutual_info_classif(
                encoded.values, y, random_state=42
            )
        else:
            y = pd.to_numeric(target, errors="coerce").fillna(target.median())
            scores = mutual_info_regression(
                encoded.values, y, random_state=42
            )

        # Normalize to 0-1 so the bands are interpretable; raw mutual
        # information has no fixed upper bound.
        peak = float(np.max(scores)) if len(scores) else 0.0
        normalized = (
            scores / peak if peak > 0 else np.zeros_like(scores)
        )

        for column, raw_score, relative in zip(usable, scores, normalized):
            name = str(column)
            score = float(relative)
            score += _goal_keyword_bonus(name, goal)
            score = max(0.0, min(1.0, score - structural_penalty.get(name, 0)))

            ranked.append({
                "column": name,
                "relevance": _relevance_band(score),
                "score": round(score, 3),
                "mutual_information": round(float(raw_score), 4),
                "reasons": structural_reasons.get(name, []) or [
                    "measured association with the target"
                ],
            })

        # Columns dropped during encoding still deserve a verdict.
        assessed = {item["column"] for item in ranked}
        for column in df.columns:
            name = str(column)
            if name in assessed or name == target_column:
                continue
            ranked.append({
                "column": name,
                "relevance": "low_relevance",
                "score": 0.0,
                "mutual_information": None,
                "reasons": structural_reasons.get(name, []) or [
                    "too high-cardinality or constant to be usable"
                ],
            })

        method = (
            f"mutual information against '{target_column}' "
            f"({'classification' if is_classification else 'regression'})"
        )

    else:
        # No target: rank by intrinsic usable signal only.
        for column in df.columns:
            name = str(column)
            series = df[column]
            non_null = int(series.notna().sum())
            completeness = non_null / max(len(df), 1)
            distinct = series.nunique(dropna=True)

            if pd.api.types.is_numeric_dtype(series):
                spread = float(series.std(skipna=True) or 0)
                signal = 0.7 if spread > 0 else 0.0
            elif 1 < distinct <= _MAX_USEFUL_CARDINALITY:
                signal = 0.6
            else:
                signal = 0.2

            score = signal * completeness
            score += _goal_keyword_bonus(name, goal)
            score = max(0.0, min(1.0, score - structural_penalty.get(name, 0)))

            ranked.append({
                "column": name,
                "relevance": _relevance_band(score),
                "score": round(score, 3),
                "mutual_information": None,
                "reasons": structural_reasons.get(name, []) or [
                    "ranked on completeness and variability only"
                ],
            })

        method = (
            "structural ranking - no target column was identified, so "
            "no relationship to an outcome was measured"
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)

    grouped = {"highly_relevant": [], "potentially_relevant": [], "low_relevance": []}
    for item in ranked:
        grouped[item["relevance"]].append(item["column"])

    return {
        "status": "success",
        "goal": goal,
        "target_column": target_column,
        "method": method,
        "highly_relevant": grouped["highly_relevant"],
        "potentially_relevant": grouped["potentially_relevant"],
        "low_relevance": grouped["low_relevance"],
        "ranking": ranked,
        "note": (
            "This is a recommendation, not a filter. No column was "
            "removed from the dataset - pass the columns you want to "
            "train_model explicitly."
        ),
    }
