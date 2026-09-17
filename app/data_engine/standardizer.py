"""
Surface-level normalization: names, whitespace, casing, dates.

Everything here is presentation-level and reversible in principle -
no rows are dropped and no values are invented. The cleaner (types,
missing values) runs after this, because type coercion is much more
reliable once '  1,234 ' has become '1,234' and 'N/A' has become a
real null.

Every change appends an entry to the action log so the report can
tell the user exactly what happened to their data.
"""

import re

import pandas as pd

from app.data_engine.profiler import NULL_LIKE_TOKENS, looks_datetime


# ISO 8601. Chosen because DuckDB, pandas, and every downstream tool
# parse it unambiguously - unlike 03/04/2024, which is two dates.
DATE_OUTPUT_FORMAT = "%Y-%m-%d"
DATETIME_OUTPUT_FORMAT = "%Y-%m-%d %H:%M:%S"


def normalize_column_name(name: str) -> str:
    """
    'Customer ID ' -> 'customer_id', 'Total($)' -> 'total',
    'address.city' -> 'address_city'.

    Consistent snake_case matters for more than tidiness here: it is
    what makes schema matching across files work at all, and it
    guarantees the name is a valid bare SQL identifier in DuckDB.
    """
    text = str(name).strip()

    # Split camelCase / PascalCase before lowering, so 'customerId'
    # becomes 'customer_id' rather than 'customerid'.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)

    text = text.replace(".", "_")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()

    if not text:
        text = "column"
    if text[0].isdigit():
        text = f"c_{text}"
    return text


def normalize_column_names(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Rename every column to snake_case, resolving collisions."""
    renames = {}
    used = set()

    for original in df.columns:
        candidate = normalize_column_name(original)

        # Two different originals can normalize to the same name
        # ('Customer ID' and 'customer_id'). Suffix rather than
        # silently drop one.
        if candidate in used:
            suffix = 2
            while f"{candidate}_{suffix}" in used:
                suffix += 1
            candidate = f"{candidate}_{suffix}"

        used.add(candidate)
        if str(original) != candidate:
            renames[original] = candidate

    if renames:
        df = df.rename(columns=renames)
        log.append({
            "stage": "standardize",
            "action": "renamed_columns",
            "detail": f"{len(renames)} column name(s) normalized.",
            "columns": {str(k): v for k, v in renames.items()},
        })

    return df


def strip_whitespace(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Trim padding and collapse internal runs of whitespace."""
    changed = []

    for column in df.columns:
        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        as_text = series.astype(str)
        stripped = (
            as_text.str.strip().str.replace(r"\s+", " ", regex=True)
        )
        # Restore genuine nulls that astype(str) turned into 'nan'.
        stripped = stripped.mask(series.isna())

        affected = int((as_text.ne(stripped) & series.notna()).sum())
        if affected:
            df[column] = stripped
            changed.append({"column": str(column), "values_changed": affected})

    if changed:
        log.append({
            "stage": "standardize",
            "action": "stripped_whitespace",
            "detail": f"Whitespace normalized in {len(changed)} column(s).",
            "columns": changed,
        })

    return df


def normalize_null_like(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Convert text placeholders ('N/A', 'unknown', '-') into real nulls.

    Without this, a column that is 30% 'N/A' looks complete to every
    downstream check, and any numeric conversion of it fails.
    """
    changed = []

    for column in df.columns:
        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        mask = series.astype(str).str.strip().str.lower().isin(NULL_LIKE_TOKENS)
        mask = mask & series.notna()
        affected = int(mask.sum())

        if affected:
            df.loc[mask, column] = None
            changed.append({"column": str(column), "values_changed": affected})

    if changed:
        total = sum(item["values_changed"] for item in changed)
        log.append({
            "stage": "standardize",
            "action": "normalized_null_placeholders",
            "detail": (
                f"{total} placeholder value(s) such as 'N/A' or "
                f"'unknown' converted to real missing values."
            ),
            "columns": changed,
        })

    return df


def normalize_categories(
    df: pd.DataFrame,
    categorical_columns: list,
    log: list,
) -> pd.DataFrame:
    """
    Merge category variants that differ only by case or spacing.

    Conservative by design. 'USA' and 'usa' are merged because they
    are unambiguously the same value written twice. 'USA' and
    'United States' are NOT merged - that is a semantic judgement,
    and guessing wrong silently corrupts the user's data. Those are
    reported as a validation warning instead.

    The surviving spelling is the most frequent one in the data, so
    the result still reads naturally rather than being force-lowered.
    """
    changed = []

    for column in categorical_columns:
        if column not in df.columns:
            continue

        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        as_text = series.dropna().astype(str)
        if as_text.empty:
            continue

        key = as_text.str.strip().str.lower()
        if key.nunique() == as_text.nunique():
            continue  # Nothing collapses; leave the column alone.

        # For each normalized key, adopt the most common original
        # spelling as the canonical form.
        frame = pd.DataFrame({"key": key, "value": as_text})
        canonical = (
            frame.groupby("key")["value"]
            .agg(lambda values: values.value_counts().idxmax())
            .to_dict()
        )

        mapped = key.map(canonical)
        affected = int((mapped != as_text).sum())

        if affected:
            df.loc[mapped.index, column] = mapped
            merged = as_text.nunique() - key.nunique()
            changed.append({
                "column": str(column),
                "values_changed": affected,
                "categories_merged": int(merged),
            })

    if changed:
        log.append({
            "stage": "standardize",
            "action": "standardized_categories",
            "detail": (
                f"Case/spacing variants merged in {len(changed)} "
                f"categorical column(s)."
            ),
            "columns": changed,
        })

    return df


def standardize_dates(
    df: pd.DataFrame,
    datetime_columns: list,
    log: list,
) -> pd.DataFrame:
    """
    Parse mixed date formats and re-emit them as ISO 8601 strings.

    Output is text, not datetime64, on purpose: the frame is written
    to CSV and re-read by every downstream tool, so an ISO string is
    the format that survives that round trip and stays sortable and
    comparable in DuckDB.

    Values that fail to parse are set to null and counted - never
    left as a half-parsed string that would poison later comparisons.
    """
    changed = []

    for column in datetime_columns:
        if column not in df.columns:
            continue

        original = df[column]
        non_null_before = int(original.notna().sum())
        if not non_null_before:
            continue

        parsed = pd.to_datetime(original, errors="coerce", format="mixed")

        parsed_count = int(parsed.notna().sum())
        if parsed_count == 0:
            continue

        # Refuse to convert a column we can barely parse - that is a
        # sign the type inference was wrong, not that the data is bad.
        if parsed_count / non_null_before < 0.5:
            continue

        has_time = bool(
            (parsed.dt.hour.fillna(0) != 0).any()
            or (parsed.dt.minute.fillna(0) != 0).any()
            or (parsed.dt.second.fillna(0) != 0).any()
        )
        output_format = (
            DATETIME_OUTPUT_FORMAT if has_time else DATE_OUTPUT_FORMAT
        )

        formatted = parsed.dt.strftime(output_format)
        formatted = formatted.where(parsed.notna(), None)

        unparseable = non_null_before - parsed_count
        df[column] = formatted

        changed.append({
            "column": str(column),
            "output_format": "ISO 8601"
                             + (" with time" if has_time else " date"),
            "unparseable_values": unparseable,
        })

    if changed:
        log.append({
            "stage": "standardize",
            "action": "standardized_dates",
            "detail": (
                f"{len(changed)} date column(s) converted to ISO 8601."
            ),
            "columns": changed,
        })

    return df


def standardize(df: pd.DataFrame, profile: dict, log: list) -> pd.DataFrame:
    """Run the full standardization stage in dependency order."""
    df = df.copy()

    df = normalize_column_names(df, log)

    # The profile was built against the pre-rename frame, so map its
    # column lists onto the new names before using them.
    rename_map = {
        name: normalize_column_name(name) for name in profile["column_names"]
    }

    def remap(names):
        return [rename_map.get(name, name) for name in names]

    df = strip_whitespace(df, log)
    df = normalize_null_like(df, log)
    df = normalize_categories(df, remap(profile["categorical_columns"]), log)
    df = standardize_dates(df, remap(profile["datetime_columns"]), log)

    return df
