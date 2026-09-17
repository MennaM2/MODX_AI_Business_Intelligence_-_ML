"""
Relationship detection and dataset integration.

Turns the column matches from ``schema_matcher`` into actual
relationships (which table is the "one" side, which is the "many"),
then materializes a joined dataset for the relationships that are
unambiguous.

The safeguards matter more than the joins here. A join on a key that
repeats on both sides produces a cartesian blow-up - 10k x 10k rows
becomes 100 million - which would exhaust memory and silently
multiply every revenue figure the agent later computes. So a join is
only materialized when:

  1. the match is confident (score >= CONFIDENT_THRESHOLD), and
  2. exactly one side's key is unique (a real one-to-many), and
  3. enough of the foreign keys actually resolve, and
  4. the projected row count stays within a hard ceiling.

Anything failing those is reported as a detected relationship the
agent can JOIN on demand in DuckDB - which it already does well -
rather than being written to disk.

pandas does the join rather than DuckDB: the output is a CSV that
DuckDB registers as a table anyway, and keeping the merge in pandas
means the preparation engine has no query-engine dependency.
"""

import pandas as pd

from app.data_engine.schema_matcher import ColumnMatch


# A materialized join may not exceed this many rows outright, nor
# grow beyond MAX_ROW_GROWTH times the larger input. Both guard
# against an unnoticed cartesian product.
MAX_JOINED_ROWS = 2_000_000
MAX_ROW_GROWTH = 1.5

# Below this fraction of foreign keys resolving, the two files are
# probably from different periods or systems; joining them would
# quietly drop most of the data or fill it with nulls.
MIN_KEY_RESOLUTION = 0.60


def detect_relationships(matches: list) -> list:
    """
    Interpret column matches as table relationships.

    Uniqueness decides direction: the side whose key is unique is the
    parent ("one"), the side where it repeats is the child ("many").
    """
    relationships = []

    for match in matches:
        if match.left_unique and match.right_unique:
            kind = "one_to_one"
            parent_table, parent_column = match.left_table, match.left_column
            child_table, child_column = match.right_table, match.right_column
        elif match.left_unique and not match.right_unique:
            kind = "one_to_many"
            parent_table, parent_column = match.left_table, match.left_column
            child_table, child_column = match.right_table, match.right_column
        elif match.right_unique and not match.left_unique:
            kind = "one_to_many"
            parent_table, parent_column = match.right_table, match.right_column
            child_table, child_column = match.left_table, match.left_column
        else:
            kind = "many_to_many"
            parent_table, parent_column = match.left_table, match.left_column
            child_table, child_column = match.right_table, match.right_column

        relationships.append({
            "type": kind,
            "parent_table": parent_table,
            "parent_column": parent_column,
            "child_table": child_table,
            "child_column": child_column,
            "score": round(match.score, 3),
            "confident": match.confident,
            "value_overlap": round(match.overlap_score, 3),
            "evidence": match.evidence,
            "joinable": bool(match.confident and kind != "many_to_many"),
        })

    return relationships


def _normalize_key(series: pd.Series) -> pd.Series:
    """Align key representations so an int key joins to a text key.
    Mirrors ``_comparable_values`` in schema_matcher so detection and
    execution agree on what counts as the same value."""
    text = series.astype(str).str.strip().str.lower()
    return text.str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def _suffixless_merge(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_key: str,
    parent_key: str,
    parent_table: str,
) -> pd.DataFrame:
    """
    LEFT JOIN child -> parent, keeping every child row.

    Overlapping non-key column names are prefixed with the parent
    table name instead of pandas' default '_x'/'_y', so the resulting
    columns are self-describing when the agent queries them later.
    """
    child_normalized = _normalize_key(child[child_key])
    parent_normalized = _normalize_key(parent[parent_key])

    left = child.copy()
    right = parent.copy()
    left["__join_key__"] = child_normalized
    right["__join_key__"] = parent_normalized

    # The parent key duplicates the child key after the join.
    right = right.drop(columns=[parent_key])

    overlapping = (set(right.columns) & set(left.columns)) - {"__join_key__"}
    if overlapping:
        right = right.rename(
            columns={
                column: f"{parent_table}_{column}" for column in overlapping
            }
        )

    # Deduplicate the parent on its key so a parent-side duplicate
    # cannot fan the child out - this is a LEFT JOIN onto a lookup.
    right = right.drop_duplicates(subset="__join_key__", keep="first")

    merged = left.merge(right, on="__join_key__", how="left")
    return merged.drop(columns=["__join_key__"])


def build_integrated_datasets(
    tables: dict,
    relationships: list,
    log: list,
) -> list:
    """
    Materialize the joins that pass every safety check.

    Returns a list of dicts describing each integrated dataset, each
    carrying the joined DataFrame under 'df'. Everything rejected is
    recorded with a reason so the report can explain the decision
    rather than leaving the user guessing.
    """
    integrated = []
    already_joined = set()

    # Each child table gets at most one materialized join - its best
    # one. A table with three valid parents would otherwise spawn
    # three derived datasets, burying the user's actual tables in the
    # agent's catalog and multiplying storage for no benefit. The
    # other relationships are still reported, and the agent can JOIN
    # them in SQL whenever a question needs them.
    enriched_children = set()

    joinable = [
        relationship for relationship in relationships
        if relationship["joinable"]
    ]
    # Strongest relationships first, so the best key wins if two
    # tables are linked by more than one candidate column.
    joinable.sort(key=lambda relationship: relationship["score"], reverse=True)

    for relationship in joinable:
        parent_table = relationship["parent_table"]
        child_table = relationship["child_table"]
        parent_column = relationship["parent_column"]
        child_column = relationship["child_column"]

        pair = frozenset((parent_table, child_table))
        if pair in already_joined:
            continue
        if child_table in enriched_children:
            relationship["join_skipped_reason"] = (
                f"'{child_table}' was already enriched by a "
                f"higher-confidence join"
            )
            continue

        parent = tables.get(parent_table)
        child = tables.get(child_table)
        if parent is None or child is None:
            continue
        if parent_column not in parent.columns or child_column not in child.columns:
            continue

        child_keys = _normalize_key(child[child_column]).dropna()
        parent_keys = set(_normalize_key(parent[parent_column]).dropna())

        if child_keys.empty or not parent_keys:
            continue

        matched = int(child_keys.isin(parent_keys).sum())
        resolution = matched / len(child_keys)
        unmatched = len(child_keys) - matched

        if resolution < MIN_KEY_RESOLUTION:
            relationship["join_skipped_reason"] = (
                f"only {resolution:.0%} of '{child_table}.{child_column}' "
                f"values were found in '{parent_table}.{parent_column}' "
                f"(minimum {MIN_KEY_RESOLUTION:.0%})"
            )
            continue

        projected = len(child)
        ceiling = max(len(child), len(parent)) * MAX_ROW_GROWTH
        if projected > MAX_JOINED_ROWS or projected > ceiling:
            relationship["join_skipped_reason"] = (
                f"the joined result would have {projected:,} rows, "
                f"beyond the safe limit"
            )
            continue

        try:
            merged = _suffixless_merge(
                child, parent, child_column, parent_column, parent_table
            )
        except Exception as exc:
            relationship["join_skipped_reason"] = f"join failed: {exc}"
            continue

        # Final backstop: even with a deduplicated parent, verify the
        # row count did not grow. If it did, something about the key
        # was wrong and the result is not trustworthy.
        if len(merged) > len(child):
            relationship["join_skipped_reason"] = (
                f"row count grew unexpectedly "
                f"({len(child):,} -> {len(merged):,}); join discarded"
            )
            continue

        name = f"{child_table}_with_{parent_table}"
        already_joined.add(pair)
        enriched_children.add(child_table)
        relationship["joined_as"] = name

        integrated.append({
            "table_name": name,
            "df": merged,
            "join_type": "LEFT JOIN",
            "left_table": child_table,
            "right_table": parent_table,
            "left_key": child_column,
            "right_key": parent_column,
            "rows": len(merged),
            "columns": len(merged.columns),
            "matched_rows": matched,
            "unmatched_rows": unmatched,
            "unmatched_percent": round(unmatched / len(child_keys) * 100, 2),
        })

        log.append({
            "stage": "integrate",
            "action": "materialized_join",
            "detail": (
                f"'{child_table}' LEFT JOIN '{parent_table}' on "
                f"{child_column} = {parent_column} -> '{name}' "
                f"({len(merged):,} rows, {unmatched:,} unmatched)."
            ),
        })

    skipped = [
        relationship for relationship in relationships
        if relationship.get("join_skipped_reason")
    ]
    if skipped:
        log.append({
            "stage": "integrate",
            "action": "skipped_joins",
            "detail": (
                f"{len(skipped)} candidate join(s) were detected but not "
                f"materialized; the agent can still JOIN them in SQL."
            ),
            "relationships": [
                {
                    "tables": f"{item['child_table']} -> {item['parent_table']}",
                    "reason": item["join_skipped_reason"],
                }
                for item in skipped
            ],
        })

    return integrated
