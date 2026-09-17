"""
Preparation report: assembly, persistence, retrieval.

The report is the user-facing product of the engine. It is written to
``data/prepared/<session_id>/report.json`` rather than into
sessions.db, so the existing SQLite schema in app/agent/memory.py is
left completely untouched - one less thing that can break for
sessions created before the engine existed.

The stored document is plain JSON so the frontend can render it
directly and the agent can be handed a condensed text version.
"""

import json
import math
import os
from datetime import datetime, timezone


REPORT_FILENAME = "report.json"


def _json_safe(value):
    """
    NaN and Infinity are valid Python floats but invalid JSON.

    The same trap app/agent/agent.py guards against in ``_json_safe``:
    json.dumps emits bare NaN tokens that break strict parsers,
    including the frontend's fetch().json().
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        # numpy scalars
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def session_report_path(prepared_dir: str, session_id: str) -> str:
    return os.path.join(prepared_dir, session_id, REPORT_FILENAME)


def build_report(
    session_id: str,
    datasets: list,
    integrated: list,
    relationships: list,
    matches: list,
    errors: list,
) -> dict:
    """
    Assemble the structured report from every stage's output.

    Sections mirror what the user needs to decide whether to trust an
    answer: what came in, what was wrong with it, what was changed,
    what was joined, what still needs attention.
    """
    total_rows = sum(dataset["rows_after"] for dataset in datasets)
    total_warnings = sum(
        len(dataset["validation"]["warnings"]) for dataset in datasets
    )
    total_failures = sum(
        len(dataset["validation"]["failed"]) for dataset in datasets
    )

    report = {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if datasets else "failed",

        "summary": {
            "files_uploaded": len({d["source_file"] for d in datasets}),
            "datasets_prepared": len(datasets),
            "integrated_datasets": len(integrated),
            "total_rows": total_rows,
            "total_warnings": total_warnings,
            "total_failed_checks": total_failures,
            "failed_files": len(errors),
        },

        "datasets": [
            {
                "table_name": dataset["table_name"],
                "source_file": dataset["source_file"],
                "source_format": dataset["source_format"],
                "rows_before": dataset["rows_before"],
                "rows_after": dataset["rows_after"],
                "columns": dataset["columns_after"],
                "column_types": dataset["column_types"],
                # Per-column detail (missing %, distinct values, type)
                # computed by the profiler over the FULL cleaned
                # dataset - not a client-side guess from a 10-row
                # preview. Powers the frontend's Data Overview cards.
                "column_profiles": [
                    {
                        "name": column["name"],
                        "type": column["inferred_type"],
                        "missing_percent": column["missing_percent"],
                        "distinct": column["distinct"],
                        "unique_ratio": column["unique_ratio"],
                        "is_constant": column["is_constant"],
                        "id_candidate": column["id_candidate"],
                    }
                    for column in dataset["profile_after"]["columns_detail"]
                ],
                "prepared_path": dataset["prepared_path"],
                "raw_path": dataset["raw_path"],
                "load_notes": dataset["load_notes"],
            }
            for dataset in datasets
        ],

        "quality": [
            {
                "table_name": dataset["table_name"],
                "missing_percent_before": dataset["profile_before"]["missing_percent"],
                "missing_percent_after": dataset["profile_after"]["missing_percent"],
                "duplicate_rows_removed": dataset["duplicates"]["exact_duplicates_removed"],
                "duplicate_key_columns": dataset["duplicates"]["duplicate_key_columns"],
                "invalid_values": dataset["invalid_findings"],
                "id_candidates": dataset["profile_after"]["id_candidates"],
                "constant_columns": dataset["profile_after"]["constant_columns"],
            }
            for dataset in datasets
        ],

        "transformations": [
            {
                "table_name": dataset["table_name"],
                "actions": dataset["actions"],
            }
            for dataset in datasets
        ],

        "integration": {
            "column_matches": [match.to_dict() for match in matches],
            "relationships": relationships,
            "joined_datasets": [
                {
                    key: value for key, value in dataset.items()
                    if key != "df"
                }
                for dataset in integrated
            ],
        },

        "validation": {
            "per_dataset": [
                {
                    "table_name": dataset["table_name"],
                    "passed": dataset["validation"]["passed"],
                    "warnings": dataset["validation"]["warnings"],
                    "failed": dataset["validation"]["failed"],
                }
                for dataset in datasets
            ],
            "integration_checks": [],
        },

        "feature_relevance": [
            {
                "table_name": dataset["table_name"],
                **dataset["feature_relevance"],
            }
            for dataset in datasets
        ],

        "errors": errors,
    }

    return _json_safe(report)


def save_report(prepared_dir: str, session_id: str, report: dict) -> str:
    path = session_report_path(prepared_dir, session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    return path


def load_report(prepared_dir: str, session_id: str):
    """Return the stored report, or None if this session predates the
    preparation engine or has no uploads yet."""
    path = session_report_path(prepared_dir, session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _merge_keyed(old_items: list, new_items: list, key) -> list:
    """Combine two lists of dicts, newer entry winning on key collision,
    preserving the newer list's ordering for its own items and
    appending anything from the old list that wasn't superseded."""
    new_keys = {key(item) for item in new_items}
    carried_over = [item for item in old_items if key(item) not in new_keys]
    return carried_over + new_items


def merge_reports(previous: dict, new: dict) -> dict:
    """
    Combine an existing session report with a fresh one.

    Uploading a second batch of files must not erase the record of
    the first - the same additive guarantee ``memory.add_dataset``
    gives for datasets themselves. Same table name uploaded twice
    means the newer entry wins.
    """
    if not previous:
        return new

    def merge_by(key_field, old_items, new_items):
        combined = {item[key_field]: item for item in old_items}
        for item in new_items:
            combined[item[key_field]] = item
        return list(combined.values())

    merged = dict(new)

    merged["datasets"] = merge_by(
        "table_name", previous.get("datasets", []), new.get("datasets", [])
    )
    merged["quality"] = merge_by(
        "table_name", previous.get("quality", []), new.get("quality", [])
    )
    merged["transformations"] = merge_by(
        "table_name",
        previous.get("transformations", []),
        new.get("transformations", []),
    )
    merged["feature_relevance"] = merge_by(
        "table_name",
        previous.get("feature_relevance", []),
        new.get("feature_relevance", []),
    )
    merged["validation"] = {
        "per_dataset": merge_by(
            "table_name",
            previous.get("validation", {}).get("per_dataset", []),
            new.get("validation", {}).get("per_dataset", []),
        ),
        "integration_checks": new.get("validation", {}).get(
            "integration_checks", []
        ),
    }

    # Integration results are keyed on a synthetic 'pair' so a table
    # that becomes matchable again in a later batch overwrites its old
    # entry, but every match/relationship/join from earlier batches
    # that is still valid survives - this is what keeps
    # summarize_for_agent mentioning a join made in batch 1 after
    # batch 2 has been uploaded.
    old_integration = previous.get("integration", {})
    new_integration = new.get("integration", {})

    merged["integration"] = {
        "column_matches": _merge_keyed(
            old_integration.get("column_matches", []),
            new_integration.get("column_matches", []),
            key=lambda item: (item["left"], item["right"]),
        ),
        "relationships": _merge_keyed(
            old_integration.get("relationships", []),
            new_integration.get("relationships", []),
            key=lambda item: (
                item["parent_table"], item["parent_column"],
                item["child_table"], item["child_column"],
            ),
        ),
        "joined_datasets": _merge_keyed(
            old_integration.get("joined_datasets", []),
            new_integration.get("joined_datasets", []),
            key=lambda item: item["table_name"],
        ),
    }

    merged["summary"] = {
        "files_uploaded": len(
            {item["source_file"] for item in merged["datasets"]}
        ),
        "datasets_prepared": len(merged["datasets"]),
        "integrated_datasets": len(
            merged.get("integration", {}).get("joined_datasets", [])
        ),
        "total_rows": sum(
            item.get("rows_after", 0) for item in merged["datasets"]
        ),
        "total_warnings": sum(
            len(item.get("warnings", []))
            for item in merged["validation"]["per_dataset"]
        ),
        "total_failed_checks": sum(
            len(item.get("failed", []))
            for item in merged["validation"]["per_dataset"]
        ),
        "failed_files": len(merged.get("errors", [])),
    }

    return merged


def summarize_for_agent(report: dict) -> str:
    """
    Condense the report into a few lines for the agent's context.

    Kept deliberately short: this rides along with every turn, and
    the agent needs to know what was changed and what to distrust,
    not the full transformation log.
    """
    if not report:
        return ""

    lines = ["Data preparation summary:"]

    for dataset in report.get("datasets", []):
        removed = dataset["rows_before"] - dataset["rows_after"]
        note = f" ({removed:,} duplicate rows removed)" if removed else ""
        lines.append(
            f"- '{dataset['table_name']}': {dataset['rows_after']:,} rows, "
            f"{dataset['columns']} columns, cleaned from "
            f"{dataset['source_file']}{note}"
        )

    for joined in report.get("integration", {}).get("joined_datasets", []):
        lines.append(
            f"- '{joined['table_name']}' is a prepared {joined['join_type']} "
            f"of '{joined['left_table']}' and '{joined['right_table']}' "
            f"({joined['unmatched_rows']:,} unmatched rows)"
        )

    warnings = []
    for entry in report.get("validation", {}).get("per_dataset", []):
        for warning in entry.get("warnings", []) + entry.get("failed", []):
            warnings.append(f"{entry['table_name']}: {warning['message']}")

    if warnings:
        lines.append("Known data-quality caveats:")
        lines.extend(f"- {warning}" for warning in warnings[:8])
        if len(warnings) > 8:
            lines.append(f"- ... and {len(warnings) - 8} more")

    lines.append(
        "Column names were normalized to snake_case and dates to "
        "ISO 8601 (YYYY-MM-DD)."
    )

    return "\n".join(lines)