"""
Data Preparation Engine - orchestrator.

This is the only module FastAPI imports. ``prepare_uploads`` is the
whole public surface: hand it saved file paths and a session, get
back prepared datasets and a report.

Stage order, and why:

  load         format-agnostic read, everything as text
  profile      inspect the raw truth before touching anything
  standardize  names, whitespace, null placeholders, categories, dates
  clean        type conversion, missing values, invalid-value flags
  deduplicate  exact duplicates removed, duplicate keys reported
  re-profile   the post-clean state, for before/after reporting
  match        cross-dataset column matching (deterministic)
  integrate    materialize only the joins that are provably safe
  validate     what should the user not trust?
  persist      write CSVs + report.json

Standardization runs before cleaning because type inference is far
more reliable once '  1,234 ' is '1,234' and 'N/A' is a real null.
Deduplication runs after cleaning because rows that differ only by
'USA' vs 'usa ' are not duplicates until categories are normalized.

The raw upload is never modified. Prepared output is written
alongside it, and only prepared paths are registered with the
session - which is the single seam that connects this engine to the
existing agent.
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from app.config import settings
from app.data_engine import (
    cleaner,
    deduplicator,
    feature_selector,
    loader,
    merger,
    profiler,
    report as report_module,
    schema_matcher,
    standardizer,
    validator,
)
from app.data_engine.llm_assist import get_resolver


# Prepared output lives beside the uploads, one directory per
# session, so a session's artifacts can be inspected or deleted as a
# unit.
def prepared_dir() -> str:
    return os.getenv("PREPARED_DIR", "data/prepared")


@dataclass
class PreparedDataset:
    table_name: str
    prepared_path: str
    rows: int
    columns: int


@dataclass
class PreparationResult:
    """What FastAPI hands back to the frontend."""

    session_id: str
    datasets: List[PreparedDataset] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    errors: List[dict] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.datasets)


def _unique_table_name(base: str, taken: set) -> str:
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # index=False matters: a stray index column would look like a
    # feature to the ML tools and an ID to the schema matcher.
    df.to_csv(path, index=False, encoding="utf-8")


def _column_types(df: pd.DataFrame, profile: dict) -> dict:
    by_name = {
        column["name"]: column["inferred_type"]
        for column in profile["columns_detail"]
    }
    return {str(column): by_name.get(str(column), "text") for column in df.columns}


def prepare_uploads(
    session_id: str,
    saved_files: list,
    existing_table_names=None,
) -> PreparationResult:
    """
    Run the full preparation pipeline over one batch of uploads.

    ``saved_files`` is a list of (original_filename, saved_path)
    tuples - FastAPI's only job is to write the bytes and pass them
    here.

    ``existing_table_names`` lets a second upload into the same
    session avoid colliding with tables prepared earlier.

    Returns a PreparationResult. A file that cannot be read produces
    an entry in ``errors`` and does not stop the rest of the batch.
    """
    output_root = prepared_dir()
    session_dir = os.path.join(output_root, session_id)
    os.makedirs(session_dir, exist_ok=True)

    taken_names = set(existing_table_names or [])
    errors = []
    prepared = []
    frames = {}

    # ---- per-file preparation ---------------------------------------
    for original_filename, saved_path in saved_files:
        try:
            loaded_tables = loader.load_file(saved_path)
        except loader.LoadError as exc:
            errors.append({
                "file": original_filename,
                "error": str(exc),
            })
            continue
        except Exception as exc:
            errors.append({
                "file": original_filename,
                "error": f"Unexpected error reading the file: {exc}",
            })
            continue

        for table in loaded_tables:
            base_name = loader.sanitize_table_name(original_filename)
            if table.sheet_name:
                base_name = (
                    f"{base_name}_"
                    f"{loader.sanitize_table_name(table.sheet_name)}"
                )
            table_name = _unique_table_name(base_name, taken_names)

            actions = []
            raw_df = table.df

            profile_before = profiler.profile_dataframe(raw_df, table_name)

            working = standardizer.standardize(raw_df, profile_before, actions)

            # Re-profile after standardization: column names changed,
            # and null placeholders are now real nulls, so the
            # cleaner needs the updated picture to decide anything.
            profile_mid = profiler.profile_dataframe(working, table_name)

            working, invalid_findings = cleaner.clean(
                working, profile_mid, actions
            )
            working, duplicate_info = deduplicator.deduplicate(
                working, profile_mid, actions
            )

            profile_after = profiler.profile_dataframe(working, table_name)

            validation = validator.validate_dataset(
                working,
                profile_after,
                table_name,
                invalid_findings=invalid_findings,
                duplicate_info=duplicate_info,
            )

            relevance = feature_selector.assess_structural_relevance(
                working, profile_after
            )

            prepared_path = os.path.join(session_dir, f"{table_name}.csv")
            _write_csv(working, prepared_path)

            frames[table_name] = working

            prepared.append({
                "table_name": table_name,
                "source_file": original_filename,
                "source_format": table.source_format,
                "sheet_name": table.sheet_name,
                "raw_path": saved_path,
                "prepared_path": prepared_path,
                "rows_before": profile_before["rows"],
                "rows_after": profile_after["rows"],
                "columns_before": profile_before["columns"],
                "columns_after": profile_after["columns"],
                "column_types": _column_types(working, profile_after),
                "profile_before": profile_before,
                "profile_after": profile_after,
                "actions": actions,
                "invalid_findings": invalid_findings,
                "duplicates": duplicate_info,
                "validation": validation,
                "feature_relevance": relevance,
                "load_notes": table.notes,
            })

    # ---- cross-dataset stages ----------------------------------------
    matches = []
    relationships = []
    integrated = []

    if len(frames) >= 2:
        profiles = {
            item["table_name"]: item["profile_after"] for item in prepared
        }
        matches = schema_matcher.match_schemas(
            frames, profiles, llm_resolver=get_resolver()
        )
        relationships = merger.detect_relationships(matches)

        integration_log = []
        integrated = merger.build_integrated_datasets(
            frames, relationships, integration_log
        )

        for dataset in integrated:
            table_name = _unique_table_name(dataset["table_name"], taken_names)
            dataset["table_name"] = table_name

            prepared_path = os.path.join(session_dir, f"{table_name}.csv")
            _write_csv(dataset["df"], prepared_path)
            dataset["prepared_path"] = prepared_path

        # Attach the integration log to the first dataset's action
        # list so it appears in the transformations section rather
        # than being lost.
        if integration_log and prepared:
            prepared[0]["actions"].extend(integration_log)

    # ---- report -------------------------------------------------------
    new_report = report_module.build_report(
        session_id=session_id,
        datasets=prepared,
        integrated=integrated,
        relationships=relationships,
        matches=matches,
        errors=errors,
    )

    if integrated:
        new_report["validation"]["integration_checks"] = (
            validator.validate_integration(integrated)
        )

    previous = report_module.load_report(output_root, session_id)
    final_report = report_module.merge_reports(previous, new_report)
    report_module.save_report(output_root, session_id, final_report)

    result_datasets = [
        PreparedDataset(
            table_name=item["table_name"],
            prepared_path=item["prepared_path"],
            rows=item["rows_after"],
            columns=item["columns_after"],
        )
        for item in prepared
    ]

    result_datasets.extend(
        PreparedDataset(
            table_name=dataset["table_name"],
            prepared_path=dataset["prepared_path"],
            rows=dataset["rows"],
            columns=dataset["columns"],
        )
        for dataset in integrated
    )

    return PreparationResult(
        session_id=session_id,
        datasets=result_datasets,
        report=final_report,
        errors=errors,
    )


def get_report(session_id: str):
    """Fetch a session's stored preparation report, or None."""
    return report_module.load_report(prepared_dir(), session_id)


def agent_context(session_id: str) -> str:
    """One-paragraph summary of preparation for the agent's prompt."""
    return report_module.summarize_for_agent(get_report(session_id))


def delete_session_artifacts(session_id: str) -> None:
    """Remove a session's prepared directory. Called when a session is
    deleted so prepared CSVs don't outlive the session that owns
    them."""
    session_dir = os.path.join(prepared_dir(), session_id)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
