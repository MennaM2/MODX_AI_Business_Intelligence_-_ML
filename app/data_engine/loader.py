"""
Format-agnostic ingestion.

The rest of the application is CSV/path based: every tool does
``pd.read_csv(file_path)`` and DuckDB registers those same paths as
tables. So the job of this module is narrow and important - take
whatever the user uploaded (CSV, TSV, Excel, JSON, JSONL, Parquet)
and turn it into one or more in-memory DataFrames that the pipeline
can normalize and write back out as plain CSV.

Nothing downstream ever needs to know the original format.

One upload can legitimately produce several tables: an Excel
workbook with three sheets becomes three datasets, because that is
what the user actually gave us.
"""

import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# Extensions we will attempt. Anything else is rejected early with a
# clear message rather than being fed to pandas and failing obscurely.
SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xlsm", ".xls",
    ".json", ".jsonl", ".ndjson",
    ".parquet",
}

_ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# Read at most this many bytes when sniffing the delimiter - enough
# to be reliable, small enough to stay cheap on a large file.
_SNIFF_BYTES = 64 * 1024


@dataclass
class LoadedTable:
    """One logical table extracted from one uploaded file."""

    df: pd.DataFrame
    source_file: str
    source_format: str
    # Set only when a single file yields several tables (Excel
    # sheets). Used to build a unique table name downstream.
    sheet_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class LoadError(Exception):
    """Raised when a file cannot be read at all."""


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename or "")[1].lower() in SUPPORTED_EXTENSIONS


def _read_raw_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _decode(raw: bytes) -> tuple:
    """Return (text, encoding_used). latin-1 never fails, so this
    always succeeds - but we try the likelier encodings first so we
    don't silently mangle UTF-8 text into mojibake."""
    for encoding in _ENCODINGS_TO_TRY:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _sniff_delimiter(sample: str, default: str = ",") -> str:
    """Detect the field separator of a delimited text file.

    csv.Sniffer is the first attempt; when it fails (it does, on
    short or irregular files) we fall back to counting candidate
    delimiters in the header line, which is crude but effective for
    the semicolon-separated exports that European Excel produces.
    """
    head = sample[:_SNIFF_BYTES]
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except Exception:
        pass

    first_line = head.splitlines()[0] if head.splitlines() else ""
    counts = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else default


def _load_delimited(path: str) -> LoadedTable:
    raw = _read_raw_bytes(path)
    text, encoding = _decode(raw)

    if not text.strip():
        raise LoadError("The file is empty.")

    delimiter = _sniff_delimiter(text)

    notes = []
    if encoding != "utf-8":
        notes.append(f"Decoded using '{encoding}' (not valid UTF-8).")
    if delimiter != ",":
        notes.append(f"Detected '{delimiter}' as the field separator.")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            # Keep everything as text on the way in. Type inference
            # is the cleaner's job, where it is deliberate, logged,
            # and reversible - not an invisible side effect of
            # reading the file.
            dtype=str,
            keep_default_na=True,
            skip_blank_lines=True,
        )
    except Exception as exc:
        raise LoadError(f"Could not parse the delimited file: {exc}") from exc

    return LoadedTable(
        df=df,
        source_file=os.path.basename(path),
        source_format="delimited",
        notes=notes,
    )


def _load_excel(path: str) -> List[LoadedTable]:
    try:
        workbook = pd.read_excel(path, sheet_name=None, dtype=str)
    except ImportError as exc:
        raise LoadError(
            "Reading Excel files requires the 'openpyxl' package. "
            "Install it with: pip install openpyxl"
        ) from exc
    except Exception as exc:
        raise LoadError(f"Could not read the Excel workbook: {exc}") from exc

    tables = []
    non_empty = {
        name: frame for name, frame in workbook.items() if not frame.empty
    }

    if not non_empty:
        raise LoadError("The workbook contains no non-empty sheets.")

    multi_sheet = len(non_empty) > 1
    for sheet_name, frame in non_empty.items():
        notes = []
        if multi_sheet:
            notes.append(
                f"Sheet '{sheet_name}' was loaded as its own dataset."
            )
        tables.append(
            LoadedTable(
                df=frame,
                source_file=os.path.basename(path),
                source_format="excel",
                sheet_name=sheet_name if multi_sheet else None,
                notes=notes,
            )
        )

    return tables


def _load_json(path: str) -> LoadedTable:
    raw = _read_raw_bytes(path)
    text, _encoding = _decode(raw)

    if not text.strip():
        raise LoadError("The file is empty.")

    notes = []
    records = None

    # Try strict JSON first.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        # Fall back to JSON Lines: one object per line.
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LoadError(
                    f"Not valid JSON, and line {line_number} is not "
                    f"valid JSON Lines either: {exc}"
                ) from exc
        records = rows
        notes.append("Parsed as JSON Lines (one object per line).")
    elif isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        # A common shape is {"data": [...]} or {"results": [...]}.
        # Prefer the longest list value; otherwise treat the dict
        # itself as a single record.
        list_values = {
            key: value for key, value in parsed.items()
            if isinstance(value, list)
        }
        if list_values:
            key = max(list_values, key=lambda k: len(list_values[k]))
            records = list_values[key]
            notes.append(f"Extracted the record list from key '{key}'.")
        else:
            records = [parsed]
    else:
        raise LoadError(
            "The JSON file is a bare scalar, not a table of records."
        )

    if not records:
        raise LoadError("The JSON file contains no records.")

    try:
        # json_normalize flattens nested objects into dotted columns
        # (address.city), which is exactly what a tabular downstream
        # needs. Lists-of-lists stay as-is and get stringified later.
        df = pd.json_normalize(records)
    except Exception as exc:
        raise LoadError(f"Could not flatten the JSON records: {exc}") from exc

    if any("." in str(column) for column in df.columns):
        notes.append("Nested JSON objects were flattened into columns.")

    return LoadedTable(
        df=df,
        source_file=os.path.basename(path),
        source_format="json",
        notes=notes,
    )


def _load_parquet(path: str) -> LoadedTable:
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise LoadError(
            "Reading Parquet files requires 'pyarrow'. "
            "Install it with: pip install pyarrow"
        ) from exc
    except Exception as exc:
        raise LoadError(f"Could not read the Parquet file: {exc}") from exc

    return LoadedTable(
        df=df,
        source_file=os.path.basename(path),
        source_format="parquet",
    )


def load_file(path: str) -> List[LoadedTable]:
    """
    Read one uploaded file into one or more DataFrames.

    Raises LoadError with a user-facing message on failure; the
    pipeline turns that into a per-file error in the report rather
    than failing the whole upload.
    """
    if not os.path.exists(path):
        raise LoadError("File does not exist on the server.")

    extension = os.path.splitext(path)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise LoadError(
            f"Unsupported file type '{extension}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension in (".xlsx", ".xlsm", ".xls"):
        tables = _load_excel(path)
    elif extension in (".json", ".jsonl", ".ndjson"):
        tables = [_load_json(path)]
    elif extension == ".parquet":
        tables = [_load_parquet(path)]
    else:
        tables = [_load_delimited(path)]

    for table in tables:
        if table.df.empty:
            raise LoadError("The dataset has no rows.")
        # An all-unnamed header row usually means the real header is
        # further down, or the file is header-less. Flag it rather
        # than guessing and silently dropping the first data row.
        unnamed = sum(
            1 for column in table.df.columns
            if re.fullmatch(r"Unnamed:?\s*\d*", str(column).strip())
        )
        if unnamed and unnamed == len(table.df.columns):
            table.notes.append(
                "No usable header row was detected; columns were "
                "auto-named and may need manual correction."
            )

    return tables


def sanitize_table_name(name: str) -> str:
    """
    Turn 'Order Items (2).csv' into 'order_items_2'.

    Mirrors the behaviour of ``_sanitize_table_name`` in
    app/api/main.py so table names stay consistent with datasets
    uploaded before the preparation engine existed.
    """
    base = os.path.splitext(str(name or ""))[0]
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", base).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "dataset"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned
