"""
Cross-dataset column matching.

Recognizes that ``Customer_ID`` in customers.csv, ``customer_id`` in
orders.csv, and ``cust_id`` in a third file all refer to the same
concept. This is what makes automatic join-key detection possible.

Entirely deterministic by default. Four independent signals are
combined into one score:

  name similarity   normalized names, an abbreviation table, and
                    token overlap
  type agreement    two columns of different types rarely mean the
                    same thing
  value overlap     the strongest signal by far - if 95% of one
                    column's values appear in the other, they are
                    almost certainly the same key
  cardinality       a unique key matching a repeating foreign key

Value overlap is weighted highest because it is evidence from the
data itself rather than from naming conventions, which users break
constantly.

An LLM fallback exists for genuinely ambiguous pairs but is OFF by
default. When enabled it receives only column names, types, and three
sample values - never the datasets.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# A match is reported at or above this score, and treated as
# confident (usable for an automatic join) at the higher threshold.
MATCH_THRESHOLD = 0.55
CONFIDENT_THRESHOLD = 0.80

# An automatic join additionally requires real agreement on BOTH
# independent signals - see ColumnMatch.confident.
MIN_CONFIDENT_NAME_SCORE = 0.50
MIN_CONFIDENT_OVERLAP = 0.70

# Columns that are artifacts of the export process. They often match
# perfectly across files (every row carries the same import
# timestamp) while meaning nothing, so they are never join keys.
_ARTIFACT_COLUMN_PATTERN = re.compile(
    r"(^|_)(index|unnamed|row_?num(ber)?|import|ingest|etl|load|"
    r"extract|batch|source_?file|created_?at|updated_?at|"
    r"raw_|_raw$|sys_|tmp_)",
    re.IGNORECASE,
)

# Pairs in this band are where an LLM could genuinely add something;
# outside it the deterministic answer is already clear.
AMBIGUOUS_BAND = (0.45, 0.80)

# Compare at most this many distinct values per column when measuring
# overlap. Keeps a 5-million-row join check cheap without materially
# changing the ratio.
_MAX_OVERLAP_SAMPLE = 20000

# Common abbreviations, expanded before comparing names. Small and
# hand-curated on purpose - a large fuzzy dictionary produces more
# false matches than it prevents.
_ABBREVIATIONS = {
    "cust": "customer",
    "custid": "customerid",
    "usr": "user",
    "acct": "account",
    "prod": "product",
    "qty": "quantity",
    "amt": "amount",
    "num": "number",
    "no": "number",
    "dt": "date",
    "ts": "timestamp",
    "addr": "address",
    "org": "organization",
    "emp": "employee",
    "dept": "department",
    "txn": "transaction",
    "trans": "transaction",
    "inv": "invoice",
    "ref": "reference",
    "desc": "description",
    "cat": "category",
    "pymt": "payment",
    "pmt": "payment",
    "tel": "phone",
    "mob": "phone",
    "mobile": "phone",
    "fname": "firstname",
    "lname": "lastname",
    "dob": "dateofbirth",
    "pk": "id",
    "key": "id",
    "code": "id",
}

# Suffixes that carry no meaning for matching: 'customer_id' and
# 'customer_key' should collapse to the same concept.
_NOISE_TOKENS = {"the", "a", "of", "col", "column", "field", "value"}


@dataclass
class ColumnMatch:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    score: float
    name_score: float
    type_score: float
    overlap_score: float
    left_unique: bool
    right_unique: bool
    evidence: List[str] = field(default_factory=list)
    llm_opinion: Optional[str] = None

    @property
    def confident(self) -> bool:
        """
        Confident means "safe to join on automatically".

        Both signals are required, not just a high combined score.
        Value overlap alone is not enough: two unrelated integer
        sequences can overlap completely. Name agreement alone is not
        enough either: a 'code' column in two files may be entirely
        different code systems. Demanding both is what keeps an
        automatic join from silently producing wrong totals.
        """
        return (
            self.score >= CONFIDENT_THRESHOLD
            and self.name_score >= MIN_CONFIDENT_NAME_SCORE
            and self.overlap_score >= MIN_CONFIDENT_OVERLAP
        )

    def to_dict(self) -> dict:
        return {
            "left": f"{self.left_table}.{self.left_column}",
            "right": f"{self.right_table}.{self.right_column}",
            "left_table": self.left_table,
            "left_column": self.left_column,
            "right_table": self.right_table,
            "right_column": self.right_column,
            "score": round(self.score, 3),
            "confident": self.confident,
            "name_similarity": round(self.name_score, 3),
            "type_agreement": round(self.type_score, 3),
            "value_overlap": round(self.overlap_score, 3),
            "left_is_unique": self.left_unique,
            "right_is_unique": self.right_unique,
            "evidence": self.evidence,
            "llm_opinion": self.llm_opinion,
        }


def _tokenize(name: str) -> list:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).lower()
    tokens = [token for token in text.split("_") if token]
    expanded = []
    for token in tokens:
        if token in _NOISE_TOKENS:
            continue
        expanded.append(_ABBREVIATIONS.get(token, token))
    return expanded or tokens


def _canonical_name(name: str) -> str:
    return "".join(_tokenize(name))


def name_similarity(left: str, right: str) -> float:
    """Blend exact-canonical, token-overlap, and character-level
    similarity so that 'cust_id'/'customer_id' scores high while
    'order_date'/'order_amount' does not."""
    left_canonical = _canonical_name(left)
    right_canonical = _canonical_name(right)

    if left_canonical == right_canonical:
        return 1.0

    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(
            left_tokens | right_tokens
        )
    else:
        jaccard = 0.0

    ratio = difflib.SequenceMatcher(
        None, left_canonical, right_canonical
    ).ratio()

    return max(jaccard, 0.85 * ratio)


def _type_agreement(left_type: str, right_type: str) -> float:
    if left_type == right_type:
        return 1.0
    # IDs legitimately arrive as text in one file and numbers in
    # another; that mismatch should cost something but not disqualify.
    numeric_like = {"numeric", "boolean"}
    if left_type in numeric_like and right_type in numeric_like:
        return 0.7
    if {left_type, right_type} <= {"categorical", "text", "numeric"}:
        return 0.5
    return 0.0


def _comparable_values(series: pd.Series) -> set:
    """
    Normalize values so a key stored as 1001 (int) matches '1001'
    (text) and ' 1001 ' (padded text). Without this, value overlap
    misses the single most common real-world case.
    """
    values = series.dropna()
    if values.empty:
        return set()

    if len(values) > _MAX_OVERLAP_SAMPLE:
        values = values.head(_MAX_OVERLAP_SAMPLE)

    text = values.astype(str).str.strip().str.lower()
    # '1001.0' and '1001' are the same key.
    text = text.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return set(text.unique())


def _is_contiguous_integer_range(values: set) -> bool:
    """
    True when a value set is essentially 1..N.

    This is the single biggest source of false join keys. Row
    numbers, ticket IDs, customer IDs, and an 'age' column are all
    dense integer sequences, so any two of them overlap almost
    perfectly by coincidence. Containment cannot tell 'these are the
    same key' from 'these are both small integers'.
    """
    if len(values) < 3:
        return False

    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if number != int(number):
            return False
        numbers.append(int(number))

    span = max(numbers) - min(numbers) + 1
    # Allow gaps, but a genuinely sparse key (order IDs scattered
    # across a huge range) is informative and should not be penalized.
    return span <= len(numbers) * 1.5


def value_overlap(left: pd.Series, right: pd.Series) -> tuple:
    """
    Returns (containment, coincidence_risk).

    Containment, not symmetric similarity: a foreign key column with
    50,000 rows pointing at a 1,000-row primary key should score
    ~1.0. Jaccard would score that pair around 0.02 and miss every
    real join in the dataset.

    ``coincidence_risk`` flags the cases where a containment of 1.0
    means nothing on its own - two dense integer sequences (every
    primary key is 1..N, and so is an 'age' column), or a handful of
    distinct values (status flags always overlap).

    The risk is returned rather than applied here because whether it
    matters depends on the names. 'customer_id' matching 'cust_id' is
    still a real key even though both are 1..N; 'age' matching
    'ticket_id' is not. Only the caller knows the name score, so only
    the caller can make that call.
    """
    left_values = _comparable_values(left)
    right_values = _comparable_values(right)

    if not left_values or not right_values:
        return 0.0, False

    intersection = len(left_values & right_values)
    if not intersection:
        return 0.0, False

    smaller = min(len(left_values), len(right_values))
    if smaller < 2:
        # A constant column contains everything and identifies
        # nothing, regardless of what it is called.
        return 0.0, False

    containment = intersection / smaller

    coincidence_risk = (
        smaller < 8
        or (
            _is_contiguous_integer_range(left_values)
            and _is_contiguous_integer_range(right_values)
        )
    )

    return containment, coincidence_risk


def _column_detail(profile: dict, column_name: str) -> dict:
    for column in profile.get("columns_detail", []):
        if column["name"] == column_name:
            return column
    return {}


def match_schemas(
    tables: dict,
    profiles: dict,
    llm_resolver=None,
) -> List[ColumnMatch]:
    """
    Compare every column of every dataset against every other.

    ``tables``   maps table_name -> cleaned DataFrame
    ``profiles`` maps table_name -> profile dict (post-clean)
    ``llm_resolver`` is an optional callable used only for pairs that
    land in the ambiguous band; see ``llm_assist.py``. When None
    (the default) the process is fully deterministic.
    """
    matches = []
    table_names = list(tables.keys())

    for i, left_table in enumerate(table_names):
        for right_table in table_names[i + 1:]:
            left_df = tables[left_table]
            right_df = tables[right_table]
            left_profile = profiles[left_table]
            right_profile = profiles[right_table]

            for left_column in left_df.columns:
                left_detail = _column_detail(left_profile, str(left_column))
                left_type = left_detail.get("inferred_type", "text")

                if _ARTIFACT_COLUMN_PATTERN.search(str(left_column)):
                    continue
                if left_detail.get("is_constant"):
                    continue

                for right_column in right_df.columns:
                    if _ARTIFACT_COLUMN_PATTERN.search(str(right_column)):
                        continue
                    right_detail = _column_detail(
                        right_profile, str(right_column)
                    )
                    right_type = right_detail.get("inferred_type", "text")

                    if right_detail.get("is_constant"):
                        continue

                    name_score = name_similarity(
                        str(left_column), str(right_column)
                    )
                    type_score = _type_agreement(left_type, right_type)

                    # Skip the expensive set comparison for pairs
                    # that are obviously unrelated by name and type.
                    if name_score < 0.3 and type_score < 0.5:
                        continue

                    overlap_score, coincidence_risk = value_overlap(
                        left_df[left_column], right_df[right_column]
                    )

                    # Overlap that could be coincidence only counts
                    # when the names independently agree. This is the
                    # guard that separates 'customer_id' matching
                    # 'cust_id' (both 1..N, but genuinely the same
                    # key) from 'age' matching 'ticket_id' (both
                    # 1..N, and pure coincidence).
                    if coincidence_risk and name_score < 0.5:
                        overlap_score *= 0.3

                    # Free text and high-cardinality descriptions
                    # overlap by coincidence; don't let that count.
                    if left_type == "text" and right_type == "text":
                        overlap_score *= 0.5

                    score = (
                        0.30 * name_score
                        + 0.15 * type_score
                        + 0.55 * overlap_score
                    )

                    # A name match with literally no shared values is
                    # a naming coincidence, not a relationship.
                    if overlap_score == 0.0:
                        score *= 0.45

                    if score < AMBIGUOUS_BAND[0]:
                        continue

                    evidence = []
                    if name_score >= 0.9:
                        evidence.append("column names are equivalent")
                    elif name_score >= 0.6:
                        evidence.append("column names are similar")
                    if overlap_score >= 0.9:
                        evidence.append(
                            f"{overlap_score:.0%} of values are shared"
                        )
                    elif overlap_score >= 0.5:
                        evidence.append(
                            f"{overlap_score:.0%} of values overlap"
                        )
                    if type_score == 1.0:
                        evidence.append(f"both are {left_type}")

                    left_unique = bool(
                        left_detail.get("unique_ratio", 0) >= 0.99
                    )
                    right_unique = bool(
                        right_detail.get("unique_ratio", 0) >= 0.99
                    )
                    if left_unique != right_unique:
                        evidence.append(
                            "one side is unique, the other repeats "
                            "(one-to-many)"
                        )

                    matches.append(
                        ColumnMatch(
                            left_table=left_table,
                            left_column=str(left_column),
                            right_table=right_table,
                            right_column=str(right_column),
                            score=score,
                            name_score=name_score,
                            type_score=type_score,
                            overlap_score=overlap_score,
                            left_unique=left_unique,
                            right_unique=right_unique,
                            evidence=evidence,
                        )
                    )

    # Optional semantic tie-break, only for the ambiguous middle.
    if llm_resolver is not None:
        ambiguous = [
            match for match in matches
            if AMBIGUOUS_BAND[0] <= match.score < AMBIGUOUS_BAND[1]
        ]
        if ambiguous:
            _apply_llm_opinions(ambiguous, tables, llm_resolver)

    matches = [match for match in matches if match.score >= MATCH_THRESHOLD]
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


def _apply_llm_opinions(ambiguous, tables, llm_resolver) -> None:
    """
    Ask the resolver about ambiguous pairs, passing metadata only.

    The payload is column names, inferred types, and three sample
    values per side. The datasets themselves never leave the engine.
    A returned verdict nudges the score; it cannot by itself create a
    confident match out of nothing.
    """
    payload = []
    for match in ambiguous:
        left_samples = (
            tables[match.left_table][match.left_column]
            .dropna().astype(str).head(3).tolist()
        )
        right_samples = (
            tables[match.right_table][match.right_column]
            .dropna().astype(str).head(3).tolist()
        )
        payload.append({
            "left": f"{match.left_table}.{match.left_column}",
            "right": f"{match.right_table}.{match.right_column}",
            "left_samples": left_samples,
            "right_samples": right_samples,
        })

    try:
        verdicts = llm_resolver(payload) or {}
    except Exception:
        # The engine must stay deterministic and working even if the
        # LLM is unreachable, rate-limited, or misconfigured.
        return

    for match in ambiguous:
        key = f"{match.left_table}.{match.left_column}|" \
              f"{match.right_table}.{match.right_column}"
        verdict = verdicts.get(key)
        if verdict is None:
            continue

        match.llm_opinion = verdict
        if verdict == "same":
            match.score = min(1.0, match.score + 0.15)
            match.evidence.append("semantic review agreed")
        elif verdict == "different":
            match.score = max(0.0, match.score - 0.25)
            match.evidence.append("semantic review disagreed")
