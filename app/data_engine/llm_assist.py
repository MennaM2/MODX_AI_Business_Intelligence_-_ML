"""
Optional semantic tie-breaker for ambiguous column matches.

Off by default. Enable with DATA_ENGINE_LLM_MATCHING=true.

This is the single place the preparation engine is allowed to talk to
Gemini, and it is deliberately hemmed in:

  - it runs only for pairs the deterministic matcher scored in the
    ambiguous band, never for every column pair
  - it sends column names, inferred types, and three sample values -
    never a dataset, never a full column
  - its verdict adjusts a score; it cannot create a match on its own
  - any failure (no key, timeout, bad JSON) degrades silently back to
    the deterministic result

Everything else in the engine - cleaning, typing, deduplication,
merging, validation - stays completely free of the LLM.
"""

import json
import os

from app.config import settings


# Hard cap on how many pairs are ever sent. Ambiguity beyond this is
# a sign the datasets are unrelated, not that more LLM calls are
# needed.
MAX_PAIRS_PER_CALL = 25

_PROMPT = """You are helping match columns across database tables.

For each pair below, decide whether the two columns represent the
SAME real-world concept (for example "cust_id" and "customer_id"
both identify a customer).

Answer "same", "different", or "unsure". Prefer "unsure" over a
guess.

Respond with ONLY a JSON object mapping each pair's "id" to your
answer. No markdown, no explanation.

Example response:
{"orders.cust_id|customers.customer_id": "same"}

Pairs:
"""


def llm_matching_enabled() -> bool:
    flag = os.getenv("DATA_ENGINE_LLM_MATCHING", "false")
    return (
        flag.strip().lower() in {"1", "true", "yes", "on"}
        and bool(settings.gemini_api_key)
    )


def resolve_ambiguous_matches(pairs: list) -> dict:
    """
    Ask Gemini about ambiguous column pairs.

    ``pairs`` is a list of {"left", "right", "left_samples",
    "right_samples"} dicts built by the schema matcher. Returns a
    mapping of "left|right" -> "same"/"different"/"unsure".

    Returns {} on any failure; the caller treats that as "no opinion".
    """
    if not pairs:
        return {}

    # Imported lazily so the engine has no import-time dependency on
    # the agent module or the OpenAI client.
    from app.agent.agent import _init_client

    trimmed = pairs[:MAX_PAIRS_PER_CALL]

    described = []
    for pair in trimmed:
        described.append({
            "id": f"{pair['left']}|{pair['right']}",
            "column_a": pair["left"],
            "samples_a": pair["left_samples"][:3],
            "column_b": pair["right"],
            "samples_b": pair["right_samples"][:3],
        })

    prompt = _PROMPT + json.dumps(described, indent=2, default=str)

    try:
        client = _init_client()
        response = client.chat.completions.create(
            model=settings.gemini_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception:
        return {}

    # Models wrap JSON in code fences often enough to be worth
    # handling rather than failing on.
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    allowed = {"same", "different", "unsure"}
    return {
        str(key): str(value).strip().lower()
        for key, value in parsed.items()
        if str(value).strip().lower() in allowed
    }


def get_resolver():
    """Return the resolver callable, or None when disabled - which is
    what keeps the matcher deterministic by default."""
    return resolve_ambiguous_matches if llm_matching_enabled() else None
