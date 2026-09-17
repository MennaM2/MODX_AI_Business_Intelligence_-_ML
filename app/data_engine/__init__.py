"""
Data Preparation Engine.

Deterministic Python/data-engineering code that turns messy,
heterogeneous uploads into clean, typed, integrated datasets before
the AI Data Analyst agent ever sees them.

Public surface - everything else is internal:

    from app.data_engine.pipeline import prepare_uploads, get_report

The engine does not import the agent and does not call Gemini, with
one narrow, opt-in exception (``llm_assist``) used only to break ties
between ambiguous column matches, and only on metadata.
"""

__all__ = ["pipeline"]
