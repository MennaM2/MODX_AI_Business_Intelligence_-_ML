# python -m uvicorn app.api.main:app --reload --port 8000
#  python -m http.server 3000
# http://localhost:3000/#
"""
FastAPI backend for the AI-Powered Business Analysis & Automation
Agent.

This is the "production-oriented" surface: the Streamlit UI is a thin
client that talks to these same endpoints over HTTP, so the agent
logic is exercised identically whether it's driven from the browser
UI, curl, or an automated integration.

Sessions can hold multiple datasets (uploading another file adds a
table, never replaces one) and their chat history is persisted to
disk via app.agent.memory (SQLite) - nothing is wiped by a new
upload, and a session survives an API restart.
"""

import logging
import os
import re

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from app.agent import memory
from app.agent.agent import MODEL, run_agent
from app.api.schemas import (
    AnalyzeRequest,
    ChatRequest,
    ChatResponse,
    DatasetMeta,
    HealthResponse,
    ReportRequest,
    ReportResponse,
    SessionDetail,
    SessionSummary,
    UploadResponse,
)
from app.config import settings
from app.data_engine import pipeline
from app.data_engine.loader import SUPPORTED_EXTENSIONS, is_supported
from app.tools.automation_tools import deliver_business_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-api")


app = FastAPI(
    title="AI-Powered Business Analysis & Automation Agent",
    version="1.1.0",
    description=(
        "Agentic backend that turns natural-language business "
        "questions into real tool calls over one or more uploaded "
        "datasets: SQL (including joins across datasets), "
        "statistics, trend analysis, anomaly detection, "
        "visualization, and automated report generation/delivery."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated charts and reports (outputs/chart_*.png,
# outputs/business_report.pdf, etc.) over HTTP so the frontend can
# actually display/download them instead of only being told the file
# exists on the server's disk. Files are reachable at /files/<name>.
os.makedirs(settings.output_dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=settings.output_dir), name="files")


def _title_from_text(text: str, max_len: int = 60) -> str:
    """Turn a filename or a user question into a short, readable
    chat title - used so a session in 'Resume a chat' reads like
    'Orders' or 'Which country has the most orders?' instead of a
    raw uuid."""
    cleaned = " ".join((text or "").replace("_", " ").split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= max_len:
        return cleaned[:1].upper() + cleaned[1:]
    return cleaned[: max_len - 1].rstrip() + "…"


def _sanitize_table_name(filename: str) -> str:
    """
    Turn 'Order Items (2).csv' into a valid, boring SQL identifier
    like 'order_items_2'. This becomes the table name in run_sql_query
    and the dataset_name every other tool expects.
    """
    base = os.path.splitext(filename)[0]
    name = re.sub(r"[^0-9a-zA-Z_]", "_", base).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _resolve_report_file_path(request: ReportRequest):
    """Return (file_path, error_detail_or_None) for /report."""
    if request.file_path:
        return request.file_path, None

    if not request.session_id:
        return None, "Provide either file_path or session_id."

    session = memory.get_session(request.session_id)
    if session is None:
        return None, "Unknown session_id."

    if not session.datasets:
        return None, "This session has no uploaded datasets yet."

    if request.dataset_name:
        path = session.datasets.get(request.dataset_name)
        if path is None:
            return None, (
                f"Unknown dataset_name '{request.dataset_name}'. "
                f"Available: {list(session.datasets.keys())}"
            )
        return path, None

    if len(session.datasets) == 1:
        return next(iter(session.datasets.values())), None

    return None, (
        "This session has multiple datasets - specify dataset_name. "
        f"Available: {list(session.datasets.keys())}"
    )

@app.get("/download/{session_id}/{table_name}", tags=["dataset"])
def download_dataset(session_id: str, table_name: str):
    """Download the prepared and cleaned CSV file for an uploaded table."""
    session = memory.get_session(session_id)
    if not session or table_name not in session.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found in this session.")
    file_path = session.datasets[table_name]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Cleaned dataset file not found on disk.")
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=f"{table_name}_cleaned.csv",
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness/readiness plus safe  configuration status."""
    return HealthResponse(
        status="ok",
        provider="gemini",
        provider_configured=bool(settings.gemini_api_key),
        model=MODEL,
    )


@app.post("/upload", response_model=UploadResponse, tags=["dataset"])
async def upload(
    files: list[UploadFile] = File(...),
    session_id: str = None,
):
    """
    Upload one or more datasets (CSV, TSV, Excel, JSON, Parquet).

    This route stays deliberately thin: it receives the files, writes
    the raw bytes to disk, hands the paths to the Data Preparation
    Engine, and registers whatever the engine produced. All profiling,
    cleaning, schema matching, integration, and validation happens
    inside app/data_engine - never here.

    If `session_id` is given and exists, the datasets are ADDED to
    that session - nothing already there is replaced or cleared. If
    it is omitted, a new session is created.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    unsupported = [
        file.filename for file in files
        if not file.filename or not is_supported(file.filename)
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type(s): {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    if session_id:
        session = memory.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session_id.")
    else:
        session = memory.create_session()

    # --- 1. receive and store the raw uploads -------------------------
    os.makedirs(settings.upload_dir, exist_ok=True)
    saved_files = []

    for file in files:
        raw_path = os.path.join(
            settings.upload_dir, f"{session.session_id}_{file.filename}"
        )
        try:
            content = await file.read()
            with open(raw_path, "wb") as destination:
                destination.write(content)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save '{file.filename}': {exc}",
            ) from exc

        saved_files.append((file.filename, raw_path))

    # --- 2. hand everything to the preparation engine -----------------
    try:
        result = pipeline.prepare_uploads(
            session_id=session.session_id,
            saved_files=saved_files,
            existing_table_names=set(session.datasets.keys()),
        )
    except Exception as exc:
        logger.exception("Data preparation failed")
        raise HTTPException(
            status_code=500, detail=f"Data preparation failed: {exc}"
        ) from exc

    if not result.ready:
        detail = "; ".join(
            f"{error['file']}: {error['error']}" for error in result.errors
        )
        raise HTTPException(
            status_code=400,
            detail=detail or "No dataset could be prepared from these files.",
        )

    # --- 3. register the PREPARED datasets with the session -----------
    # This one line is the entire seam between the new engine and the
    # existing agent: every tool resolves a table name to a path and
    # reads it, so pointing those paths at the cleaned CSVs makes the
    # whole analyst stack work on prepared data with no other change.
    for dataset in result.datasets:
        memory.add_dataset(
            session.session_id, dataset.table_name, dataset.prepared_path
        )

    if not session.title:
        # Placeholder title, shown in "Resume a chat" until the first
        # question gives it a better one - see chat_endpoint below.
        memory.set_title(
            session.session_id,
            _title_from_text(result.datasets[0].table_name),
        )

    refreshed = memory.get_session(session.session_id)

    return UploadResponse(
        session_id=session.session_id,
        datasets=[
            DatasetMeta(
                name=dataset.table_name,
                table_name=dataset.table_name,
                rows=dataset.rows,
                columns=dataset.columns,
            )
            for dataset in result.datasets
        ],
        all_datasets=list(refreshed.datasets.keys()),
        preparation=result.report,
        errors=result.errors,
    )


@app.get("/sessions", response_model=list[SessionSummary], tags=["dataset"])
def list_sessions():
    """List all persisted sessions (past and current chats)."""
    return memory.list_sessions()


@app.get(
    "/sessions/{session_id}", response_model=SessionDetail, tags=["dataset"]
)
def get_session(session_id: str):
    """Full detail for one session: its datasets and complete chat
    history - used to resume a past chat exactly where it left off."""
    session = memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    return SessionDetail(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        datasets=session.datasets,
        history=memory.get_full_history(session.session_id),
        # Served from the stored report rather than recomputed - no
        # new endpoint, and resuming a chat shows the same data-quality
        # context the user saw when they first uploaded.
        preparation=pipeline.get_report(session_id),
    )


@app.delete("/sessions/{session_id}", tags=["dataset"])
def delete_session(session_id: str):
    deleted = memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    # Prepared CSVs should not outlive the session that owns them.
    pipeline.delete_session_artifacts(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat_endpoint(request: ChatRequest):
    """
    Ask the agent a question within an existing session. Every
    dataset ever uploaded to this session is available to the agent;
    history is loaded from and saved to persistent storage, so it
    survives across uploads and API restarts.
    """
    session = memory.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown session_id. Upload a dataset first via /upload.",
        )
    if not session.datasets:
        raise HTTPException(
            status_code=400,
            detail="No datasets are associated with this session yet.",
        )

    is_first_turn = not memory.get_full_history(session.session_id)
    recent_history = memory.get_recent_history(session.session_id)

    try:
        result = run_agent(
            request.message,
            session.datasets,
            history=recent_history,
            data_context=pipeline.agent_context(session.session_id),
        )
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(
            status_code=502, detail=f"Agent failed: {exc}"
        ) from exc

    if is_first_turn:
        # The first real question is a better chat title than the
        # dataset-name placeholder set at upload time - this is what
        # makes "Resume a chat" read like an actual conversation
        # instead of a table name once you've started chatting.
        memory.set_title(session.session_id, _title_from_text(request.message))

    memory.append_turn(session.session_id, "user", request.message)
    if result.get("answer"):
        memory.append_turn(session.session_id, "assistant", result["answer"])

    return ChatResponse(
        session_id=session.session_id,
        answer=result.get("answer") or "",
        tools_used=result.get("tools_used", []),
        tool_log=result.get("tool_log", []),
        plot_path=result.get("plot_path"),
        report_files=result.get("report_files"),
        ml_result=result.get("ml_result"),
        latency_seconds=(result.get("latency") or {}).get("total_seconds"),
        error=result.get("error"),
        error_detail=result.get("error_detail"),
    )


@app.post("/analyze", response_model=ChatResponse, tags=["agent"])
def analyze(request: AnalyzeRequest):
    """
    Stateless variant of /chat: pass a file_path directly (no prior
    /upload needed, single dataset, no persisted history) - mainly
    useful for quick testing/scripting.
    """
    if not request.file_path:
        raise HTTPException(
            status_code=400, detail="Provide file_path for /analyze."
        )

    try:
        result = run_agent(
            request.question,
            {"dataset": request.file_path},
        )
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(
            status_code=502, detail=f"Agent failed: {exc}"
        ) from exc

    return ChatResponse(
        session_id="",
        answer=result.get("answer") or "",
        tools_used=result.get("tools_used", []),
        tool_log=result.get("tool_log", []),
        plot_path=result.get("plot_path"),
        report_files=result.get("report_files"),
        ml_result=result.get("ml_result"),
        latency_seconds=(result.get("latency") or {}).get("total_seconds"),
        error=result.get("error"),
        error_detail=result.get("error_detail"),
    )


@app.post("/report", response_model=ReportResponse, tags=["automation"])
def report(request: ReportRequest):
    """
    Deterministically generate (and save, and optionally email) a
    business report for one dataset - bypasses the LLM entirely for a
    fast, reliable automation path. If the session has more than one
    dataset, `dataset_name` must say which one.
    """
    file_path, error_detail = _resolve_report_file_path(request)

    if not file_path:
        raise HTTPException(status_code=400, detail=error_detail)

    result = deliver_business_report(
        file_path,
        date_column=request.date_column,
        value_column=request.value_column,
        category_column=request.category_column,
        target_column=request.target_column,
        email_to=request.email_to,
    )

    if "error" in result:
        return ReportResponse(status="error", error=result["error"])

    return ReportResponse(
        status="success",
        report_path_markdown=result.get("report_path_markdown"),
        report_path_html=result.get("report_path_html"),
        report_path_pdf=result.get("report_path_pdf"),
        report_path_docx=result.get("report_path_docx"),
        detected_columns=result.get("detected_columns"),
        key_insights=result.get("key_insights", []),
        recommendations=result.get("recommendations", []),
        email=result.get("email"),
    )