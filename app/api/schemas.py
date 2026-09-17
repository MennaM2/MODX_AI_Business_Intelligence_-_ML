from typing import Optional

from pydantic import BaseModel, Field


class DatasetMeta(BaseModel):
    name: str
    table_name: str
    rows: int
    columns: int
    # Post-preparation figure. Optional because a prepared dataset
    # reports its remaining gaps through the preparation report,
    # which is far more informative than a single total.
    missing_values: Optional[int] = None


class UploadResponse(BaseModel):
    session_id: str
    # One upload can yield several datasets: several files at once,
    # a multi-sheet workbook, or a materialized join.
    datasets: list[DatasetMeta]
    # Every table now available in this session, including ones
    # uploaded in earlier calls - confirms nothing was wiped.
    all_datasets: list[str]
    # The full structured report from the Data Preparation Engine.
    # Left as a free-form dict on purpose: the engine owns its shape,
    # and pinning it down here would mean editing this file every
    # time a new check is added.
    preparation: Optional[dict] = None
    # Files that could not be read. Other files in the same batch are
    # still prepared - one bad file does not fail the upload.
    errors: list[dict] = []


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)


class ToolLogEntry(BaseModel):
    tool: str
    arguments: dict
    success: bool


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tools_used: list[str] = []
    tool_log: list[ToolLogEntry] = []
    plot_path: Optional[str] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    report_files: Optional[dict] = None
    ml_result: Optional[dict] = None


class AnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    file_path: Optional[str] = None
    question: str = Field(..., min_length=1)


class ReportRequest(BaseModel):
    session_id: Optional[str] = None
    # Which uploaded table to report on. Optional only when the
    # session has exactly one dataset.
    dataset_name: Optional[str] = None
    file_path: Optional[str] = None
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None
    target_column: Optional[str] = None
    email_to: Optional[str] = None


class ReportResponse(BaseModel):
    status: str
    report_path_markdown: Optional[str] = None
    report_path_html: Optional[str] = None
    detected_columns: Optional[dict] = None
    key_insights: list[str] = []
    recommendations: list[str] = []
    email: Optional[dict] = None
    error: Optional[str] = None
    report_path_pdf: Optional[str] = None
    report_path_docx: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    provider_configured: bool
    model: str


class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: float
    message_count: int
    datasets: list[str] = []


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: float


class SessionDetail(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: float
    datasets: dict[str, str]
    history: list[SessionMessage]
    # None for sessions created before the preparation engine existed.
    preparation: Optional[dict] = None