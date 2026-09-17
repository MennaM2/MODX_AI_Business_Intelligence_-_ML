"""Environment-driven application configuration."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv() 


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_timeout_seconds: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "75"))
    gemini_max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "2"))

    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    sessions_db_path: str = os.getenv("SESSIONS_DB_PATH", "data/sessions.db")
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USER", "")
    smtp_use_tls: bool = _get_bool("SMTP_USE_TLS", True)

    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)


settings = Settings()