"""
Persistent Session Store (SQLite).

Replaces the earlier in-process dict. Three properties this version
guarantees that the old one didn't:

1. Uploading a new file to an existing session ADDS a dataset/table -
   it never replaces or wipes the ones already there.
2. Chat history is written to disk on every turn and is never lost -
   not on a new upload, not on an API restart.
3. Multiple sessions ("chats") can exist side by side and be listed
   or resumed later via session_id.

Nothing here is exotic: three tables (sessions, session_datasets,
session_messages) in a single SQLite file. That's an intentional
choice for a portfolio project - no extra service to run - documented
as a scaling limit in the README (SQLite is fine for one process; a
real multi-worker deployment would move this to Postgres/Redis).
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import settings


@dataclass
class Session:
    session_id: str
    title: Optional[str] = None
    datasets: dict = field(default_factory=dict)  # table_name -> file_path
    created_at: float = field(default_factory=time.time)


def _db_path() -> Path:
    path = Path(settings.sessions_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect():
    connection = sqlite3.connect(_db_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_datasets (
                session_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                added_at REAL NOT NULL,
                PRIMARY KEY (session_id, table_name),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )


# Ensure tables exist as soon as this module is imported.
init_db()


def create_session(title: Optional[str] = None) -> Session:
    session_id = str(uuid.uuid4())
    created_at = time.time()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (session_id, title, created_at) "
            "VALUES (?, ?, ?)",
            (session_id, title, created_at),
        )
    return Session(session_id=session_id, title=title, created_at=created_at)


def get_session(session_id: str) -> Optional[Session]:
    if not session_id:
        return None

    with _connect() as connection:
        row = connection.execute(
            "SELECT session_id, title, created_at FROM sessions "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            return None

        dataset_rows = connection.execute(
            "SELECT table_name, file_path FROM session_datasets "
            "WHERE session_id = ? ORDER BY added_at",
            (session_id,),
        ).fetchall()

    datasets = {r["table_name"]: r["file_path"] for r in dataset_rows}

    return Session(
        session_id=row["session_id"],
        title=row["title"],
        datasets=datasets,
        created_at=row["created_at"],
    )


def add_dataset(session_id: str, table_name: str, file_path: str) -> None:
    """Attach a dataset/table to a session. Adds it - never removes
    or overwrites any dataset already attached under a different
    name. Re-uploading the same table_name does update its path."""
    with _connect() as connection:
        connection.execute(
            "INSERT INTO session_datasets "
            "(session_id, table_name, file_path, added_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id, table_name) "
            "DO UPDATE SET file_path = excluded.file_path, "
            "added_at = excluded.added_at",
            (session_id, table_name, file_path, time.time()),
        )


def append_turn(session_id: str, role: str, content: str) -> None:
    if not content:
        return
    with _connect() as connection:
        connection.execute(
            "INSERT INTO session_messages "
            "(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_full_history(session_id: str) -> list:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT role, content, created_at FROM session_messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        for r in rows
    ]


def get_recent_history(session_id: str, limit: int = None) -> list:
    """The trimmed window actually sent to the LLM as context - full
    history is kept on disk regardless (see get_full_history)."""
    limit = limit or settings.max_history_messages
    full = get_full_history(session_id)
    trimmed = full[-limit:] if limit else full
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]


def list_sessions() -> list:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                s.session_id,
                s.title,
                s.created_at,
                (SELECT COUNT(*) FROM session_messages m
                 WHERE m.session_id = s.session_id) AS message_count,
                (SELECT GROUP_CONCAT(table_name) FROM session_datasets d
                 WHERE d.session_id = s.session_id) AS dataset_names
            FROM sessions s
            ORDER BY s.created_at DESC
            """
        ).fetchall()

    return [
        {
            "session_id": r["session_id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "message_count": r["message_count"],
            "datasets": r["dataset_names"].split(",") if r["dataset_names"] else [],
        }
        for r in rows
    ]


def set_title(session_id: str, title: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )


def delete_session(session_id: str) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM session_datasets WHERE session_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM session_messages WHERE session_id = ?", (session_id,)
        )
    return cursor.rowcount > 0