"""
Remembers which repo each chat session belongs to.

Takes:  the SQLite connection the CLI already uses for checkpoints
Gives:  `bind_session` - register a session, or report that it belongs to
        a different repo

Why: a session's saved messages talk about one codebase. Resuming it with
`--repo` pointing somewhere else would mix the old conversation with new
code and give confusing answers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def init_sessions(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS reposcout_sessions (session TEXT PRIMARY KEY, repo TEXT NOT NULL)")
    conn.commit()


def bind_session(conn: sqlite3.Connection, session: str, repo: Path) -> str | None:
    """Register `session` for `repo`.

    Returns None when it is fine to continue (new session, or same repo),
    or the other repo's path when the session already belongs to a different one.
    """
    init_sessions(conn)
    repo_text = str(Path(repo).resolve())
    row = conn.execute("SELECT repo FROM reposcout_sessions WHERE session = ?", (session,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO reposcout_sessions (session, repo) VALUES (?, ?)", (session, repo_text))
        conn.commit()
        return None
    return None if row[0] == repo_text else row[0]
