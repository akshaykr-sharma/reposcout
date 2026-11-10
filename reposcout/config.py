"""
Settings for RepoScout.

All configuration comes from environment variables. A `.env` file in the
project folder is loaded automatically, so you normally just copy
`.env.example` to `.env` and edit it.

Takes:  environment variables (OLLAMA_MODEL, REPOSCOUT_REPO_PATH, ...)
Gives:  a `Settings` dataclass via `load_settings()`

The same settings object is used by the CLI, by `langgraph dev` (graph.py)
and by the eval script, which keeps the three entry points consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# override=False means a variable you export in the shell wins over .env
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_IGNORES = [
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".idea", ".vscode", ".next", "target", ".reposcout",
]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _env_path(name: str, default: Path) -> Path:
    # empty values in .env count as "not set"
    raw = os.getenv(name, "").strip()
    if not raw:
        return Path(default).resolve()
    path = Path(raw).expanduser()
    # relative paths in .env are relative to the project folder, not the shell
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs to know about its environment."""

    ollama_model: str
    ollama_base_url: str
    num_ctx: int
    temperature: float
    repo_path: Path
    docs_dir: Path
    data_dir: Path
    extra_ignores: list[str] = field(default_factory=list)

    @property
    def ignores(self) -> set[str]:
        return set(DEFAULT_IGNORES) | set(self.extra_ignores)

    def with_repo(self, repo: str | Path, docs: str | Path | None = None) -> "Settings":
        """Return a copy pointing at a different repo (used by the CLI flags)."""
        repo = Path(repo).expanduser().resolve()
        docs_dir = Path(docs).expanduser().resolve() if docs else self.docs_dir
        return replace(self, repo_path=repo, docs_dir=docs_dir)


def load_settings() -> Settings:
    """Read settings from the environment and make sure the folders exist."""
    repo = _env_path("REPOSCOUT_REPO_PATH", PROJECT_ROOT)
    docs = _env_path("REPOSCOUT_DOCS_DIR", PROJECT_ROOT / "output" / "docs")
    data = _env_path("REPOSCOUT_DATA_DIR", PROJECT_ROOT / ".reposcout")

    extra = [x.strip() for x in os.getenv("REPOSCOUT_IGNORE", "").split(",") if x.strip()]

    settings = Settings(
        ollama_model=os.getenv("OLLAMA_MODEL") or "qwen2.5:7b",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
        num_ctx=_env_int("OLLAMA_NUM_CTX", 16384),
        temperature=_env_float("OLLAMA_TEMPERATURE", 0.1),
        repo_path=repo,
        docs_dir=docs,
        data_dir=data,
        extra_ignores=extra,
    )
    ensure_dirs(settings)
    return settings


def ensure_dirs(settings: Settings) -> None:
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)


def langsmith_enabled() -> bool:
    """True when tracing is switched on and a key is present."""
    on = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    return on and bool(os.getenv("LANGSMITH_API_KEY"))
