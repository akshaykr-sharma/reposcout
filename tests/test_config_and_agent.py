import sqlite3

import pytest
from deepagents.middleware.filesystem import _check_fs_permission

from reposcout.agent import build_agent, build_permissions
from reposcout.config import ConfigError, load_settings
from reposcout.llm import OllamaUnavailable, check_ollama
from reposcout.sessions import bind_session


# --- config ------------------------------------------------------------------

def test_bad_number_gives_friendly_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "lots")
    with pytest.raises(ConfigError, match="OLLAMA_NUM_CTX"):
        load_settings()


def test_load_settings_has_no_disk_side_effects(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOSCOUT_DOCS_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("REPOSCOUT_DATA_DIR", str(tmp_path / "data"))
    load_settings()
    assert not (tmp_path / "d").exists() and not (tmp_path / "data").exists()


def test_cache_dir_is_per_repo(tmp_path):
    s = load_settings()
    assert s.with_repo(tmp_path / "a").cache_dir != s.with_repo(tmp_path / "b").cache_dir


# --- permissions -------------------------------------------------------------

def decision(rules, op, path):
    return _check_fs_permission(rules, op, path)


@pytest.mark.parametrize(
    "path",
    [
        "/repo/.env", "/repo/sub/.env.production", "/repo/keys/server.pem", "/repo/id_rsa",
        "/repo/deep/er/credentials.json", "/repo/.npmrc",
    ],
)
def test_secrets_unreadable(path):
    assert decision(build_permissions(False), "read", path) == "deny"


@pytest.mark.parametrize("path", ["/repo/.env.example", "/repo/app/main.py", "/repo/README.md"])
def test_normal_files_readable(path):
    assert decision(build_permissions(False), "read", path) == "allow"


def test_write_rules():
    rules = build_permissions(False)
    assert decision(rules, "write", "/repo/app.py") == "deny"
    assert decision(rules, "write", "/docs/OVERVIEW.md") == "interrupt"
    assert decision(rules, "write", "/memories/AGENTS.md") == "interrupt"
    assert decision(rules, "write", "/scratch/x.md") == "allow"
    assert decision(rules, "write", "/cache/modules/a.md") == "allow"


def test_read_only_mode_denies_docs_and_memory_writes():
    rules = build_permissions(True)
    assert decision(rules, "write", "/docs/OVERVIEW.md") == "deny"
    assert decision(rules, "write", "/memories/AGENTS.md") == "deny"


def test_build_agent_constructs(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOSCOUT_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("REPOSCOUT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLLAMA_WRITER_MODEL", "other-model:1b")
    assert build_agent(load_settings(), read_only=True) is not None
    assert (tmp_path / "data" / "cache").exists()


# --- sessions ----------------------------------------------------------------

def test_session_bound_to_repo(tmp_path):
    conn = sqlite3.connect(":memory:")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert bind_session(conn, "s1", a) is None
    assert bind_session(conn, "s1", a) is None
    assert bind_session(conn, "s1", b) == str(a.resolve())


# --- ollama check ------------------------------------------------------------

def test_check_ollama_unreachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        check_ollama(load_settings(), timeout=0.5)
