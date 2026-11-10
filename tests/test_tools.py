import os

from langchain_core.messages import ToolMessage

from reposcout.config import DEFAULT_IGNORES
from reposcout.fsutil import is_secret, walk_files
from reposcout.tools import make_repo_tools


def call(tool, **args):
    return tool.invoke(args)


def call_as_agent(tool, **args):
    """Invoke the way the agent runtime does, to see the ToolMessage status."""
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "1"})


# --- path handling & secrets -------------------------------------------------

def test_path_outside_repo_is_refused_with_error_status(tools):
    msg = call_as_agent(tools["python_outline"], path="/repo/../../etc/passwd.py")
    assert isinstance(msg, ToolMessage) and msg.status == "error"
    assert "outside the repo" in msg.content


def test_secret_files_hidden_and_blocked(tools):
    listing = call(tools["repo_map"])
    assert ".env.example" in listing
    assert ".env\n" not in listing + "\n" and "\n.env " not in listing
    msg = call_as_agent(tools["code_outline"], path="/repo/.env")
    assert msg.status == "error" and "credentials" in msg.content


def test_is_secret():
    assert is_secret(".env") and is_secret("id_rsa") and is_secret("server.PEM")
    assert not is_secret(".env.example") and not is_secret("main.py")


# --- repo_map ----------------------------------------------------------------

def test_repo_map_skips_ignored_and_counts_languages(tools):
    out = call(tools["repo_map"])
    assert "node_modules" not in out
    assert "Python" in out and "Go" in out and "app.py" in out


def test_repo_map_truncates_big_trees(tmp_path):
    root = tmp_path / "big"
    root.mkdir()
    for i in range(500):
        (root / f"f{i}.py").write_text("x = 1\n")
    out = make_repo_tools(root, set(DEFAULT_IGNORES))["repo_map"].invoke({})
    assert "truncated" in out


def test_walk_files_never_enters_ignored_dirs(repo):
    seen = [p.relative_to(repo).as_posix() for p in walk_files(repo, set(DEFAULT_IGNORES))]
    assert seen and not any(s.startswith("node_modules") for s in seen)


# --- python_outline / code_outline -------------------------------------------

def test_python_outline_full_signature_and_decorators(tools):
    out = call(tools["python_outline"], path="pkg/core.py")
    assert "@decorator('x')" in out
    assert "async def run(a, *args, b=1, **kw) -> int" in out
    assert "class Engine(Base)" in out and "def go(self, speed: int=3) -> None" in out
    assert "MAX_ITEMS = 10" in out


def test_python_outline_finds_imports_inside_try(tools):
    assert "json" in call(tools["python_outline"], path="pkg/util.py")


def test_python_outline_syntax_error_is_tool_error(tools, repo):
    (repo / "bad.py").write_text("def (:\n")
    msg = call_as_agent(tools["python_outline"], path="bad.py")
    assert msg.status == "error" and "could not parse" in msg.content


def test_code_outline_js_and_go(tools):
    js = call(tools["code_outline"], path="server.js")
    assert "function start" in js and "class Server" in js
    go = call(tools["code_outline"], path="main.go")
    assert "func main" in go and "type Store struct" in go


def test_code_outline_unsupported_type(tools):
    assert call_as_agent(tools["code_outline"], path="README.md").status == "error"


# --- entry points ------------------------------------------------------------

def test_entry_points_real_hits_only(tools):
    out = call(tools["find_entry_points"])
    assert "app.py:4: python __main__ block" in out
    assert "argparse CLI" in out
    assert "express app" in out and "go main()" in out
    assert "README.md" not in out          # FastAPI( in a README is not an app
    assert "node_modules" not in out
    assert "tests/" not in out


# --- risk tools --------------------------------------------------------------

def test_find_todos_only_real_comments(tools):
    out = call(tools["find_todos"])
    assert "pkg/core.py:11: TODO handle errors" in out
    assert "README.md" not in out


def test_test_coverage_map(tools):
    out = call(tools["test_coverage_map"])
    assert "pkg/core.py" not in out        # test_core.py matches core.py
    assert "pkg/util.py" in out and "server.js" in out


def test_import_graph_resolves_relative_and_absolute(tools):
    out = call(tools["import_graph"])
    assert "app.py -> pkg/core.py" in out
    assert "pkg/core.py -> pkg/util.py" in out


def test_dependency_manifests(tools):
    assert "requests>=2" in call(tools["dependency_manifests"])


def test_git_hotspots_without_git_is_error(tools):
    assert call_as_agent(tools["git_hotspots"]).status == "error"


# --- module cache ------------------------------------------------------------

def test_module_fingerprint_fresh_then_stale(tools, repo, tmp_path):
    first = call(tools["module_fingerprint"], path="/repo/pkg")
    assert "missing or changed" in first
    note_path = [ln for ln in first.splitlines() if "/cache/modules/" in ln][0].split()[-1]
    note_file = tmp_path / "cache" / "modules" / os.path.basename(note_path)
    note_file.write_text("## pkg\nnote")

    assert "fresh" in call(tools["module_fingerprint"], path="/repo/pkg")

    (repo / "pkg" / "util.py").write_text("def helper():\n    return 2  # changed\n")
    again = call(tools["module_fingerprint"], path="/repo/pkg")
    assert "missing or changed" in again
    assert not note_file.exists()          # outdated note is dropped


# --- regression: patterns quoted in comments / strings ----------------------------

def test_quoted_patterns_are_not_hits(tools, repo):
    (repo / "notes.py").write_text(
        '# we should call FastAPI( somewhere, see Flask(__name__)\n'
        'MSG = "# TODO: not a real todo"\n'
        'DOC = "if __name__ == \'__main__\'"\n'
    )
    entry = call(tools["find_entry_points"])
    assert "notes.py" not in entry
    assert "notes.py" not in call(tools["find_todos"])
