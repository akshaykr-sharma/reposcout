"""The eval fixture repos must stay analysable by the tools (no model needed)."""
from pathlib import Path

import pytest

from reposcout.config import DEFAULT_IGNORES
from reposcout.tools import make_repo_tools

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"


def tools_for(name, tmp_path):
    return make_repo_tools(FIXTURES / name, set(DEFAULT_IGNORES), tmp_path)


def test_flask_fixture(tmp_path):
    t = tools_for("flask_todo", tmp_path)
    assert "Flask app" in t["find_entry_points"].invoke({})
    assert "TODO" in t["find_todos"].invoke({})
    assert "app.py -> models.py, storage.py" in t["import_graph"].invoke({})


def test_node_fixture(tmp_path):
    t = tools_for("node_cli", tmp_path)
    entry = t["find_entry_points"].invoke({})
    assert "npm script 'start'" in entry and "package bin 'wc-lite'" in entry
    assert "function countFile" in t["code_outline"].invoke({"path": "src/index.js"})
    assert "commander" in t["dependency_manifests"].invoke({})


def test_go_fixture(tmp_path):
    t = tools_for("go_service", tmp_path)
    assert "go main()" in t["find_entry_points"].invoke({})
    assert "type Cache struct" in t["code_outline"].invoke({"path": "cache.go"})


@pytest.mark.parametrize("name", ["flask_todo", "node_cli", "go_service"])
def test_every_fixture_has_questions(name):
    assert (FIXTURES / name / "questions.jsonl").read_text().strip()
