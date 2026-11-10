from reposcout.config import DEFAULT_IGNORES
from reposcout.validate import validate_docs

GOOD_MERMAID = "```mermaid\nsequenceDiagram\n  A->>B: hello\n```\n"


def make_docs(tmp_path, **overrides):
    docs = tmp_path / "docs"
    docs.mkdir()
    files = {
        n: f"# {n}\nSee `app.py` and `pkg/core.py:run`.\n\n{GOOD_MERMAID}"
        for n in ("OVERVIEW.md", "MODULES.md", "FLOW.md", "RISKS.md")
    }
    files.update(overrides)
    for name, text in files.items():
        if text is not None:
            (docs / name).write_text(text)
    return docs


def run(docs, repo):
    return validate_docs(docs, repo, set(DEFAULT_IGNORES))


def test_clean_pack_has_no_issues(tmp_path, repo):
    assert run(make_docs(tmp_path), repo) == []


def test_missing_file(tmp_path, repo):
    assert run(make_docs(tmp_path, **{"RISKS.md": None}), repo) == ["RISKS.md: missing or empty"]


def test_invented_path_is_flagged(tmp_path, repo):
    docs = make_docs(tmp_path, **{"MODULES.md": "Look at `pkg/ghost.py` and `/repo/app.py`.\n"})
    issues = run(docs, repo)
    assert any("pkg/ghost.py" in i for i in issues)
    assert len(issues) == 1


def test_non_path_backticks_ignored(tmp_path, repo):
    docs = make_docs(tmp_path, **{"MODULES.md": "Run `pip install -e .` then `e.g.` or `http://x.io/a.py`\n"})
    assert run(docs, repo) == []


def test_bad_mermaid(tmp_path, repo):
    bad = "```mermaid\nsequenceDiagram\n  A->>B hello\n  C[unclosed\n```\n"
    issues = run(make_docs(tmp_path, **{"FLOW.md": bad}), repo)
    assert any("needs ': text'" in i for i in issues)
    assert any("unclosed" in i or "unbalanced" in i for i in issues)


def test_unknown_diagram_type_and_open_fence(tmp_path, repo):
    text = "```mermaid\nfoo\n  A-->B\n```\n```python\nx = 1\n"
    issues = run(make_docs(tmp_path, **{"OVERVIEW.md": text}), repo)
    assert any("not a known diagram type" in i for i in issues)
    assert any("never closed" in i for i in issues)
