"""
Checks on the generated onboarding pack.

Takes:  the docs folder, the repo folder and the ignore set
Gives:  - `validate_docs`        a list of problems (empty list = all good)
        - `make_check_docs_tool` the same check as an agent tool, so the
                                 doc-writer can fix what it got wrong

What is checked (cheap, deterministic, no model needed):
  1. all four files exist and are not empty
  2. every file path written in `backticks` exists in the repo - the quickest
     way to catch an invented file or function location
  3. every ```mermaid block starts with a known diagram type and has balanced
     brackets / quotes (a light syntax check; it does not run mermaid itself)
  4. code fences are closed
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from .fsutil import walk_files
from .tools import TEXT_SUFFIXES

EXPECTED_FILES = ("OVERVIEW.md", "MODULES.md", "FLOW.md", "RISKS.md")

DIAGRAM_TYPES = {
    "graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram",
    "stateDiagram-v2", "erDiagram", "gantt", "pie", "journey", "mindmap", "timeline",
    "gitGraph", "C4Context", "quadrantChart", "requirementDiagram", "sankey-beta", "block-beta",
}
SEQUENCE_ARROW = re.compile(r"(-->>|->>|--x|-x|--\)|-\)|-->|->)")
BACKTICKED = re.compile(r"`([^`\n]{2,120})`")
PATH_LIKE = re.compile(r"^[\w./\\-]+\.[A-Za-z0-9]{1,10}$")
MERMAID_BLOCK = re.compile(r"```mermaid[^\n]*\n(.*?)```", re.S)
PAIRS = {")": "(", "]": "[", "}": "{"}


def _repo_index(repo_root: Path, ignores: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for f in walk_files(repo_root, ignores, skip_secrets=False):
        rel = f.relative_to(repo_root).as_posix()
        index[f.name].append(rel)
    return index


def _exists(path: str, index: dict[str, list[str]]) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(rel == path or rel.endswith("/" + path) for rel in index.get(name, ()))


def _candidate_paths(text: str) -> set[str]:
    found = set()
    for span in BACKTICKED.findall(text):
        token = span.strip().replace("\\", "/")
        token = re.sub(r":[\w.]+$", "", token)  # `agent.py:build_agent` or `agent.py:12`
        if token.startswith("/repo/"):
            token = token[len("/repo/"):]
        if token.startswith("./"):
            token = token[2:]
        if "://" in token or any(c in token for c in "*<>{}$ ") or token.startswith(("/", "~")):
            continue
        if not PATH_LIKE.match(token) or Path(token).suffix not in TEXT_SUFFIXES:
            continue
        found.add(token)
    return found


def _check_mermaid(block: str) -> list[str]:
    problems = []
    lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    if not lines:
        return ["empty mermaid block"]
    first = lines[0].strip().split()[0]
    if first not in DIAGRAM_TYPES:
        problems.append(f"mermaid block starts with {first!r}, not a known diagram type")
    stack: list[str] = []
    for number, line in enumerate(lines, 1):
        bare = re.sub(r'"[^"\n]*"', '""', line)
        if bare.count('"') % 2:
            problems.append(f"mermaid line {number}: unbalanced quote")
        for ch in bare:
            if ch in "([{":
                stack.append(ch)
            elif ch in PAIRS:
                if not stack or stack.pop() != PAIRS[ch]:
                    problems.append(f"mermaid line {number}: unbalanced '{ch}'")
                    stack = []
                    break
        if first == "sequenceDiagram" and SEQUENCE_ARROW.search(bare) and ":" not in bare:
            problems.append(f"mermaid line {number}: sequence message needs ': text'")
    if stack:
        problems.append("mermaid block has unclosed brackets")
    return problems


def validate_docs(docs_dir: Path, repo_root: Path, ignores: set[str], expected: bool = True) -> list[str]:
    """Return a list of human-readable problems with the docs in `docs_dir`."""
    issues: list[str] = []
    files = list(EXPECTED_FILES) if expected else sorted(p.name for p in docs_dir.glob("*.md"))
    index: dict[str, list[str]] | None = None

    for name in files:
        path = docs_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"{name}: missing or empty")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")

        if sum(1 for ln in text.splitlines() if ln.strip().startswith("```")) % 2:
            issues.append(f"{name}: a code fence is never closed")

        candidates = _candidate_paths(text)
        if candidates:
            index = index if index is not None else _repo_index(repo_root, ignores)
            for missing in sorted(c for c in candidates if not _exists(c, index)):
                issues.append(f"{name}: `{missing}` does not exist in the repo")

        for block in MERMAID_BLOCK.findall(text):
            issues += [f"{name}: {p}" for p in _check_mermaid(block)]
    return issues


def format_issues(issues: list[str]) -> str:
    return "\n".join(f"- {i}" for i in issues)


def make_check_docs_tool(docs_dir: Path, repo_root: Path, ignores: set[str]) -> BaseTool:
    @tool
    def check_docs() -> str:
        """Check the onboarding files already written to /docs/.

        Reports missing files, file paths that do not exist in the repo,
        and broken mermaid diagrams. Call it after all four files are
        written (and approved); fix every problem it lists, then call it again.
        """
        issues = validate_docs(docs_dir, repo_root, ignores)
        if not issues:
            return "OK: all four files exist, every mentioned path exists, mermaid blocks look valid."
        return "Problems found:\n" + format_issues(issues)

    check_docs.handle_tool_error = True
    return check_docs


__all__ = ["validate_docs", "format_issues", "make_check_docs_tool", "EXPECTED_FILES"]
