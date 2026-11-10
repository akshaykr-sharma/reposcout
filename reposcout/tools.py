"""
Repo-aware tools for RepoScout.

The deep agent already gets generic file tools from deepagents (ls, read_file,
grep, glob, write_file, edit_file). Those are fine but "low level" - a 7B
model wastes a lot of turns doing `ls` folder by folder. The tools here give
it a few higher level views of the code so it can plan faster.

Takes:  the repo root on disk (passed once to `make_repo_tools`)
Gives:  a list of LangChain tools:
          - repo_map            folder tree with line counts, skipping junk dirs
          - python_outline      classes / functions / imports of one .py file
          - find_entry_points   places where the program starts (main blocks,
                                web apps, CLIs, package scripts, Dockerfiles)

Paths: the agent sees the repo mounted at `/repo/`, so every tool accepts
either `/repo/src/app.py` or plain `src/app.py`. Anything that resolves
outside the repo root is refused.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool

REPO_MOUNT = "/repo/"

TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".scala", ".sql",
    ".sh", ".ps1", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".md",
    ".rst", ".txt", ".html", ".css", ".scss", ".vue", ".dockerfile",
}

LANG_BY_SUFFIX = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++", ".c": "C", ".swift": "Swift",
    ".scala": "Scala", ".sql": "SQL", ".sh": "Shell", ".vue": "Vue",
}

# Cheap regex heuristics - good enough to point the agent in the right direction
ENTRY_PATTERNS = [
    (re.compile(r"""if\s+__name__\s*==\s*['"]__main__['"]"""), "python __main__ block"),
    (re.compile(r"\bFastAPI\s*\("), "FastAPI app"),
    (re.compile(r"\bFlask\s*\(\s*__name__"), "Flask app"),
    (re.compile(r"\bargparse\.ArgumentParser\s*\("), "argparse CLI"),
    (re.compile(r"@click\.(command|group)"), "click CLI"),
    (re.compile(r"\btyper\.Typer\s*\("), "typer CLI"),
    (re.compile(r"^\s*(import|from)\s+streamlit\b", re.M), "streamlit app"),
    (re.compile(r"\bexpress\s*\(\s*\)"), "express app"),
    (re.compile(r"\bfunc\s+main\s*\(\s*\)"), "go main()"),
    (re.compile(r"public\s+static\s+void\s+main\s*\("), "java main()"),
]

MAX_FILE_BYTES = 400_000


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def make_repo_tools(repo_root: Path, ignores: set[str]) -> list:
    """Build the tools bound to one repo folder."""
    root = repo_root.resolve()

    def resolve(path: str) -> Path:
        rel = path.strip()
        if rel.startswith(REPO_MOUNT):
            rel = rel[len(REPO_MOUNT):]
        elif rel.rstrip("/") == REPO_MOUNT.rstrip("/"):
            rel = ""
        rel = rel.lstrip("/\\")
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"{path} is outside the repo")
        return target

    def to_virtual(path: Path) -> str:
        rel = path.relative_to(root).as_posix()
        return REPO_MOUNT if rel == "." else REPO_MOUNT + rel

    def is_ignored(path: Path) -> bool:
        return any(part in ignores for part in path.relative_to(root).parts)

    @tool
    def repo_map(subpath: str = "/repo/", max_depth: int = 3) -> str:
        """Show the folder tree of the repo with line counts per file.

        Use this first to get the lay of the land instead of calling `ls`
        many times. Junk folders (.git, node_modules, venvs, build output)
        are skipped.

        Args:
            subpath: folder to map, e.g. "/repo/" or "/repo/src".
            max_depth: how many folder levels to expand (1-6).
        """
        try:
            start = resolve(subpath)
        except ValueError as exc:
            return f"Error: {exc}"
        if not start.is_dir():
            return f"Error: {subpath} is not a folder"

        max_depth = max(1, min(int(max_depth), 6))
        lines: list[str] = []
        langs: Counter[str] = Counter()
        total_files = 0

        def walk(folder: Path, depth: int) -> None:
            nonlocal total_files
            try:
                entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return
            for entry in entries:
                if is_ignored(entry) or entry.name.startswith(".") and entry.is_dir():
                    continue
                indent = "  " * depth
                if entry.is_dir():
                    lines.append(f"{indent}{entry.name}/")
                    if depth + 1 < max_depth:
                        walk(entry, depth + 1)
                    else:
                        count = sum(1 for p in entry.rglob("*") if p.is_file() and not is_ignored(p))
                        lines.append(f"{indent}  ... {count} files")
                else:
                    total_files += 1
                    if entry.suffix in LANG_BY_SUFFIX:
                        langs[LANG_BY_SUFFIX[entry.suffix]] += 1
                    loc = _count_lines(entry) if entry.suffix in TEXT_SUFFIXES else 0
                    suffix = f"  ({loc} lines)" if loc else ""
                    lines.append(f"{indent}{entry.name}{suffix}")

        walk(start, 0)

        if len(lines) > 400:
            lines = lines[:400] + ["... truncated, call again with a smaller subpath"]

        header = f"{to_virtual(start)} - {total_files} files shown"
        if langs:
            header += " | languages: " + ", ".join(f"{k} ({v})" for k, v in langs.most_common())
        return header + "\n" + "\n".join(lines)

    @tool
    def python_outline(path: str) -> str:
        """Summarise one Python file without reading it all.

        Returns the module docstring, imports, classes (with their methods)
        and top-level functions with signatures. Much cheaper than read_file
        for big files. Only works on .py files.

        Args:
            path: file path like "/repo/app/main.py".
        """
        try:
            target = resolve(path)
        except ValueError as exc:
            return f"Error: {exc}"
        if target.suffix != ".py" or not target.is_file():
            return "Error: python_outline only works on existing .py files. Use read_file for others."
        if target.stat().st_size > MAX_FILE_BYTES:
            return "Error: file too large to outline, read it in chunks with read_file."

        source = target.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return f"Error: could not parse {path}: {exc}"

        out = [f"# {to_virtual(target)} ({len(source.splitlines())} lines)"]
        doc = ast.get_docstring(tree)
        if doc:
            out.append("docstring: " + doc.strip().splitlines()[0])

        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports.append(("." * node.level) + (node.module or ""))
        if imports:
            out.append("imports: " + ", ".join(sorted(set(imports))))

        def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            args = [a.arg for a in fn.args.args]
            prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
            first_doc = (ast.get_docstring(fn) or "").strip().splitlines()
            hint = f"  # {first_doc[0]}" if first_doc else ""
            return f"{prefix} {fn.name}({', '.join(args)}) [line {fn.lineno}]{hint}"

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                out.append(f"class {node.name}({bases}) [line {node.lineno}]")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append("    " + signature(item))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(signature(node))

        return "\n".join(out)

    @tool
    def find_entry_points() -> str:
        """Find where the program starts.

        Looks for __main__ blocks, web app objects (FastAPI, Flask, Express),
        CLIs (argparse, click, typer), scripts in pyproject.toml /
        package.json, Dockerfile CMD/ENTRYPOINT and manage.py. Use this
        before tracing a request flow.
        """
        hits: list[str] = []

        for file in root.rglob("*"):
            if not file.is_file() or is_ignored(file):
                continue
            name = file.name

            if name == "Dockerfile" or name.endswith(".dockerfile"):
                for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip().upper().startswith(("CMD", "ENTRYPOINT")):
                        hits.append(f"{to_virtual(file)}: {line.strip()}")
                continue

            if name == "package.json":
                try:
                    scripts = json.loads(file.read_text(encoding="utf-8")).get("scripts", {})
                    for key, cmd in scripts.items():
                        hits.append(f"{to_virtual(file)}: npm script '{key}' -> {cmd}")
                except (json.JSONDecodeError, OSError):
                    pass
                continue

            if name == "pyproject.toml":
                text = file.read_text(encoding="utf-8", errors="replace")
                block = re.search(r"\[project\.scripts\](.*?)(\n\[|\Z)", text, re.S)
                if block:
                    for line in block.group(1).strip().splitlines():
                        if "=" in line:
                            hits.append(f"{to_virtual(file)}: console script {line.strip()}")
                continue

            if name == "manage.py":
                hits.append(f"{to_virtual(file)}: Django manage.py")
                continue

            if file.suffix not in TEXT_SUFFIXES or file.stat().st_size > MAX_FILE_BYTES:
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            for pattern, label in ENTRY_PATTERNS:
                match = pattern.search(text)
                if match:
                    line_no = text[: match.start()].count("\n") + 1
                    hits.append(f"{to_virtual(file)}:{line_no}: {label}")

        if not hits:
            return "No obvious entry points found. Try grep for 'main' or look at the README."
        if len(hits) > 80:
            hits = hits[:80] + ["... more hits truncated"]
        return "\n".join(hits)

    return [repo_map, python_outline, find_entry_points]
