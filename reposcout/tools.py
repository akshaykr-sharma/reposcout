"""
Repo-aware tools for RepoScout.

The deep agent already gets generic file tools from deepagents (ls, read_file,
grep, glob, write_file, edit_file). Those are fine but "low level" - a 7B
model wastes a lot of turns doing `ls` folder by folder. The tools here give
it higher level views of the code so it can plan faster.

Takes:  the repo root on disk, the folder names to ignore, and (optionally)
        the cache folder for module notes - passed once to `make_repo_tools`
Gives:  a dict name -> LangChain tool:
          repo_map              folder tree with line counts, skipping junk dirs
          python_outline        classes / functions / decorators / imports of one .py file
          code_outline          the same idea for JS/TS/Go/Java/Rust/Ruby/PHP/C# files
          find_entry_points     places where the program starts
          import_graph          which Python files import which (internal imports only)
          dependency_manifests  requirements.txt, package.json, go.mod, ... summarised
          find_todos            TODO / FIXME / HACK / XXX comments
          test_coverage_map     source files that have no matching test file
          git_hotspots          files that change most often in git history
          module_fingerprint    change detector so unchanged modules are not re-analysed

Paths: the agent sees the repo mounted at `/repo/`, so every tool accepts
either `/repo/src/app.py` or plain `src/app.py`. Anything that resolves
outside the repo root is refused, and so are files that look like secrets.

Errors: tools raise `ToolException`; the resulting ToolMessage carries
`status="error"`, which the terminal UI uses to colour failures.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.tools import BaseTool, ToolException, tool

from .fsutil import NON_ENTRY_DIRS, count_files, is_secret, walk_files

REPO_MOUNT = "/repo/"
CACHE_MOUNT = "/cache/"
MODULE_NOTES_DIR = "modules"

TEXT_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs",
    ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".scala", ".sql",
    ".sh", ".ps1", ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".md",
    ".rst", ".txt", ".html", ".css", ".scss", ".vue", ".dockerfile",
}

LANG_BY_SUFFIX = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".kt": "Kotlin", ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".cs": "C#", ".cpp": "C++", ".c": "C", ".swift": "Swift", ".scala": "Scala",
    ".sql": "SQL", ".sh": "Shell", ".vue": "Vue",
}

CODE_SUFFIXES = {s for s in LANG_BY_SUFFIX if s not in {".sql", ".sh", ".vue"}}

PY = {".py"}
JS = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}

# Cheap regex heuristics - good enough to point the agent in the right direction.
# Each pattern only runs on the file types it makes sense for (no more matching
# "FastAPI(" inside a README).
ENTRY_PATTERNS = [
    (re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]""", re.M), "python __main__ block", PY),
    (re.compile(r"\bFastAPI\s*\("), "FastAPI app", PY),
    (re.compile(r"\bFlask\s*\(\s*__name__"), "Flask app", PY),
    (re.compile(r"^\s*[\w.]*\bargparse\.ArgumentParser\s*\(", re.M), "argparse CLI", PY),
    (re.compile(r"^\s*@click\.(command|group)", re.M), "click CLI", PY),
    (re.compile(r"\btyper\.Typer\s*\("), "typer CLI", PY),
    (re.compile(r"^\s*(import|from)\s+streamlit\b", re.M), "streamlit app", PY),
    (re.compile(r"\bexpress\s*\(\s*\)"), "express app", JS),
    (re.compile(r"^func\s+main\s*\(\s*\)", re.M), "go main()", {".go"}),
    (re.compile(r"public\s+static\s+void\s+main\s*\("), "java main()", {".java"}),
]

OUTLINE_PATTERNS: dict[str, list[re.Pattern]] = {}
_JS_DEFS = [
    r"^\s*(export\s+)?(default\s+)?(async\s+)?function\*?\s+\w+",
    r"^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+\w+",
    r"^\s*(export\s+)?(const|let|var)\s+\w+\s*(:[^=]+)?=\s*(async\s*)?(\([^)]*\)|\w+)\s*(:[^=]+)?=>",
    r"^\s*(export\s+)?(interface|type|enum)\s+\w+",
]
_GO_DEFS = [r"^func\s+(\([^)]*\)\s*)?\w+", r"^type\s+\w+\s+(struct|interface)"]
_JVM_DEFS = [
    r"^\s*((public|private|protected|internal|static|abstract|final|open|sealed|data)\s+)*"
    r"(class|interface|enum|record|object)\s+\w+",
    r"^\s*((public|private|protected|internal|static|abstract|final|override|suspend)\s+)+"
    r"[\w<>\[\],.? ]+\s+\w+\s*\([^;]*\)\s*(\{|throws|:)",
    r"^\s*(override\s+)?fun\s+\w+",
]
_RUST_DEFS = [r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(fn|struct|enum|trait|impl|mod)\b\s*[\w<]*"]
_RUBY_DEFS = [r"^\s*(def|class|module)\s+[\w.:]+"]
_PHP_DEFS = [r"^\s*((public|private|protected|static|abstract|final)\s+)*(function|class|interface|trait)\s+\w+"]
for _suffixes, _pats in (
    (JS, _JS_DEFS), ({".go"}, _GO_DEFS), ({".java", ".kt", ".cs", ".scala", ".swift"}, _JVM_DEFS),
    ({".rs"}, _RUST_DEFS), ({".rb"}, _RUBY_DEFS), ({".php"}, _PHP_DEFS),
):
    for _s in _suffixes:
        OUTLINE_PATTERNS[_s] = [re.compile(p) for p in _pats]

IMPORT_LINE = re.compile(r"""^\s*(import\s|from\s+\S+\s+import|using\s|use\s|require\b|const\s+.*=\s*require\()""")
TODO_RE = re.compile(r"(#|//|/\*|\*|--|<!--)\s*(TODO|FIXME|HACK|XXX)\b[:\s(]*(.*)")
TEST_DIRS = {"tests", "test", "__tests__", "spec", "specs"}

MANIFEST_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "package.json", "go.mod",
    "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "composer.json",
}

MAX_FILE_BYTES = 400_000
MAX_MAP_LINES = 400


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _in_comment_or_string(text: str, pos: int, suffix: str) -> bool:
    """True if `pos` sits inside a quoted string or after a line comment marker.

    Single-line heuristic (multi-line docstrings are not detected), but it
    removes the common false hit of a pattern quoted in a comment or a string.
    """
    line_start = text.rfind("\n", 0, pos) + 1
    prefix = text[line_start:pos]
    if prefix.count('"') % 2 or prefix.count("'") % 2 or prefix.count("`") % 2:
        return True
    marker = "#" if suffix in (".py", ".rb", ".sh") else "//"
    return marker in prefix


def _safe_tool(fn) -> BaseTool:
    """`@tool` plus: raised ToolException becomes a ToolMessage with status='error'."""
    built = tool(fn)
    built.handle_tool_error = True
    return built


def _is_test_file(rel: Path) -> bool:
    name = rel.name.lower()
    if any(part.lower() in TEST_DIRS for part in rel.parts[:-1]):
        return True
    return (
        name.startswith("test_") or name.endswith("_test.py") or name.endswith("_test.go")
        or ".test." in name or ".spec." in name or name.endswith("test.java")
        or name.endswith("tests.py") or name.endswith("_spec.rb")
    )


def _test_stem(name: str) -> str:
    stem = name.lower().split(".")[0]
    for pre in ("test_",):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    for suf in ("_test", "_spec", "tests", "test"):
        if stem.endswith(suf) and len(stem) > len(suf):
            stem = stem[: -len(suf)].rstrip("_")
    return stem


def make_repo_tools(repo_root: Path, ignores: set[str], cache_dir: Path | None = None) -> dict[str, BaseTool]:
    """Build the tools bound to one repo folder."""
    root = repo_root.resolve()

    def resolve(path: str) -> Path:
        rel = path.strip()
        if rel.startswith(REPO_MOUNT):
            rel = rel[len(REPO_MOUNT):]
        elif rel.rstrip("/") == REPO_MOUNT.rstrip("/"):
            rel = ""
        rel = rel.lstrip("/\\")
        try:
            target = (root / rel).resolve()
        except (OSError, ValueError) as exc:
            raise ToolException(f"invalid path {path!r}: {exc}") from None
        if target != root and root not in target.parents:
            raise ToolException(f"{path} is outside the repo")
        if target.is_file() and is_secret(target.name):
            raise ToolException(f"{path} looks like a credentials file and is blocked")
        return target

    def to_virtual(path: Path) -> str:
        rel = path.relative_to(root).as_posix()
        return REPO_MOUNT if rel == "." else REPO_MOUNT + rel

    def rel_of(path: Path) -> Path:
        return path.relative_to(root)

    def source_files(*, skip_dirs: set[str] | None = None):
        return walk_files(root, ignores, skip_dirs=skip_dirs)

    # ------------------------------------------------------------------ repo_map
    @_safe_tool
    def repo_map(subpath: str = "/repo/", max_depth: int = 3) -> str:
        """Show the folder tree of the repo with line counts per file.

        Use this first to get the lay of the land instead of calling `ls`
        many times. Junk folders (.git, node_modules, venvs, build output)
        and credentials files are skipped.

        Args:
            subpath: folder to map, e.g. "/repo/" or "/repo/src".
            max_depth: how many folder levels to expand (1-6).
        """
        start = resolve(subpath)
        if not start.is_dir():
            raise ToolException(f"{subpath} is not a folder")

        max_depth = max(1, min(int(max_depth), 6))
        lines: list[str] = []
        langs: Counter[str] = Counter()
        total_files = 0
        truncated = False

        def walk(folder: Path, depth: int) -> None:
            nonlocal total_files, truncated
            try:
                entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return
            for entry in entries:
                if truncated:
                    return
                if len(lines) >= MAX_MAP_LINES:
                    truncated = True
                    return
                if entry.name in ignores:
                    continue
                indent = "  " * depth
                if entry.is_dir():
                    if entry.name.startswith("."):
                        continue
                    lines.append(f"{indent}{entry.name}/")
                    if depth + 1 < max_depth:
                        walk(entry, depth + 1)
                    else:
                        count = count_files(entry, ignores)
                        lines.append(f"{indent}  ... {count}{'+' if count >= 5000 else ''} files")
                else:
                    if is_secret(entry.name):
                        continue
                    total_files += 1
                    if entry.suffix in LANG_BY_SUFFIX:
                        langs[LANG_BY_SUFFIX[entry.suffix]] += 1
                    loc = _count_lines(entry) if entry.suffix in TEXT_SUFFIXES else 0
                    suffix = f"  ({loc} lines)" if loc else ""
                    lines.append(f"{indent}{entry.name}{suffix}")

        walk(start, 0)

        if truncated:
            lines.append("... truncated, call again with a smaller subpath")

        header = f"{to_virtual(start)} - {total_files} files shown"
        if langs:
            header += " | languages: " + ", ".join(f"{k} ({v})" for k, v in langs.most_common())
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------ python_outline
    def _py_outline(target: Path) -> str:
        source = _read(target)
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            raise ToolException(f"could not parse {to_virtual(target)}: {exc}") from None

        out = [f"# {to_virtual(target)} ({len(source.splitlines())} lines)"]
        doc = ast.get_docstring(tree)
        if doc:
            out.append("docstring: " + doc.strip().splitlines()[0])

        # ast.walk also finds imports inside try/if blocks and lazy imports in functions
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(("." * node.level) + (node.module or ""))
        if imports:
            out.append("imports: " + ", ".join(sorted(imports)))

        def short(node: ast.AST, limit: int = 70) -> str:
            text = " ".join(ast.unparse(node).split())
            return text if len(text) <= limit else text[: limit - 3] + "..."

        def decorators(node) -> str:
            return "".join(f"@{short(d, 60)} " for d in node.decorator_list)

        def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
            ret = f" -> {short(fn.returns, 40)}" if fn.returns else ""
            first_doc = (ast.get_docstring(fn) or "").strip().splitlines()
            hint = f"  # {first_doc[0]}" if first_doc else ""
            return f"{decorators(fn)}{prefix} {fn.name}({short(fn.args, 110)}){ret} [line {fn.lineno}]{hint}"

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                out.append(f"{decorators(node)}class {node.name}({bases}) [line {node.lineno}]")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append("    " + signature(item))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(signature(node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and node.value is not None:
                        out.append(f"{t.id} = {short(node.value, 50)} [line {node.lineno}]")
        return "\n".join(out)

    @_safe_tool
    def python_outline(path: str) -> str:
        """Summarise one Python file without reading it all.

        Returns the module docstring, imports (including ones inside try/if
        blocks), constants, classes (with their methods) and top-level
        functions with full signatures, decorators and return types. Much
        cheaper than read_file for big files. Only works on .py files; use
        code_outline for other languages.

        Args:
            path: file path like "/repo/app/main.py".
        """
        target = resolve(path)
        if target.suffix != ".py" or not target.is_file():
            raise ToolException("python_outline only works on existing .py files. Use code_outline or read_file.")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ToolException("file too large to outline, read it in chunks with read_file.")
        return _py_outline(target)

    # -------------------------------------------------------------- code_outline
    @_safe_tool
    def code_outline(path: str) -> str:
        """Outline any source file: imports plus class/function/type definitions with line numbers.

        Works for Python (full outline) and, using pattern matching, for
        JavaScript/TypeScript, Go, Java, Kotlin, C#, Scala, Swift, Rust,
        Ruby and PHP. Much cheaper than read_file on a big file.

        Args:
            path: file path like "/repo/src/server.ts".
        """
        target = resolve(path)
        if not target.is_file():
            raise ToolException(f"{path} is not a file")
        patterns = OUTLINE_PATTERNS.get(target.suffix)
        if target.suffix != ".py" and not patterns:
            raise ToolException(f"no outline support for {target.suffix or 'this file type'}; use read_file")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ToolException("file too large to outline, read it in chunks with read_file.")
        if target.suffix == ".py":
            return _py_outline(target)

        text_lines = _read(target).splitlines()
        imports: list[str] = []
        defs: list[str] = []
        for number, line in enumerate(text_lines, 1):
            if IMPORT_LINE.match(line):
                imports.append(line.strip()[:100])
            elif any(p.match(line) for p in patterns):
                defs.append(f"L{number}: {line.strip()[:110]}")
        out = [f"# {to_virtual(target)} ({len(text_lines)} lines)"]
        if imports:
            out.append("imports:")
            out += [f"  {i}" for i in imports[:25]]
            if len(imports) > 25:
                out.append(f"  ... {len(imports) - 25} more")
        out.append("definitions:" if defs else "no definitions matched (use read_file)")
        out += [f"  {d}" for d in defs[:150]]
        if len(defs) > 150:
            out.append(f"  ... {len(defs) - 150} more")
        return "\n".join(out)

    # ----------------------------------------------------------- find_entry_points
    @_safe_tool
    def find_entry_points() -> str:
        """Find where the program starts.

        Looks for __main__ blocks, web app objects (FastAPI, Flask, Express),
        CLIs (argparse, click, typer), scripts in pyproject.toml /
        package.json, Dockerfile CMD/ENTRYPOINT, Go/Java main and manage.py.
        Test, docs and example folders are skipped. Use this before tracing
        a request flow.
        """
        hits: list[str] = []

        for file in source_files(skip_dirs=NON_ENTRY_DIRS):
            name = file.name
            vpath = to_virtual(file)

            if name == "Dockerfile" or name.endswith(".dockerfile"):
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue
                for line in _read(file).splitlines():
                    if line.strip().upper().startswith(("CMD", "ENTRYPOINT")):
                        hits.append(f"{vpath}: {line.strip()}")
                continue

            if name == "package.json":
                try:
                    data = json.loads(_read(file))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, dict):
                    continue
                for key, cmd in (data.get("scripts") or {}).items():
                    hits.append(f"{vpath}: npm script '{key}' -> {cmd}")
                if data.get("main"):
                    hits.append(f"{vpath}: package main -> {data['main']}")
                bins = data.get("bin")
                if isinstance(bins, str):
                    hits.append(f"{vpath}: package bin -> {bins}")
                elif isinstance(bins, dict):
                    hits += [f"{vpath}: package bin '{k}' -> {v}" for k, v in bins.items()]
                continue

            if name == "pyproject.toml":
                block = re.search(r"\[project\.scripts\](.*?)(\n\[|\Z)", _read(file), re.S)
                if block:
                    for line in block.group(1).strip().splitlines():
                        if "=" in line:
                            hits.append(f"{vpath}: console script {line.strip()}")
                continue

            if name == "manage.py":
                hits.append(f"{vpath}: Django manage.py")
                continue

            if file.suffix not in TEXT_SUFFIXES or file.stat().st_size > MAX_FILE_BYTES:
                continue
            text = None
            for pattern, label, suffixes in ENTRY_PATTERNS:
                if file.suffix not in suffixes:
                    continue
                text = text if text is not None else _read(file)
                match = next(
                    (m for m in pattern.finditer(text)
                     if not _in_comment_or_string(text, m.start(), file.suffix)),
                    None,
                )
                if match:
                    line_no = text[: match.start()].count("\n") + 1
                    hits.append(f"{vpath}:{line_no}: {label}")

        if not hits:
            return "No obvious entry points found. Try grep for 'main' or look at the README."
        if len(hits) > 80:
            hits = hits[:80] + ["... more hits truncated"]
        return "\n".join(hits)

    # --------------------------------------------------------------- import_graph
    @_safe_tool
    def import_graph(subpath: str = "/repo/") -> str:
        """Show which Python files import which other files of the same repo.

        Only internal imports are listed (third-party and stdlib are left
        out). Use it to see which modules are central (many importers) and
        which depend on which. Python only.

        Args:
            subpath: restrict the "importing" side to this folder, e.g. "/repo/src".
        """
        scope = resolve(subpath)
        py_files = [f for f in source_files() if f.suffix == ".py" and f.stat().st_size <= MAX_FILE_BYTES]
        if not py_files:
            return "No Python files found."

        def dotted(path: Path) -> str:
            parts = list(rel_of(path).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            return ".".join(parts)

        by_module: dict[str, Path] = {}
        for f in py_files:
            name = dotted(f)
            by_module[name] = f
            if name.startswith("src."):  # src-layout: imports omit the "src" part
                by_module.setdefault(name[4:], f)

        def lookup(name: str) -> Path | None:
            # `from pkg import mod` may name a module or just a symbol - try the longest match
            while name:
                if name in by_module:
                    return by_module[name]
                name = name.rpartition(".")[0]
            return None

        edges: dict[Path, set[Path]] = defaultdict(set)
        for f in py_files:
            if scope != root and scope not in f.parents:
                continue
            try:
                tree = ast.parse(_read(f))
            except (SyntaxError, ValueError):
                continue
            is_pkg = f.name == "__init__.py"
            pkg_parts = rel_of(f).with_suffix("").parts[:-1] if not is_pkg else rel_of(f).parts[:-1]
            for node in ast.walk(tree):
                dests: list[Path | None] = []
                if isinstance(node, ast.Import):
                    dests = [lookup(a.name) for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = list(pkg_parts[: len(pkg_parts) - (node.level - 1)]) if node.level > 1 else list(pkg_parts)
                        base_name = ".".join(base + ([node.module] if node.module else []))
                    else:
                        base_name = node.module or ""
                    # `from pkg import mod` imports the submodule if there is one,
                    # otherwise a symbol that lives in pkg itself
                    submodules = [by_module.get(f"{base_name}.{a.name}".strip(".")) for a in node.names]
                    dests = [d for d in submodules if d] or [lookup(base_name)]
                for dest in dests:
                    if dest and dest != f:
                        edges[f].add(dest)

        if not edges:
            return "No internal imports found between Python files."
        incoming = Counter(d for dests in edges.values() for d in dests)
        lines = ["internal imports (file -> files it imports):"]
        for f in sorted(edges, key=lambda p: to_virtual(p))[:80]:
            lines.append(f"{to_virtual(f)} -> " + ", ".join(sorted(rel_of(d).as_posix() for d in edges[f])))
        lines.append("most imported: " + ", ".join(f"{rel_of(p).as_posix()} ({n})" for p, n in incoming.most_common(8)))
        return "\n".join(lines)

    # ------------------------------------------------------- dependency_manifests
    @_safe_tool
    def dependency_manifests() -> str:
        """Summarise dependency and build files (requirements*.txt, pyproject.toml,
        package.json, go.mod, Cargo.toml, pom.xml, Gemfile, ...).

        Use it for the "tech stack" and "how to run it" parts of the docs.
        """
        found: list[Path] = []
        for f in source_files():
            if f.name in MANIFEST_NAMES or (f.name.startswith("requirements") and f.suffix == ".txt"):
                found.append(f)
        if not found:
            return "No dependency manifests found."
        out: list[str] = []
        for f in sorted(found, key=lambda p: (len(rel_of(p).parts), str(p)))[:8]:
            out.append(f"## {to_virtual(f)}")
            text = _read(f) if f.stat().st_size <= MAX_FILE_BYTES else ""
            if f.name == "package.json":
                try:
                    data = json.loads(text)
                    for key in ("name", "version", "main"):
                        if data.get(key):
                            out.append(f"{key}: {data[key]}")
                    for key in ("dependencies", "devDependencies"):
                        if data.get(key):
                            out.append(f"{key}: " + ", ".join(f"{k}@{v}" for k, v in data[key].items()))
                    if data.get("scripts"):
                        out.append("scripts: " + ", ".join(data["scripts"]))
                except (json.JSONDecodeError, AttributeError):
                    out.append("(could not parse JSON)")
            else:
                shown = [ln for ln in text.splitlines() if ln.strip()][:60]
                out += shown
        return "\n".join(out)

    # ----------------------------------------------------------------- find_todos
    @_safe_tool
    def find_todos() -> str:
        """List TODO / FIXME / HACK / XXX comments with file and line number.

        Only real comments are matched. Use it for the "unclear or fragile
        code" part of RISKS.md.
        """
        hits: list[str] = []
        total = 0
        for f in source_files(skip_dirs={"fixtures"}):
            if f.suffix not in TEXT_SUFFIXES or f.suffix in {".md", ".rst", ".txt", ".json"}:
                continue
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            for number, line in enumerate(_read(f).splitlines(), 1):
                m = TODO_RE.search(line)
                if m:
                    before = line[: m.start()]
                    if before.count('"') % 2 or before.count("'") % 2:
                        continue  # the marker is inside a string literal, not a comment
                    total += 1
                    if len(hits) < 100:
                        hits.append(f"{to_virtual(f)}:{number}: {m.group(2)} {m.group(3).strip()[:100]}")
        if not hits:
            return "No TODO/FIXME/HACK comments found."
        if total > len(hits):
            hits.append(f"... {total - len(hits)} more")
        return f"{total} marker(s):\n" + "\n".join(hits)

    # ----------------------------------------------------------- test_coverage_map
    @_safe_tool
    def test_coverage_map() -> str:
        """Find source files that have no matching test file.

        Matching is by file name (`parser.py` <-> `test_parser.py`,
        `parser.test.ts`, `parser_test.go`), so it is a rough guide, not
        real coverage. Use it for the "missing tests" part of RISKS.md.
        """
        sources: list[tuple[int, Path]] = []
        test_stems: set[str] = set()
        test_count = 0
        for f in source_files(skip_dirs={"fixtures"}):
            if f.suffix not in CODE_SUFFIXES:
                continue
            rel = rel_of(f)
            if _is_test_file(rel):
                test_count += 1
                test_stems.add(_test_stem(f.name))
            elif f.name not in {"__init__.py", "conftest.py", "setup.py"} and not f.name.endswith(".d.ts"):
                sources.append((_count_lines(f), f))
        if not sources:
            return "No source files found."
        untested = [(n, f) for n, f in sources if f.name.split(".")[0].lower() not in test_stems]
        untested.sort(key=lambda item: -item[0])
        lines = [
            f"{len(sources)} source files, {test_count} test files, "
            f"{len(untested)} source files without a name-matching test"
        ]
        if test_count == 0:
            lines.append("WARNING: no tests found at all")
        lines += [f"  {to_virtual(f)} ({n} lines)" for n, f in untested[:40]]
        if len(untested) > 40:
            lines.append(f"  ... {len(untested) - 40} more")
        return "\n".join(lines)

    # --------------------------------------------------------------- git_hotspots
    @_safe_tool
    def git_hotspots() -> str:
        """Show the files changed most often in the last 500 commits.

        Frequently changed files are where bugs and merge conflicts
        cluster. Needs the repo to be a git checkout.
        """
        try:
            proc = subprocess.run(
                ["git", "--no-pager", "-c", "core.quotepath=off", "-C", str(root),
                 "log", "--name-only", "--pretty=format:", "-n", "500"],
                capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            raise ToolException("git is not installed") from None
        except subprocess.TimeoutExpired:
            raise ToolException("git log timed out") from None
        if proc.returncode != 0:
            raise ToolException("not a git repository (or no commits yet)")
        counts: Counter[str] = Counter()
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = Path(line).parts
            if any(p in ignores for p in parts) or is_secret(parts[-1]):
                continue
            counts[line] += 1
        if not counts:
            return "No history found."
        return "most changed files (changes in last 500 commits):\n" + "\n".join(
            f"  {n:>3}  {REPO_MOUNT}{path}" for path, n in counts.most_common(15)
        )

    # --------------------------------------------------------- module_fingerprint
    def _fingerprint(folder: Path) -> str:
        digest = hashlib.sha1()
        for f in walk_files(folder, ignores):
            try:
                st = f.stat()
            except OSError:
                continue
            digest.update(f"{f.relative_to(folder).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        return digest.hexdigest()[:12]

    @_safe_tool
    def module_fingerprint(path: str) -> str:
        """Check whether a module was already analysed and has not changed since.

        Call this BEFORE delegating a folder to module-analyst. It returns a
        fingerprint of the folder and the exact note path to use. If the
        status is "fresh", read that note instead of re-analysing.

        Args:
            path: folder like "/repo/src/payments".
        """
        if cache_dir is None:
            raise ToolException("module cache is not configured")
        folder = resolve(path)
        if not folder.is_dir():
            raise ToolException(f"{path} is not a folder")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel_of(folder).as_posix()).strip("_") or "root"
        fp = _fingerprint(folder)
        notes = cache_dir / MODULE_NOTES_DIR
        notes.mkdir(parents=True, exist_ok=True)
        current = notes / f"{name}.{fp}.md"
        fresh = current.is_file() and current.stat().st_size > 0
        if not fresh:  # drop outdated notes for this module so docs never use stale ones
            for old in notes.glob(f"{name}.*.md"):
                if old != current:
                    old.unlink(missing_ok=True)
        note_path = f"{CACHE_MOUNT}{MODULE_NOTES_DIR}/{current.name}"
        if fresh:
            return f"module: {name}\nfingerprint: {fp}\nstatus: fresh - read {note_path}, do not re-analyse"
        return (
            f"module: {name}\nfingerprint: {fp}\nstatus: missing or changed - "
            f"delegate to module-analyst and tell it to write its note to {note_path}"
        )

    return {
        t.name: t
        for t in (
            repo_map, python_outline, code_outline, find_entry_points, import_graph,
            dependency_manifests, find_todos, test_coverage_map, git_hotspots, module_fingerprint,
        )
    }
