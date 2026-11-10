import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reposcout.config import DEFAULT_IGNORES  # noqa: E402
from reposcout.tools import make_repo_tools  # noqa: E402


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    write(root, "app.py", 'import argparse\nfrom pkg.core import run\n\nif __name__ == "__main__":\n    argparse.ArgumentParser()\n    run()\n')
    write(root, "pkg/__init__.py", "")
    write(
        root,
        "pkg/core.py",
        '"""Core logic."""\nfrom . import util\nimport os\n\nMAX_ITEMS = 10\n\n\n@decorator("x")\n'
        'async def run(a, *args, b=1, **kw) -> int:\n    """Run it."""\n    # TODO: handle errors\n    return 1\n\n\n'
        "class Engine(Base):\n    def go(self, speed: int = 3) -> None: ...\n",
    )
    write(root, "pkg/util.py", "def helper():\n    try:\n        import json\n    except ImportError:\n        pass\n")
    write(root, "tests/test_core.py", "def test_x():\n    pass\n")
    write(root, "server.js", "const express = require('express');\nconst app = express();\nexport async function start() {}\nclass Server {}\n")
    write(root, "main.go", 'package main\n\nimport "fmt"\n\nfunc main() {}\n\ntype Store struct {}\n')
    write(root, "README.md", "Use FastAPI( here\n<!-- TODO: docs -->\n")
    write(root, "node_modules/lib/index.js", "if (x) {}\napp = express()\n")
    write(root, ".env", "SECRET=1\n")
    write(root, ".env.example", "SECRET=\n")
    write(root, "requirements.txt", "requests>=2\n")
    return root


@pytest.fixture
def tools(repo, tmp_path):
    return make_repo_tools(repo, set(DEFAULT_IGNORES), tmp_path / "cache")
