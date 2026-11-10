"""
Filesystem helpers shared by the tools, the agent permissions and the validator.

Takes:  a repo root and a set of folder names to ignore
Gives:  - `walk_files`   fast recursive file listing that never enters ignored folders
        - `is_secret`    True for files that may hold credentials
        - `SECRET_READ_GLOBS` permission globs that block the agent from reading secrets

Why not `Path.rglob`? It descends into `node_modules` and `.venv` and only
then lets you filter, which is very slow on real repos. `os.walk` lets us
prune a folder in place so it is never entered.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

# File names (matched case-insensitively) that commonly hold secrets.
SECRET_NAME_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa", "id_rsa.*", "id_dsa", "id_ecdsa", "id_ed25519", "id_ed25519.*",
    "credentials", "credentials.json", ".netrc", ".npmrc", ".pypirc", "secrets.*",
    "*.tfstate",
)
# Names that look secret but are meant to be shared templates.
SECRET_ALLOW_GLOBS = (".env.example", ".env.sample", ".env.template", "*.pub")

# Permission rules for deepagents (first matching rule wins, so allows come first).
# deepagents' `**` does not match dot-files, so dot-file rules are explicit.
SECRET_READ_GLOBS = (
    "/repo/**/*.pem", "/repo/**/*.key", "/repo/**/*.p12", "/repo/**/*.pfx",
    "/repo/**/*.jks", "/repo/**/*.keystore", "/repo/**/id_rsa", "/repo/**/id_rsa.*",
    "/repo/**/id_dsa", "/repo/**/id_ecdsa", "/repo/**/id_ed25519",
    "/repo/**/id_ed25519.*", "/repo/**/credentials", "/repo/**/credentials.json",
    "/repo/**/secrets.*", "/repo/**/*.tfstate",
    "/repo/**/.env", "/repo/**/.env.*", "/repo/**/.netrc", "/repo/**/.npmrc",
    "/repo/**/.pypirc",
)
SECRET_ALLOW_READ_GLOBS = (
    "/repo/**/.env.example", "/repo/**/.env.sample", "/repo/**/.env.template",
    "/repo/**/*.pub",
)

# Folders that hold tests/docs - not useful as program entry points.
NON_ENTRY_DIRS = {"tests", "test", "__tests__", "docs", "doc", "examples", "example", "fixtures"}


def is_secret(name: str) -> bool:
    lowered = name.lower()
    if any(fnmatch.fnmatch(lowered, g) for g in SECRET_ALLOW_GLOBS):
        return False
    return any(fnmatch.fnmatch(lowered, g) for g in SECRET_NAME_GLOBS)


def walk_files(
    root: Path,
    ignores: set[str],
    *,
    skip_dirs: set[str] | None = None,
    skip_secrets: bool = True,
) -> Iterator[Path]:
    """Yield every file under `root`, never entering ignored folders."""
    skip_dirs = skip_dirs or set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ignores and d not in skip_dirs)
        for name in sorted(filenames):
            if name in ignores or (skip_secrets and is_secret(name)):
                continue
            yield Path(dirpath) / name


def count_files(folder: Path, ignores: set[str], cap: int = 5000) -> int:
    """Count files below `folder`, stopping at `cap` so huge trees stay cheap."""
    total = 0
    for _ in walk_files(folder, ignores):
        total += 1
        if total >= cap:
            break
    return total
