"""
Builds the RepoScout deep agent.

Takes:  Settings, plus an optional LangGraph checkpointer and store
Gives:  a compiled LangGraph graph you can .invoke() / .stream()

How the pieces fit:

    CompositeBackend  - one virtual filesystem made of several mounts
        /repo/      -> FilesystemBackend on the real repo folder (read only)
        /docs/      -> FilesystemBackend on the output folder
        /memories/  -> StoreBackend (LangGraph store, survives restarts)
        /cache/     -> FilesystemBackend on .reposcout/cache/<repo>: per-repo
                       module notes named <module>.<fingerprint>.md, so
                       unchanged modules are not analysed again
        everything else (/scratch/...) -> StateBackend, lives in graph state

    Permissions (first matching rule wins; subagents inherit them)
        reads of credentials (.env, *.pem, id_rsa, ...) -> denied
        writes to /repo/**      -> denied
        writes to /memories/**  -> interrupt (repo text is untrusted, so a
                                   poisoned note must not persist silently)
        writes to /docs/**      -> interrupt (human approves / rejects)

    Subagents (called through the built-in `task` tool)
        module-analyst, flow-tracer, risk-analyst, doc-writer
        Each can run on its own model (OLLAMA_ANALYST_MODEL / OLLAMA_WRITER_MODEL).

    Middleware
        TodoListMiddleware gives the agent `write_todos` for planning.
        deepagents adds filesystem, subagent and summarization middleware
        on its own.

Checkpointer and store are optional because `langgraph dev` provides its
own and refuses custom ones. The CLI passes SQLite versions of both.
"""

from __future__ import annotations

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from langchain.agents.middleware import TodoListMiddleware

from .config import Settings, ensure_dirs
from .fsutil import SECRET_ALLOW_READ_GLOBS, SECRET_READ_GLOBS
from .llm import get_chat_model
from .prompts import (
    DOC_WRITER_PROMPT,
    FLOW_TRACER_PROMPT,
    MAIN_PROMPT,
    MODULE_ANALYST_PROMPT,
    RISK_ANALYST_PROMPT,
)
from .tools import make_repo_tools
from .validate import make_check_docs_tool

MEMORY_FILE = "/memories/AGENTS.md"
MEMORY_NAMESPACE = ("reposcout", "memories")


def build_backend(settings: Settings) -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/repo/": FilesystemBackend(root_dir=settings.repo_path, virtual_mode=True),
            "/docs/": FilesystemBackend(root_dir=settings.docs_dir, virtual_mode=True),
            "/cache/": FilesystemBackend(root_dir=settings.cache_dir, virtual_mode=True),
            "/memories/": StoreBackend(namespace=lambda _runtime: MEMORY_NAMESPACE),
        },
    )


def build_permissions(read_only: bool) -> list[FilesystemPermission]:
    mode = "deny" if read_only else "interrupt"
    return [
        # shareable templates first (.env.example), then block real credentials
        FilesystemPermission(operations=["read"], paths=list(SECRET_ALLOW_READ_GLOBS), mode="allow"),
        FilesystemPermission(operations=["read"], paths=list(SECRET_READ_GLOBS), mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/repo/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/memories/**"], mode=mode),
        FilesystemPermission(operations=["write"], paths=["/docs/**"], mode=mode),
    ]


def _role_model(settings: Settings, name: str):
    """A dedicated model for a role, or None to inherit the main one."""
    if not name or name == settings.ollama_model:
        return None
    return get_chat_model(settings, model=name)


def _spec(name: str, description: str, prompt: str, tools: list, role_model) -> dict:
    item = {"name": name, "description": description, "system_prompt": prompt, "tools": tools}
    if role_model is not None:
        item["model"] = role_model
    return item


def build_agent(settings: Settings, checkpointer=None, store=None, read_only: bool = False):
    """Create the RepoScout agent.

    read_only=True blocks writes to /docs/ and /memories/ as well. The eval
    script uses it so a test run can never touch your output folder, change
    your saved preferences or stop for approval.
    """
    ensure_dirs(settings)
    model = get_chat_model(settings)
    analyst_model = _role_model(settings, settings.analyst_model)
    writer_model = _role_model(settings, settings.writer_model)
    t = make_repo_tools(settings.repo_path, settings.ignores, settings.cache_dir)
    check_docs = make_check_docs_tool(settings.docs_dir, settings.repo_path, settings.ignores)

    subagents = [
        _spec(
            "module-analyst",
            "Studies ONE folder/package of the repo and writes a markdown note: purpose, "
            "key files, main classes, dependencies, gotchas. Pass the exact folder path "
            "(e.g. '/repo/src/payments') AND the exact note path from module_fingerprint.",
            MODULE_ANALYST_PROMPT,
            [t["repo_map"], t["python_outline"], t["code_outline"], t["import_graph"]],
            analyst_model,
        ),
        _spec(
            "flow-tracer",
            "Traces one request or command from its entry point through the code and "
            "writes a numbered walkthrough plus a mermaid sequence diagram to /scratch/flow.md.",
            FLOW_TRACER_PROMPT,
            [t["find_entry_points"], t["python_outline"], t["code_outline"]],
            analyst_model,
        ),
        _spec(
            "risk-analyst",
            "Collects evidence of fragile or unclear areas (TODOs, untested files, git "
            "hotspots, dependencies) and writes /scratch/risks.md. Takes no arguments.",
            RISK_ANALYST_PROMPT,
            [t["find_todos"], t["test_coverage_map"], t["git_hotspots"],
             t["dependency_manifests"], t["repo_map"]],
            analyst_model,
        ),
        _spec(
            "doc-writer",
            "Turns the notes in /scratch/ and /cache/modules/ into the final onboarding "
            "files under /docs/ and checks them. Call it once the analysis is done.",
            DOC_WRITER_PROMPT,
            [check_docs],
            writer_model,
        ),
    ]

    return create_deep_agent(
        model=model,
        tools=[t["repo_map"], t["find_entry_points"], t["python_outline"], t["code_outline"],
               t["dependency_manifests"], t["module_fingerprint"]],
        system_prompt=MAIN_PROMPT,
        subagents=subagents,
        middleware=[TodoListMiddleware()],
        backend=build_backend(settings),
        permissions=build_permissions(read_only),
        memory=[MEMORY_FILE],
        checkpointer=checkpointer,
        store=store,
        name="reposcout",
    )
