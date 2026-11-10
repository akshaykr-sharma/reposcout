"""
Builds the RepoScout deep agent.

Takes:  Settings, plus an optional LangGraph checkpointer and store
Gives:  a compiled LangGraph graph you can .invoke() / .stream()

How the pieces fit:

    CompositeBackend  - one virtual filesystem made of several mounts
        /repo/      -> FilesystemBackend on the real repo folder (read only)
        /docs/      -> FilesystemBackend on the output folder
        /memories/  -> StoreBackend (LangGraph store, survives restarts)
        everything else (/scratch/...) -> StateBackend, lives in graph state

    Permissions
        writes to /repo/**  -> denied
        writes to /docs/**  -> interrupt (human approves / rejects)

    Subagents (called through the built-in `task` tool)
        module-analyst, flow-tracer, doc-writer

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

from .config import Settings
from .llm import get_chat_model
from .prompts import (
    DOC_WRITER_PROMPT,
    FLOW_TRACER_PROMPT,
    MAIN_PROMPT,
    MODULE_ANALYST_PROMPT,
)
from .tools import make_repo_tools

MEMORY_FILE = "/memories/AGENTS.md"
MEMORY_NAMESPACE = ("reposcout", "memories")


def build_backend(settings: Settings) -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/repo/": FilesystemBackend(root_dir=settings.repo_path, virtual_mode=True),
            "/docs/": FilesystemBackend(root_dir=settings.docs_dir, virtual_mode=True),
            "/memories/": StoreBackend(namespace=lambda _runtime: MEMORY_NAMESPACE),
        },
    )


def build_permissions(read_only: bool) -> list[FilesystemPermission]:
    rules = [FilesystemPermission(operations=["write"], paths=["/repo/**"], mode="deny")]
    if read_only:
        rules.append(FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="deny"))
    else:
        rules.append(FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="interrupt"))
    return rules


def build_agent(settings: Settings, checkpointer=None, store=None, read_only: bool = False):
    """Create the RepoScout agent.

    read_only=True blocks writes to /docs/ as well. The eval script uses it
    so a test run can never touch your output folder or stop for approval.
    """
    model = get_chat_model(settings)
    repo_map, python_outline, find_entry_points = make_repo_tools(
        settings.repo_path, settings.ignores
    )

    subagents = [
        {
            "name": "module-analyst",
            "description": (
                "Studies ONE folder/package of the repo and returns a markdown "
                "summary: purpose, key files, main classes, dependencies, gotchas. "
                "Pass the exact folder path, e.g. '/repo/src/payments'."
            ),
            "system_prompt": MODULE_ANALYST_PROMPT,
            "tools": [repo_map, python_outline],
        },
        {
            "name": "flow-tracer",
            "description": (
                "Traces one request or command from its entry point through the "
                "code and returns a numbered walkthrough plus a mermaid sequence diagram."
            ),
            "system_prompt": FLOW_TRACER_PROMPT,
            "tools": [find_entry_points, python_outline],
        },
        {
            "name": "doc-writer",
            "description": (
                "Turns the notes in /scratch/ into the final onboarding files "
                "under /docs/. Call it once the analysis is done."
            ),
            "system_prompt": DOC_WRITER_PROMPT,
            "tools": [],
        },
    ]

    return create_deep_agent(
        model=model,
        tools=[repo_map, python_outline, find_entry_points],
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
