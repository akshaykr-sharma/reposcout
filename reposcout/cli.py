"""
Command line chat for RepoScout.

Usage:
    python -m reposcout.cli --repo ../some-project            # chat about a repo
    python -m reposcout.cli --repo ../some-project --pack     # build the full pack
    python -m reposcout.cli --session old-run                 # continue a session

Takes:  CLI flags (repo folder, output folder, session name) + what you type
Gives:  an interactive session. Docs land in the output folder only after you
        approve each file. Conversation state is saved in
        .reposcout/reposcout.db, so `--session <name>` picks up where you left off.

Chat commands:
    /todos   show the agent's current plan
    /memory  show what the agent remembers about your preferences
    /new     start a fresh session (new thread id)
    /quit    exit
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from rich.prompt import Prompt

from .agent import MEMORY_NAMESPACE, build_agent
from .config import ensure_dirs, langsmith_enabled, load_settings
from .console import console, run_turn, show_todos

PACK_REQUEST = (
    "Build the full onboarding pack for this repo: OVERVIEW.md, MODULES.md, "
    "FLOW.md and RISKS.md in /docs/."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoScout - local codebase onboarding agent")
    parser.add_argument("--repo", help="folder of the code to study (default: REPOSCOUT_REPO_PATH)")
    parser.add_argument("--docs", help="where to write the onboarding pack (default: output/docs)")
    parser.add_argument("--session", help="session name to resume; a new one is made if omitted")
    parser.add_argument("--pack", action="store_true", help="start by building the full onboarding pack")
    return parser.parse_args()


def print_memory(store) -> None:
    item = store.get(MEMORY_NAMESPACE, "/AGENTS.md")
    if not item:
        console.print("[dim]nothing remembered yet - tell the agent your preferences and it will save them[/]")
        return
    console.print(item.value.get("content", ""))


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if args.repo or args.docs:
        settings = settings.with_repo(args.repo or settings.repo_path, args.docs)
        ensure_dirs(settings)

    if not settings.repo_path.is_dir():
        console.print(f"[red]Repo folder not found:[/] {settings.repo_path}")
        raise SystemExit(1)

    db_path = settings.data_dir / "reposcout.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    store = SqliteStore(conn)
    store.setup()

    agent = build_agent(settings, checkpointer=checkpointer, store=store)
    session = args.session or f"scout-{uuid.uuid4().hex[:8]}"

    console.rule("[bold]RepoScout")
    console.print(f"repo     : {settings.repo_path}")
    console.print(f"docs out : {settings.docs_dir}")
    console.print(f"model    : {settings.ollama_model} (ctx {settings.num_ctx})")
    console.print(f"session  : {session}")
    console.print(f"langsmith: {'tracing on' if langsmith_enabled() else 'off'}")
    console.print("[dim]type /quit to exit, /todos for the plan, /memory for saved preferences[/]\n")

    first = PACK_REQUEST if args.pack else None

    try:
        while True:
            text = first or Prompt.ask("[bold]you[/]")
            first = None
            text = text.strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/todos":
                state = agent.get_state({"configurable": {"thread_id": session}})
                show_todos(state.values.get("todos", []) if state.values else [])
                continue
            if text == "/memory":
                print_memory(store)
                continue
            if text == "/new":
                session = f"scout-{uuid.uuid4().hex[:8]}"
                console.print(f"[dim]new session {session}[/]")
                continue

            config = {"configurable": {"thread_id": session}, "recursion_limit": 150}
            run_turn(agent, {"messages": [{"role": "user", "content": text}]}, config)
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/]")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
