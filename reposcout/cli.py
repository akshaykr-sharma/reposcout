"""
Command line chat for RepoScout.

Usage:
    python -m reposcout.cli --repo ../some-project            # chat about a repo
    python -m reposcout.cli --repo ../some-project --pack     # build the full pack
    python -m reposcout.cli --session old-run                 # continue a session
    reposcout --repo ../some-project                          # same, after `pip install -e .`

Takes:  CLI flags (repo folder, output folder, session name) + what you type
Gives:  an interactive session. Docs land in the output folder only after you
        approve each file. Conversation state is saved in
        .reposcout/reposcout.db, so `--session <name>` picks up where you left off.
        A session is tied to the repo it started with.

After `--pack` finishes, the pack is checked (files exist, mentioned paths
exist, mermaid looks valid) and you are offered a fix-up round if it is not clean.

Chat commands:
    /todos   show the agent's current plan
    /memory  show what the agent remembers about your preferences
    /new     start a fresh session (new thread id)
    /quit    exit

Ctrl-C while the agent is working cancels that turn and returns to the prompt;
Ctrl-C at the prompt exits.
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from dataclasses import replace

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from rich.prompt import Confirm, Prompt

from .agent import MEMORY_NAMESPACE, build_agent
from .config import ConfigError, ensure_dirs, langsmith_enabled, load_settings
from .console import console, run_turn, show_todos
from .llm import ModelUnavailable, check_models, cloud_providers, parse_spec
from .sessions import bind_session
from .validate import format_issues, validate_docs

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
    parser.add_argument("--model", help="main model spec, e.g. anthropic:claude-sonnet-5-5 (default: REPOSCOUT_MODEL)")
    parser.add_argument("--no-check", action="store_true", help="skip the model availability check")
    return parser.parse_args()


def print_memory(store) -> None:
    item = store.get(MEMORY_NAMESPACE, "/AGENTS.md")
    if not item:
        console.print("[dim]nothing remembered yet - tell the agent your preferences and it will save them[/]")
        return
    console.print(item.value.get("content", ""))


def check_pack(agent, settings, config) -> None:
    """Validate the written pack; offer one fix-up round if it has problems."""
    for attempt in (1, 2):
        issues = validate_docs(settings.docs_dir, settings.repo_path, settings.ignores)
        if not issues:
            console.print("[green]pack check: all files present, paths exist, mermaid looks valid[/]")
            return
        console.print(f"[yellow]pack check found {len(issues)} problem(s):[/]\n{format_issues(issues)}")
        if attempt == 2 or not Confirm.ask("ask the agent to fix them?", default=True):
            return
        request = "check_docs found these problems in /docs/. Fix each one:\n" + format_issues(issues)
        run_turn(agent, {"messages": [{"role": "user", "content": request}]}, config)


def main() -> None:
    args = parse_args()
    try:
        settings = load_settings()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise SystemExit(2) from None
    if args.repo or args.docs:
        settings = settings.with_repo(args.repo or settings.repo_path, args.docs)
    if args.model:
        settings = replace(settings, model=args.model)

    if not settings.repo_path.is_dir():
        console.print(f"[red]Repo folder not found:[/] {settings.repo_path}")
        raise SystemExit(1)

    if not args.no_check:
        try:
            check_models(settings)
        except ModelUnavailable as exc:
            console.print(f"[red]{exc}[/]")
            raise SystemExit(1) from None

    ensure_dirs(settings)
    db_path = settings.data_dir / "reposcout.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    store = SqliteStore(conn)
    store.setup()

    session = args.session or f"scout-{uuid.uuid4().hex[:8]}"
    other_repo = bind_session(conn, session, settings.repo_path)
    if other_repo:
        console.print(
            f"[red]Session '{session}' belongs to {other_repo}[/], not {settings.repo_path}.\n"
            "Use --repo with that folder, or pick a different --session name."
        )
        conn.close()
        raise SystemExit(1)

    agent = build_agent(settings, checkpointer=checkpointer, store=store)

    console.rule("[bold]RepoScout")
    console.print(f"repo     : {settings.repo_path}")
    console.print(f"docs out : {settings.docs_dir}")
    main = parse_spec(settings.main_model)
    ctx = f" (ctx {settings.num_ctx})" if main.provider == "ollama" else ""
    console.print(f"model    : {main}{ctx}")
    if settings.analyst_model or settings.writer_model:
        console.print(
            f"           analysts: {settings.analyst_model or 'same'}, writer: {settings.writer_model or 'same'}"
        )
    if cloud_providers(settings):
        console.print(
            f"[yellow]cloud model: code the agent reads is sent to {', '.join(cloud_providers(settings))}[/] "
            "[dim](credential files stay blocked; use an ollama: model to stay fully local)[/]"
        )
    console.print(f"session  : {session}")
    console.print(f"langsmith: {'tracing on' if langsmith_enabled() else 'off'}")
    console.print("[dim]type /quit to exit, /todos for the plan, /memory for saved preferences[/]\n")

    first = PACK_REQUEST if args.pack else None
    pack_pending = args.pack

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
                bind_session(conn, session, settings.repo_path)
                console.print(f"[dim]new session {session}[/]")
                continue

            config = {"configurable": {"thread_id": session}, "recursion_limit": 150}
            try:
                run_turn(agent, {"messages": [{"role": "user", "content": text}]}, config)
                if pack_pending:
                    pack_pending = False
                    check_pack(agent, settings, config)
            except KeyboardInterrupt:
                # Dangling tool calls from the cancelled turn are repaired by
                # deepagents' PatchToolCallsMiddleware on the next message.
                pack_pending = False
                console.print("\n[yellow]turn cancelled[/] - the session keeps its last saved state")
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/]")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
