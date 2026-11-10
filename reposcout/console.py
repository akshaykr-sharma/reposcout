"""
Terminal rendering and human-in-the-loop prompts.

Takes:  a compiled agent, an input (a message dict or a resume Command) and
        a LangGraph config with a thread_id
Gives:  `run_turn()` - streams one turn to the terminal, stops for approval
        whenever the agent hits an interrupt, and resumes with your decision.

What you see while it runs:
    plan    the todo list each time the agent updates it
    ->      a tool call made by the main agent
    |  ->   a tool call made inside a subagent (indented)
    !       a tool error (for example a blocked write)

Nothing project specific lives here, so the same file can be dropped into
any deep agent project that needs a terminal front end.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()

STATUS_ICON = {"completed": "[green]done[/]", "in_progress": "[yellow]now [/]", "pending": "[dim]todo[/]"}


def _short(value: Any, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _as_list(messages: Any) -> list:
    # Depending on the reducer, an update can be a list, a single message or a wrapper
    if messages is None:
        return []
    if hasattr(messages, "value"):
        messages = messages.value
    if isinstance(messages, list):
        return messages
    return [messages]


def show_todos(todos: list[dict]) -> None:
    if not todos:
        return
    table = Table(title="plan", show_header=False, box=None, padding=(0, 1))
    for item in todos:
        table.add_row(STATUS_ICON.get(item.get("status", ""), "?"), item.get("content", ""))
    console.print(table)


def _render_update(namespace: tuple, update: dict, seen: set) -> None:
    in_subagent = bool(namespace)
    pad = "[dim]|[/]  " if in_subagent else ""

    if "todos" in update and update["todos"] and not in_subagent:
        show_todos(update["todos"])

    for msg in _as_list(update.get("messages")):
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                # after a resume the same tool call is emitted again - print it once
                if call["name"] == "write_todos" or call.get("id") in seen:
                    continue
                seen.add(call.get("id"))
                if call["name"] == "task":
                    who = call["args"].get("subagent_type", "?")
                    what = _short(call["args"].get("description", ""), 110)
                    console.print(f"{pad}[magenta]-> delegating to {who}[/]: {what}")
                else:
                    console.print(f"{pad}[cyan]-> {call['name']}[/] {_short(call['args'])}")
            if not msg.tool_calls and msg.text.strip() and not in_subagent:
                console.print(Panel(Markdown(msg.text), border_style="green", title="agent"))
        elif isinstance(msg, ToolMessage):
            text = msg.text
            if msg.status == "error" or text.startswith("Error"):
                console.print(f"{pad}[red]! {_short(text, 160)}[/]")


def _stream(agent, payload, config, seen: set) -> None:
    for namespace, chunk in agent.stream(payload, config, stream_mode="updates", subgraphs=True):
        if not isinstance(chunk, dict):
            continue
        for update in chunk.values():
            if isinstance(update, dict):
                _render_update(namespace, update, seen)


def _preview_action(action: dict) -> None:
    name, args = action["name"], action.get("args", {})
    if name in ("write_file", "edit_file"):
        path = args.get("file_path", "?")
        body = args.get("content") or args.get("new_string") or ""
        lines = body.splitlines()
        shown = "\n".join(lines[:40]) + ("\n..." if len(lines) > 40 else "")
        console.print(Panel(Syntax(shown, "markdown", word_wrap=True),
                            title=f"{name} -> {path}  ({len(lines)} lines)", border_style="yellow"))
    else:
        console.print(Panel(json.dumps(args, indent=2, ensure_ascii=False, default=str)[:3000],
                            title=f"{name}", border_style="yellow"))


def _ask_decision(action: dict, allowed: list[str], respond_tools: set[str]) -> dict:
    name, args = action["name"], action.get("args", {})

    # "respond" style tools: the human's answer IS the tool result
    if name in respond_tools and "respond" in allowed:
        question = args.get("question") or _short(args, 300)
        console.print(Panel(question, title="the agent is asking you", border_style="blue"))
        answer = Prompt.ask("[blue]your answer[/]")
        return {"type": "respond", "message": answer}

    _preview_action(action)
    while True:
        choice = Prompt.ask("[yellow]approve, reject or view full?[/]", choices=["a", "r", "v"], default="a")
        if choice == "v":
            console.print(args.get("content") or json.dumps(args, indent=2, default=str))
            continue
        if choice == "a":
            return {"type": "approve"}
        reason = Prompt.ask("what should change? (sent back to the agent)")
        return {"type": "reject", "message": reason or "Rejected by the reviewer."}


def _decisions_for(interrupt_value: dict, respond_tools: set[str]) -> dict:
    actions = interrupt_value.get("action_requests", [])
    configs = {c["action_name"]: c for c in interrupt_value.get("review_configs", [])}
    decisions = []
    for action in actions:
        allowed = configs.get(action["name"], {}).get("allowed_decisions", ["approve", "reject"])
        decisions.append(_ask_decision(action, allowed, respond_tools))
    return {"decisions": decisions}


def run_turn(agent, payload, config, respond_tools: Iterable[str] = ()) -> None:
    """Stream one user turn, handling any number of approval stops."""
    respond_tools = set(respond_tools)
    seen: set = set()
    while True:
        _stream(agent, payload, config, seen)
        pending = agent.get_state(config).interrupts
        if not pending:
            return
        if len(pending) == 1:
            payload = Command(resume=_decisions_for(pending[0].value, respond_tools))
        else:
            payload = Command(resume={i.id: _decisions_for(i.value, respond_tools) for i in pending})
