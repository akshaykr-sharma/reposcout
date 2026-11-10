"""
RepoScout - a local codebase onboarding agent.

Point it at a folder on your machine and it writes an onboarding pack for
that code: an architecture overview, per-module notes, a request-flow
walkthrough and a list of risky areas.

Package layout:
    config.py   -> settings read from env / .env
    llm.py      -> the local Ollama chat model
    tools.py    -> repo-aware tools (tree map, python outline, entry points)
    prompts.py  -> system prompts for the main agent and its subagents
    agent.py    -> wires everything into a deep agent
    graph.py    -> entry point used by `langgraph dev` / LangGraph Studio
    console.py  -> terminal rendering + human-in-the-loop prompts
    cli.py      -> `python -m reposcout.cli`
"""

__version__ = "0.1.0"
