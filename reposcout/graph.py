"""
Graph entry point for `langgraph dev` and LangGraph Studio.

Takes:  settings from .env (REPOSCOUT_REPO_PATH decides which repo is studied)
Gives:  the module-level `agent` that langgraph.json points at

The local dev server brings its own checkpointer and store, so we build the
agent without them here. To study a different repo in Studio, change
REPOSCOUT_REPO_PATH in .env and restart `langgraph dev`.
"""

from reposcout.agent import build_agent
from reposcout.config import load_settings

agent = build_agent(load_settings())
