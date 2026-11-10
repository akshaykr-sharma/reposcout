"""
The chat model used by RepoScout.

Takes:  a `Settings` object (model name, Ollama URL, context size, temperature)
Gives:  a LangChain `ChatOllama` instance that supports tool calling

Why num_ctx matters: Ollama defaults to a small context window (a few
thousand tokens). A deep agent's system prompt plus tool schemas is already
several thousand tokens, so with the default Ollama silently cuts the start
of the prompt and the model "forgets" its tools. 16k is a safe floor for a
7B model on a laptop.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from .config import Settings


def get_chat_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
    )
