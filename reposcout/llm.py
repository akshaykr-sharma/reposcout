"""
The chat models used by RepoScout.

Takes:  a `Settings` object (model names, Ollama URL, context size, temperature)
Gives:  - `get_chat_model`  a LangChain `ChatOllama` that supports tool calling;
                            pass `model=` to use a different one for one role
        - `check_ollama`    fail early, with a fix, if Ollama or the model is missing

Why num_ctx matters: Ollama defaults to a small context window (a few
thousand tokens). A deep agent's system prompt plus tool schemas is already
several thousand tokens, so with the default Ollama silently cuts the start
of the prompt and the model "forgets" its tools. 16k is a safe floor for a
7B model on a laptop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_ollama import ChatOllama

from .config import Settings


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable, or a configured model is not pulled."""


def get_chat_model(settings: Settings, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
    )


def configured_models(settings: Settings) -> list[str]:
    names = [settings.ollama_model, settings.analyst_model, settings.writer_model]
    return sorted({n for n in names if n})


def check_ollama(settings: Settings, timeout: float = 3.0) -> None:
    """Raise OllamaUnavailable with a how-to-fix message if the setup is broken."""
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            installed = {m["name"] for m in json.load(resp).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {settings.ollama_base_url} ({exc}).\n"
            "Start it with `ollama serve`, or fix OLLAMA_BASE_URL in .env."
        ) from None

    # "qwen2.5:7b" and "qwen2.5:7b-instruct" are different tags; "name" alone means ":latest"
    def present(model: str) -> bool:
        return model in installed or (":" not in model and f"{model}:latest" in installed)

    missing = [m for m in configured_models(settings) if not present(m)]
    if missing:
        pulls = "\n".join(f"  ollama pull {m}" for m in missing)
        raise OllamaUnavailable(f"Model(s) not installed in Ollama: {', '.join(missing)}\nRun:\n{pulls}")
