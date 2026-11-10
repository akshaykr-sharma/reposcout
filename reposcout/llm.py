"""
The chat models used by RepoScout: local Ollama, Anthropic Claude or Google Gemini.

Takes:  a `Settings` object and, optionally, a model "spec" string
Gives:  - `get_chat_model`  a LangChain chat model that supports tool calling
        - `check_models`    fail early, with a fix, if a configured model cannot be used
        - `parse_spec`      "provider:model" -> ModelSpec

Model specs
    ollama:qwen2.5:7b          local, free, private (the default)
    anthropic:claude-sonnet-5-5   needs ANTHROPIC_API_KEY
    google:gemini-2.5-flash    needs GOOGLE_API_KEY (or GEMINI_API_KEY)
    anthropic / google         provider alone = that provider's default model
    qwen2.5:7b                 no known provider prefix = Ollama (old .env files still work)
    (aliases: claude = anthropic, gemini = google)

Each role can use a different spec, for example a fast cheap model for the
analysts and a strong one for the doc-writer. See `.env.example`.

Privacy: with Anthropic or Google, the file excerpts the agent reads (and the
questions you ask) are sent to that provider. Credential files stay blocked
either way. Ollama keeps everything on your machine.

Why num_ctx matters (Ollama only): Ollama defaults to a small context window
(a few thousand tokens). A deep agent's system prompt plus tool schemas is
already several thousand tokens, so with the default Ollama silently cuts the
start of the prompt and the model "forgets" its tools. 16k is a safe floor for
a 7B model on a laptop. Cloud models have large windows and ignore it.
Temperature is also applied to Ollama only; cloud models use the provider
default because newer models may reject custom sampling settings.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings

PROVIDER_ALIASES = {"ollama": "ollama", "anthropic": "anthropic", "claude": "anthropic",
                    "google": "google", "gemini": "google"}
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "google": "gemini-2.5-flash"}
API_KEY_VARS = {"anthropic": ("ANTHROPIC_API_KEY",), "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
PROVIDER_LABEL = {"ollama": "Ollama (local)", "anthropic": "Anthropic Claude", "google": "Google Gemini"}
MAX_OUTPUT_TOKENS = 8192


class ModelUnavailable(RuntimeError):
    """A configured model cannot be used (server down, model missing, no API key...)."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str   # "ollama" | "anthropic" | "google"
    name: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"

    @property
    def is_cloud(self) -> bool:
        return self.provider != "ollama"


def parse_spec(spec: str) -> ModelSpec:
    """Turn "provider:model" (or a bare Ollama model name) into a ModelSpec."""
    spec = spec.strip()
    if not spec:
        raise ModelUnavailable("empty model name")
    head, _, rest = spec.partition(":")
    provider = PROVIDER_ALIASES.get(head.lower())
    if provider is None:
        return ModelSpec("ollama", spec)       # e.g. "qwen2.5:7b"
    name = rest.strip() or DEFAULT_MODELS.get(provider, "")
    if not name:
        raise ModelUnavailable("an Ollama model needs a name, e.g. ollama:qwen2.5:7b")
    return ModelSpec(provider, name)


def api_key(provider: str) -> str | None:
    for var in API_KEY_VARS.get(provider, ()):
        value = os.getenv(var, "").strip()
        if value:
            return value
    return None


def get_chat_model(settings: Settings, model: str | None = None):
    """Build the chat model for `model` (a spec) or the main model from settings."""
    spec = parse_spec(model or settings.main_model)

    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.name,
            base_url=settings.ollama_base_url,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
        )

    key = api_key(spec.provider)
    if not key:
        raise ModelUnavailable(_missing_key_message(spec))

    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ModelUnavailable('Claude support is not installed: pip install "reposcout[anthropic]"') from None
        return ChatAnthropic(model=spec.name, api_key=key, max_tokens=MAX_OUTPUT_TOKENS)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ModelUnavailable('Gemini support is not installed: pip install "reposcout[google]"') from None
    return ChatGoogleGenerativeAI(model=spec.name, google_api_key=key)


def _missing_key_message(spec: ModelSpec) -> str:
    names = " or ".join(API_KEY_VARS[spec.provider])
    return f"{PROVIDER_LABEL[spec.provider]} model '{spec.name}' is configured but {names} is not set. Add it to .env."


def configured_specs(settings: Settings, extra: tuple[str, ...] = ()) -> list[ModelSpec]:
    """Every distinct model the run will use (main + roles + extras such as the eval judge)."""
    raw = [settings.main_model, settings.analyst_model, settings.writer_model, *extra]
    return sorted({parse_spec(r) for r in raw if r}, key=str)


def cloud_providers(settings: Settings) -> list[str]:
    return sorted({PROVIDER_LABEL[s.provider] for s in configured_specs(settings) if s.is_cloud})


def _check_ollama_models(settings: Settings, names: list[str], timeout: float) -> None:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            installed = {m["name"] for m in json.load(resp).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ModelUnavailable(
            f"Cannot reach Ollama at {settings.ollama_base_url} ({exc}).\n"
            "Start it with `ollama serve`, fix OLLAMA_BASE_URL in .env, "
            "or switch to a cloud model with REPOSCOUT_MODEL."
        ) from None

    # "qwen2.5:7b" and "qwen2.5:7b-instruct" are different tags; "name" alone means ":latest"
    def present(model: str) -> bool:
        return model in installed or (":" not in model and f"{model}:latest" in installed)

    missing = [m for m in names if not present(m)]
    if missing:
        pulls = "\n".join(f"  ollama pull {m}" for m in missing)
        raise ModelUnavailable(f"Model(s) not installed in Ollama: {', '.join(missing)}\nRun:\n{pulls}")


def check_models(settings: Settings, extra: tuple[str, ...] = (), timeout: float = 3.0) -> None:
    """Raise ModelUnavailable with a how-to-fix message if any configured model cannot be used.

    Checks that Ollama is up and has the models, and that cloud API keys exist.
    It does not call the cloud APIs, so a wrong key is only discovered on first use.
    """
    specs = configured_specs(settings, extra)
    for spec in specs:
        if spec.is_cloud and not api_key(spec.provider):
            raise ModelUnavailable(_missing_key_message(spec))
    local = [s.name for s in specs if s.provider == "ollama"]
    if local:
        _check_ollama_models(settings, local, timeout)
