import pytest

from reposcout.agent import build_agent
from reposcout.config import load_settings
from reposcout.llm import (
    ModelSpec,
    ModelUnavailable,
    check_models,
    cloud_providers,
    get_chat_model,
    parse_spec,
)

KEY_VARS = ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
MODEL_VARS = (
    "REPOSCOUT_MODEL", "REPOSCOUT_ANALYST_MODEL", "REPOSCOUT_WRITER_MODEL", "REPOSCOUT_JUDGE_MODEL",
    "OLLAMA_ANALYST_MODEL", "OLLAMA_WRITER_MODEL", "OLLAMA_JUDGE_MODEL", "OLLAMA_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in KEY_VARS + MODEL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REPOSCOUT_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("REPOSCOUT_DATA_DIR", str(tmp_path / "data"))


# --- spec parsing ------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("qwen2.5:7b", ModelSpec("ollama", "qwen2.5:7b")),            # bare = Ollama (old .env files)
        ("llama3", ModelSpec("ollama", "llama3")),
        ("ollama:qwen2.5:7b", ModelSpec("ollama", "qwen2.5:7b")),
        ("anthropic:claude-sonnet-5-5", ModelSpec("anthropic", "claude-sonnet-5-5")),
        ("claude:claude-haiku-5-5", ModelSpec("anthropic", "claude-haiku-5-5")),
        ("anthropic", ModelSpec("anthropic", "claude-sonnet-5-5")),    # provider alone = default model
        ("google:gemini-2.5-pro", ModelSpec("google", "gemini-2.5-pro")),
        ("gemini", ModelSpec("google", "gemini-2.5-flash")),
    ],
)
def test_parse_spec(text, expected):
    assert parse_spec(text) == expected


def test_ollama_needs_a_name():
    with pytest.raises(ModelUnavailable):
        parse_spec("ollama")


# --- settings ----------------------------------------------------------------

def test_default_main_model_is_local_ollama():
    assert parse_spec(load_settings().main_model) == ModelSpec("ollama", "qwen2.5:7b")


def test_old_ollama_var_names_still_work(monkeypatch):
    monkeypatch.setenv("OLLAMA_WRITER_MODEL", "big:70b")
    assert load_settings().writer_model == "big:70b"
    monkeypatch.setenv("REPOSCOUT_WRITER_MODEL", "anthropic")      # new name wins
    assert load_settings().writer_model == "anthropic"


# --- credentials -------------------------------------------------------------

def test_cloud_model_without_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_MODEL", "anthropic:claude-sonnet-5-5")
    with pytest.raises(ModelUnavailable, match="ANTHROPIC_API_KEY"):
        check_models(load_settings())
    with pytest.raises(ModelUnavailable, match="ANTHROPIC_API_KEY"):
        get_chat_model(load_settings())


def test_gemini_accepts_either_key_name(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_MODEL", "google:gemini-2.5-flash")
    with pytest.raises(ModelUnavailable, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        check_models(load_settings())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    check_models(load_settings())      # all-cloud setup: no Ollama needed


def test_all_cloud_setup_does_not_contact_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REPOSCOUT_MODEL", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    check_models(load_settings(), timeout=0.2)


def test_mixed_setup_still_checks_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REPOSCOUT_MODEL", "anthropic")
    monkeypatch.setenv("REPOSCOUT_ANALYST_MODEL", "ollama:qwen2.5:7b")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(ModelUnavailable, match="ollama serve"):
        check_models(load_settings(), timeout=0.2)


# --- building the chat models (no network call is made) ----------------------

def test_builds_each_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    s = load_settings()
    assert type(get_chat_model(s)).__name__ == "ChatOllama"
    assert type(get_chat_model(s, "anthropic:claude-sonnet-5-5")).__name__ == "ChatAnthropic"
    assert type(get_chat_model(s, "gemini:gemini-2.5-flash")).__name__ == "ChatGoogleGenerativeAI"


def test_build_agent_with_cloud_main_and_local_writer(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("REPOSCOUT_WRITER_MODEL", "google:gemini-2.5-flash")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert build_agent(load_settings(), read_only=True) is not None


def test_cloud_providers_listed_for_privacy_notice(monkeypatch):
    assert cloud_providers(load_settings()) == []
    monkeypatch.setenv("REPOSCOUT_MODEL", "claude")
    monkeypatch.setenv("REPOSCOUT_WRITER_MODEL", "gemini")
    assert cloud_providers(load_settings()) == ["Anthropic Claude", "Google Gemini"]
