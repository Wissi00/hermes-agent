"""OpenRouter stays out of every fallback layer for a profile shaped like the coder lane: the main
chain carries no OpenRouter entry and, with auxiliary.openrouter_fallback off, the auxiliary
auto-detect chains (text and vision) never pick OpenRouter even with OPENROUTER_API_KEY set."""
import pytest

CODER_LIKE_CONFIG = """\
model:
  default: claude-opus-5-5
  provider: anthropic
fallback_providers:
  - provider: openai-codex
    model: gpt-5.6-sol
  - provider: custom
    model: ministral-8b-latest
    base_url: https://api.mistral.ai/v1
    api_mode: chat_completions
    key_env: MISTRAL_API_KEY
auxiliary:
  openrouter_fallback: false
"""


@pytest.fixture
def coder_home(monkeypatch):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(CODER_LIKE_CONFIG)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return home


def test_main_chain_has_no_openrouter_entry(coder_home):
    from hermes_cli.config import load_config
    chain = load_config().get("fallback_providers") or []
    assert chain, "fixture must carry a fallback chain"
    assert all((e.get("provider") or "").lower() != "openrouter" for e in chain)
    assert all("openrouter" not in str(e.get("base_url") or "") for e in chain)


def test_aux_auto_chains_never_select_openrouter(coder_home, monkeypatch):
    from agent import auxiliary_client as ac
    assert ac.openrouter_auto_allowed() is False
    assert "openrouter" not in [label for label, _ in ac._get_provider_chain()]
    assert "openrouter" not in ac._vision_auto_order()
    engaged = []
    monkeypatch.setattr(ac, "_try_openrouter", lambda *a, **k: engaged.append(1) or (None, None))
    for label, try_fn in ac._get_provider_chain():
        try:
            try_fn()
        except Exception:
            pass
    assert engaged == []


def test_default_keeps_openrouter_in_the_aux_chain(coder_home):
    """The switch is opt-out: without it the upstream auto chain is unchanged."""
    (coder_home / "config.yaml").write_text(CODER_LIKE_CONFIG.replace("openrouter_fallback: false", "free_only: false"))
    from agent import auxiliary_client as ac
    assert ac.openrouter_auto_allowed() is True
    assert ac._get_provider_chain()[0][0] == "openrouter"
