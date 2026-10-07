"""``compression.model_threshold_tokens``: per-model absolute compaction triggers.

The GPT Sol family ships with a 300K-token trigger (``gpt-*-sol*``). It must fire on the real
prompt-token count at exactly 300,000, leave every other model on its ratio trigger, override the
Codex autoraise / small-window floor when it fits, and step aside (never overflow) when the
model's usable input window cannot hold it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_init import _parse_compression_config
from agent.compression_model_trigger import (
    DEFAULT_MODEL_THRESHOLD_TOKENS,
    parse_model_threshold_tokens,
    resolve_model_threshold,
)
from agent.context_compressor import ContextCompressor

SOL = {"gpt-*-sol*": 300_000}


def _cc(model, ctx, threshold=0.50, mapping=SOL, provider="openai-codex"):
    with patch("agent.context_compressor.get_model_context_length", return_value=ctx):
        cc = ContextCompressor(model=model, threshold_percent=threshold, quiet_mode=True,
                               provider=provider, model_threshold_tokens=mapping)
        _ = cc.threshold_tokens
    return cc


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol", "openai/gpt-6.1-sol",
                                   "gpt-6-sol-2026-09-22", "gpt-7-sol"])
def test_sol_family_fires_at_exactly_300k(model):
    cc = _cc(model, 1_050_000)
    assert cc.threshold_tokens == 300_000
    assert cc.should_compress(299_999) is False
    assert cc.should_compress(300_000) is True
    assert cc.model_trigger_status["applied"] is True
    assert cc.threshold_percent == pytest.approx(300_000 / 1_050_000)


@pytest.mark.parametrize("model,provider", [("claude-opus-5-5", "anthropic"), ("gpt-6-luna", "openai-codex"),
                                            ("gpt-6-astra", "openai-codex"), ("solar-pro3", "upstage")])
def test_non_sol_models_keep_their_ratio_trigger(model, provider):
    cc = _cc(model, 1_000_000, provider=provider)
    assert cc.threshold_tokens == 500_000
    assert cc.model_trigger_status is None
    assert cc.should_compress(300_000) is False


def test_window_too_small_keeps_safe_ratio_trigger():
    """Codex OAuth serves Sol with a 272K window: 300K is unreachable, so the autoraised 85% stays."""
    cc = _cc("gpt-6.1-sol", 272_000, threshold=0.85)
    assert cc.threshold_tokens == 231_200
    assert cc.model_trigger_status["applied"] is False
    assert cc.model_trigger_status["reason"] == "window_too_small"
    assert cc.should_compress(231_200) is True


def test_trigger_needs_15_percent_headroom():
    assert _cc("gpt-6.1-sol", 352_942).threshold_tokens == 300_000  # 0.85 * 352,942 = 300,000.7
    assert _cc("gpt-6.1-sol", 352_940).threshold_tokens != 300_000


def test_model_switch_applies_and_drops_the_trigger():
    cc = _cc("claude-opus-5-5", 1_000_000, provider="anthropic")
    assert cc.threshold_tokens == 500_000
    cc.update_model("gpt-6.1-sol-900k", 872_000, provider="openai-codex")
    assert cc.threshold_tokens == 300_000
    assert cc.preview_threshold_tokens("claude-opus-5-5", 1_000_000, "anthropic") == 500_000
    assert cc.model_trigger_status["applied"] is True  # preview did not clobber the live verdict
    cc.update_model("claude-opus-5-5", 1_000_000, provider="anthropic")
    assert cc.threshold_tokens == 500_000
    assert cc.model_trigger_status is None


def test_global_cap_still_lowers_the_trigger():
    with patch("agent.context_compressor.get_model_context_length", return_value=1_050_000):
        cc = ContextCompressor(model="gpt-6.1-sol", quiet_mode=True, model_threshold_tokens=SOL,
                               threshold_tokens_cap=200_000)
        assert cc.threshold_tokens == 200_000


def test_parse_layers_user_entries_over_the_shipped_default():
    assert parse_model_threshold_tokens(None) == DEFAULT_MODEL_THRESHOLD_TOKENS == SOL
    assert parse_model_threshold_tokens({}) == SOL
    assert parse_model_threshold_tokens({"glm-5.2": 120_000}) == {**SOL, "glm-5.2": 120_000}
    assert parse_model_threshold_tokens({"gpt-*-sol*": 0}) == {}
    assert parse_model_threshold_tokens({"x": "300000", "y": True, "z": -1}) == SOL


def test_glob_keys_rank_by_literal_length_and_scope():
    th = {"gpt-*-sol*": 0.3, "gpt-6.1-sol": 0.4, "openai-codex:gpt-*-sol*": 0.35}
    assert resolve_model_threshold("gpt-6.1-sol", th, 0.5, "openai-codex") == 0.4
    assert resolve_model_threshold("gpt-6-sol", th, 0.5, "openai-codex") == 0.35
    assert resolve_model_threshold("gpt-6-sol", th, 0.5, "openrouter") == 0.3
    assert resolve_model_threshold("solar-pro3", th, 0.5, "upstage") == 0.5


def test_config_parse_overrides_codex_autoraise():
    """Through the real config parser: the Codex autoraise (0.85) feeds the ratio, the Sol trigger wins."""
    agent = SimpleNamespace(model="gpt-6.1-sol-900k", provider="openai-codex", base_url="", quiet_mode=True)
    cs = _parse_compression_config(agent, {"compression": {"threshold": 0.5}})
    assert cs.model_threshold_tokens == SOL
    with patch("agent.context_compressor.get_model_context_length", return_value=872_000):
        cc = ContextCompressor(model=agent.model, threshold_percent=cs.threshold, quiet_mode=True,
                               provider=agent.provider, model_threshold_tokens=cs.model_threshold_tokens)
        assert cc.threshold_tokens == 300_000
