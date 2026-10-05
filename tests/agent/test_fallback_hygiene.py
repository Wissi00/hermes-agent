"""Fallback hygiene contracts: same-route retry window, context-aware chain skip, mid-turn
primary restore cadence, paid-lane entry/exit alerts."""
import time
from types import SimpleNamespace

import pytest

from agent import fallback_hygiene as fh
from agent.error_classifier import FailoverReason


@pytest.fixture
def cfg(monkeypatch):
    values = dict(fh._DEFAULTS)
    monkeypatch.setattr(fh, "settings", lambda: values)
    return values


def _agent(**kw):
    base = dict(provider="custom", model="qwen-local", session_id="s1",
                _primary_runtime={"provider": "anthropic", "model": "claude-opus-5-5",
                                  "compressor_context_length": 200000},
                _fallback_chain=[], _fallback_index=0, context_compressor=SimpleNamespace(last_prompt_tokens=0))
    base.update(kw)
    return SimpleNamespace(**base)


def test_transient_window_holds_then_releases(cfg, monkeypatch):
    cfg["transient_retry_window_seconds"] = 60
    clock = [1000.0]
    monkeypatch.setattr(fh.time, "monotonic", lambda: clock[0])
    agent = _agent(provider="openai-codex", model="gpt-5.6-sol")
    assert fh.transient_window_open(agent, FailoverReason.overloaded)
    clock[0] += 59
    assert fh.transient_window_open(agent, FailoverReason.server_error)
    clock[0] += 2
    assert not fh.transient_window_open(agent, FailoverReason.timeout)
    # A delivered call closes the window; the next outage gets a fresh one.
    fh.note_call_succeeded(agent)
    assert fh.transient_window_open(agent, FailoverReason.overloaded)


@pytest.mark.parametrize("reason", [FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit])
def test_rate_limits_and_quota_never_wait(cfg, reason):
    assert not fh.transient_window_open(_agent(), reason)


def test_window_restarts_for_a_different_route(cfg, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(fh.time, "monotonic", lambda: clock[0])
    agent = _agent()
    fh.transient_window_open(agent, FailoverReason.overloaded)
    clock[0] += 100
    agent.provider, agent.model = "custom", "ministral-8b-latest"
    assert fh.transient_window_open(agent, FailoverReason.overloaded)


def test_entry_too_small_compares_request_to_window(monkeypatch):
    monkeypatch.setattr(fh, "entry_context_length", lambda *a, **k: 131072)
    agent = _agent(_last_request_tokens=200_000)
    assert fh.entry_too_small(agent, "custom", "ministral-8b-latest")
    agent._last_request_tokens = 90_000
    assert not fh.entry_too_small(agent, "custom", "ministral-8b-latest")
    # The provider's last real prompt size floors the estimate.
    agent.context_compressor.last_prompt_tokens = 140_000
    assert fh.entry_too_small(agent, "custom", "ministral-8b-latest")


def test_unknown_window_never_skips(monkeypatch):
    monkeypatch.setattr(fh, "entry_context_length", lambda *a, **k: 0)
    assert not fh.entry_too_small(_agent(_last_request_tokens=10**7), "custom", "x")


def test_paid_lane_posts_once_on_entry_and_once_on_exit(cfg, monkeypatch):
    sent = []
    monkeypatch.setattr(fh, "_alert", sent.append)
    agent = _agent(provider="anthropic", model="claude-opus-5-5")
    fh.note_route_change(agent, new_model="gpt-5.6-sol", new_provider="openai-codex",
                         reason=FailoverReason.overloaded, leaving_primary=True)
    assert sent == []
    fh.note_route_change(agent, new_model="deepseek/deepseek-v4.1-flash", new_provider="openrouter",
                         reason=FailoverReason.format_error)
    for _ in range(3):
        fh.note_api_call(agent)
    # Re-activating the same paid model does not repeat the entry line.
    fh.note_route_change(agent, new_model="deepseek/deepseek-v4.1-flash", new_provider="openrouter")
    fh.note_route_change(agent, new_model="claude-opus-5-5", new_provider="anthropic")
    assert len(sent) == 2
    assert "on paid fallback deepseek/deepseek-v4.1-flash" in sent[0]
    assert "primary claude-opus-5-5 down: overloaded" in sent[0]
    assert "left paid fallback deepseek/deepseek-v4.1-flash" in sent[1] and "3 calls" in sent[1]


def test_any_openrouter_route_is_alerted():
    # OpenRouter is out of every fallback chain: even a :free SKU there is a misroute.
    assert fh.is_paid_route("openrouter", "meta-llama/llama-4:free")
    assert fh.is_paid_route("openrouter", "deepseek/deepseek-v4.1-flash")
    assert not fh.is_paid_route("custom", "ministral-8b-latest")


def test_aux_openrouter_alert_fires_once_per_model(cfg, monkeypatch):
    sent = []
    monkeypatch.setattr(fh, "_alert", sent.append)
    monkeypatch.setattr(fh, "_aux_openrouter_alerted", set())
    fh.note_aux_openrouter("google/gemini-3.6-flash")
    fh.note_aux_openrouter("google/gemini-3.6-flash")
    assert len(sent) == 1 and "auxiliary call routed to OpenRouter google/gemini-3.6-flash" in sent[0]


def _fallback_agent(monkeypatch, cfg, **kw):
    agent = _agent(_provider_fallback_active=True, _fallback_activated=True,
                   _fallback_chain=[{"provider": "openai-codex", "model": "gpt-5.6-sol"},
                                    {"provider": "custom", "model": "ministral-8b-latest", "base_url": "https://m"},
                                    {"provider": "custom", "model": "qwen-local", "base_url": "http://127.0.0.1:8080/v1"}],
                   _fallback_index=3, **kw)
    agent._fallback_probe_state = {"last_probe": time.monotonic(), "calls": 0}
    return agent


def test_probe_waits_for_cadence(cfg, monkeypatch):
    probes = []
    monkeypatch.setattr(fh, "probe_route", lambda *a, **k: probes.append(a) or (False, "down"))
    agent = _fallback_agent(monkeypatch, cfg)
    for _ in range(9):
        fh.note_api_call(agent)
        assert not fh.maybe_restore_midturn(agent)
    assert probes == []
    fh.note_api_call(agent)  # 10th call: due
    monkeypatch.setattr(fh, "entry_context_length", lambda *a, **k: 0)
    assert not fh.maybe_restore_midturn(agent)
    # Primary first, then each higher-ranked entry; the active entry is never probed.
    assert [p[:2] for p in probes] == [("anthropic", "claude-opus-5-5"), ("openai-codex", "gpt-5.6-sol"),
                                       ("custom", "ministral-8b-latest")]


def test_probe_due_on_wall_clock(cfg, monkeypatch):
    agent = _fallback_agent(monkeypatch, cfg)
    agent._fallback_probe_state["last_probe"] -= 181
    assert fh._probe_due(agent)


def test_primary_answering_restores_mid_turn(cfg, monkeypatch):
    monkeypatch.setattr(fh, "probe_route", lambda provider, model, **k: (provider == "anthropic", "x"))
    agent = _fallback_agent(monkeypatch, cfg, _rate_limited_until=10**12)

    def _restore():
        assert agent._rate_limited_until == 0  # a proven-live primary is not vetoed by a stale cooldown
        agent.provider, agent.model = "anthropic", "claude-opus-5-5"
        return True
    agent._restore_primary_runtime = _restore
    agent._fallback_probe_state["calls"] = 10
    assert fh.maybe_restore_midturn(agent)
    assert agent.model == "claude-opus-5-5"


def test_higher_ranked_entry_is_promoted_when_primary_stays_down(cfg, monkeypatch):
    monkeypatch.setattr(fh, "probe_route", lambda provider, model, **k: (provider == "openai-codex", "x"))
    monkeypatch.setattr(fh, "entry_context_length", lambda *a, **k: 0)
    agent = _fallback_agent(monkeypatch, cfg)
    agent._restore_primary_runtime = lambda: pytest.fail("primary is down")

    def _activate(reason=None):
        entry = agent._fallback_chain[agent._fallback_index]
        agent._fallback_index += 1
        agent.provider, agent.model = entry["provider"], entry["model"]
        return True
    agent._try_activate_fallback = _activate
    agent._fallback_probe_state["calls"] = 10
    assert fh.maybe_restore_midturn(agent)
    assert (agent.provider, agent.model, agent._fallback_index) == ("openai-codex", "gpt-5.6-sol", 1)


def test_probe_skips_routes_too_small_for_the_request(cfg, monkeypatch):
    probes = []
    monkeypatch.setattr(fh, "probe_route", lambda provider, model, **k: probes.append(model) or (False, "x"))
    monkeypatch.setattr(fh, "entry_context_length", lambda agent, provider, model, *a: 131072 if "ministral" in model else 400000)
    agent = _fallback_agent(monkeypatch, cfg, _last_request_tokens=180_000)
    agent._fallback_probe_state["calls"] = 10
    fh.maybe_restore_midturn(agent)
    assert "ministral-8b-latest" not in probes and "gpt-5.6-sol" in probes


def test_no_probe_on_primary(cfg, monkeypatch):
    monkeypatch.setattr(fh, "probe_route", lambda *a, **k: pytest.fail("no fallback active"))
    agent = _agent(_provider_fallback_active=False, _fallback_activated=False)
    agent._fallback_probe_state = {"last_probe": 0, "calls": 99}
    assert not fh.maybe_restore_midturn(agent)
