"""Fallback hygiene: a turn stays on a fallback route no longer than it must.

Three rules shared by the fallback walk (``chat_completion_helpers.try_activate_fallback``),
the error router (``turn_recovery`` / ``turn_api_error``) and the turn loop
(``conversation_loop``):

* **Retry before falling.** A transient failure (overloaded / 5xx / timeout) retries the SAME
  route with backoff for ``fallback.transient_retry_window_seconds`` before the chain advances.
  Rate limits and quota exhaustion never enter the window: they advance immediately.
* **Context-aware chain.** A chain entry whose context window is smaller than the request being
  sent is skipped instead of being handed a request it can only reject.
* **Mid-turn restore.** Kanban workers and cron jobs run ONE long turn, so the turn-start
  ``restore_primary_runtime`` never fires for them. While a provider fallback is active, between
  API calls, the primary (then every chain entry ranked above the active one) is probed with a
  tiny request at most every ``fallback.restore_probe_interval_seconds`` of wall time or every
  ``fallback.restore_probe_every_calls`` calls, whichever comes first; the first route that
  answers is switched back in.

Plus the paid-lane guard: engaging an OpenRouter model that is not a ``:free`` SKU posts one line
to ``fallback.paid_lane_alert_channel`` (Discord channel id) and another when the route leaves it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Optional

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

TRANSIENT_REASONS = frozenset({FailoverReason.overloaded, FailoverReason.server_error, FailoverReason.timeout})

_DEFAULTS = {
    "transient_retry_window_seconds": 60,
    "restore_probe_interval_seconds": 180,
    "restore_probe_every_calls": 10,
    "paid_lane_alert_channel": "",
}
_PROBE_TIMEOUT_S = 25.0


def settings() -> dict:
    """The ``fallback:`` config block merged over the defaults (a load failure keeps defaults)."""
    merged = dict(_DEFAULTS)
    try:
        from hermes_cli.config import load_config
        block = (load_config() or {}).get("fallback") or {}
        if isinstance(block, dict):
            merged.update({k: v for k, v in block.items() if k in _DEFAULTS and v is not None})
    except Exception:
        logger.debug("fallback settings: config load failed; using defaults", exc_info=True)
    return merged


def _num(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Retry before falling ────────────────────────────────────────────────────────────────────


def transient_window_open(agent, reason) -> bool:
    """True while a transient failure of the CURRENT route is inside its same-route retry window.

    The window opens on the first transient failure of a (provider, model) pair and is closed by
    ``note_call_succeeded``; a different route failing restarts it."""
    if reason not in TRANSIENT_REASONS:
        return False
    window = _num(settings().get("transient_retry_window_seconds"), 60.0)
    if window <= 0:
        return False
    route = (str(getattr(agent, "provider", "") or ""), str(getattr(agent, "model", "") or ""))
    now = time.monotonic()
    started = getattr(agent, "_transient_failure_started", None)
    if not started or started[0] != route:
        agent._transient_failure_started = (route, now)
        logger.info("Transient %s on %s/%s: retrying the same route for up to %.0f s before falling back",
                    getattr(reason, "value", reason), route[0], route[1], window)
        return True
    if now - started[1] < window:
        return True
    logger.warning("Transient failures on %s/%s outlasted the %.0f s retry window: advancing the fallback chain",
                   route[0], route[1], window)
    return False


def note_call_succeeded(agent) -> None:
    """A delivered response closes the transient window and feeds the probe call counter."""
    agent._transient_failure_started = None


# ── Context-aware chain ─────────────────────────────────────────────────────────────────────


def current_request_tokens(agent) -> int:
    """Size of the request in flight: the assembled request's pressure figure, floored at the
    provider's last real prompt size."""
    tokens = int(getattr(agent, "_last_request_tokens", 0) or 0)
    compressor = getattr(agent, "context_compressor", None)
    real = int(getattr(compressor, "last_prompt_tokens", 0) or 0) if compressor is not None else 0
    return max(tokens, real)


def entry_context_length(agent, provider: str, model: str, base_url: str = "", api_key: str = "") -> int:
    """Context window of a chain entry; 0 when it cannot be resolved (never skip on ignorance)."""
    try:
        from agent.model_metadata import get_model_context_length
        return int(get_model_context_length(
            model, base_url=base_url or "", api_key=api_key if isinstance(api_key, str) else "",
            provider=provider, custom_providers=getattr(agent, "_custom_providers", None)) or 0)
    except Exception:
        logger.debug("context length lookup failed for %s/%s", provider, model, exc_info=True)
        return 0


def entry_too_small(agent, provider: str, model: str, base_url: str = "", api_key: str = "") -> bool:
    need = current_request_tokens(agent)
    if need <= 0:
        return False
    window = entry_context_length(agent, provider, model, base_url, api_key)
    if window and need > window:
        logger.warning("Fallback skip: %s/%s context window %d is smaller than the current request (~%d tokens)",
                       provider, model, window, need)
        return True
    return False


# ── Paid-lane guard ─────────────────────────────────────────────────────────────────────────


def is_paid_route(provider: str, model: str) -> bool:
    """Every OpenRouter route counts, ``:free`` SKUs included: OpenRouter is out of the fallback
    chains, so any call that lands there is a misroute worth an alert."""
    return (provider or "").strip().lower() == "openrouter"


def _profile_name() -> str:
    try:
        from hermes_constants import get_hermes_home, profile_name_for_home
        return profile_name_for_home(get_hermes_home()) or "default"
    except Exception:
        return "default"


def _task_label(agent) -> str:
    task = os.environ.get("HERMES_KANBAN_TASK") or ""
    return task or f"session {getattr(agent, 'session_id', '') or '?'}"


def _bot_token() -> str:
    """The Discord bot token: the profile's secret scope first, then the root install's ``.env``
    (named profiles do not carry the gateway's bot token, the alert channel belongs to it)."""
    try:
        from agent.secret_scope import get_secret
        token = get_secret("DISCORD_BOT_TOKEN") or ""
    except Exception:
        token = ""
    if token:
        return token
    try:
        from dotenv import dotenv_values
        from hermes_constants import get_default_hermes_root
        return str(dotenv_values(get_default_hermes_root() / ".env").get("DISCORD_BOT_TOKEN") or "")
    except Exception:
        return ""


def _post_discord(channel: str, text: str) -> None:
    token = _bot_token()
    if not token:
        logger.warning("Paid-lane alert not posted (no DISCORD_BOT_TOKEN): %s", text)
        return
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel}/messages",
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                 "User-Agent": "hermes-fallback-guard/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        logger.info("Paid-lane alert posted to Discord channel %s: %s", channel, text)
    except Exception as exc:
        logger.warning("Paid-lane alert to Discord channel %s failed (%s): %s", channel, exc, text)


def _alert(text: str) -> Optional[threading.Thread]:
    channel = str(settings().get("paid_lane_alert_channel") or "").strip()
    if not channel:
        logger.info("Paid-lane alert (no fallback.paid_lane_alert_channel configured): %s", text)
        return None
    worker = threading.Thread(target=_post_discord, args=(channel, text), name="paid-lane-alert", daemon=True)
    worker.start()
    return worker


def _reason_text(reason) -> str:
    return str(getattr(reason, "value", reason) or "unknown")


def note_route_change(agent, *, new_model: str, new_provider: str, reason=None, leaving_primary: bool = False) -> None:
    """Every runtime switch (fallback activation, restore, promotion) reports here: it arms the
    mid-turn probe clock and fires the paid-lane entry/exit lines exactly once per stay."""
    now = time.monotonic()
    if leaving_primary:
        agent._fallback_origin_reason = _reason_text(reason)
    agent._fallback_probe_state = {"last_probe": now, "calls": 0}
    paid = getattr(agent, "_paid_lane", None)
    entering_paid = is_paid_route(new_provider, new_model)
    if paid and (not entering_paid or paid.get("model") != new_model):
        minutes = (now - paid["since"]) / 60.0
        _alert(f"{_profile_name()} worker left paid fallback {paid['model']} ({_task_label(agent)}) after "
               f"{minutes:.1f} min, {paid['calls']} calls; now on {new_model} via {new_provider}")
        agent._paid_lane = None
    if entering_paid and not getattr(agent, "_paid_lane", None):
        primary = str((getattr(agent, "_primary_runtime", None) or {}).get("model") or "?")
        origin = getattr(agent, "_fallback_origin_reason", None) or _reason_text(reason)
        agent._paid_lane = {"model": new_model, "since": now, "calls": 0}
        _alert(f"{_profile_name()} worker on paid fallback {new_model} (task {_task_label(agent)}), "
               f"primary {primary} down: {origin}")


_aux_openrouter_alerted: set = set()
_main_route_resolution = threading.local()


class main_route_resolution:
    """Marks client resolution done for the MAIN chain (fallback walk, probes): those report
    through note_route_change, so the auxiliary OpenRouter alert stays quiet inside."""

    def __enter__(self):
        _main_route_resolution.depth = getattr(_main_route_resolution, "depth", 0) + 1

    def __exit__(self, *exc):
        _main_route_resolution.depth -= 1
        return False


def note_aux_openrouter(model: str) -> None:
    """An auxiliary call resolved to OpenRouter: one alert per model per process."""
    if getattr(_main_route_resolution, "depth", 0) > 0 or model in _aux_openrouter_alerted:
        return
    _aux_openrouter_alerted.add(model)
    _alert(f"{_profile_name()} auxiliary call routed to OpenRouter {model} "
           f"(task {os.environ.get('HERMES_KANBAN_TASK') or '-'}): OpenRouter is not in any fallback chain")


def note_api_call(agent) -> None:
    """Count calls served while on a fallback (probe cadence + paid-lane exit line)."""
    state = getattr(agent, "_fallback_probe_state", None)
    if state is not None:
        state["calls"] = int(state.get("calls", 0)) + 1
    paid = getattr(agent, "_paid_lane", None)
    if paid is not None:
        paid["calls"] += 1


# ── Mid-turn restore ────────────────────────────────────────────────────────────────────────


def probe_route(provider: str, model: str, *, base_url: str = "", api_key: Any = None, api_mode: str = "") -> tuple[bool, str]:
    """One tiny request against a route. Returns (ok, detail); never raises."""
    try:
        from agent.auxiliary_client import resolve_provider_client
        with main_route_resolution():
            client, resolved = resolve_provider_client(
                provider, model=model, explicit_base_url=base_url or None,
                explicit_api_key=api_key or None, api_mode=api_mode or None)
        if client is None:
            return False, "provider not configured"
        client.chat.completions.create(
            model=resolved or model, messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16, timeout=_PROBE_TIMEOUT_S)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def _probe_due(agent) -> bool:
    state = getattr(agent, "_fallback_probe_state", None)
    if state is None:
        state = agent._fallback_probe_state = {"last_probe": time.monotonic(), "calls": 0}
    cfg = settings()
    interval = _num(cfg.get("restore_probe_interval_seconds"), 180.0)
    every = int(_num(cfg.get("restore_probe_every_calls"), 10))
    if interval <= 0 and every <= 0:
        return False
    elapsed = time.monotonic() - state["last_probe"]
    return (interval > 0 and elapsed >= interval) or (every > 0 and state["calls"] >= every)


def _primary_candidate(agent) -> Optional[dict]:
    rt = getattr(agent, "_primary_runtime", None) or {}
    provider = str(rt.get("provider") or "").strip().lower()
    model = str(rt.get("model") or "").strip()
    if not provider or not model:
        return None
    custom = provider == "custom" or provider.startswith("custom:")
    return {"provider": provider, "model": model, "primary": True,
            "base_url": str(rt.get("base_url") or "") if custom else "",
            "api_key": rt.get("api_key") if custom and isinstance(rt.get("api_key"), str) else None,
            "api_mode": str(rt.get("api_mode") or "") if custom else "",
            "context_length": int(rt.get("compressor_context_length") or 0)}


def _chain_candidates(agent) -> list[dict]:
    """Chain entries ranked above the active one (``_fallback_index`` points past it)."""
    chain = list(getattr(agent, "_fallback_chain", None) or [])
    active = int(getattr(agent, "_fallback_index", 0) or 0) - 1
    out = []
    for index, entry in enumerate(chain[:max(active, 0)]):
        provider = (entry.get("provider") or "").strip().lower()
        model = (entry.get("model") or "").strip()
        if not provider or not model:
            continue
        try:
            from hermes_cli.fallback_config import resolve_entry_api_key
            api_key = resolve_entry_api_key(entry)
        except Exception:
            api_key = None
        out.append({"provider": provider, "model": model, "primary": False, "index": index,
                    "base_url": (entry.get("base_url") or "").strip(), "api_key": api_key,
                    "api_mode": (entry.get("api_mode") or "").strip(), "context_length": 0})
    return out


def _same_route(agent, cand: dict) -> bool:
    return (cand["provider"] == str(getattr(agent, "provider", "") or "").strip().lower()
            and cand["model"] == str(getattr(agent, "model", "") or "").strip())


def maybe_restore_midturn(agent) -> bool:
    """Between API calls: while a provider fallback is active and a probe is due, probe the primary
    then each higher-ranked chain entry; switch to the first one that answers."""
    if not getattr(agent, "_provider_fallback_active", False) or not getattr(agent, "_fallback_activated", False):
        return False
    if not _probe_due(agent):
        return False
    agent._fallback_probe_state = {"last_probe": time.monotonic(), "calls": 0}
    from_model, from_provider = str(agent.model), str(agent.provider)
    candidates = [c for c in [_primary_candidate(agent), *_chain_candidates(agent)] if c]
    need = current_request_tokens(agent)
    for cand in candidates:
        if _same_route(agent, cand):
            continue
        from agent.fallback_cooldown import _is_entitlement_rejected
        if _is_entitlement_rejected(agent, cand["provider"], cand["model"]):
            continue
        window = cand["context_length"] or entry_context_length(
            agent, cand["provider"], cand["model"], cand["base_url"], cand["api_key"] or "")
        if need and window and need > window:
            logger.info("Mid-turn probe skip: %s/%s window %d < request ~%d tokens",
                        cand["provider"], cand["model"], window, need)
            continue
        ok, detail = probe_route(cand["provider"], cand["model"], base_url=cand["base_url"],
                                 api_key=cand["api_key"], api_mode=cand["api_mode"])
        if not ok:
            logger.info("Mid-turn probe: %s/%s still unavailable (%s)", cand["provider"], cand["model"], detail)
            continue
        if cand["primary"]:
            # The probe just proved the primary answers: a stale cooldown must not veto it.
            agent._rate_limited_until = 0
            if agent._restore_primary_runtime():
                logger.info("Primary restored: %s → %s (mid-turn, %s)", from_model, agent.model, agent.provider)
                return True
            logger.info("Mid-turn probe: primary %s answered but restore was refused", cand["model"])
            continue
        agent._fallback_index = cand["index"]
        if agent._try_activate_fallback() and _same_route(agent, cand):
            logger.info("Fallback promoted: %s → %s (mid-turn, %s)", from_model, agent.model, agent.provider)
            return True
        logger.info("Mid-turn probe: %s answered but activation failed", cand["model"])
        return False
    return False
