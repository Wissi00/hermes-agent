"""Per-model compaction trigger resolution: ``compression.model_thresholds`` (ratios) and
``compression.model_threshold_tokens`` (absolute input-token triggers).

Keys are substrings of the model name, optionally provider-scoped as ``"<provider>:<substr>"``; a key
containing a glob metacharacter (``*``, ``?``, ``[``) is matched as a glob anywhere in the name, so
``gpt-*-sol*`` covers ``gpt-6.1-sol``, ``openai/gpt-6-sol`` and future Sol tiers without catching
``solar-*``. The most specific match wins (literal characters, then provider scope).
"""

from __future__ import annotations

import logging
from fnmatch import fnmatchcase
from typing import Any

logger = logging.getLogger(__name__)

# Shipped default for ``compression.model_threshold_tokens``: the GPT Sol family compacts at a fixed
# 300K input tokens wherever its window leaves room for that (see ``model_trigger_tokens``).
DEFAULT_MODEL_THRESHOLD_TOKENS: dict[str, int] = {"gpt-*-sol*": 300_000}

# An absolute trigger only applies when it leaves at least 15% of the usable input window free;
# otherwise the ratio trigger (with its small-window floor and Codex autoraise) stays in charge.
ABSOLUTE_TRIGGER_MAX_WINDOW_RATIO = 0.85

_GLOB_CHARS = frozenset("*?[")
# (key, model, window) already warned about, so a rebuilt agent does not re-log the same verdict.
_WARNED_UNFIT: set[tuple[str, str, int]] = set()


def _model_threshold_key_rank(key: str, model: str, provider: str) -> "tuple[int, int] | None":
    """Match rank for one per-model key, or None when it does not apply.
    ``"<provider>:<substr>"`` keys apply only on that provider; bare keys apply on every route.
    The same slug means different windows on different routes (Codex caps Astra at 272K; OpenRouter
    serves the full window), so a bare ``astra: 0.85`` written for Codex silently leaks everywhere.
    Rank = (literal length, scoped): the most specific model match wins, scope breaks ties."""
    scope, sep, substr = key.partition(":")
    if sep and scope.strip().lower() != provider:
        return None
    pattern = substr if sep else key
    if _GLOB_CHARS.intersection(pattern):
        matched = fnmatchcase(model, f"*{pattern}*")
        literal = sum(1 for ch in pattern if ch not in _GLOB_CHARS)
    else:
        matched, literal = pattern in model, len(pattern)
    return (literal, 1 if sep else 0) if matched else None


def _best_key(model: str, mapping: dict | None, provider: str) -> str | None:
    if not mapping or not model:
        return None
    provider = (provider or "").strip().lower()
    ranked = ((_model_threshold_key_rank(key, model, provider), key) for key in mapping)
    best = max(((rank, key) for rank, key in ranked if rank is not None), default=None)
    return best[1] if best else None


def resolve_model_threshold(
    model: str, model_thresholds: dict[str, float] | None, default: float, provider: str = "",
) -> float:
    """Per-model threshold: longest matching ``model_thresholds`` key wins, else ``default``.
    A scoped key outranks a bare one of the same specificity. Module-level so plugin context
    engines can reuse it (re-exported from ``agent.context_compressor``)."""
    key = _best_key(model, model_thresholds, provider)
    return float(model_thresholds[key]) if key is not None and model_thresholds else default


def parse_model_threshold_tokens(raw: Any) -> dict[str, int]:
    """``compression.model_threshold_tokens`` → ``{key: tokens}``: user entries layered over the shipped
    default, the same under every loader (the gateway reads raw YAML, the CLI a defaults-merged copy).
    A non-positive value (``0``) disables a key, the shipped one included."""
    merged: dict[str, Any] = dict(DEFAULT_MODEL_THRESHOLD_TOKENS)
    if isinstance(raw, dict):
        merged.update({str(key): value for key, value in raw.items()})
    elif raw is not None:
        logger.warning("Invalid compression.model_threshold_tokens=%r; expected a mapping, ignoring it.", raw)
    parsed: dict[str, int] = {}
    for key, value in merged.items():
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            parsed[key] = value
        elif value != 0:
            logger.warning("Ignoring compression.model_threshold_tokens[%r]=%r: not a positive integer.", key, value)
    return parsed


def model_trigger_tokens(
    ratio_trigger: int, *, model: str, provider: str, mapping: dict[str, int] | None,
    effective_input_window: int,
) -> "tuple[int, dict | None]":
    """Apply a matching absolute trigger to the ratio-derived one.

    Returns ``(trigger, status)``; ``status`` is None when no key matches. A configured count that
    leaves less than 15% of the usable input window is NOT applied (compacting at, or past, the window
    would overflow the provider); the ratio trigger is kept and the status says why."""
    key = _best_key(model, mapping, provider)
    if key is None or not mapping:
        return ratio_trigger, None
    configured = int(mapping[key])
    ceiling = int(effective_input_window * ABSOLUTE_TRIGGER_MAX_WINDOW_RATIO)
    status = {"key": key, "configured": configured, "input_window": effective_input_window, "ceiling": ceiling}
    if configured <= ceiling:
        return configured, {**status, "applied": True, "trigger": configured}
    if (key, model, effective_input_window) not in _WARNED_UNFIT:
        _WARNED_UNFIT.add((key, model, effective_input_window))
        logger.warning(
            "compression.model_threshold_tokens[%r]=%d does not fit %s's %d-token input window "
            "(max %d); keeping the ratio trigger at %d tokens.",
            key, configured, model, effective_input_window, ceiling, ratio_trigger,
        )
    return ratio_trigger, {**status, "applied": False, "trigger": ratio_trigger, "reason": "window_too_small"}
