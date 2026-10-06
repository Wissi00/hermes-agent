"""Interval jobs re-anchor on completion; that slot must not miss the tick meant to fire it.

``next_run_at = completion + interval`` lands ``runtime`` seconds after the fixed-rate tick at
``dispatch + interval``, so an ``every 1m`` job taking 4s fired only every other tick (live:
usage-dashboard ran 30/60 times an hour, ~56s late each run). The slot is now snapped to the
nearest tick on the lattice given by ``last_dispatch.dispatched_at``.
"""

from datetime import datetime, timedelta, timezone

from cron.jobs import (
    INTERVAL_TICK_SNAP_LEAD_SECONDS,
    TICKER_INTERVAL_SECONDS,
    _advance_after_run,
    _snap_interval_next_to_tick,
)

T0 = datetime(2026, 10, 6, 16, 30, 39, 600000, tzinfo=timezone.utc)


def _job(minutes, dispatched_at=T0):
    job = {
        "id": "j1",
        "schedule": {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"},
        "repeat": {"times": None, "completed": 0},
        "state": "scheduled",
    }
    if dispatched_at is not None:
        job["last_dispatch"] = {"dispatched_at": dispatched_at.isoformat()}
    return job


def _next(job):
    return datetime.fromisoformat(job["next_run_at"]).astimezone(timezone.utc)


def test_one_minute_job_is_due_on_the_very_next_tick():
    finished = T0 + timedelta(seconds=4.5)
    job = _job(1)
    _advance_after_run(job, finished.isoformat())
    nxt = _next(job)
    next_tick = T0 + timedelta(seconds=TICKER_INTERVAL_SECONDS) - timedelta(seconds=0.2)  # jitter
    assert nxt <= next_tick, "slot must be due on the next tick, not the one after"
    assert nxt > finished
    assert nxt == T0 + timedelta(seconds=TICKER_INTERVAL_SECONDS - INTERVAL_TICK_SNAP_LEAD_SECONDS)


def test_long_run_rounds_to_nearest_tick_and_never_before_completion():
    finished = T0 + timedelta(seconds=40)  # target = T0+100s -> nearest tick T0+120s
    job = _job(1)
    _advance_after_run(job, finished.isoformat())
    assert _next(job) == T0 + timedelta(seconds=120 - INTERVAL_TICK_SNAP_LEAD_SECONDS)

    finished = T0 + timedelta(seconds=200)  # overran the interval: still after completion
    job = _job(1)
    _advance_after_run(job, finished.isoformat())
    assert _next(job) > finished


def test_hourly_job_moves_at_most_half_a_tick():
    finished = T0 + timedelta(minutes=5)
    job = _job(60)
    _advance_after_run(job, finished.isoformat())
    target = finished + timedelta(hours=1)
    assert abs((_next(job) - target).total_seconds()) <= TICKER_INTERVAL_SECONDS / 2 + INTERVAL_TICK_SNAP_LEAD_SECONDS


def test_without_dispatch_stamp_keeps_completion_anchor():
    finished = T0 + timedelta(seconds=4)
    job = _job(1, dispatched_at=None)
    _advance_after_run(job, finished.isoformat())
    assert _next(job) == finished + timedelta(minutes=1)


def test_future_dispatch_stamp_is_ignored():
    finished = T0
    nxt = (finished + timedelta(minutes=1)).isoformat()
    job = _job(1, dispatched_at=T0 + timedelta(minutes=5))
    assert _snap_interval_next_to_tick(job, nxt, finished.isoformat()) == nxt


def test_cron_kind_is_untouched():
    job = _job(1)
    job["schedule"] = {"kind": "cron", "expr": "*/5 * * * *"}
    _advance_after_run(job, (T0 + timedelta(seconds=4)).isoformat())
    assert _next(job).minute % 5 == 0 and _next(job).second == 0
