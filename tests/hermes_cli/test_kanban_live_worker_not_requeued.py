"""A live worker is never requeued as "exited", so the dispatcher never spawns a duplicate.

Incident (t_a3485a28, October 5): run 124's worker (pid 4047) was still running when the
dead-worker sweep booked it ``rate_limited`` and requeued the card; run 127 was spawned beside it.
Two faults combined:

* macOS re-reads of a live process's psutil ``create_time`` drift by ~1 s (``kern.boottime``
  adjustment), and ``_pid_recycled`` compared the spawn fingerprint by exact string equality, so
  the live worker read as a recycled PID, i.e. dead;
* the sweep then classified the "death" from the LAST exit trailer in the append-mode task log,
  which was run 122's ``rc=75`` — a previous worker's rate-limit exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway.status import get_process_start_time
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query

# One second of start-time drift, in fingerprint units (Linux ticks / psutil centiseconds).
ONE_SECOND_DRIFT = 100


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kbd._recent_worker_exits.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def lingering_worker():
    """A real live child standing in for the worker that outlived its 'rate-limited' booking."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        deadline = time.time() + 5
        while get_process_start_time(proc.pid) is None and time.time() < deadline:
            time.sleep(0.05)
        yield proc
    finally:
        proc.kill()
        proc.wait()


def _drifted_fingerprint(pid: int) -> str:
    """The spawn fingerprint as macOS recorded it: same process, start time read 1 s apart."""
    live = kbd._process_fingerprint(pid)
    assert live is not None
    epoch, start = live.rsplit("|", 1)
    return f"{epoch}|{int(start) + ONE_SECOND_DRIFT}"


def _running_with_stale_rate_limit_trailer(conn, pid: int, fingerprint: "str | None") -> str:
    """Card claimed by ``pid`` whose log still ends with the PREVIOUS run's ``rc=75`` trailer."""
    tid = kb.create_task(conn, title="candidatures move", assignee="coder")
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:dispatcher")
    prev_run = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    old = int(time.time()) - 150
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid=?, worker_started_at=?, started_at=? WHERE id=?",
            (pid, fingerprint, old, tid),
        )
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        # Previous worker: legacy untagged trailer AND a tagged one for an earlier run.
        f.write(f"older run output\n\n{KANBAN_WORKER_EXIT_TRAILER}75\n")
        f.write(f"older run output\n\n{KANBAN_WORKER_EXIT_TRAILER}75 run={int(prev_run) + 1000}\n")
        f.write("current worker is still writing tool output...\n")
    return tid


def test_drifted_start_time_still_identifies_the_live_worker(lingering_worker):
    fingerprint = _drifted_fingerprint(lingering_worker.pid)
    assert kbd._pid_recycled(lingering_worker.pid, fingerprint) is False
    assert kbd._worker_alive(lingering_worker.pid, fingerprint) is True


def test_another_boot_or_far_start_time_is_still_foreign(lingering_worker):
    live = kbd._process_fingerprint(lingering_worker.pid)
    epoch, start = live.rsplit("|", 1)
    assert kbd._pid_recycled(lingering_worker.pid, f"deadbeef-boot:1|{start}") is True
    assert kbd._pid_recycled(lingering_worker.pid, f"{epoch}|{int(start) - 10_000}") is True


def test_live_worker_is_not_requeued_and_card_cannot_be_reclaimed(board, lingering_worker):
    """Lingering child + stale rate-limit trailer: the sweep must leave the card running, book
    nothing, and a second claim (the dispatcher's duplicate spawn) must fail."""
    pid = lingering_worker.pid
    tid = _running_with_stale_rate_limit_trailer(board, pid, _drifted_fingerprint(pid))

    crashed = kbd.detect_crashed_workers(board)

    assert crashed == []
    assert kbd.detect_crashed_workers._last_rate_limited == []
    task = kb.get_task(board, tid)
    assert task.status == "running"
    assert task.worker_pid == pid
    kinds = [r["kind"] for r in board.execute(
        "SELECT kind FROM task_events WHERE task_id=?", (tid,)).fetchall()]
    assert "rate_limited" not in kinds and "crashed" not in kinds
    assert kb.claim_task(board, tid, claimer="other-host:dup") is None


def test_dead_worker_ignores_a_previous_runs_rate_limit_trailer(board, monkeypatch):
    """Once the worker really is dead, a trailer left by an earlier run must not book this run
    as a rate-limit requeue: without its own trailer it is a plain crash."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    tid = _running_with_stale_rate_limit_trailer(board, 70123, None)

    kbd.detect_crashed_workers(board)

    run = board.execute(
        "SELECT outcome, error FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert run["outcome"] == "crashed"
    assert "rate-limited" not in (run["error"] or "")


def test_dead_worker_books_its_own_tagged_rate_limit_trailer(board, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    tid = _running_with_stale_rate_limit_trailer(board, 70124, None)
    run_id = board.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    with open(kb.worker_log_path(tid), "a", encoding="utf-8") as f:
        f.write(f"\n{KANBAN_WORKER_EXIT_TRAILER}75 run={run_id}\n")

    kbd.detect_crashed_workers(board)

    run = board.execute(
        "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert run["outcome"] == "rate_limited"


def test_exit_trailer_carries_the_run_id(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "124")
    with pytest.raises(SystemExit):
        exit_single_query(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert f"{KANBAN_WORKER_EXIT_TRAILER}75 run=124" in capsys.readouterr().err
