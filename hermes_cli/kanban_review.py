"""Bounded implementation/review/repair lifecycle.

This module is deliberately a thin policy layer over :mod:`kanban_db`.  The
existing DB transitions remain the source of truth and historical cards are
never rewritten.  Cards opting into this policy carry ``bounded_review`` in
run metadata; legacy review cards retain their old semantics.
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import kanban_db as kb

REVIEW_OUTCOMES = frozenset({
    "accepted", "changes_requested", "infrastructure_blocked", "authority_blocked",
})
MAX_REPAIRS = 1
MAX_REVIEWS = 2
# SQLite serializes writers, but the policy check and transition span two
# transactions in the legacy review primitive. Serialize this lifecycle in a
# process so two verifier callbacks cannot overshoot a bound.
_COUNTER_LOCK = threading.Lock()

@dataclass(frozen=True)
class ReviewPolicy:
    max_repairs: int = MAX_REPAIRS
    max_reviews: int = MAX_REVIEWS


def _columns(conn):
    return {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}


def _event_payload(conn, task_id: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    try:
        value = json.loads(row["payload"]) if row and row["payload"] else {}
    except (TypeError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _bounded_counts(conn, task_id: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT review_count, repair_count FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    return (int(row["review_count"] or 0), int(row["repair_count"] or 0)) if row else (0, 0)


def request_implementation_review(
    conn, task_id: str, *, implementation_commit: str,
    reviewer: str, implementation_session_id: str,
    review_workspace: str, summary: str, metadata: Optional[dict] = None,
    expected_run_id: Optional[int] = None, policy: ReviewPolicy = ReviewPolicy(),
) -> bool:
    """Hand an implementation to a separate, clean verifier (at most twice)."""
    if not implementation_commit or not implementation_session_id or not review_workspace:
        return False
    if not _clean_at_commit(review_workspace, implementation_commit):
        return False
    task = kb.get_task(conn, task_id)
    if task is None or task.assignee == reviewer:
        return False
    md = dict(metadata or {})
    md.update({"bounded_review": True, "implementation_commit": implementation_commit,
               "implementation_session_id": implementation_session_id,
               "review_workspace": review_workspace})
    with _COUNTER_LOCK:
        reviews, repairs = _bounded_counts(conn, task_id)
        if reviews >= policy.max_reviews or repairs > policy.max_repairs:
            return False
        ok = kb.request_review(conn, task_id, summary=summary, reviewer=reviewer,
                               metadata=md, expected_run_id=expected_run_id)
        if not ok:
            return False
        # The policy flag and provenance are durable task state, rather than
        # relying on lossy event payloads. The guarded update makes increments
        # deterministic and keeps legacy DBs readable after migration.
        with kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET bounded_review=1, review_count=review_count+1, "
                "implementation_commit=?, implementation_session_id=?, review_workspace=? "
                "WHERE id=? AND review_count=?",
                (implementation_commit, implementation_session_id, review_workspace,
                 task_id, reviews),
            )
            if cur.rowcount != 1:
                return False
        return True


def _clean_at_commit(worktree: str, commit: str) -> bool:
    """Require a real git worktree at the declared implementation commit."""
    p = Path(worktree)
    if not p.is_dir() or not commit:
        return False
    try:
        head = subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(p), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return head == commit and not status.strip()


def submit_review_outcome(
    conn, task_id: str, *, outcome: str, summary: str,
    expected_run_id: Optional[int] = None, reviewer_session_id: Optional[str] = None,
    policy: ReviewPolicy = ReviewPolicy(),
) -> tuple[bool, str]:
    """Apply one structured verifier outcome with bounded transitions."""
    if outcome not in REVIEW_OUTCOMES:
        return False, "invalid outcome"
    task = kb.get_task(conn, task_id)
    if task is None or task.status != "running" or task.current_run_id is None:
        return False, "review is not active"
    if not reviewer_session_id:
        return False, "reviewer session is required"
    if reviewer_session_id == task.implementation_session_id:
        return False, "reviewer must use a separate session"
    if not getattr(task, "bounded_review", False):
        return False, "task is not opted into bounded review"
    review_payload = _event_payload(conn, task_id, "review_requested")
    expected_reviewer = review_payload.get("reviewer")
    if expected_reviewer and task.assignee != expected_reviewer:
        return False, "reviewer provenance mismatch"
    claimed_payload = _event_payload(conn, task_id, "claimed")
    if claimed_payload.get("source_status") != "review":
        return False, "run was not claimed from review"
    reviews, repairs = _bounded_counts(conn, task_id)
    if reviews > policy.max_reviews:
        return False, "review limit exceeded"
    if outcome == "changes_requested":
        if repairs >= policy.max_repairs:
            return False, "repair allowance exhausted"
        ok, detail = kb.request_changes(conn, task_id, reason=summary, expected_run_id=expected_run_id)
        if ok:
            with _COUNTER_LOCK, kb.write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET repair_count=repair_count+1 "
                    "WHERE id=? AND repair_count < ?",
                    (task_id, policy.max_repairs),
                )
                if cur.rowcount != 1:
                    return False, "repair allowance exhausted"
            return True, detail or "repair requested"
        return False, detail or "changes request failed"
    if outcome == "accepted":
        if not kb.complete_task(conn, task_id, summary=summary, result="accepted", expected_run_id=expected_run_id):
            return False, "acceptance failed"
        return True, "done"
    # Blockers do not consume repair allowance. Authority decisions go to triage;
    # infrastructure failures remain a retryable blocked review.
    with kb.write_txn(conn):
        row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row or (expected_run_id is not None and row["current_run_id"] != expected_run_id):
            return False, "run_id mismatch"
        destination = "triage" if outcome == "authority_blocked" else "blocked"
        conn.execute("UPDATE tasks SET status=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                     (destination, task_id))
        run_id = kb._end_run(conn, task_id, outcome=outcome, status=destination, summary=summary)
        kb._append_event(conn, task_id, outcome, {"outcome": outcome, "summary": summary}, run_id=run_id)
    return True, destination


def verifier_can_mutate_tracked_source(worktree: str) -> bool:
    """Return false for verifier contexts; tracked source is read-only."""
    return False


def verifier_workspace_snapshot(worktree: str) -> dict[str, Any]:
    """Read-only evidence used by a verifier, without running mutating git ops."""
    p = Path(worktree)
    if not p.is_dir():
        raise ValueError("review worktree does not exist")
    head = subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(p), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    return {
        "head": head,
        "dirty": bool(status),
        "tracked_source_read_only": True,
        "untracked_files": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
    }


def ensure_verifier_profile(name: str = "verifier") -> Path:
    """Create an internal no-gateway/no-personality/no-model verifier profile."""
    from . import profiles
    if profiles.profile_exists(name):
        return profiles.get_profile_dir(name)
    path = profiles.create_profile(name, no_alias=True, no_skills=True,
                                   description="Independent read-only implementation verifier")
    (path / "VERIFIER.md").write_text(
        "Inspect only. Do not edit tracked source, delegate, publish, or mutate Kanban.\n",
        encoding="utf-8",
    )
    return path
