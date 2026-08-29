"""Deterministic execution preflight and repository write coordination.

This module is deliberately independent of model/provider code.  Callers can
run it before constructing an execution request; a failed preflight is an
infrastructure failure, not a model failure and must not be retried unchanged.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class PreflightEvidence:
    """Machine-readable evidence for one repository preflight."""

    workspace: str
    repository: str
    base_revision: str
    head_revision: str
    branch: str


class PreflightError(RuntimeError):
    """An execution cannot safely start because its infrastructure is invalid."""

    def __init__(self, evidence: str):
        super().__init__(evidence)
        self.evidence = evidence


class LockContention(RuntimeError):
    """The repository is currently owned by another implementation worker."""


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise PreflightError(
            f"preflight git {' '.join(args)} failed (exit {result.returncode}): {detail}"
        )
    return result.stdout.strip()


def repository_preflight(
    workspace: str | Path,
    *,
    repository: str | Path | None = None,
    base_revision: str,
    expected_branch: str | None = None,
    expected_remote: str | None = None,
    expected_task_id: str | None = None,
    expected_origin_main: str | None = None,
    require_clean: bool = False,
) -> PreflightEvidence:
    """Validate the canonical repository and task-owned checkout.

    Every failure preserves the exact command evidence in ``PreflightError``.
    No mutating git command is run here.  ``base_revision`` is required rather
    than inferred so a caller cannot silently execute against a moving base.
    """
    ws = Path(workspace).expanduser().resolve(strict=False)
    if not ws.is_dir():
        raise PreflightError(f"preflight workspace missing: {ws}")
    if not base_revision or not base_revision.strip():
        raise PreflightError("preflight base revision missing")
    actual_repo = Path(_git(ws, "rev-parse", "--show-toplevel")).resolve()
    if not actual_repo.is_dir() or not (actual_repo / ".git").exists():
        raise PreflightError(f"preflight repository invalid: {actual_repo}")
    if expected_remote is not None:
        remote = _git(ws, "remote", "get-url", "origin")
        if remote != expected_remote:
            raise PreflightError(f"preflight origin mismatch: expected {expected_remote}, got {remote}")
    if expected_origin_main is not None:
        origin_main = _git(ws, "rev-parse", "origin/main")
        if origin_main != expected_origin_main:
            raise PreflightError(
                f"preflight origin/main mismatch: expected {expected_origin_main}, got {origin_main}"
            )
    if expected_task_id is not None:
        marker = ws / ".hermes-task-id"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected_task_id:
            raise PreflightError(f"preflight task ownership mismatch: expected {expected_task_id} at {marker}")
    worktrees = _git(actual_repo, "worktree", "list", "--porcelain")
    entries = worktrees.split("\n\n")
    matches = [e for e in entries if f"worktree {ws}" in e]
    if len(matches) != 1:
        raise PreflightError(f"preflight worktree metadata mismatch: {ws}")
    canonical_repo = (
        Path(repository).expanduser().resolve(strict=False)
        if repository is not None
        else actual_repo
    )
    if actual_repo != canonical_repo:
        raise PreflightError(
            f"preflight canonical repository mismatch: expected {canonical_repo}, "
            f"got {actual_repo}"
        )
    head = _git(ws, "rev-parse", "HEAD")
    # Verify the requested revision is a real object, without accepting a
    # branch name or silently resolving it to another object.
    resolved_base = _git(ws, "rev-parse", "--verify", f"{base_revision}^{{commit}}")
    if resolved_base != base_revision:
        raise PreflightError(
            f"preflight base revision mismatch: expected {base_revision}, "
            f"resolved {resolved_base}"
        )
    branch = _git(ws, "branch", "--show-current")
    if expected_branch and branch != expected_branch:
        raise PreflightError(
            f"preflight branch mismatch: expected {expected_branch}, got {branch or '<detached>'}"
        )
    if require_clean:
        dirty = _git(ws, "status", "--porcelain")
        if dirty:
            raise PreflightError(f"preflight workspace not clean: {dirty}")
    return PreflightEvidence(str(ws), str(canonical_repo), base_revision, head, branch)


@contextlib.contextmanager
def repository_implementation_lock(
    repository: str | Path, *, read_only: bool = False, blocking: bool = True
) -> Iterator[None]:
    """Serialize implementation work per repository; read-only work bypasses it.

    The lock is advisory and process-safe on POSIX.  Non-POSIX callers retain
    the same API and rely on the repository's own serialization guarantees.
    """
    if read_only:
        yield
        return
    lock_path = Path(repository).expanduser().resolve() / ".git" / "hermes-implementation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError:
                handle.close()
                raise LockContention(str(lock_path))
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def preflight_failure_result(error: PreflightError) -> dict[str, object]:
    """Return a stable, non-retryable result for an infrastructure failure."""
    return {
        "status": "infrastructure_failed",
        "error": error.evidence,
        "api_calls": 0,
        "retryable": False,
        "repair_consumed": False,
    }
