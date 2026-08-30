"""Tests for repository execution preflight and implementation serialization."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent.reliability import (
    LockContention,
    PreflightError,
    preflight_failure_result,
    repository_implementation_lock,
    repository_preflight,
)


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    (repo / "README").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, revision


def test_preflight_returns_canonical_evidence(tmp_path: Path) -> None:
    repo, revision = _repo(tmp_path)
    evidence = repository_preflight(repo, repository=repo, base_revision=revision, expected_branch="main")
    assert evidence.repository == str(repo.resolve())
    assert evidence.base_revision == revision
    assert evidence.head_revision == revision
    assert evidence.branch == "main"


def test_preflight_failure_is_exact_and_zero_call(tmp_path: Path) -> None:
    repo, revision = _repo(tmp_path)
    with pytest.raises(PreflightError) as caught:
        repository_preflight(repo, repository=repo, base_revision=revision, expected_branch="wrong")
    result = preflight_failure_result(caught.value)
    assert result["api_calls"] == 0
    assert result["retryable"] is False
    assert result["repair_consumed"] is False
    assert result["error"] == caught.value.evidence
    assert "expected wrong" in caught.value.evidence


def test_implementation_lock_serializes_writers(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with repository_implementation_lock(repo):
            entered.set()
            release.wait(2)

    def second() -> None:
        entered.wait(2)
        with repository_implementation_lock(repo):
            second_entered.set()

    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    assert entered.wait(2)
    b.start()
    assert not second_entered.wait(0.1)
    release.set()
    a.join(2)
    b.join(2)
    assert second_entered.is_set()


def _add_worktree(repo: Path, path: Path) -> None:
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", str(path), "HEAD"],
        check=True,
    )


def test_lock_resolves_to_shared_common_dir_across_sibling_worktrees(tmp_path: Path) -> None:
    import subprocess

    repo, _ = _repo(tmp_path)
    wt1 = tmp_path / "wt1"
    wt2 = tmp_path / "wt2"
    _add_worktree(repo, wt1)
    _add_worktree(repo, wt2)

    common_dir = subprocess.check_output(
        ["git", "-C", str(wt1), "rev-parse", "--git-common-dir"], text=True
    ).strip()
    common = Path(common_dir)
    if not common.is_absolute():
        common = (wt1 / common).resolve()
    expected_lock = common.resolve() / "hermes-implementation.lock"

    captured: dict[str, Path] = {}

    def capture(worktree: Path) -> None:
        # Reach into one lock acquisition to observe the resolved lock path.
        original_open = Path.open

        def spy_open(self, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
            captured.setdefault("path", self)
            return original_open(self, mode, *args, **kwargs)

        Path.open = spy_open  # type: ignore[method-assign]
        try:
            with repository_implementation_lock(worktree):
                pass
        finally:
            Path.open = original_open  # type: ignore[method-assign]

    capture(wt1)
    capture(wt2)
    assert captured["path"] == expected_lock


def test_sibling_worktrees_contend_on_one_lock(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    wt1 = tmp_path / "wt1"
    wt2 = tmp_path / "wt2"
    _add_worktree(repo, wt1)
    _add_worktree(repo, wt2)

    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with repository_implementation_lock(wt1):
            entered.set()
            release.wait(2)

    def second() -> None:
        entered.wait(2)
        with repository_implementation_lock(wt2):
            second_entered.set()

    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    assert entered.wait(2)
    b.start()
    assert not second_entered.wait(0.1)
    release.set()
    a.join(2)
    b.join(2)
    assert second_entered.is_set()


def test_unrelated_repositories_do_not_collide(tmp_path: Path) -> None:
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    base_a.mkdir()
    base_b.mkdir()
    repo_a, _ = _repo(base_a)
    repo_b, _ = _repo(base_b)

    entered = threading.Event()
    release = threading.Event()
    other_entered = threading.Event()

    def hold_a() -> None:
        with repository_implementation_lock(repo_a):
            entered.set()
            release.wait(2)

    def try_b() -> None:
        entered.wait(2)
        with repository_implementation_lock(repo_b):
            other_entered.set()

    a = threading.Thread(target=hold_a)
    b = threading.Thread(target=try_b)
    a.start()
    assert entered.wait(2)
    b.start()
    # Unrelated repositories have independent locks: B acquires immediately
    # while A is still held.
    assert other_entered.wait(1)
    release.set()
    a.join(2)
    b.join(2)


def test_lock_contention_raises_and_is_not_masked(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    release = threading.Event()

    def hold() -> None:
        with repository_implementation_lock(repo):
            release.wait(2)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        # Wait until the holder actually owns the flock.
        lock_path = repo / ".git" / "hermes-implementation.lock"
        deadline = 2.0
        while deadline > 0:
            try:
                probe = lock_path.open("a+b")
                import fcntl

                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
                probe.close()
                deadline -= 0.05
                import time

                time.sleep(0.05)
            except BlockingIOError:
                break
        else:
            pytest.fail("holder never acquired the lock")

        with pytest.raises(LockContention):
            with repository_implementation_lock(repo, blocking=False):
                pass
    finally:
        release.set()
        holder.join(2)


def test_normal_lock_acquisition_still_succeeds(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    entered = threading.Event()
    inside = threading.Event()

    with repository_implementation_lock(repo):
        entered.set()
        inside.wait(0.1)
    assert entered.is_set()
