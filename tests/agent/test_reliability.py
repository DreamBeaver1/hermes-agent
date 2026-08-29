"""Tests for repository execution preflight and implementation serialization."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent.reliability import (
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
