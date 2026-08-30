"""Focused pre-dispatch task-contract/prerequisite tests."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _contract(repo: Path, mode: str = "local_commit", **extra):
    pub = {"commit": mode in {"local_commit", "push_branch", "pull_request"},
           "push": mode in {"push_branch", "pull_request"},
           "pull_request": mode == "pull_request"}
    value = {
        "repository": str(repo), "base_revision": "HEAD",
        "implementation_scope": ["the dispatcher"], "non_goals": ["Sales OS"],
        "acceptance_criteria": ["focused tests pass"],
        "validation_commands": ["python -m pytest"], "required_runtime": ["python3"],
        "required_harness": ["git"], "publication_mode": mode,
        "publication_requirements": pub, "authority_requirements": {},
        "dependencies": [],
    }
    value.update(extra)
    return value


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "init"], check=True)
    return repo


def test_valid_local_commit_pass(tmp_path):
    from hermes_cli.task_contract import validate_contract
    assert validate_contract(_contract(_repo(tmp_path))) is None


def test_missing_https_harness_zero_calls(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path)
    calls = []
    contract = _contract(repo, required_harness=["https"])
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="contract", assignee="default", task_contract=contract)
        result = kb.dispatch_once(conn, spawn_fn=lambda *a: calls.append(a))
        status = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()["status"]
    assert tid in result.contract_blocked
    assert status == "blocked"
    assert calls == []


def test_push_required_without_write_authority_zero_calls(tmp_path, monkeypatch):
    from hermes_cli.task_contract import validate_contract
    calls = []
    monkeypatch.setattr("hermes_cli.task_contract._run", lambda *a, **k: calls.append(a) or (True, "ok"))
    contract = _contract(_repo(tmp_path), "push_branch", authority_requirements={"credentials": ["MISSING_TOKEN"]})
    failure = validate_contract(contract, worker_env={})
    assert failure is not None and failure.kind == "authority_unavailable"
    assert calls == []


def test_contradictory_publication_requirements_pre_dispatch(tmp_path):
    from hermes_cli.task_contract import validate_contract
    contract = _contract(_repo(tmp_path), publication_requirements={"push": True})
    failure = validate_contract(contract)
    assert failure is not None and failure.kind == "publication_contradiction"
    assert failure.repair_consumed is False
    assert "push" in failure.evidence


# Keep the imported names above intentional: this file tests worker-env checks.
os
