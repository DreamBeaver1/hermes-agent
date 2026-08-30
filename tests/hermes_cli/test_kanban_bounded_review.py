import subprocess
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_review import (
    ReviewPolicy,
    request_implementation_review,
    submit_review_outcome,
    verifier_can_mutate_tracked_source,
    verifier_workspace_snapshot,
)


def _repo(tmp_path):
    repo = tmp_path / "implementation"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.txt").write_text("initial\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return repo, commit


def _task(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    task_id = kb.create_task(
        conn, title="bounded implementation", assignee="implementer",
        session_id="implementation-session",
    )
    return conn, task_id


def _request(conn, task_id, repo, commit):
    task = kb.get_task(conn, task_id)
    return request_implementation_review(
        conn, task_id, implementation_commit=commit, reviewer="verifier",
        implementation_session_id="implementation-session",
        review_workspace=str(repo), summary="implementation ready",
        expected_run_id=task.current_run_id,
    )


def test_bounded_review_allows_one_repair_then_exhausts(tmp_path):
    repo, commit = _repo(tmp_path)
    conn, task_id = _task(tmp_path)

    assert _request(conn, task_id, repo, commit)
    assert kb.get_task(conn, task_id).review_count == 1
    review_run = kb.claim_review_task(conn, task_id, claimer="verifier")
    assert review_run is not None
    assert submit_review_outcome(
        conn, task_id, outcome="changes_requested", summary="fix the regression",
        expected_run_id=review_run.current_run_id, reviewer_session_id="verifier-session",
    ) == (True, "implementer")
    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    assert task.repair_count == 1

    implementation_run = kb.claim_task(conn, task_id, claimer="implementer")
    assert implementation_run is not None
    assert _request(conn, task_id, repo, commit)
    assert kb.get_task(conn, task_id).review_count == 2
    second_review = kb.claim_review_task(conn, task_id, claimer="verifier")
    assert second_review is not None
    assert submit_review_outcome(
        conn, task_id, outcome="changes_requested", summary="a second repair is forbidden",
        expected_run_id=second_review.current_run_id, reviewer_session_id="verifier-session",
    ) == (False, "repair allowance exhausted")


def test_acceptance_requires_separate_verifier_session_and_completes(tmp_path):
    repo, commit = _repo(tmp_path)
    conn, task_id = _task(tmp_path)
    assert _request(conn, task_id, repo, commit)
    review_run = kb.claim_review_task(conn, task_id, claimer="verifier")
    assert review_run is not None
    assert submit_review_outcome(
        conn, task_id, outcome="accepted", summary="verified",
        expected_run_id=review_run.current_run_id,
        reviewer_session_id="implementation-session",
    ) == (False, "reviewer must use a separate session")
    assert submit_review_outcome(
        conn, task_id, outcome="accepted", summary="verified",
        expected_run_id=review_run.current_run_id, reviewer_session_id="verifier-session",
    ) == (True, "done")
    assert kb.get_task(conn, task_id).status == "done"


def test_verifier_snapshot_is_read_only_and_requires_clean_commit(tmp_path):
    repo, commit = _repo(tmp_path)
    snapshot = verifier_workspace_snapshot(str(repo))
    assert snapshot == {"head": commit, "dirty": False, "tracked_source_read_only": True}
    assert verifier_can_mutate_tracked_source(str(repo)) is False
    (repo / "source.txt").write_text("changed\n")
    assert verifier_workspace_snapshot(str(repo))["dirty"] is True
    conn, task_id = _task(tmp_path)
    assert not request_implementation_review(
        conn, task_id, implementation_commit=commit, reviewer="verifier",
        implementation_session_id="implementation-session", review_workspace=str(repo),
        summary="dirty workspace",
    )


def test_custom_policy_bounds_are_respected(tmp_path):
    repo, commit = _repo(tmp_path)
    conn, task_id = _task(tmp_path)
    assert _request(conn, task_id, repo, commit)
    assert not request_implementation_review(
        conn, task_id, implementation_commit=commit, reviewer="verifier",
        implementation_session_id="implementation-session", review_workspace=str(repo),
        summary="policy allows no second review", policy=ReviewPolicy(max_reviews=1),
    )
