"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


def test_correctly_owned_clean_worktree_is_reused(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "owned"
    _add_worktree(repo, target, "wt/owned")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="owned", workspace_kind="worktree", workspace_path=str(target), branch_name="wt/owned")
        task = kb.get_task(conn, tid)
        assert task is not None
    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == target.resolve()
    assert branch == "wt/owned"


def test_dirty_worktree_is_preserved_collision(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    target = _add_worktree(repo, repo / ".worktrees" / "dirty", "wt/dirty")
    (target / "untracked.txt").write_text("keep me\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workspace collision"):
        kb._ensure_git_worktree(repo, target, "wt/dirty", owner_id="dirty")
    assert (target / "untracked.txt").read_text(encoding="utf-8") == "keep me\n"


def test_partial_unregistered_git_path_is_collision_without_add(kanban_home, tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "partial"
    target.mkdir(parents=True)
    (target / ".git").write_text("gitdir: nowhere\n", encoding="utf-8")
    calls = []
    real_run = subprocess.run
    def run(*args, **kwargs):
        if args and isinstance(args[0], list) and args[0][0:4] == ["git", "-C", str(repo), "worktree"] and "add" in args[0]:
            calls.append(args[0])
        return real_run(*args, **kwargs)
    monkeypatch.setattr(kb.subprocess, "run", run)
    with pytest.raises(ValueError, match="workspace collision"):
        kb._ensure_git_worktree(repo, target, "wt/partial", owner_id="partial")
    assert calls == []


def test_collision_is_blocked_and_not_retried(kanban_home, monkeypatch):
    spawns = []
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="collision", assignee="alice", workspace_kind="worktree")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        monkeypatch.setattr(
            kb, "_resolve_worktree_workspace",
            lambda task, board=None: (_ for _ in ()).throw(
                kb._WorkspaceCollision("workspace collision: unchanged")
            ),
        )
        first = kb._dispatch_once_locked(
            conn, spawn_fn=lambda *args, **kwargs: spawns.append(args), max_spawn=1,
        )
        second = kb._dispatch_once_locked(
            conn, spawn_fn=lambda *args, **kwargs: spawns.append(args), max_spawn=1,
        )
        row = conn.execute("SELECT status, consecutive_failures FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0
    assert not spawns
    assert not first.spawned and not second.spawned


def test_workspace_owner_metadata_blocks_active_reuse(kanban_home, tmp_path):
    workspace = (tmp_path / "repo" / ".worktrees" / "owner").resolve()
    with kb.connect() as conn:
        owner_id = kb.create_task(
            conn, title="owner", workspace_kind="worktree",
            workspace_path=str(workspace), branch_name="wt/owner",
        )
        contender_id = kb.create_task(
            conn, title="contender", workspace_kind="worktree",
            workspace_path=str(workspace), branch_name="wt/contender",
        )
        contender = kb.get_task(conn, contender_id)
        assert contender is not None
        with pytest.raises(ValueError, match=f"owned by task {owner_id}"):
            kb._assert_task_workspace_owner(conn, contender, workspace)


def test_collision_evidence_is_idempotent(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="collision")
        kb._record_workspace_collision(conn, tid, "first collision")
        kb._record_workspace_collision(conn, tid, "second collision")
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "AND kind='infrastructure_failed' ORDER BY id",
            (tid,),
        ).fetchall()
        task = conn.execute(
            "SELECT status, last_failure_error FROM tasks WHERE id=?", (tid,)
        ).fetchone()

    assert len(events) == 1
    assert events[0]["payload"] is not None
    assert "first collision" in events[0]["payload"]
    assert task["status"] == "blocked"
    assert task["last_failure_error"] == "first collision"



