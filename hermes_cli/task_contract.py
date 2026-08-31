"""Task contracts and deterministic dispatcher preflight.

This module is intentionally model-independent: validation runs before a task is
claimed, a workspace is allocated, or a worker process is started.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.environments.local import build_subprocess_env

MODES = {"evidence_only", "local_commit", "push_branch", "pull_request"}
REQUIRED_FIELDS = (
    "repository", "base_revision", "implementation_scope", "non_goals",
    "acceptance_criteria", "validation_commands", "required_runtime",
    "required_harness", "publication_mode", "authority_requirements",
    "dependencies",
)
_SECRET_RE = re.compile(r"(?i)(token|password|secret|key|authorization)\s*[:=]\s*\S+")


@dataclass(frozen=True)
class ContractFailure:
    kind: str
    evidence: str
    retryable: bool = False
    repair_consumed: bool = False


def _redact(text: str) -> str:
    text = _SECRET_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text or "")
    return re.sub(r"(?i)\b(ghp_|github_pat_|sk-|xox[baprs]-)[A-Za-z0-9_./-]+", "<redacted>", text)


def _nonempty(value: Any) -> bool:
    return bool(value) and (not isinstance(value, str) or bool(value.strip()))


def _run(argv: list[str], *, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        p = subprocess.run(argv, cwd=str(cwd) if cwd else None, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", check=False,
                           timeout=30,
                           # Exact legacy os.environ.copy() semantics: no secret
                           # scrubbing, no profile-home rewriting — pre-flight git
                           # and subprocess probes must see the full inherited env.
                           env=build_subprocess_env(scrub_secrets=False,
                                                    inherit_profile_home=False))
        detail = (p.stderr or p.stdout or "").strip()
        return p.returncode == 0, _redact(f"$ {' '.join(argv)}\n{detail}")
    except (OSError, subprocess.SubprocessError) as exc:
        return False, _redact(f"$ {' '.join(argv)}\n{exc}")


def _failure(kind: str, evidence: str) -> ContractFailure:
    return ContractFailure(kind=kind, evidence=_redact(evidence))


def validate_contract(contract: Mapping[str, Any], *, worker_env: Mapping[str, str] | None = None) -> ContractFailure | None:
    """Validate contract fields and prerequisites without model or filesystem mutation."""
    # Empty objects/lists are valid declarations for authority (no special
    # credentials) and dependencies (no dependencies); presence is what the
    # contract requires for those fields.
    missing = [name for name in REQUIRED_FIELDS if name not in contract or contract[name] is None or (isinstance(contract[name], str) and not contract[name].strip())]
    if missing:
        return _failure("contract_invalid", "missing required task-contract fields: " + ", ".join(missing))
    mode = contract["publication_mode"]
    if mode not in MODES:
        return _failure("contract_invalid", f"publication_mode must be one of {sorted(MODES)}, got {mode!r}")
    if not isinstance(contract["non_goals"], list) or not contract["non_goals"]:
        return _failure("contract_invalid", "non_goals must be a non-empty list")
    if not isinstance(contract["acceptance_criteria"], list) or not contract["acceptance_criteria"]:
        return _failure("contract_invalid", "acceptance_criteria must be a non-empty list")
    if not isinstance(contract["validation_commands"], list) or not contract["validation_commands"]:
        return _failure("contract_invalid", "validation_commands must be a non-empty list")
    if not isinstance(contract["dependencies"], list):
        return _failure("contract_invalid", "dependencies must be a list")
    pub = contract.get("publication_requirements") or {}
    if not isinstance(pub, Mapping):
        return _failure("contract_invalid", "publication_requirements must be an object")
    required = {k for k, v in pub.items() if v}
    allowed = {
        "evidence_only": set(),
        "local_commit": {"commit"},
        "push_branch": {"commit", "push"},
        "pull_request": {"commit", "push", "pull_request"},
    }[mode]
    if required != allowed:
        return _failure("publication_contradiction", f"publication requirements {sorted(required)} contradict mode {mode}")
    if contract.get("dependencies") and not contract.get("dependencies_complete", False):
        return _failure("dependency_incomplete", "declared dependencies are incomplete")

    env = dict(worker_env or os.environ)
    runtime = contract["required_runtime"]
    runtime_cmds = runtime if isinstance(runtime, list) else [runtime]
    for item in runtime_cmds:
        command = item.get("command") if isinstance(item, Mapping) else item
        if not isinstance(command, str) or not command.strip() or not shutil.which(shlex.split(command)[0]):
            return _failure("harness_unavailable", f"required runtime unavailable: {command!r}")
    for item in contract["required_harness"]:
        command = item.get("command") if isinstance(item, Mapping) else item
        if not isinstance(command, str) or not command.strip() or not shutil.which(shlex.split(command)[0]):
            return _failure("harness_unavailable", f"required harness unavailable: {command!r}")
    auth = contract["authority_requirements"]
    if not isinstance(auth, Mapping):
        return _failure("authority_unavailable", "authority_requirements must be an object")
    for variable in auth.get("credentials", []) or []:
        if not env.get(str(variable)):
            return _failure("authority_unavailable", f"required credential unavailable: {variable}")
    repo = Path(str(contract["repository"])).expanduser()
    if not repo.is_dir():
        return _failure("contract_invalid", f"repository unavailable: {repo}")
    ok, evidence = _run(["git", "-C", str(repo), "rev-parse", "--verify", f"{contract['base_revision']}^{{commit}}"])
    if not ok:
        return _failure("contract_invalid", evidence)
    if mode == "local_commit":
        ok, evidence = _run(["git", "-C", str(repo), "status", "--porcelain"])
        if not ok:
            return _failure("contract_invalid", evidence)
        # A local-commit publication starts from a clean checkout; otherwise
        # the worker cannot produce an attributable commit.
        if evidence.strip().splitlines()[1:]:
            return _failure("contract_invalid", "repository is not clean before local commit: " + evidence)
    if mode in {"push_branch", "pull_request"}:
        remote = str(auth.get("remote", "origin"))
        branch = str(contract.get("branch") or "HEAD")
        ok, evidence = _run(["git", "-C", str(repo), "push", "--dry-run", remote, f"HEAD:refs/heads/{branch}"])
        if not ok:
            return _failure("authority_unavailable", evidence)
    if mode == "pull_request":
        tooling = auth.get("pr_tool") or "gh"
        if not shutil.which(str(tooling)):
            return _failure("harness_unavailable", f"pull-request tooling unavailable: {tooling}")
    return None


def contract_from_task(task: Any) -> Mapping[str, Any] | None:
    value = getattr(task, "task_contract", None)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, Mapping) else None
        except (TypeError, ValueError):
            return None
    return None
