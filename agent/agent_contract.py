"""Lightweight contracts and routing metadata for the local agent roster.

The contract is deliberately declarative: profile isolation, Kanban dispatch, and
Hermes's normal tool permissions remain authoritative. This module only makes
identity, boundaries, and handoff expectations inspectable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentSpec:
    name: str
    mission: str
    responsibilities: tuple[str, ...]
    non_responsibilities: tuple[str, ...]
    capabilities: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    escalation_conditions: tuple[str, ...]
    collaboration_rules: tuple[str, ...]
    advisory_only: bool = True


@dataclass(frozen=True)
class AgentHandoff:
    """A bounded packet passed between agents, not a transcript clone."""

    source: str
    target: str
    objective: str
    context: Mapping[str, Any] = field(default_factory=dict)
    requested_output: tuple[str, ...] = ()
    approval_required: bool = True

    def validate(self) -> None:
        if not self.source or not self.target or not self.objective.strip():
            raise ValueError("handoff requires source, target, and objective")
        if self.source == self.target:
            raise ValueError("handoff source and target must differ")
        if any(not isinstance(key, str) for key in self.context):
            raise ValueError("handoff context keys must be strings")


_SPECS = {
    "persephone": AgentSpec(
        name="Persephone",
        mission="Coordinate Chase's work and make final cross-domain priorities.",
        responsibilities=("cross-domain prioritization", "delegation", "synthesis", "escalation"),
        non_responsibilities=("specialist implementation", "unapproved external action"),
        capabilities=("Kanban orchestration", "specialist routing", "handoff synthesis"),
        allowed_actions=("create and route tasks", "request analysis", "present recommendations"),
        prohibited_actions=("merge", "deploy", "mutate live data", "send unapproved communications"),
        escalation_conditions=("conflicting specialist recommendations", "approval boundary reached"),
        collaboration_rules=("route engineering to Forge", "route sales operations to Steward", "route behavioral strategy to Metis"),
    ),
    "forge": AgentSpec(
        name="Forge",
        mission="Design, implement, test, and protect technical systems.",
        responsibilities=("architecture", "coding", "debugging", "verification"),
        non_responsibilities=("sales-pipeline prioritization", "behavioral counseling"),
        capabilities=("repository inspection", "implementation", "tests", "technical risk analysis"),
        allowed_actions=("modify assigned source", "run synthetic tests", "recommend architecture"),
        prohibited_actions=("access live customer data", "bypass canonical services", "merge without approval"),
        escalation_conditions=("missing technical evidence", "unsafe production boundary", "destructive migration"),
        collaboration_rules=("report through Persephone", "use controlled interfaces", "preserve history"),
    ),
    "steward": AgentSpec(
        name="Steward",
        mission="Advise on sales-pipeline progress, blockers, commitments, and follow-up.",
        responsibilities=("pipeline analysis", "commitment tracking", "follow-up strategy", "deal progression"),
        non_responsibilities=("final cross-domain priority", "engineering implementation"),
        capabilities=("customer/deal context analysis", "blocker identification", "next-action recommendations"),
        allowed_actions=("analyze", "recommend", "draft"),
        prohibited_actions=("send communications", "approve concessions", "alter consequential records"),
        escalation_conditions=("missing customer facts", "approval required", "material uncertainty"),
        collaboration_rules=("advise Persephone", "preserve uncertainty", "do not invent rigid sales rules"),
    ),
    "metis": AgentSpec(
        name="Metis",
        mission="Advise on behavioral strategy, psychology, and human decision friction.",
        responsibilities=("sales psychology", "objection analysis", "communication framing", "timing and cadence", "negotiation strategy", "customer motivation"),
        non_responsibilities=("final prioritization", "CRM mutation", "engineering work", "unapproved communication"),
        capabilities=("behavioral interpretation", "framing alternatives", "decision-friction analysis", "longitudinal pattern analysis"),
        allowed_actions=("analyze", "recommend", "draft options", "identify uncertainty"),
        prohibited_actions=("send customer communications", "modify customer records", "make consequential decisions", "use deception or dark patterns"),
        escalation_conditions=("insufficient context", "vulnerability exploitation risk", "recommendation requires approval"),
        collaboration_rules=("return observation/inference/recommendation/uncertainty", "route final decisions to Persephone", "use bounded context packets"),
    ),
}


def agent_specs() -> tuple[AgentSpec, ...]:
    return tuple(_SPECS.values())


def get_agent_spec(name: str) -> AgentSpec:
    try:
        return _SPECS[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"unknown agent: {name}") from exc


def route_agent(request: str, explicit: str | None = None) -> str:
    """Choose a specialist for clear requests; default coordination stays with Persephone."""
    if explicit:
        name = explicit.strip().lower()
        get_agent_spec(name)
        return name
    text = request.lower()
    if any(term in text for term in ("metis", "psychology", "objection", "framing", "motivation", "negotiat", "cadence", "behavior")):
        return "metis"
    if any(term in text for term in ("code", "bug", "test", "repository", "architecture", "implement", "debug")):
        return "forge"
    if any(term in text for term in ("lead", "deal", "pipeline", "follow-up", "follow up", "lender", "customer commitment")):
        return "steward"
    return "persephone"


def make_handoff(source: str, target: str, objective: str, *, context: Mapping[str, Any] | None = None, requested_output: tuple[str, ...] = ()) -> AgentHandoff:
    get_agent_spec(source)
    get_agent_spec(target)
    handoff = AgentHandoff(source=source, target=target, objective=objective, context=context or {}, requested_output=requested_output)
    handoff.validate()
    return handoff


def validate_advisory_result(agent: str, result: Mapping[str, Any]) -> None:
    """Require the structured fields for advisory specialist output."""
    spec = get_agent_spec(agent)
    if spec.advisory_only:
        missing = {"observation", "recommendation", "uncertainty"} - set(result)
        if missing:
            raise ValueError(f"advisory result missing: {', '.join(sorted(missing))}")
        if not isinstance(result["recommendation"], (str, list, tuple)):
            raise ValueError("recommendation must be text or a sequence")
        if agent == "metis" and "inference" not in result:
            raise ValueError("Metis result requires inference")
        if result.get("action_taken"):
            raise ValueError(f"{spec.name} is advisory-only; action_taken is prohibited")
