"""Behavioral tests for the common local agent contract and Metis routing."""

import pytest

from agent.agent_contract import (
    agent_specs,
    get_agent_spec,
    make_handoff,
    route_agent,
    validate_advisory_result,
)


def test_inventory_contains_only_the_four_current_agents():
    assert {spec.name.lower() for spec in agent_specs()} == {
        "persephone", "forge", "steward", "metis",
    }
    with pytest.raises(KeyError):
        get_agent_spec("scout")
    with pytest.raises(KeyError):
        get_agent_spec("scribe")


def test_existing_and_new_routing_boundaries():
    assert route_agent("implement and test the repository change") == "forge"
    assert route_agent("review the customer pipeline and lender blocker") == "steward"
    assert route_agent("analyze the objection and improve the negotiation framing") == "metis"
    assert route_agent("coordinate these unrelated items") == "persephone"
    assert route_agent("write code", explicit="metis") == "metis"


def test_handoff_is_bounded_and_validated():
    handoff = make_handoff(
        "persephone", "metis", "Analyze the customer's objection",
        context={"facts": ["customer asked about price"]},
        requested_output=("recommendation", "uncertainty"),
    )
    assert handoff.approval_required is True
    with pytest.raises(ValueError):
        make_handoff("metis", "metis", "self route")


def test_advisory_contract_rejects_incomplete_or_actionful_metis_results():
    good = {
        "observation": "The customer asked for a lower price.",
        "inference": "Price may be a proxy for fairness concerns.",
        "recommendation": "Ask one clarifying question before discussing concessions.",
        "uncertainty": "Motivation is not confirmed.",
    }
    validate_advisory_result("metis", good)
    with pytest.raises(ValueError):
        validate_advisory_result("metis", {"observation": "x", "recommendation": "y"})
    with pytest.raises(ValueError):
        validate_advisory_result("metis", {**good, "action_taken": "sent message"})


def test_metis_boundary_is_explicit():
    spec = get_agent_spec("metis")
    assert "send customer communications" in spec.prohibited_actions
    assert "modify customer records" in spec.prohibited_actions
    assert "use deception or dark patterns" in spec.prohibited_actions
    assert "route final decisions to Persephone" in spec.collaboration_rules


def test_persephone_routes_explicit_requests_without_redefining_existing_roles():
    assert route_agent("anything", explicit="persephone") == "persephone"
    assert "engineering" not in get_agent_spec("steward").responsibilities
    assert "pipeline analysis" not in get_agent_spec("forge").responsibilities
    assert "final prioritization" in get_agent_spec("metis").non_responsibilities
