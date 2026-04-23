"""
Unit tests for the parent agent (deepagents-based).

Tests the creation of parent agents that route requests to specialized child agents
using deepagents' create_deep_agent, including plan approval via submit_plan interrupt.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import InterruptOnConfig


@pytest.fixture
def mock_llm():
    """Mock LLM for routing decisions."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return InMemorySaver()


@pytest.fixture
def mock_subagents():
    """Mock SubAgent dicts for testing."""
    return [
        {
            "name": "Rancher",
            "description": "Expert in Rancher UI and Kubernetes management through Rancher",
            "system_prompt": "You are a Rancher expert",
            "tools": [MagicMock()],
        },
        {
            "name": "Fleet",
            "description": "Expert in Fleet GitOps continuous delivery for Kubernetes",
            "system_prompt": "You are a Fleet expert",
            "tools": [MagicMock()],
        },
        {
            "name": "Harvester",
            "description": "Expert in Harvester HCI and virtual machine management",
            "system_prompt": "You are a Harvester expert",
            "tools": [MagicMock()],
        },
    ]


# ============================================================================
# Factory Function Tests
# ============================================================================

@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_calls_create_deep_agent(mock_create_deep, mock_llm, mock_subagents, mock_checkpointer):
    """Verify that create_parent_agent delegates to create_deep_agent with planning config."""
    from app.services.agent.parent import create_parent_agent, PLANNING_SYSTEM_PROMPT

    mock_graph = MagicMock()
    mock_create_deep.return_value = mock_graph

    result = create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    assert result == mock_graph
    mock_create_deep.assert_called_once()
    call_kwargs = mock_create_deep.call_args[1]
    assert call_kwargs["model"] == mock_llm
    assert call_kwargs["subagents"] == mock_subagents
    assert call_kwargs["checkpointer"] == mock_checkpointer
    assert call_kwargs["system_prompt"] == PLANNING_SYSTEM_PROMPT
    assert "submit_plan" in call_kwargs["interrupt_on"]


@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_passes_submit_plan_tool(mock_create_deep, mock_llm, mock_subagents, mock_checkpointer):
    """Verify that submit_plan is passed as a tool to create_deep_agent."""
    from app.services.agent.parent import create_parent_agent, submit_plan

    mock_create_deep.return_value = MagicMock()
    create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    tools = call_kwargs["tools"]
    assert any(t.name == "submit_plan" for t in tools)


@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_passes_all_subagents(mock_create_deep, mock_llm, mock_subagents, mock_checkpointer):
    """Verify that all subagents are passed to create_deep_agent."""
    from app.services.agent.parent import create_parent_agent

    mock_create_deep.return_value = MagicMock()

    create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    subagents = call_kwargs["subagents"]
    assert len(subagents) == 3
    assert subagents[0]["name"] == "Rancher"
    assert subagents[1]["name"] == "Fleet"
    assert subagents[2]["name"] == "Harvester"


@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_with_many_subagents(mock_create_deep, mock_llm, mock_checkpointer):
    """Verify that create_parent_agent works with many subagents."""
    mock_create_deep.return_value = MagicMock()

    from app.services.agent.parent import create_parent_agent

    subagents = [
        {
            "name": f"Agent{i}",
            "description": f"Agent {i} description",
            "system_prompt": f"You are Agent {i}",
            "tools": [MagicMock()],
        }
        for i in range(5)
    ]

    create_parent_agent(mock_llm, subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    assert len(call_kwargs["subagents"]) == 5


@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_with_interrupt_on(mock_create_deep, mock_llm, mock_checkpointer):
    """Verify that subagents with interrupt_on are passed correctly."""
    mock_create_deep.return_value = MagicMock()

    from app.services.agent.parent import create_parent_agent

    subagents = [
        {
            "name": "Rancher",
            "description": "Rancher agent",
            "system_prompt": "You are Rancher",
            "tools": [MagicMock()],
            "interrupt_on": {"createKubernetesResource": True, "deleteKubernetesResource": True},
        },
        {
            "name": "Fleet",
            "description": "Fleet agent",
            "system_prompt": "You are Fleet",
            "tools": [MagicMock()],
        },
    ]

    create_parent_agent(mock_llm, subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    assert "interrupt_on" in call_kwargs["subagents"][0]
    assert call_kwargs["subagents"][0]["interrupt_on"]["createKubernetesResource"] is True
    assert "interrupt_on" not in call_kwargs["subagents"][1]


# ============================================================================
# Planning / submit_plan Interrupt Tests
# ============================================================================

@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_configures_submit_plan_interrupt(mock_create_deep, mock_llm, mock_subagents, mock_checkpointer):
    """Verify that submit_plan is configured as an interrupt for plan approval."""
    from app.services.agent.parent import create_parent_agent

    mock_create_deep.return_value = MagicMock()
    create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    interrupt_on = call_kwargs["interrupt_on"]
    assert "submit_plan" in interrupt_on

    submit_plan_config = interrupt_on["submit_plan"]
    assert "approve" in submit_plan_config["allowed_decisions"]
    assert "reject" in submit_plan_config["allowed_decisions"]


@patch("app.services.agent.parent.create_deep_agent")
def test_create_parent_agent_includes_planning_system_prompt(mock_create_deep, mock_llm, mock_subagents, mock_checkpointer):
    """Verify that the planning system prompt is passed to create_deep_agent."""
    from app.services.agent.parent import create_parent_agent, PLANNING_SYSTEM_PROMPT

    mock_create_deep.return_value = MagicMock()
    create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    call_kwargs = mock_create_deep.call_args[1]
    assert call_kwargs["system_prompt"] == PLANNING_SYSTEM_PROMPT
    assert "submit_plan" in PLANNING_SYSTEM_PROMPT
    assert "approval" in PLANNING_SYSTEM_PROMPT.lower()


def test_format_plan_description_produces_plan_approval_tag():
    """Verify _format_plan_description wraps plan args in <plan-approval> tags."""
    from app.services.agent.parent import _format_plan_description

    tool_call = {
        "name": "submit_plan",
        "args": {
            "goal": "List all pods in the cluster",
            "steps": [
                {"title": "List all namespaces", "description": "Get namespaces first"},
                {"title": "List pods in each namespace", "description": ""},
            ]
        },
        "id": "call_123",
    }

    description = _format_plan_description(tool_call, {}, None)
    assert description.startswith("<plan-approval>")
    assert description.endswith("</plan-approval>")

    inner_json = description[len("<plan-approval>"):-len("</plan-approval>")]
    parsed = json.loads(inner_json)
    assert "goal" in parsed
    assert parsed["goal"] == "List all pods in the cluster"
    assert len(parsed["steps"]) == 2
    assert parsed["steps"][0]["title"] == "List all namespaces"


def test_format_plan_description_handles_empty_steps():
    """Verify _format_plan_description handles empty steps list."""
    from app.services.agent.parent import _format_plan_description

    tool_call = {"name": "submit_plan", "args": {"goal": "test", "steps": []}, "id": "call_456"}

    description = _format_plan_description(tool_call, {}, None)
    inner_json = description[len("<plan-approval>"):-len("</plan-approval>")]
    parsed = json.loads(inner_json)
    assert parsed["steps"] == []


def test_format_plan_description_handles_missing_args():
    """Verify _format_plan_description handles missing args gracefully."""
    from app.services.agent.parent import _format_plan_description

    tool_call = {"name": "submit_plan", "args": {}, "id": "call_789"}

    description = _format_plan_description(tool_call, {}, None)
    inner_json = description[len("<plan-approval>"):-len("</plan-approval>")]
    parsed = json.loads(inner_json)
    assert parsed == {}


# ============================================================================
# submit_plan Tool Tests
# ============================================================================

def test_submit_plan_tool_exists_and_has_correct_name():
    """Verify submit_plan tool has the correct name."""
    from app.services.agent.parent import submit_plan
    assert submit_plan.name == "submit_plan"


def test_submit_plan_tool_description_mentions_plan():
    """Verify submit_plan tool description mentions planning."""
    from app.services.agent.parent import submit_plan
    assert "plan" in submit_plan.description.lower()

