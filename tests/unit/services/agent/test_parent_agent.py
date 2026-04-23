"""
Unit tests for the parent agent (deepagents-based).

Tests the creation of parent agents that route requests to specialized child agents
using deepagents' create_deep_agent.
"""
import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import InMemorySaver


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
    """Verify that create_parent_agent delegates to create_deep_agent."""
    from app.services.agent.parent import create_parent_agent

    mock_graph = MagicMock()
    mock_create_deep.return_value = mock_graph

    result = create_parent_agent(mock_llm, mock_subagents, mock_checkpointer)

    assert result == mock_graph
    mock_create_deep.assert_called_once_with(
        model=mock_llm,
        subagents=mock_subagents,
        checkpointer=mock_checkpointer,
    )


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
