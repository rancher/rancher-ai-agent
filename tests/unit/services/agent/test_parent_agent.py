"""
Unit tests for the supervisor (parent) agent.

Tests the creation of supervisor agents that coordinate specialized child agents
using create_supervisor_agent.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agent.parent import (
    ChildAgent,
    create_supervisor_agent,
    _build_child_config,
    _extract_last_message,
    _create_agent_tool,
)


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
def mock_child_agents():
    """Mock ChildAgent list for testing."""
    agents = []
    for name, desc in [
        ("Rancher", "Expert in Rancher and Kubernetes management"),
        ("Fleet", "Expert in Fleet GitOps continuous delivery"),
        ("Harvester", "Expert in Harvester HCI and VM management"),
    ]:
        config = MagicMock()
        config.name = name
        config.description = desc
        agent = MagicMock()
        agents.append(ChildAgent(config=config, agent=agent))
    return agents


# ============================================================================
# Factory Function Tests
# ============================================================================

@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_creates_graph(mock_create_agent, mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that create_supervisor_agent delegates to langchain create_agent."""
    mock_graph = MagicMock()
    mock_create_agent.return_value = mock_graph

    result = create_supervisor_agent(mock_llm, mock_child_agents, mock_checkpointer)

    assert result == mock_graph
    mock_create_agent.assert_called_once()


@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_creates_tools_for_all_children(mock_create_agent, mock_llm, mock_child_agents, mock_checkpointer):
    """Verify that each child agent becomes a tool."""
    mock_create_agent.return_value = MagicMock()

    create_supervisor_agent(mock_llm, mock_child_agents, mock_checkpointer)

    call_kwargs = mock_create_agent.call_args[1]
    tools = call_kwargs["tools"]
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "Rancher" in tool_names
    assert "Fleet" in tool_names
    assert "Harvester" in tool_names


@patch("app.services.agent.parent.create_agent")
def test_create_supervisor_agent_with_many_agents(mock_create_agent, mock_llm, mock_checkpointer):
    """Verify that create_supervisor_agent works with many child agents."""
    mock_create_agent.return_value = MagicMock()

    child_agents = []
    for i in range(5):
        config = MagicMock()
        config.name = f"Agent{i}"
        config.description = f"Agent {i} description"
        child_agents.append(ChildAgent(config=config, agent=MagicMock()))

    create_supervisor_agent(mock_llm, child_agents, mock_checkpointer)

    call_kwargs = mock_create_agent.call_args[1]
    assert len(call_kwargs["tools"]) == 5


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_build_child_config_namespaces_thread_id():
    """Verify child config has namespaced thread_id."""
    with patch("app.services.agent.parent.ensure_config") as mock_ensure:
        mock_ensure.return_value = {
            "configurable": {
                "thread_id": "parent-thread-1",
                "request_id": "req-123",
            }
        }
        config = _build_child_config("rancher")
        assert config["configurable"]["thread_id"] == "parent-thread-1::rancher"
        assert config["configurable"]["request_id"] == "req-123"
        assert config["callbacks"] == []


def test_extract_last_message_returns_content():
    """Verify extraction of last non-empty message content."""
    msg = MagicMock()
    msg.content = "Hello from agent"
    result = _extract_last_message({"messages": [msg]})
    assert result == "Hello from agent"


def test_extract_last_message_empty():
    """Verify fallback when no messages."""
    result = _extract_last_message({"messages": []})
    assert result == "No response from agent."


def test_create_agent_tool_returns_structured_tool():
    """Verify _create_agent_tool wraps child agent as a StructuredTool."""
    config = MagicMock()
    config.name = "rancher"
    config.description = "Rancher agent"
    child = ChildAgent(config=config, agent=MagicMock())

    tool = _create_agent_tool(child)

    assert tool.name == "rancher"
    assert "Rancher agent" in tool.description
