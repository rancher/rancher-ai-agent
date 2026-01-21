"""
Unit tests for the ChildAgentBuilder class.

Tests the creation and behavior of child agents that delegate summarization
to their parent agents.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from app.services.agent.base import BaseAgentBuilder
from app.services.agent.child import ChildAgentBuilder, create_child_agent
from app.services.agent.loader import AgentConfig, HumanValidationTool, ToolActionType, AuthenticationType


class MockTool:
    """Mock tool for testing."""
    def __init__(self, name, return_value="mock_result"):
        self.name = name
        self._return_value = return_value


@pytest.fixture
def mock_llm():
    """Mock LLM with bound tools."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="child_agent_response"))
    return llm


@pytest.fixture
def mock_tools():
    """Mock tools for testing."""
    return [
        MockTool("getKubernetesResource"),
        MockTool("patchKubernetesResource")
    ]


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return MagicMock()


@pytest.fixture
def agent_config():
    """Mock agent configuration."""
    return AgentConfig(
        name="test-child-agent",
        description="Test child agent",
        system_prompt="You are a test child agent",
        mcp_url="http://test:8080",
        authentication=AuthenticationType.NONE,
        human_validation_tools=[
            HumanValidationTool(name="patchKubernetesResource", type=ToolActionType.UPDATE)
        ]
    )


# ============================================================================
# Builder Initialization Tests
# ============================================================================

def test_child_agent_builder_initialization(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that ChildAgentBuilder correctly initializes with all required parameters."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    assert builder is not None
    assert builder.llm == mock_llm
    assert builder.tools == mock_tools
    assert builder.system_prompt == "test system prompt"
    assert builder.checkpointer == mock_checkpointer
    assert builder.agent_config == agent_config
    assert isinstance(builder, BaseAgentBuilder)
    mock_llm.bind_tools.assert_called_once_with(mock_tools)



# ============================================================================
# Graph Building Tests
# ============================================================================

def test_child_agent_build_returns_compiled_graph(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that build() returns a compiled StateGraph."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    graph = builder.build()

    assert graph is not None
    # Check that it's a compiled graph
    assert hasattr(graph, 'invoke')
    assert hasattr(graph, 'stream')


def test_child_agent_graph_has_correct_nodes(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that the child agent graph contains the expected nodes."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    graph = builder.build()

    # Access the underlying graph structure
    nodes = graph.nodes
    assert "agent" in nodes
    assert "tools" in nodes
