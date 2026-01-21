"""
Unit tests for the RootAgentBuilder class.

Tests the creation and behavior of root agents that run independently without a parent.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from app.services.agent.base import BaseAgentBuilder
from app.services.agent.root import RootAgentBuilder, create_root_agent
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
    llm.invoke = MagicMock(return_value=AIMessage(content="root_agent_response"))
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
        name="test-root-agent",
        description="Test root agent",
        system_prompt="You are a test root agent",
        mcp_url="http://test:8080",
        authentication=AuthenticationType.NONE,
        human_validation_tools=[
            HumanValidationTool(name="patchKubernetesResource", type=ToolActionType.UPDATE)
        ]
    )


# ============================================================================
# Builder Initialization Tests
# ============================================================================

def test_root_agent_builder_initialization(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that RootAgentBuilder correctly initializes with all required parameters."""
    builder = RootAgentBuilder(
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

def test_root_agent_graph_has_correct_nodes(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that the root agent graph contains the expected nodes."""
    builder = RootAgentBuilder(
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
    assert "summarize_conversation" in nodes

# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_root_agent_factory(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that create_root_agent factory function creates a valid agent."""
    graph = create_root_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    assert graph is not None
    assert hasattr(graph, 'invoke')
    assert hasattr(graph, 'stream')
    assert graph.checkpointer == mock_checkpointer


def test_create_root_agent_with_empty_tools(mock_llm, mock_checkpointer, agent_config):
    """Verify that create_root_agent works with an empty tools list."""
    graph = create_root_agent(
        llm=mock_llm,
        tools=[],
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    assert graph is not None


def test_create_root_agent_with_custom_system_prompt(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that create_root_agent respects custom system prompts."""
    custom_prompt = "Custom root agent instructions"
    
    graph = create_root_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt=custom_prompt,
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    assert graph is not None


# ============================================================================
# Agent Behavior Tests
# ============================================================================

def test_root_agent_has_nodes(mock_llm, mock_tools, mock_checkpointer, agent_config):
    """Verify that root agent includes a summarize_conversation_node.
    
    Root agents handle their own summarization, unlike child agents which
    delegate to their parent.
    """
    builder = RootAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="test system prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config
    )

    graph = builder.build()
    nodes = graph.nodes

    assert "summarize_conversation" in nodes
    assert "agent" in nodes
    assert "tools" in nodes
