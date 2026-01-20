import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from app.services.agent.child import ChildAgentBuilder
from langchain_core.messages import AIMessage
from app.services.agent.builtin_agents import HumanValidationTool

class FakeMessage:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []

class MockTool:
    def __init__(self, name, return_value):
        self.name = name
        self._return_value = return_value
        self.ainvoke = AsyncMock(return_value=return_value)

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="llm_response"))
    return llm

fake_get_tool_execution_message = "fake k8s resource fetched"
fake_patch_tool_execution_message = "fake k8s resource patched"

@pytest.fixture
def mock_tools():
    return [MockTool("getKubernetesResource", fake_get_tool_execution_message), MockTool("patchKubernetesResource", fake_patch_tool_execution_message)]

@pytest.fixture
def mock_checkpointer():
    return MagicMock()

@pytest.fixture
def mock_config():
    return {
        "configurable": {
            "request_id": "test_id",
            "request_metadata": {"tags": []}
        }
    }

@pytest.fixture
def agent_config_with_validation():
    config = MagicMock()
    config.human_validation_tools = [
        HumanValidationTool(name="patchKubernetesResource", type="UPDATE")
    ]
    return config

def test_builder_initialization_binds_tools(mock_llm, mock_tools, mock_checkpointer):
    """Tests that the K8sAgentBuilder correctly initializes and binds tools to the LLM."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )

    assert builder is not None
    mock_llm.bind_tools.assert_called_once_with(mock_tools)

@pytest.mark.asyncio
@patch("langgraph.types.interrupt", new=MagicMock(return_value="no"))
async def test_tool_node_human_verification_cancelled(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_with_validation):
    """Tests that tool execution is cancelled if the user responds 'no' to the confirmation prompt."""
    builder = ChildAgentBuilder(
        llm=mock_llm, 
        tools=mock_tools, 
        system_prompt="system_prompt", 
        checkpointer=mock_checkpointer,
        agent_config=agent_config_with_validation
    )
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local", "patch": "[]"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}
    result = await builder.tool_node(state, mock_config)

    assert result["messages"][0].name == "patchKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == "tool execution cancelled by the user"

@pytest.mark.asyncio
@patch("langgraph.types.interrupt", new=MagicMock(return_value="yes"))
async def test_tool_node_human_verification_approved(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_with_validation):
    """Tests that tool execution proceeds if the user responds 'yes' to the confirmation prompt."""
    builder = ChildAgentBuilder(
        llm=mock_llm, 
        tools=mock_tools, 
        system_prompt="system_prompt", 
        checkpointer=mock_checkpointer,
        agent_config=agent_config_with_validation
    )
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local", "patch": "[]"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}
    result = await builder.tool_node(state, mock_config)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "patchKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == fake_patch_tool_execution_message

@pytest.mark.asyncio
async def test_tool_node_executes_tool_without_confirmation(mock_llm, mock_tools, mock_checkpointer, mock_config):
    """Tests that a tool not requiring confirmation is executed directly."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    tool_call = {
        "id": "123",
        "name": "getKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await builder.tool_node(state, mock_config)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "getKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == fake_get_tool_execution_message

def test_call_model_node_includes_system_prompt(mock_llm, mock_tools, mock_checkpointer, mock_config):
    """Tests that the system prompt is included in the messages sent to the LLM."""
    builder = ChildAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    fake_message = FakeMessage()
    state = {"messages": [fake_message]}

    result = builder.call_model_node(state, mock_config)

    assert result["messages"][0].content == "llm_response"
    mock_llm.invoke.assert_called_once()