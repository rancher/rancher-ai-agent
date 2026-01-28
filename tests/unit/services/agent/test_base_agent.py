"""
Unit tests for agent building functionality including tool execution,
human validation, conversation summarization, and error handling.
"""
import pytest
import json

from unittest.mock import AsyncMock, MagicMock, patch
from app.services.agent.base import (
    create_confirmation_response,
    should_interrupt,
    handle_interrupt,
    process_tool_result,
    convert_to_string_if_needed,
    INTERRUPT_CANCEL_MESSAGE,
    BaseAgentBuilder
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage, SystemMessage
from langchain_core.tools import ToolException
from app.services.agent.loader import HumanValidationTool
from ollama import ResponseError

class FakeMessage:
    """Mock message class for testing."""
    def __init__(self, tool_calls=None, content="test"):
        self.tool_calls = tool_calls or []
        self.content = content
        self.id = "msg_123"

class MockTool:
    """Mock tool for testing tool execution."""
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

@pytest.fixture
def mock_llm():
    """Mock LLM with bound tools."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="llm_response"))
    return llm

@pytest.fixture
def mock_tools():
    """Mock tools for testing."""
    return [
        MockTool("getKubernetesResource", "fake k8s resource fetched"),
        MockTool("patchKubernetesResource", "fake k8s resource patched")
    ]

@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return MagicMock()

@pytest.fixture
def mock_config():
    """Mock runnable configuration."""
    return {
        "configurable": {
            "request_id": "test_id",
            "request_metadata": {"tags": []}
        }
    }

@pytest.fixture
def agent_config_with_validation():
    """Mock agent configuration with human validation enabled."""
    config = MagicMock()
    config.human_validation_tools = [
        HumanValidationTool(name="patchKubernetesResource", type="UPDATE"),
        HumanValidationTool(name="createKubernetesResource", type="CREATE")
    ]
    return config

@pytest.fixture
def agent_config_without_validation():
    """Mock agent configuration without human validation."""
    config = MagicMock()
    config.human_validation_tools = []
    return config

# ============================================================================
# Builder Initialization Tests
# ============================================================================

def test_builder_initialization_binds_tools(mock_llm, mock_tools, mock_checkpointer):
    """Verify that BaseAgentBuilder correctly initializes and binds tools to the LLM."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )

    assert builder is not None
    assert builder.tools == mock_tools
    assert builder.system_prompt == "system_prompt"
    mock_llm.bind_tools.assert_called_once_with(mock_tools)

def test_builder_creates_tools_by_name_dict(mock_llm, mock_tools, mock_checkpointer):
    """Verify that builder creates a tools_by_name dictionary for quick lookup."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )

    assert "getKubernetesResource" in builder.tools_by_name
    assert "patchKubernetesResource" in builder.tools_by_name
    assert builder.tools_by_name["getKubernetesResource"].name == "getKubernetesResource"

# ============================================================================
# Human Validation Tests
# ============================================================================

@pytest.mark.asyncio
@patch("langgraph.types.interrupt", new=MagicMock(return_value="no"))
async def test_tool_node_human_verification_cancelled(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_with_validation):
    """Verify that tool execution is cancelled when user responds 'no' to confirmation."""
    builder = BaseAgentBuilder(
        llm=mock_llm, 
        tools=mock_tools, 
        system_prompt="system_prompt", 
        checkpointer=mock_checkpointer,
        agent_config=agent_config_with_validation
    )
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {
            "namespace": "cattle-system",
            "name": "rancher",
            "kind": "Deployment",
            "cluster": "local",
            "patch": "[]"
        }
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}
    result = await builder.tool_node(state, mock_config)

    assert result["messages"][0].name == "patchKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == INTERRUPT_CANCEL_MESSAGE
    assert result["messages"][0].additional_kwargs["confirmation"] is False

@pytest.mark.asyncio
@patch("langgraph.types.interrupt", new=MagicMock(return_value="yes"))
async def test_tool_node_human_verification_approved(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_with_validation):
    """Verify that tool execution proceeds when user responds 'yes' to confirmation."""
    builder = BaseAgentBuilder(
        llm=mock_llm, 
        tools=mock_tools, 
        system_prompt="system_prompt", 
        checkpointer=mock_checkpointer,
        agent_config=agent_config_with_validation
    )
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {
            "namespace": "cattle-system",
            "name": "rancher",
            "kind": "Deployment",
            "cluster": "local",
            "patch": "[]"
        }
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}
    result = await builder.tool_node(state, mock_config)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "patchKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == "fake k8s resource patched"
    assert result["messages"][0].additional_kwargs["confirmation"] is True

@pytest.mark.asyncio
@patch("langgraph.types.interrupt", new=MagicMock(return_value="yes"))
async def test_tool_node_human_verification_create_operation(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_with_validation):
    """Verify that CREATE operations also trigger human validation correctly."""
    create_tool = MockTool("createKubernetesResource", "resource created")
    tools = [create_tool]
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_with_validation
    )
    tool_call = {
        "id": "456",
        "name": "createKubernetesResource",
        "args": {
            "namespace": "default",
            "name": "test-pod",
            "kind": "Pod",
            "cluster": "local",
            "resource": '{"spec": {}}'
        }
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}
    result = await builder.tool_node(state, mock_config)

    assert result["messages"][0].content == "resource created"
    assert result["messages"][0].additional_kwargs["confirmation"] is True

# ============================================================================
# Tool Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tool_node_executes_tool_without_confirmation(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_without_validation):
    """Verify that tools not requiring confirmation execute directly."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_without_validation
    )
    tool_call = {
        "id": "123",
        "name": "getKubernetesResource",
        "args": {
            "namespace": "cattle-system",
            "name": "rancher",
            "kind": "Deployment",
            "cluster": "local"
        }
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await builder.tool_node(state, mock_config)

    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
    assert result["messages"][0].name == "getKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == "fake k8s resource fetched"
    assert result["messages"][0].additional_kwargs.get("request_id") == "test_id"

@pytest.mark.asyncio
async def test_tool_node_handles_tool_exception(mock_llm, mock_checkpointer, mock_config, agent_config_without_validation):
    """Verify that ToolException is caught and returned as an error message."""
    error_tool = MockTool("errorTool", "success")
    error_tool.ainvoke = AsyncMock(side_effect=ToolException("Tool error occurred"))
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=[error_tool],
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_without_validation
    )
    tool_call = {
        "id": "error_123",
        "name": "errorTool",
        "args": {}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await builder.tool_node(state, mock_config)

    assert result["messages"][0].content == "Tool error occurred"
    assert result["messages"][0].tool_call_id == "error_123"

@pytest.mark.asyncio
async def test_tool_node_handles_unexpected_exception(mock_llm, mock_checkpointer, mock_config, agent_config_without_validation):
    """Verify that unexpected exceptions are caught and logged properly."""
    error_tool = MockTool("errorTool", "success")
    error_tool.ainvoke = AsyncMock(side_effect=ValueError("Unexpected error"))
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=[error_tool],
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_without_validation
    )
    tool_call = {
        "id": "error_456",
        "name": "errorTool",
        "args": {}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await builder.tool_node(state, mock_config)

    assert "unexpected error during tool call" in result["messages"][0].content
    assert "Unexpected error" in result["messages"][0].content

@pytest.mark.asyncio
async def test_tool_node_handles_multiple_tool_calls(mock_llm, mock_tools, mock_checkpointer, mock_config, agent_config_without_validation):
    """Verify that multiple tool calls in a single message are all executed."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_without_validation
    )
    tool_calls = [
        {
            "id": "call_1",
            "name": "getKubernetesResource",
            "args": {"namespace": "default", "name": "pod1", "kind": "Pod", "cluster": "local"}
        },
        {
            "id": "call_2",
            "name": "getKubernetesResource",
            "args": {"namespace": "default", "name": "pod2", "kind": "Pod", "cluster": "local"}
        }
    ]
    state = {"messages": [FakeMessage(tool_calls=tool_calls)]}

    result = await builder.tool_node(state, mock_config)

    assert len(result["messages"]) == 2
    assert result["messages"][0].tool_call_id == "call_1"
    assert result["messages"][1].tool_call_id == "call_2"

@pytest.mark.asyncio
@patch("app.services.agent.base.dispatch_custom_event")
async def test_tool_node_processes_mcp_response(mock_dispatch, mock_llm, mock_checkpointer, mock_config, agent_config_without_validation):
    """Verify that MCP responses with uiContext are properly processed."""
    mcp_result = json.dumps({
        "llm": "Response for LLM",
        "uiContext": {"key": "value"}
    })
    mcp_tool = MockTool("mcpTool", mcp_result)
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=[mcp_tool],
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=agent_config_without_validation
    )
    tool_call = {
        "id": "mcp_123",
        "name": "mcpTool",
        "args": {}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await builder.tool_node(state, mock_config)

    assert result["messages"][0].content == "Response for LLM"
    assert "mcp_response" in result["messages"][0].additional_kwargs
    assert "<mcp-response>" in result["messages"][0].additional_kwargs["mcp_response"]

# ============================================================================
# Model Call Tests
# ============================================================================

def test_call_model_node_includes_system_prompt(mock_llm, mock_tools, mock_checkpointer, mock_config):
    """Verify that system prompt is prepended to messages when calling the LLM."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="You are a helpful assistant",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    state = {"messages": [HumanMessage(content="Hello")]}

    result = builder.call_model_node(state, mock_config)

    assert result["messages"][0].content == "llm_response"
    mock_llm.invoke.assert_called_once()
    call_args = mock_llm.invoke.call_args[0][0]
    assert any(msg.content == "You are a helpful assistant" for msg in call_args)

def test_call_model_node_retries_on_tool_parse_error(mock_llm, mock_tools, mock_checkpointer, mock_config):
    """Verify that LLM call is retried once on tool parsing errors."""
    error = ResponseError(error="error parsing tool call: invalid json")
    success_response = AIMessage(content="success")
    
    mock_llm.invoke = MagicMock(side_effect=[error, success_response])
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    state = {"messages": [HumanMessage(content="Test")]}

    result = builder.call_model_node(state, mock_config)

    assert mock_llm.invoke.call_count == 2
    assert result["messages"][0].content == "success"

def test_call_model_node_fails_after_retry_limit(mock_llm, mock_tools, mock_checkpointer, mock_config):
    """Verify that after one retry, the error is raised."""
    error = ResponseError(error="error parsing tool call: invalid json")
    mock_llm.invoke = MagicMock(side_effect=error)
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    state = {"messages": [HumanMessage(content="Test")]}

    with pytest.raises(ResponseError):
        builder.call_model_node(state, mock_config)
    
    assert mock_llm.invoke.call_count == 2

# ============================================================================
# Conversation Summarization Tests
# ============================================================================

def test_summarize_conversation_creates_new_summary(mock_llm, mock_tools, mock_checkpointer):
    """Verify that summarize_conversation creates a new summary when none exists."""
    summary_response = AIMessage(content="Summary of conversation")
    mock_llm.invoke = MagicMock(return_value=summary_response)
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    msg1 = HumanMessage(content="Hello", id="msg1")
    msg2 = AIMessage(content="Hi there", id="msg2")
    state = {"messages": [msg1, msg2]}

    result = builder.summarize_conversation_node(state)

    assert result["summary"] == "Summary of conversation"
    assert any(isinstance(msg, RemoveMessage) for msg in result["messages"])
    # The summary is now a SystemMessage with "Conversation summary: " prefix
    assert isinstance(result["messages"][-1], SystemMessage)
    assert result["messages"][-1].content == "Conversation summary: Summary of conversation"
    assert result["messages"][-1].additional_kwargs.get("is_summary") is True

def test_summarize_conversation_extends_existing_summary(mock_llm, mock_tools, mock_checkpointer):
    """Verify that summarize_conversation extends an existing summary."""
    summary_response = AIMessage(content="Extended summary")
    mock_llm.invoke = MagicMock(return_value=summary_response)
    
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    msg1 = HumanMessage(content="New message", id="msg1")
    state = {
        "messages": [msg1],
        "summary": "Previous summary"
    }

    result = builder.summarize_conversation_node(state)

    assert result["summary"] == "Extended summary"
    call_args = mock_llm.invoke.call_args[0][0]
    assert any("Previous summary" in str(msg.content) for msg in call_args)

def test_should_summarize_conversation_returns_summarize_for_long_conversations(mock_llm, mock_tools, mock_checkpointer):
    """Verify that conversations with > 7 messages trigger summarization."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    messages = [AIMessage(content=f"Message {i}") for i in range(10)]
    state = {"messages": messages}

    result = builder.should_summarize_conversation(state)

    assert result == "summarize_conversation"

def test_should_summarize_conversation_returns_end_for_short_conversations(mock_llm, mock_tools, mock_checkpointer):
    """Verify that conversations with ≤ 7 messages end without summarization."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    messages = [AIMessage(content=f"Message {i}") for i in range(5)]
    state = {"messages": messages}

    result = builder.should_summarize_conversation(state)

    assert result == "end"

def test_should_summarize_conversation_returns_continue_for_tool_calls(mock_llm, mock_tools, mock_checkpointer):
    """Verify that messages with tool calls continue rather than summarize."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    msg_with_tool = AIMessage(content="Calling tool")
    msg_with_tool.tool_calls = [{"id": "1", "name": "test"}]
    state = {"messages": [msg_with_tool]}

    result = builder.should_summarize_conversation(state)

    assert result == "continue"

# ============================================================================
# Control Flow Tests
# ============================================================================

def test_should_continue_returns_end_without_tool_calls(mock_llm, mock_tools, mock_checkpointer):
    """Verify that should_continue returns 'end' when no tool calls present."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    state = {"messages": [AIMessage(content="Done")]}

    result = builder.should_continue(state)

    assert result == "end"

def test_should_continue_returns_continue_with_tool_calls(mock_llm, mock_tools, mock_checkpointer):
    """Verify that should_continue returns 'continue' when tool calls present."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    msg = AIMessage(content="Calling tool")
    msg.tool_calls = [{"id": "1", "name": "test"}]
    state = {"messages": [msg]}

    result = builder.should_continue(state)

    assert result == "continue"

def test_should_continue_after_interrupt_ends_on_cancel(mock_llm, mock_tools, mock_checkpointer):
    """Verify that workflow ends when user cancels tool execution."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    cancel_msg = ToolMessage(
        content=INTERRUPT_CANCEL_MESSAGE,
        tool_call_id="123",
        name="test"
    )
    state = {"messages": [cancel_msg]}

    result = builder.should_continue_after_interrupt(state)

    assert result == "end"

def test_should_continue_after_interrupt_continues_on_success(mock_llm, mock_tools, mock_checkpointer):
    """Verify that workflow continues when tool execution succeeds."""
    builder = BaseAgentBuilder(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt",
        checkpointer=mock_checkpointer,
        agent_config=MagicMock()
    )
    
    success_msg = ToolMessage(
        content="Success",
        tool_call_id="123",
        name="test"
    )
    state = {"messages": [success_msg]}

    result = builder.should_continue_after_interrupt(state)

    assert result == "continue"

# ============================================================================
# Helper Function Tests
# ============================================================================

def test_create_confirmation_response_formats_correctly():
    """Verify confirmation response JSON structure."""
    result = create_confirmation_response(
        payload='{"key": "value"}',
        type="update",
        name="test-resource",
        kind="Deployment",
        cluster="local",
        namespace="default"
    )
    
    expected = '<confirmation-response>{"payload": "{\\"key\\": \\"value\\"}", "type": "update", "resource": {"name": "test-resource", "kind": "Deployment", "cluster": "local", "namespace": "default"}}</confirmation-response>'
    assert result == expected

def test_should_interrupt_returns_message_for_update_tools():
    """Verify interrupt message is generated for UPDATE validation tools."""
    validation_tools = [HumanValidationTool(name="patchKubernetesResource", type="UPDATE")]
    tool_call = {
        "name": "patchKubernetesResource",
        "args": {
            "patch": "[]",
            "name": "test",
            "kind": "Pod",
            "cluster": "local",
            "namespace": "default"
        }
    }
    
    result = should_interrupt(validation_tools, tool_call)
    
    expected = '<confirmation-response>{"payload": "[]", "type": "update", "resource": {"name": "test", "kind": "Pod", "cluster": "local", "namespace": "default"}}</confirmation-response>'
    assert result == expected

def test_should_interrupt_returns_empty_for_non_validated_tools():
    """Verify no interrupt message for tools without validation."""
    validation_tools = [HumanValidationTool(name="patchKubernetesResource", type="UPDATE")]
    tool_call = {
        "name": "getKubernetesResource",
        "args": {}
    }
    
    result = should_interrupt(validation_tools, tool_call)
    
    assert result == ""

def test_handle_interrupt_cancels_on_no_response():
    """Verify handle_interrupt returns False when user says no."""
    validation_tools = [HumanValidationTool(name="testTool", type="UPDATE")]
    tool_call = {
        "name": "testTool",
        "args": {
            "patch": "[]",
            "name": "test",
            "kind": "Pod",
            "cluster": "local",
            "namespace": "default"
        }
    }
    
    with patch("langgraph.types.interrupt", return_value="no"):
        should_continue, interrupt_msg = handle_interrupt(validation_tools, tool_call)
    
    assert should_continue is False
    assert interrupt_msg is not None

def test_handle_interrupt_continues_on_yes_response():
    """Verify handle_interrupt returns True when user says yes."""
    validation_tools = [HumanValidationTool(name="testTool", type="UPDATE")]
    tool_call = {
        "name": "testTool",
        "args": {
            "patch": "[]",
            "name": "test",
            "kind": "Pod",
            "cluster": "local",
            "namespace": "default"
        }
    }
    
    with patch("langgraph.types.interrupt", return_value="yes"):
        should_continue, interrupt_msg = handle_interrupt(validation_tools, tool_call)
    
    assert should_continue is True
    assert interrupt_msg is not None

def test_process_tool_result_handles_mcp_response_with_ui_context():
    """Verify MCP responses with uiContext are properly extracted."""
    tool_result = json.dumps({
        "llm": "LLM response",
        "uiContext": {"display": "data"}
    })
    
    with patch("app.services.agent.base.dispatch_custom_event"):
        processed, mcp_response = process_tool_result(tool_result, {})
    
    assert processed == "LLM response"
    assert mcp_response is not None
    assert "<mcp-response>" in mcp_response

def test_process_tool_result_handles_plain_string():
    """Verify plain string tool results are returned as-is."""
    tool_result = "Simple string response"
    
    processed, mcp_response = process_tool_result(tool_result, {})
    
    assert processed == "Simple string response"
    assert mcp_response is None

def test_process_tool_result_handles_list_format():
    """Verify list-formatted tool results are properly extracted."""
    tool_result = [{"type": "text", "text": "Extracted text", "id": "123"}]
    
    processed, mcp_response = process_tool_result(tool_result, {})
    
    assert processed == "Extracted text"
    assert mcp_response is None

def test_process_tool_result_handles_doc_links():
    """Verify docLinks are dispatched as custom events."""
    tool_result = json.dumps({
        "llm": "Response",
        "docLinks": ["https://docs.example.com"]
    })
    
    with patch("app.services.agent.base.dispatch_custom_event") as mock_dispatch:
        processed, _ = process_tool_result(tool_result, {})
        
        # Check that dock_link event was dispatched
        calls = [call for call in mock_dispatch.call_args_list if call[0][0] == "dock_link"]
        assert len(calls) == 1
        assert "https://docs.example.com" in calls[0][0][1]

def test_convert_to_string_if_needed_converts_dict():
    """Verify dicts are converted to JSON strings."""
    result = convert_to_string_if_needed({"key": "value"})
    assert result == '{"key": "value"}'

def test_convert_to_string_if_needed_converts_list():
    """Verify lists are converted to JSON strings."""
    result = convert_to_string_if_needed([1, 2, 3])
    assert result == '[1, 2, 3]'

def test_convert_to_string_if_needed_preserves_strings():
    """Verify strings are returned unchanged."""
    result = convert_to_string_if_needed("already a string")
    assert result == "already a string"

def test_convert_to_string_if_needed_preserves_primitives():
    """Verify primitive types are returned unchanged."""
    assert convert_to_string_if_needed(42) == 42
    assert convert_to_string_if_needed(True) is True
    assert convert_to_string_if_needed(None) is None