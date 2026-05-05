"""
Unit tests for the child agent module (app.services.agent.child).

Tests the module-level functions: _should_interrupt, _dispatch_ui_tools_event,
_dispatch_ui_tools, process_tool_result, convert_to_string_if_needed,
_build_interrupt_ui_tools, and create_child_agent.
"""
from app.services.ui_tools.models import UIToolsConfig
import pytest
import json

from unittest.mock import AsyncMock, MagicMock, patch
from app.services.agent.child import (
    _process_tool_result,
    convert_to_string_if_needed,
    create_child_agent,
    INTERRUPT_CANCEL_MESSAGE,
    INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE,
    _should_interrupt,
    _dispatch_ui_tools,
    _dispatch_ui_tools_event,
    _build_interrupt_ui_tools,
    _cancel_remaining_tool_calls,
    _build_agent_metadata,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, tool as langchain_tool


class MockTool:
    """Mock tool for testing tool execution (not a real langchain tool)."""
    def __init__(self, name, return_value):
        self.name = name
        self._return_value = return_value
        self.ainvoke = AsyncMock(return_value=return_value)
        self.metadata = {}


def _make_langchain_tool(name: str) -> "BaseTool":
    """Create a real langchain tool for tests that require it."""
    @langchain_tool(name)
    def _tool(x: str = "") -> str:
        """A test tool."""
        return "result"
    return _tool


@pytest.fixture
def mock_llm():
    """Mock LLM with bound tools."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="llm_response"))
    return llm


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for state persistence."""
    return False


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
        "patchKubernetesResource",
        "createKubernetesResource"
    ]
    return config


@pytest.fixture
def agent_config_without_validation():
    """Mock agent configuration without human validation."""
    config = MagicMock()
    config.human_validation_tools = []
    return config


# ============================================================================
# create_child_agent Tests
# ============================================================================

def test_create_child_agent_returns_compiled_graph(mock_llm, mock_checkpointer):
    """Verify create_child_agent returns a compiled graph."""
    tools = [_make_langchain_tool("testTool")]
    agent_config = MagicMock()
    agent_config.human_validation_tools = []

    graph = create_child_agent(
        llm=mock_llm,
        tools=tools,
        system_prompt="You are a helpful assistant",
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    assert graph is not None
    # It should have an invoke method (compiled graph)
    assert hasattr(graph, "ainvoke") or hasattr(graph, "invoke")


def test_create_child_agent_separates_planning_tools(mock_llm, mock_checkpointer):
    """Verify that Plan tools are separated from execution tools."""
    regular_tool = _make_langchain_tool("myTool")
    plan_tool = _make_langchain_tool("myToolPlan")
    agent_config = MagicMock()
    agent_config.human_validation_tools = []

    # Should not raise — plan tools are separated out
    graph = create_child_agent(
        llm=mock_llm,
        tools=[regular_tool, plan_tool],
        system_prompt="system",
        checkpointer=mock_checkpointer,
        agent_config=agent_config,
    )

    assert graph is not None


# ============================================================================
# _should_interrupt Tests
# ============================================================================

@pytest.mark.asyncio
async def test_should_interrupt_returns_message_for_validated_tools():
    """Verify interrupt message is generated for tools requiring human validation."""
    validation_tools = ["patchKubernetesResource"]
    plan_tool = MockTool("patchKubernetesResourcePlan", "plan response for patching")
    planning_tools_by_name = {"patchKubernetesResourcePlan": plan_tool}

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

    result = await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)

    assert "<confirmation-response>" in result
    assert "plan response for patching" in result


@pytest.mark.asyncio
async def test_should_interrupt_returns_empty_for_non_validated_tools():
    """Verify no interrupt message for tools without validation."""
    validation_tools = ["patchKubernetesResource"]
    planning_tools_by_name = {}

    tool_call = {
        "name": "getKubernetesResource",
        "args": {}
    }

    result = await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)

    assert result == ""


@pytest.mark.asyncio
async def test_should_interrupt_raises_if_plan_tool_missing():
    """Verify ValueError raised when Plan tool is missing for a validated tool."""
    validation_tools = ["myTool"]
    planning_tools_by_name = {}  # No plan tool available

    tool_call = {
        "name": "myTool",
        "args": {}
    }

    with pytest.raises(ValueError, match="planning tool 'myToolPlan' not found"):
        await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)


@pytest.mark.asyncio
async def test_should_interrupt_normalizes_list_response():
    """Verify list-format plan responses are normalized."""
    validation_tools = ["myTool"]
    plan_tool = MockTool("myToolPlan", [{"type": "text", "text": '{"action": "patch"}'}])
    planning_tools_by_name = {"myToolPlan": plan_tool}

    tool_call = {"name": "myTool", "args": {}}

    result = await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)

    assert "<confirmation-response>" in result
    assert "patch" in result


# ============================================================================
# _build_interrupt_ui_tools Tests
# ============================================================================

@patch('app.services.agent.child.dispatch_custom_event')
def test_build_interrupt_ui_tools_patch_operation(mock_dispatch):
    """Verify show-yaml-diff tools are built for patch operations."""
    interrupt_message = '<confirmation-response>{"type": "patch", "resource": {"kind": "Pod", "name": "test", "namespace": "ns"}, "payload": {"original": "a", "patched": "b"}}</confirmation-response>'
    state = {}
    config = {
        "configurable": {
            "request_metadata": {
                "ui_tools": {"name": "default", "tools": ["show-yaml-diff"]}
            }
        }
    }

    result = _build_interrupt_ui_tools(interrupt_message, state, config)

    assert len(result) == 1
    assert result[0]["toolName"] == "show-yaml-diff"
    assert result[0]["input"]["original"] == "a"
    assert result[0]["input"]["patched"] == "b"


@patch('app.services.agent.child.dispatch_custom_event')
def test_build_interrupt_ui_tools_create_operation(mock_dispatch):
    """Verify show-yaml tools are built for create operations."""
    interrupt_message = '<confirmation-response>{"type": "create", "resource": {"kind": "Pod", "name": "test", "namespace": "ns"}, "payload": {"yaml": "apiVersion: v1"}}</confirmation-response>'
    state = {}
    config = {
        "configurable": {
            "request_metadata": {
                "ui_tools": {"name": "default", "tools": ["show-yaml"]}
            }
        }
    }

    result = _build_interrupt_ui_tools(interrupt_message, state, config)

    assert len(result) == 1
    assert result[0]["toolName"] == "show-yaml"
    assert "yaml" in result[0]["input"]


def test_build_interrupt_ui_tools_missing_name_returns_empty():
    """Verify empty list when ui_tools config name is missing."""
    interrupt_message = '<confirmation-response>{"type": "patch"}</confirmation-response>'
    state = {}
    config = {"configurable": {"request_metadata": {"ui_tools": {}}}}

    result = _build_interrupt_ui_tools(interrupt_message, state, config)

    assert result == []


# ============================================================================
# _dispatch_ui_tools Tests
# ============================================================================

@patch('app.services.agent.child.dispatch_custom_event')
def test_dispatch_ui_tools_single_tool(mock_dispatch):
    """Test dispatching a single UI tool."""
    tools = [{"toolName": "selector", "input": {"resource": "pod"}}]

    _dispatch_ui_tools(tools)

    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert call_args[0][0] == "ui_tools"
    assert "selector" in call_args[0][1]


@patch('app.services.agent.child.dispatch_custom_event')
def test_dispatch_ui_tools_multiple_tools(mock_dispatch):
    """Test dispatching multiple UI tools."""
    tools = [
        {"toolName": "selector", "input": {"resource": "pod"}},
        {"toolName": "viewer", "input": {"format": "yaml"}}
    ]

    _dispatch_ui_tools(tools)

    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert "selector" in call_args[0][1]
    assert "viewer" in call_args[0][1]


@patch('app.services.agent.child.dispatch_custom_event', side_effect=Exception("dispatch error"))
def test_dispatch_ui_tools_handles_exception(mock_dispatch):
    """Test that exceptions in dispatch are handled gracefully."""
    tools = [{"toolName": "selector", "input": {}}]

    # Should not raise
    _dispatch_ui_tools(tools)


# ============================================================================
# _dispatch_ui_tools_event Tests
# ============================================================================

class TestDispatchUIToolsEvent:
    """Test _dispatch_ui_tools_event function."""

    @patch('app.services.agent.child.load_ui_tools_from_configmap')
    @patch('app.services.agent.child.create_ui_tools_selector')
    @patch('app.services.agent.child.dispatch_custom_event')
    def test_dispatch_ui_tools_event_success(
        self, mock_dispatch, mock_create_selector, mock_load_configmap, mock_llm
    ):
        """Test successfully dispatching UI tools."""
        from app.services.ui_tools.models import UITool, UIToolSchema, UIToolsConfig, UIToolsConfigData

        schema = UIToolSchema(type="object", properties={}, required=[])
        tool = UITool(
            name="test-selector",
            description="Test selector",
            prompt="Select",
            category="selector",
            schema=schema,
            metadata={},
            enabled=True
        )

        mock_config_data = UIToolsConfigData(
            config=UIToolsConfig(enabled=True, max_tools=5, system_prompt="Test"),
            tools=[tool]
        )
        mock_load_configmap.return_value = mock_config_data

        mock_selector = MagicMock()
        mock_selector.select_tools.return_value = [
            {"toolName": "test-selector", "input": {"resource": "pod"}}
        ]
        mock_create_selector.return_value = mock_selector

        state = {
            "messages": [HumanMessage(content="test")],
            "selected_agent": {"name": "rancher"}
        }

        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {
                    "ui_tools": {"name": "default", "tools": ["test-selector"]}
                }
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert len(result) == 1
        assert result[0]["toolName"] == "test-selector"

    def test_dispatch_ui_tools_event_missing_config_name(self, mock_llm):
        """Test skipping dispatch when config name is missing."""
        state = {"messages": [HumanMessage(content="test")]}
        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {"ui_tools": {}}  # Missing name
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert result == []

    def test_dispatch_ui_tools_event_empty_tools_filter(self, mock_llm):
        """Test skipping dispatch when tools filter list is empty."""
        state = {"messages": [HumanMessage(content="test")]}
        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {
                    "ui_tools": {"name": "default", "tools": []}
                }
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert result == []

    @patch('app.services.agent.child.load_ui_tools_from_configmap')
    @patch('app.services.agent.child.dispatch_custom_event')
    def test_dispatch_ui_tools_event_disabled_config(
        self, mock_dispatch, mock_load_configmap, mock_llm
    ):
        """Test skipping dispatch when config is disabled."""
        from app.services.ui_tools.models import UIToolsConfig, UIToolsConfigData

        mock_config_data = UIToolsConfigData(
            config=UIToolsConfig(enabled=False),
            tools=[]
        )
        mock_load_configmap.return_value = mock_config_data

        state = {"messages": [HumanMessage(content="test")]}
        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {"ui_tools": {"name": "default", "tools": ["x"]}}
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert result == []

    @patch('app.services.agent.child.load_ui_tools_from_configmap')
    @patch('app.services.agent.child.create_ui_tools_selector')
    @patch('app.services.agent.child.dispatch_custom_event')
    def test_dispatch_ui_tools_event_with_filtered_tools(
        self, mock_dispatch, mock_create_selector, mock_ui_tools_config, mock_llm
    ):
        """Test that only filtered tools are selected from available tools."""
        from app.services.ui_tools.models import UITool, UIToolSchema, UIToolsConfig, UIToolsConfigData

        schema = UIToolSchema(type="object", properties={}, required=[])

        selector_tool = UITool(
            name="test-selector", description="Test selector",
            prompt="Select", category="selector", schema=schema, metadata={}, enabled=True
        )
        viewer_tool = UITool(
            name="test-viewer", description="Test viewer",
            prompt="View", category="viewer", schema=schema, metadata={}, enabled=True
        )
        other_tool = UITool(
            name="test-other", description="Test other",
            prompt="Other", category="other", schema=schema, metadata={}, enabled=True
        )

        mock_config_data = UIToolsConfigData(
            config=UIToolsConfig(enabled=True, max_tools=5, system_prompt="Test"),
            tools=[selector_tool, viewer_tool, other_tool]
        )
        mock_ui_tools_config.return_value = mock_config_data

        mock_selector = MagicMock()
        mock_selector.select_tools.return_value = [
            {"toolName": "test-selector", "input": {"resource": "pod"}},
            {"toolName": "test-viewer", "input": {"format": "yaml"}}
        ]
        mock_create_selector.return_value = mock_selector

        state = {
            "messages": [HumanMessage(content="test")],
            "selected_agent": {"name": "rancher"}
        }

        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {
                    "ui_tools": {
                        "name": "default",
                        "tools": ["test-selector", "test-viewer"]
                    }
                }
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert len(result) == 2
        tool_names = [t["toolName"] for t in result]
        assert "test-selector" in tool_names
        assert "test-viewer" in tool_names
        assert "test-other" not in tool_names

    @patch('app.services.agent.child.load_ui_tools_from_configmap')
    @patch('app.services.agent.child.create_ui_tools_selector')
    @patch('app.services.agent.child.dispatch_custom_event')
    def test_dispatch_ui_tools_event_with_mcp_context(
        self, mock_dispatch, mock_create_selector, mock_ui_tools_config, mock_llm
    ):
        """Test that UI tools are selected with MCP response context."""
        from app.services.ui_tools.models import UITool, UIToolSchema, UIToolsConfig, UIToolsConfigData

        schema = UIToolSchema(type="object", properties={}, required=[])
        tool = UITool(
            name="resource-viewer", description="View resource",
            prompt="View", category="viewer", schema=schema, metadata={}, enabled=True
        )

        mock_config_data = UIToolsConfigData(
            config=UIToolsConfig(enabled=True, max_tools=5, system_prompt="Test"),
            tools=[tool]
        )
        mock_ui_tools_config.return_value = mock_config_data

        mock_selector = MagicMock()
        mock_selector.select_tools.return_value = [
            {"toolName": "resource-viewer", "input": {"format": "yaml"}}
        ]
        mock_create_selector.return_value = mock_selector

        mcp_response_msg = ToolMessage(
            content='{"resources": [{"id": "pod-123", "name": "test-pod"}]}',
            tool_call_id="mcp-1",
            name="get-resources",
            additional_kwargs={}
        )

        state = {
            "messages": [
                HumanMessage(content="get resources"),
                AIMessage(content="Fetching resources..."),
                mcp_response_msg
            ],
            "selected_agent": {"name": "rancher"}
        }

        config = {
            "configurable": {
                "request_id": "req-123",
                "request_metadata": {
                    "ui_tools": {
                        "name": "default",
                        "tools": ["resource-viewer"]
                    }
                }
            }
        }

        agent_config = MagicMock()
        result = _dispatch_ui_tools_event(mock_llm, agent_config, state, config)

        assert len(result) == 1
        assert result[0]["toolName"] == "resource-viewer"
        mock_selector.select_tools.assert_called_once()


# ============================================================================
# Preprocessed UI Tools with Confirmation Tests
# ============================================================================

class TestPreprocessedUIToolsWithConfirmation:
    """Test preprocessed UI tools (show-yaml, show-yaml-diff) with confirmation workflow."""

    @pytest.mark.asyncio
    @patch('app.services.agent.child.dispatch_custom_event')
    @patch('langgraph.types.interrupt')
    async def test_preprocessed_tools_dispatch_for_patch(self, mock_interrupt, mock_dispatch):
        """Test _should_interrupt + _build_interrupt_ui_tools for patch operation."""
        validation_tools = ["patchKubernetesResource"]

        plan_response = json.dumps({
            "type": "patch",
            "resource": {
                "kind": "Pod",
                "name": "test-pod",
                "namespace": "default"
            },
            "payload": {
                "original": "apiVersion: v1\nkind: Pod",
                "patched": "apiVersion: v1\nkind: Pod\nmodified: true"
            }
        })
        plan_tool = MockTool("patchKubernetesResourcePlan", plan_response)
        planning_tools_by_name = {"patchKubernetesResourcePlan": plan_tool}

        tool_call = {
            "name": "patchKubernetesResource",
            "args": {"patch": "[]", "name": "test-pod", "kind": "Pod", "cluster": "local", "namespace": "default"}
        }

        # Get the interrupt message
        interrupt_msg = await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)

        assert "<confirmation-response>" in interrupt_msg

        # Build UI tools from it
        config = {
            "configurable": {
                "request_metadata": {
                    "ui_tools": {"name": "default", "tools": ["show-yaml-diff"]}
                }
            }
        }
        ui_tools = _build_interrupt_ui_tools(interrupt_msg, {}, config)

        assert len(ui_tools) == 1
        assert ui_tools[0]["toolName"] == "show-yaml-diff"
        assert "original" in ui_tools[0]["input"]
        assert "patched" in ui_tools[0]["input"]

    @pytest.mark.asyncio
    @patch('app.services.agent.child.dispatch_custom_event')
    async def test_preprocessed_tools_dispatch_for_create(self, mock_dispatch):
        """Test _should_interrupt + _build_interrupt_ui_tools for create operation."""
        validation_tools = ["createKubernetesResource"]

        plan_response = json.dumps({
            "type": "create",
            "resource": {
                "kind": "Pod",
                "name": "test-pod",
                "namespace": "default"
            },
            "payload": {
                "yaml": "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test-pod"
            }
        })
        plan_tool = MockTool("createKubernetesResourcePlan", plan_response)
        planning_tools_by_name = {"createKubernetesResourcePlan": plan_tool}

        tool_call = {
            "name": "createKubernetesResource",
            "args": {"name": "test-pod", "kind": "Pod", "yaml": "apiVersion: v1"}
        }

        interrupt_msg = await _should_interrupt(validation_tools, tool_call, planning_tools_by_name)

        config = {
            "configurable": {
                "request_metadata": {
                    "ui_tools": {"name": "default", "tools": ["show-yaml"]}
                }
            }
        }
        ui_tools = _build_interrupt_ui_tools(interrupt_msg, {}, config)

        assert len(ui_tools) == 1
        assert ui_tools[0]["toolName"] == "show-yaml"
        assert "yaml" in ui_tools[0]["input"]


# ============================================================================
# process_tool_result Tests
# ============================================================================

def test_process_tool_result_handles_mcp_response_with_ui_context():
    """Verify MCP responses with uiContext are properly extracted."""
    tool_result = json.dumps({
        "llm": "LLM response",
        "uiContext": {"display": "data"}
    })

    with patch("app.services.agent.child.dispatch_custom_event"):
        processed, mcp_response = _process_tool_result(tool_result, {})

    assert processed == "LLM response"
    assert mcp_response is not None
    assert "<mcp-response>" in mcp_response


def test_process_tool_result_handles_plain_string():
    """Verify plain string tool results are returned as-is."""
    tool_result = "Simple string response"

    processed, mcp_response = _process_tool_result(tool_result, {})

    assert processed == "Simple string response"
    assert mcp_response is None


def test_process_tool_result_handles_list_format():
    """Verify list-formatted tool results are properly extracted."""
    tool_result = [{"type": "text", "text": "Extracted text", "id": "123"}]

    processed, mcp_response = _process_tool_result(tool_result, {})

    assert processed == "Extracted text"
    assert mcp_response is None


def test_process_tool_result_handles_doc_links():
    """Verify docLinks are dispatched as custom events."""
    tool_result = json.dumps({
        "llm": "Response",
        "docLinks": ["https://docs.example.com"]
    })

    with patch("app.services.agent.child.dispatch_custom_event") as mock_dispatch:
        processed, _ = _process_tool_result(tool_result, {})

        # Check that dock_link event was dispatched
        calls = [call for call in mock_dispatch.call_args_list if call[0][0] == "dock_link"]
        assert len(calls) == 1
        assert "https://docs.example.com" in calls[0][0][1]


def test_process_tool_result_handles_json_dict_without_llm_key():
    """Verify JSON dict without 'llm' key is returned as JSON string."""
    tool_result = json.dumps({"key": "value", "count": 42})

    processed, mcp_response = _process_tool_result(tool_result, {})

    assert processed == json.dumps({"key": "value", "count": 42})
    assert mcp_response is None


# ============================================================================
# convert_to_string_if_needed Tests
# ============================================================================

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


# ============================================================================
# build_agent_metadata Tests
# ============================================================================

def test_build_agent_metadata_basic():
    """Verify agent metadata string is built correctly."""
    result = _build_agent_metadata("rancher", "auto")
    assert '"agentName": "rancher"' in result
    assert '"selectionMode": "auto"' in result
    assert "<agent-metadata>" in result
    assert "</agent-metadata>" in result


def test_build_agent_metadata_with_extra():
    """Verify extra metadata is appended."""
    result = _build_agent_metadata("rancher", "manual", ', "extra": "data"')
    assert '"extra": "data"' in result
