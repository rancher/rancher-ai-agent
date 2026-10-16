import pytest
import json
import asyncio
from app.routers.websocket import (
    websocket_endpoint,
    build_chat_metadata,
    _extract_streaming_text,
    _extract_interrupt_value,
    _extract_text_from_chunk_content,
    _parse_websocket_request,
    _build_config,
    _resolve_target_agent,
    _build_input_data,
    _patch_tool_result,
    _call_agent,
    _record_stopped_in_history,
    _close_stopped_thread,
    _close_stopped_run,
    _cancel_waiting_plan,
    _send_stopped_plan_event,
    _stream_agent_events,
    _build_stop_reply,
    WebSocketRequest,
)
from app.constants import STOP_TAG, STOP_CANCEL_REPLY, STOP_CONTEXT_REPLY, STOP_REASON_PLANNING_MANUAL, STOP_REASON_MESSAGE_MANUAL, STOP_REASON_NEW_REQUEST
from app.services.agent._constants import NeedsOauth2
from app.services.agent.supervisor import SupervisorGraph
from app.services.oauth2.models import OAuth2Canceled
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class MockWebSocket:
    def __init__(self, messages=None):
        self.accepted = False
        self.closed = False
        self.cookies = {"R_SESS": "fake_token"}
        self.url = MagicMock()
        self.url.hostname = "fake.hostname"
        self.url.port = None
        self.client = MagicMock()
        self.client.host = "fake_client_host"
        self.client_state = WebSocketState.CONNECTED
        self._receive_queue = messages or []
        self._send_queue = []
        self.app = MagicMock()
        self.app.memory_manager = MagicMock()
        self.app.memory_manager.storage_type = MagicMock(value="in-memory")
        self.app.memory_manager.get_checkpointer = MagicMock(return_value=MagicMock())

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self._receive_queue:
            raise WebSocketDisconnect("No more messages")
        return self._receive_queue.pop(0)

    async def send_text(self, data):
        self._send_queue.append(data)

    async def close(self):
        self.closed = True


@pytest.fixture
def mock_dependencies():
    with patch('app.routers.websocket.build_agent', new_callable=AsyncMock) as mock_build_agent, \
         patch('app.routers.websocket._call_agent', new_callable=AsyncMock) as mock_call_agent, \
         patch('app.routers.websocket.get_user_id_from_token', new_callable=AsyncMock) as mock_get_user_id:

        mock_agent = MagicMock()
        mock_agent.aget_state = AsyncMock(return_value=MagicMock(interrupts=[]))
        agents_metadata = [{"name": "rancher", "status": "active"}]
        mock_build_agent.return_value = (mock_agent, agents_metadata)
        mock_get_user_id.return_value = "user-123"

        yield {
            "build_agent": mock_build_agent,
            "call_agent": mock_call_agent,
            "get_user_id": mock_get_user_id,
            "agent": mock_agent,
            "agents_metadata": agents_metadata,
        }


# --- Tests for websocket_endpoint ---

@pytest.mark.asyncio
async def test_websocket_endpoint(mock_dependencies):
    mock_ws = MockWebSocket(messages=["test message"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, None, mock_llm)

    assert mock_ws.accepted
    mock_dependencies["build_agent"].assert_called_once_with(llm=mock_llm, websocket=mock_ws)
    mock_dependencies["call_agent"].assert_awaited_once()

    call_kwargs = mock_dependencies["call_agent"].call_args.kwargs
    assert call_kwargs['websocket'] == mock_ws
    assert call_kwargs['agent'] == mock_dependencies["agent"]


@pytest.mark.asyncio
async def test_websocket_endpoint_context_message(mock_dependencies):
    mock_ws = MockWebSocket(messages=[
        '{"prompt": "show all pods", "context": {"namespace": "default", "cluster": "local"}}'
    ])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, None, mock_llm)

    mock_dependencies["build_agent"].assert_called_once()
    mock_dependencies["call_agent"].assert_awaited_once()

    call_kwargs = mock_dependencies["call_agent"].call_args.kwargs
    input_data = call_kwargs['input_data']
    assert "messages" in input_data
    assert "show all pods" in input_data["messages"][0].content
    assert "namespace" in input_data["messages"][0].content
    assert call_kwargs['websocket'] == mock_ws


@pytest.mark.asyncio
async def test_websocket_endpoint_sends_chat_metadata(mock_dependencies):
    mock_ws = MockWebSocket(messages=["hello"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, None, mock_llm)

    # First message sent should be chat metadata
    metadata_msg = mock_ws._send_queue[0]
    assert "<chat-metadata>" in metadata_msg
    assert "chatId" in metadata_msg
    assert "storageType" in metadata_msg


@pytest.mark.asyncio
async def test_websocket_endpoint_no_agent_available(mock_dependencies):
    from app.services.agent.factory import NoAgentAvailableError
    mock_dependencies["build_agent"].side_effect = NoAgentAvailableError("No agents")
    mock_ws = MockWebSocket(messages=[])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, None, mock_llm)

    assert mock_ws.accepted
    assert mock_ws.closed
    error_msg = mock_ws._send_queue[0]
    assert "<chat-error>" in error_msg
    assert "No agents" in error_msg


@pytest.mark.asyncio
async def test_websocket_endpoint_with_thread_id(mock_dependencies):
    mock_ws = MockWebSocket(messages=["hello"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, "custom-thread-id", mock_llm)

    assert mock_ws.accepted
    metadata_msg = mock_ws._send_queue[0]
    assert "custom-thread-id" in metadata_msg


@pytest.mark.asyncio
async def test_websocket_endpoint_error_during_processing(mock_dependencies):
    """Test that errors during message processing are sent to the client."""
    mock_dependencies["call_agent"].side_effect = Exception("Something went wrong")
    mock_ws = MockWebSocket(messages=["test", "second"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, None, mock_llm)

    # Should have sent error and end message
    error_messages = [m for m in mock_ws._send_queue if "<error>" in m]
    assert len(error_messages) >= 1
    assert "Something went wrong" in error_messages[0]


# --- Tests for build_chat_metadata ---

class TestBuildChatMetadata:
    def test_basic_metadata(self):
        mock_ws = MockWebSocket()
        result = build_chat_metadata("thread-123", [{"name": "rancher", "status": "active"}], mock_ws)

        assert "<chat-metadata>" in result
        assert "thread-123" in result
        assert "in-memory" in result
        assert "rancher" in result

    def test_multiple_agents(self):
        mock_ws = MockWebSocket()
        agents = [
            {"name": "rancher", "status": "active"},
            {"name": "fleet", "status": "active"},
        ]
        result = build_chat_metadata("thread-456", agents, mock_ws)

        assert "rancher" in result
        assert "fleet" in result

    def test_empty_agents(self):
        mock_ws = MockWebSocket()
        result = build_chat_metadata("thread-789", [], mock_ws)

        assert "<chat-metadata>" in result
        assert "thread-789" in result


# --- Tests for _extract_streaming_text ---

class TestExtractStreamingText:
    def test_extracts_text_from_valid_stream(self):
        chunk = MagicMock()
        chunk.content = "Hello world"
        stream = {
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": chunk},
        }
        result = _extract_streaming_text(stream)
        assert result == "Hello world"

    def test_returns_none_for_empty_chunk(self):
        stream = {
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": None},
        }
        result = _extract_streaming_text(stream)
        assert result is None

    def test_returns_none_for_empty_content(self):
        chunk = MagicMock()
        chunk.content = ""
        stream = {
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": chunk},
        }
        result = _extract_streaming_text(stream)
        assert result is None


# --- Tests for _extract_interrupt_value ---

class TestExtractInterruptValue:
    def test_extracts_string_interrupt(self):
        interrupt = MagicMock()
        interrupt.value = "Please confirm"
        stream = {
            "data": {
                "chunk": ("updates", {"__interrupt__": [interrupt]})
            }
        }
        result = _extract_interrupt_value(stream)
        assert result == "Please confirm"

    def test_extracts_dict_interrupt_as_json(self):
        interrupt = MagicMock()
        interrupt.value = {"action": "confirm", "tool": "delete"}
        stream = {
            "data": {
                "chunk": ("updates", {"__interrupt__": [interrupt]})
            }
        }
        result = _extract_interrupt_value(stream)
        assert json.loads(result) == {"action": "confirm", "tool": "delete"}

    def test_returns_none_for_non_updates_chunk(self):
        stream = {
            "data": {
                "chunk": ("messages", {"some": "data"})
            }
        }
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_no_interrupts(self):
        stream = {
            "data": {
                "chunk": ("updates", {"__interrupt__": []})
            }
        }
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_none_value(self):
        interrupt = MagicMock()
        interrupt.value = None
        stream = {
            "data": {
                "chunk": ("updates", {"__interrupt__": [interrupt]})
            }
        }
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_empty_string_value(self):
        interrupt = MagicMock()
        interrupt.value = ""
        stream = {
            "data": {
                "chunk": ("updates", {"__interrupt__": [interrupt]})
            }
        }
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_non_dict_data(self):
        stream = {"data": "not a dict"}
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_short_chunk(self):
        stream = {"data": {"chunk": ("only_one",)}}
        result = _extract_interrupt_value(stream)
        assert result is None

    def test_returns_none_for_non_dict_updates(self):
        stream = {"data": {"chunk": ("updates", "not_a_dict")}}
        result = _extract_interrupt_value(stream)
        assert result is None


# --- Tests for _extract_text_from_chunk_content ---

class TestExtractTextFromChunkContent:
    def test_simple_string(self):
        assert _extract_text_from_chunk_content("hello") == "hello"

    def test_list_of_dicts_with_text(self):
        content = [{"text": "Hello "}, {"text": "world"}]
        assert _extract_text_from_chunk_content(content) == "Hello world"

    def test_list_with_non_dict_items(self):
        content = [{"text": "a"}, "not_a_dict", {"text": "b"}]
        assert _extract_text_from_chunk_content(content) == "ab"

    def test_dict_with_text_key(self):
        content = {"text": "hello there"}
        assert _extract_text_from_chunk_content(content) == "hello there"

    def test_none_content(self):
        assert _extract_text_from_chunk_content(None) == ""

    def test_empty_list(self):
        assert _extract_text_from_chunk_content([]) == ""

    def test_list_with_missing_text_key(self):
        content = [{"type": "image"}, {"text": "caption"}]
        assert _extract_text_from_chunk_content(content) == "caption"

    def test_integer_content(self):
        assert _extract_text_from_chunk_content(42) == "42"


# --- Tests for _parse_websocket_request ---

class TestParseWebsocketRequest:
    def test_plain_text_request(self):
        result = _parse_websocket_request("hello world")
        assert result.prompt == "hello world"
        assert result.user_input == ""
        assert result.context == {}
        assert result.agent == ""
        assert result.ui_tools == {}

    def test_json_request_with_prompt(self):
        request = json.dumps({"prompt": "list pods"})
        result = _parse_websocket_request(request)
        assert result.prompt == "list pods"
        assert result.user_input == "list pods"
        assert result.context == {}

    def test_json_request_with_context(self):
        request = json.dumps({
            "prompt": "show pods",
            "context": {"namespace": "default", "cluster": "local"}
        })
        result = _parse_websocket_request(request)
        assert "show pods" in result.prompt
        assert "namespace:default" in result.prompt
        assert "cluster:local" in result.prompt
        assert result.user_input == "show pods"
        assert result.context == {"namespace": "default", "cluster": "local"}

    def test_json_request_with_agent(self):
        request = json.dumps({"prompt": "deploy app", "agent": "fleet"})
        result = _parse_websocket_request(request)
        assert result.agent == "fleet"

    def test_json_request_with_tags(self):
        request = json.dumps({"prompt": "hello", "tags": ["mytag"]})
        result = _parse_websocket_request(request)
        assert result.tags == ["mytag"]

    def test_json_request_with_labels(self):
        request = json.dumps({"prompt": "hello", "labels": {"source": "ui"}})
        result = _parse_websocket_request(request)
        assert result.labels == {"source": "ui"}

    def test_json_request_with_ui_tools(self):
        tools = {"tool1": {"param": "value"}}
        request = json.dumps({"prompt": "hello", "tools": tools})
        result = _parse_websocket_request(request)
        assert result.ui_tools == tools

    def test_invalid_json_falls_back_to_plain_text(self):
        result = _parse_websocket_request("not{valid json")
        assert result.prompt == "not{valid json"
        assert result.user_input == ""

    def test_empty_context_does_not_enrich_prompt(self):
        request = json.dumps({"prompt": "hello", "context": {}})
        result = _parse_websocket_request(request)
        assert result.prompt == "hello"


# --- Tests for _build_config ---

class TestBuildConfig:
    def test_basic_config(self):
        base_config = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="hello", user_input="hello", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )
        result = _build_config(base_config, "req-1", ws_request)

        assert result["configurable"]["thread_id"] == "t1"
        assert result["configurable"]["request_id"] == "req-1"
        assert result["configurable"]["agent"] == ""

    def test_config_with_agent(self):
        base_config = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="deploy", user_input="deploy", context={},
            tags=[], labels={}, agent="fleet", ui_tools={}
        )
        result = _build_config(base_config, "req-2", ws_request)
        assert result["configurable"]["agent"] == "fleet"

    def test_request_metadata_included(self):
        base_config = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="hello", user_input="hello", context={"ns": "dev"},
            tags=["tag1"], labels={"k": "v"}, agent="rancher", ui_tools={"t": {}}
        )
        result = _build_config(base_config, "req-4", ws_request)
        metadata = result["configurable"]["request_metadata"]
        assert metadata["agent"] == "rancher"
        assert metadata["user_input"] == "hello"
        assert metadata["context"] == {"ns": "dev"}
        assert metadata["labels"] == {"k": "v"}
        assert metadata["tags"] == ["tag1"]
        assert metadata["ui_tools"] == {"t": {}}

    def test_does_not_mutate_base_config(self):
        base_config = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="hello", user_input="hello", context={},
            tags=["mytag"], labels={}, agent="fleet", ui_tools={}
        )
        _build_config(base_config, "req-5", ws_request)
        assert base_config["configurable"]["thread_id"] == "t1"
        assert "agent" not in base_config["configurable"]


# --- Tests for _resolve_target_agent ---

class TestResolveTargetAgent:
    @pytest.mark.asyncio
    async def test_no_agent_requested_returns_supervisor(self):
        agent = MagicMock()
        config = {"configurable": {"thread_id": "t1"}}
        ws_request = WebSocketRequest(
            prompt="hi", user_input="hi", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )
        websocket = MagicMock()
        llm = MagicMock()
        result_agent, result_config = await _resolve_target_agent(agent, config, ws_request, websocket, llm)
        assert result_agent is agent
        assert result_config is config

    @pytest.mark.asyncio
    async def test_supervisor_routes_to_known_child(self):
        child_graph = MagicMock()
        child_agent = MagicMock()
        child_agent.agent = child_graph
        child_agent.needs_oauth2 = False
        supervisor = SupervisorGraph(
            graph=MagicMock(),
            child_agents={"fleet": child_agent},
        )
        config = {"configurable": {"thread_id": "t1", "request_id": "r1", "request_metadata": {}, "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="deploy", user_input="deploy", context={},
            tags=[], labels={}, agent="fleet", ui_tools={}
        )
        websocket = MagicMock()
        llm = MagicMock()
        result_agent, result_config = await _resolve_target_agent(supervisor, config, ws_request, websocket, llm)
        assert result_agent is child_graph
        assert result_config is config

    @pytest.mark.asyncio
    async def test_supervisor_falls_back_for_unknown_child(self):
        supervisor = SupervisorGraph(
            graph=MagicMock(),
            child_agents={"rancher": MagicMock()},
        )
        config = {"configurable": {"thread_id": "t1", "request_id": "r1", "request_metadata": {}, "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="hi", user_input="hi", context={},
            tags=[], labels={}, agent="unknown_agent", ui_tools={}
        )
        websocket = MagicMock()
        llm = MagicMock()
        result_agent, result_config = await _resolve_target_agent(supervisor, config, ws_request, websocket, llm)
        # Should fall back to supervisor
        assert result_config is config
        assert result_agent is supervisor


# --- Tests for _build_input_data ---

class TestBuildInputData:
    @pytest.mark.asyncio
    async def test_new_message_without_interrupt(self):
        agent = MagicMock()
        state = MagicMock()
        state.interrupts = []
        agent.aget_state = AsyncMock(return_value=state)
        config = {"configurable": {"request_id": "r1", "request_metadata": {}}}
        ws_request = WebSocketRequest(
            prompt="hello", user_input="hello", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )

        result = await _build_input_data(agent, config, ws_request)
        assert "messages" in result
        assert result["messages"][0].content == "hello"

    @pytest.mark.asyncio
    async def test_resume_on_interrupt(self):
        agent = MagicMock()
        state = MagicMock()
        state.interrupts = [MagicMock()]  # Non-empty interrupts
        agent.aget_state = AsyncMock(return_value=state)
        config = {"configurable": {"request_id": "r1", "request_metadata": {}}}
        ws_request = WebSocketRequest(
            prompt="yes, proceed", user_input="yes, proceed", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )

        result = await _build_input_data(agent, config, ws_request)
        assert isinstance(result, Command)
        assert result.resume == "yes, proceed"

    @pytest.mark.asyncio
    async def test_resume_restores_ui_tools_from_interrupted_turn(self):
        """On resume, ui_tools omitted by the confirmation message are restored
        from the interrupted turn's HumanMessage so later agents still get them."""
        from langchain_core.messages import HumanMessage

        original_ui_tools = {"name": "show-yaml", "tools": ["show-yaml-diff"]}
        human_message = HumanMessage(
            content="update the deployment",
            additional_kwargs={"request_metadata": {"ui_tools": original_ui_tools}},
        )

        agent = MagicMock()
        state = MagicMock()
        state.interrupts = [MagicMock()]
        state.values = {"messages": [human_message]}
        agent.aget_state = AsyncMock(return_value=state)

        # Resume message omits tools, so request_metadata.ui_tools is empty.
        config = {"configurable": {"request_id": "r1", "request_metadata": {"ui_tools": {}}}}
        ws_request = WebSocketRequest(
            prompt="yes", user_input="yes", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )

        result = await _build_input_data(agent, config, ws_request)
        assert isinstance(result, Command)
        assert config["configurable"]["request_metadata"]["ui_tools"] == original_ui_tools

    @pytest.mark.asyncio
    async def test_resume_keeps_client_supplied_ui_tools(self):
        """If the client resends tools on resume, they are not overwritten."""
        from langchain_core.messages import HumanMessage

        client_ui_tools = {"name": "client-tool"}
        human_message = HumanMessage(
            content="update the deployment",
            additional_kwargs={"request_metadata": {"ui_tools": {"name": "old-tool"}}},
        )

        agent = MagicMock()
        state = MagicMock()
        state.interrupts = [MagicMock()]
        state.values = {"messages": [human_message]}
        agent.aget_state = AsyncMock(return_value=state)

        config = {"configurable": {"request_id": "r1", "request_metadata": {"ui_tools": client_ui_tools}}}
        ws_request = WebSocketRequest(
            prompt="yes", user_input="yes", context={},
            tags=[], labels={}, agent="", ui_tools=client_ui_tools
        )

        result = await _build_input_data(agent, config, ws_request)
        assert isinstance(result, Command)
        assert config["configurable"]["request_metadata"]["ui_tools"] == client_ui_tools

    @pytest.mark.asyncio
    async def test_message_includes_request_metadata(self):
        agent = MagicMock()
        state = MagicMock()
        state.interrupts = []
        agent.aget_state = AsyncMock(return_value=state)
        config = {"configurable": {"request_id": "req-42", "request_metadata": {"agent": "rancher"}}}
        ws_request = WebSocketRequest(
            prompt="list pods", user_input="list pods", context={},
            tags=[], labels={}, agent="", ui_tools={}
        )

        result = await _build_input_data(agent, config, ws_request)
        msg = result["messages"][0]
        assert msg.additional_kwargs["request_id"] == "req-42"


# --- Tests for _patch_tool_result ---

class TestPatchToolResult:
    @pytest.mark.asyncio
    async def test_injects_tool_messages_for_pending_tool_calls(self):
        """When the last message has tool_calls, ToolMessages should be injected."""
        last_msg = MagicMock()
        last_msg.tool_calls = [
            {"id": "call_abc123", "name": "some_tool", "args": {}},
            {"id": "call_def456", "name": "another_tool", "args": {}},
        ]

        state = MagicMock()
        state.values = {"messages": [MagicMock(), last_msg]}

        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=state)
        agent.aupdate_state = AsyncMock()

        config = {"configurable": {"thread_id": "t1"}}

        await _patch_tool_result(agent, config, "OAuth2 cancelled")

        agent.aupdate_state.assert_awaited_once()
        call_kwargs = agent.aupdate_state.call_args.kwargs
        tool_messages = call_kwargs["values"]["messages"]
        assert len(tool_messages) == 2
        assert tool_messages[0].tool_call_id == "call_abc123"
        assert tool_messages[0].content == "OAuth2 cancelled"
        assert tool_messages[0].status == "error"
        assert tool_messages[1].tool_call_id == "call_def456"

    @pytest.mark.asyncio
    async def test_no_op_when_no_messages(self):
        """When graph state has no messages, nothing should happen."""
        state = MagicMock()
        state.values = {"messages": []}

        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=state)
        agent.aupdate_state = AsyncMock()

        config = {"configurable": {"thread_id": "t1"}}

        await _patch_tool_result(agent, config, "error")

        agent.aupdate_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_op_when_last_message_has_no_tool_calls(self):
        """When the last message has no tool_calls, nothing should happen."""
        last_msg = MagicMock()
        last_msg.tool_calls = None

        state = MagicMock()
        state.values = {"messages": [last_msg]}

        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=state)
        agent.aupdate_state = AsyncMock()

        config = {"configurable": {"thread_id": "t1"}}

        await _patch_tool_result(agent, config, "error")

        agent.aupdate_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_op_when_state_is_none(self):
        """When aget_state returns None, nothing should happen."""
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=None)
        agent.aupdate_state = AsyncMock()

        config = {"configurable": {"thread_id": "t1"}}

        await _patch_tool_result(agent, config, "error")

        agent.aupdate_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_op_when_tool_calls_is_empty_list(self):
        """When tool_calls is an empty list, nothing should happen."""
        last_msg = MagicMock()
        last_msg.tool_calls = []

        state = MagicMock()
        state.values = {"messages": [last_msg]}

        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=state)
        agent.aupdate_state = AsyncMock()

        config = {"configurable": {"thread_id": "t1"}}

        await _patch_tool_result(agent, config, "error")

        agent.aupdate_state.assert_not_awaited()


# --- Tests for _resolve_target_agent with OAuth2Cancelled ---

class TestResolveTargetAgentOAuth2:
    @pytest.mark.asyncio
    async def test_returns_none_when_oauth2_cancelled(self):
        """When OAuth2 is cancelled for a child agent, should return (None, config)."""
        child_agent = MagicMock()
        child_agent.needs_oauth2 = True
        child_agent.config = MagicMock()
        child_agent.config.name = "fleet"

        supervisor = SupervisorGraph(
            graph=MagicMock(),
            child_agents={"fleet": child_agent},
        )
        config = {"configurable": {"thread_id": "t1", "request_id": "r1", "request_metadata": {}, "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="deploy", user_input="deploy", context={},
            tags=[], labels={}, agent="fleet", ui_tools={}
        )
        websocket = AsyncMock()
        llm = MagicMock()

        with patch("app.routers.websocket.handle_oauth_authentication", new_callable=AsyncMock, side_effect=OAuth2Canceled("cancelled")):
            result_agent, result_config = await _resolve_target_agent(supervisor, config, ws_request, websocket, llm)

        assert result_agent is None
        assert result_config is config

    @pytest.mark.asyncio
    async def test_returns_none_when_oauth2_fails(self):
        """When OAuth2 raises a generic exception, should return (None, config) and send auth-error."""
        child_agent = MagicMock()
        child_agent.needs_oauth2 = True
        child_agent.config = MagicMock()
        child_agent.config.name = "fleet"

        supervisor = SupervisorGraph(
            graph=MagicMock(),
            child_agents={"fleet": child_agent},
        )
        config = {"configurable": {"thread_id": "t1", "request_id": "r1", "request_metadata": {}, "user_id": "u1"}}
        ws_request = WebSocketRequest(
            prompt="deploy", user_input="deploy", context={},
            tags=[], labels={}, agent="fleet", ui_tools={}
        )
        websocket = AsyncMock()
        llm = MagicMock()

        with patch("app.routers.websocket.handle_oauth_authentication", new_callable=AsyncMock, side_effect=Exception("token expired")):
            result_agent, result_config = await _resolve_target_agent(supervisor, config, ws_request, websocket, llm)

        assert result_agent is None
        assert result_config is config
        # Should have sent auth-error
        auth_error_sent = any("<auth-error>" in str(call) for call in websocket.send_text.call_args_list)
        assert auth_error_sent


# --- Tests for _call_agent stop functionality ---

def _make_stream_agent(events, stall_after=None):
    """
    Build a mock agent whose ``astream_events`` yields the given events.

    Args:
        events: List of event dicts to yield.
        stall_after: If set, after yielding this many events the generator awaits
            a long sleep, simulating an in-flight/long-running execution that can
            be cancelled.
    """
    agent = MagicMock()

    async def _astream_events(input_data, config=None, stream_mode=None):
        for i, event in enumerate(events):
            yield event
            if stall_after is not None and i + 1 >= stall_after:
                await asyncio.sleep(60)

    agent.astream_events = _astream_events
    agent.aget_state = AsyncMock(return_value=MagicMock(values={"messages": []}))
    agent.aupdate_state = AsyncMock()
    return agent


def _text_event(text):
    chunk = MagicMock()
    chunk.content = text
    return {"event": "on_chat_model_stream", "data": {"chunk": chunk}, "tags": [], "metadata": {}}


def _stop_payload():
    return json.dumps({"prompt": "", "tags": [STOP_TAG]})


class TestCallAgentStop:
    @pytest.mark.asyncio
    async def test_stop_message_cancels_execution(self):
        """A stop-tagged message mid-stream cancels streaming and patches tool result."""
        agent = _make_stream_agent([_text_event("partial")], stall_after=1)
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(return_value=_stop_payload())

        with patch("app.routers.websocket._close_stopped_run", new_callable=AsyncMock) as mock_close, \
             patch("app.routers.websocket._record_stopped_in_history", new_callable=AsyncMock) as mock_record:
            result = await _call_agent(agent=agent, input_data={"messages": []}, config={"configurable": {}}, websocket=websocket)

        assert result is None
        mock_close.assert_awaited_once()
        # Stopped run recorded in history with the partial streamed text
        mock_record.assert_awaited_once()
        assert mock_record.call_args.args[2] == "partial"
        assert mock_record.call_args.args[3] == STOP_REASON_MESSAGE_MANUAL
        # Opened the message block
        websocket.send_text.assert_any_await("<message>")
        # Notified the client that the execution was stopped by the user
        websocket.send_text.assert_any_await(_build_stop_reply(STOP_REASON_MESSAGE_MANUAL))
        assert _build_stop_reply(STOP_REASON_MESSAGE_MANUAL) == '<stop-message>{"reason": "message-manual-stop"}</stop-message>'

    @pytest.mark.asyncio
    async def test_normal_completion_no_patch(self):
        """When streaming completes normally, the pending receiver is cancelled and no patch occurs."""
        agent = _make_stream_agent([_text_event("hello"), _text_event(" world")])

        async def _block():
            await asyncio.sleep(60)

        websocket = AsyncMock()
        # Receiver blocks forever so streaming wins the race.
        websocket.receive_text = AsyncMock(side_effect=_block)

        with patch("app.routers.websocket._close_stopped_run", new_callable=AsyncMock) as mock_close:
            await _call_agent(agent=agent, input_data={"messages": []}, config={"configurable": {}}, websocket=websocket)

        mock_close.assert_not_awaited()
        # Streamed text was forwarded
        websocket.send_text.assert_any_await("hello")
        websocket.send_text.assert_any_await(" world")

    @pytest.mark.asyncio
    async def test_new_message_preempts_run(self):
        """A non-stop message mid-run cancels the run and is returned for processing."""
        agent = _make_stream_agent([_text_event("partial")], stall_after=1)
        new_message = json.dumps({"prompt": "new question"})
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(return_value=new_message)

        with patch("app.routers.websocket._close_stopped_run", new_callable=AsyncMock) as mock_close, \
             patch("app.routers.websocket._record_stopped_in_history", new_callable=AsyncMock) as mock_record:
            result = await _call_agent(agent=agent, input_data={"messages": []}, config={"configurable": {}}, websocket=websocket)

        assert result == new_message
        mock_close.assert_awaited_once()
        mock_record.assert_awaited_once()
        assert mock_record.call_args.args[2] == "partial"
        assert mock_record.call_args.args[3] == STOP_REASON_NEW_REQUEST
        # The client is notified that the previous execution was preempted
        websocket.send_text.assert_any_await(_build_stop_reply(STOP_REASON_NEW_REQUEST))

    @pytest.mark.asyncio
    async def test_stream_exception_propagates(self):
        """Exceptions raised by the stream (e.g. NeedsOauth2) propagate out of _call_agent."""
        agent = MagicMock()
        agent_cfg = MagicMock()

        async def _astream_events(input_data, config=None, stream_mode=None):
            raise NeedsOauth2(agent_cfg)
            yield  # pragma: no cover - makes this an async generator

        agent.astream_events = _astream_events

        async def _block():
            await asyncio.sleep(60)

        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(side_effect=_block)

        with pytest.raises(NeedsOauth2):
            await _call_agent(agent=agent, input_data={"messages": []}, config={"configurable": {}}, websocket=websocket)


class TestStreamPartialText:
    @pytest.mark.asyncio
    async def test_collects_streamed_text(self):
        """Streamed text is collected into the partial_text buffer."""
        agent = _make_stream_agent([_text_event("hello"), _text_event(" world")])
        partial_text = []

        await _stream_agent_events(agent, {"messages": []}, {"configurable": {}}, AsyncMock(), partial_text)

        assert partial_text == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_model_end_clears_buffer(self):
        """Text from a finished LLM call is dropped; only the in-progress call is kept."""
        end_event = {"event": "on_chat_model_end", "data": {}, "tags": [], "metadata": {}}
        agent = _make_stream_agent([_text_event("saved"), end_event, _text_event("in progress")])
        partial_text = []

        await _stream_agent_events(agent, {"messages": []}, {"configurable": {}}, AsyncMock(), partial_text)

        assert partial_text == ["in progress"]

    @pytest.mark.asyncio
    async def test_no_stream_model_end_keeps_buffer(self):
        """The end of an internal no-stream LLM call does not clear the buffer."""
        end_event = {"event": "on_chat_model_end", "data": {}, "tags": ["no-stream"], "metadata": {}}
        agent = _make_stream_agent([_text_event("partial"), end_event])
        partial_text = []

        await _stream_agent_events(agent, {"messages": []}, {"configurable": {}}, AsyncMock(), partial_text)

        assert partial_text == ["partial"]


class TestRecordStoppedInHistory:
    def _config(self):
        return {"configurable": {"request_id": "req1", "request_metadata": {"agent": "a", "tags": []}}}

    def _agent(self, messages, history):
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=MagicMock(values={"messages": messages, "messages_history": history}))
        agent.aupdate_state = AsyncMock()
        return agent

    def _written_history(self, agent):
        return agent.aupdate_state.call_args.kwargs["values"]["messages_history"]

    @pytest.mark.asyncio
    async def test_adds_missing_human_message_and_stopped_ai_message(self):
        human = HumanMessage(content="question", id="h1", additional_kwargs={"request_id": "req1"})
        agent = self._agent(messages=[human], history=[])

        await _record_stopped_in_history(agent, self._config(), "partial answer", STOP_REASON_MESSAGE_MANUAL)

        history = self._written_history(agent)
        assert len(history) == 2
        assert history[0] is human
        assert isinstance(history[1], AIMessage)
        assert history[1].content == "partial answer"
        assert history[1].additional_kwargs["stopped"] == STOP_REASON_MESSAGE_MANUAL
        assert history[1].additional_kwargs["request_metadata"] == {"agent": "a", "tags": []}
        assert "created_at" in history[1].additional_kwargs

    @pytest.mark.asyncio
    async def test_does_not_duplicate_human_message(self):
        human = HumanMessage(content="question", id="h1", additional_kwargs={"request_id": "req1"})
        agent = self._agent(messages=[human], history=[human])

        await _record_stopped_in_history(agent, self._config(), "", STOP_REASON_NEW_REQUEST)

        history = self._written_history(agent)
        assert [m.id for m in history if isinstance(m, HumanMessage)] == ["h1"]
        assert history[-1].additional_kwargs["stopped"] == STOP_REASON_NEW_REQUEST

    @pytest.mark.asyncio
    async def test_keeps_full_history(self):
        """messages_history is a LastValue channel, so previous entries must be carried forward."""
        old_human = HumanMessage(content="old", id="h0", additional_kwargs={"request_id": "req0"})
        old_ai = AIMessage(content="old answer", id="a0")
        human = HumanMessage(content="question", id="h1", additional_kwargs={"request_id": "req1"})
        agent = self._agent(messages=[old_human, old_ai, human], history=[old_human, old_ai])

        await _record_stopped_in_history(agent, self._config(), "partial", STOP_REASON_MESSAGE_MANUAL)

        history = self._written_history(agent)
        assert [m.id for m in history[:3]] == ["h0", "a0", "h1"]
        assert history[3].additional_kwargs["stopped"] == STOP_REASON_MESSAGE_MANUAL


class TestCloseStoppedThread:
    def _config(self, thread_id="t1"):
        return {"configurable": {"thread_id": thread_id}}

    def _agent(self, values):
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=MagicMock(values=values))
        agent.aupdate_state = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_unanswered_request_gets_stop_reply_and_planner_state_reset(self):
        human = HumanMessage(content="create ns", id="h1")
        agent = self._agent({
            "messages": [human],
            "messages_history": [human],
            "subtasks": [{"task": "x", "status": "in_progress", "agent": "rancher"}],
            "results": ["r"],
            "awaiting_input": True,
        })

        await _close_stopped_thread(agent, self._config())

        values = agent.aupdate_state.call_args.kwargs["values"]
        assert len(values["messages"]) == 1
        reply = values["messages"][0]
        assert isinstance(reply, AIMessage)
        assert reply.content == STOP_CONTEXT_REPLY
        assert reply.additional_kwargs["stopped"] is True
        assert values["subtasks"] == []
        assert values["results"] == []
        assert values["awaiting_input"] is False
        # Keys absent from the state are not written, and history is left untouched.
        assert "feedback" not in values
        assert "retry" not in values
        assert "messages_history" not in values

    @pytest.mark.asyncio
    async def test_pending_tool_call_is_patched_and_closed(self):
        tool_call_msg = AIMessage(content="", id="a1", tool_calls=[{"id": "tc1", "name": "t", "args": {}}])
        patched = ToolMessage(content=STOP_CANCEL_REPLY, tool_call_id="tc1")
        agent = MagicMock()
        agent.aget_state = AsyncMock(side_effect=[
            MagicMock(values={"messages": [tool_call_msg]}),
            MagicMock(values={"messages": [tool_call_msg, patched]}),
        ])
        agent.aupdate_state = AsyncMock()

        await _close_stopped_thread(agent, self._config())

        assert agent.aupdate_state.await_count == 2
        tool_values = agent.aupdate_state.call_args_list[0].kwargs["values"]
        assert tool_values["messages"][0].tool_call_id == "tc1"
        assert tool_values["messages"][0].content == STOP_CANCEL_REPLY
        stop_values = agent.aupdate_state.call_args_list[1].kwargs["values"]
        assert stop_values["messages"][0].content == STOP_CONTEXT_REPLY

    @pytest.mark.asyncio
    async def test_completed_thread_is_untouched(self):
        agent = self._agent({"messages": [HumanMessage(content="q"), AIMessage(content="answer")]})

        await _close_stopped_thread(agent, self._config())

        agent.aupdate_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forced_close_resets_thread_ending_with_reply(self):
        agent = self._agent({
            "messages": [HumanMessage(content="q"), AIMessage(content="Which namespace?")],
            "subtasks": [{"task": "x", "status": "in_progress", "agent": "rancher"}],
            "awaiting_input": True,
        })

        await _close_stopped_thread(agent, self._config(), force=True)

        values = agent.aupdate_state.call_args.kwargs["values"]
        assert values["messages"][0].content == STOP_CONTEXT_REPLY
        assert values["subtasks"] == []
        assert values["awaiting_input"] is False

    @pytest.mark.asyncio
    async def test_empty_thread_is_untouched(self):
        agent = self._agent({})

        await _close_stopped_thread(agent, self._config())

        agent.aupdate_state.assert_not_awaited()


class TestCloseStoppedRun:
    @pytest.mark.asyncio
    async def test_closes_target_and_child_threads(self):
        child_graph = MagicMock()
        child = MagicMock(agent=child_graph)
        supervisor = SupervisorGraph(graph=MagicMock(), child_agents={"rancher": child})
        config = {"configurable": {"thread_id": "t1", "request_id": "r1"}}

        with patch("app.routers.websocket._close_stopped_thread", new_callable=AsyncMock) as mock_close:
            await _close_stopped_run(supervisor, config)

        calls = mock_close.call_args_list
        assert calls[0].args == (supervisor, config, False)
        child_threads = [c.args[1]["configurable"]["thread_id"] for c in calls[1:]]
        assert child_threads == ["t1::child::rancher", "t1::planner::rancher"]
        assert all(c.args[0] is child_graph for c in calls[1:])
        # Other configurable keys are forwarded.
        assert all(c.args[1]["configurable"]["request_id"] == "r1" for c in calls[1:])

    @pytest.mark.asyncio
    async def test_single_agent_closes_only_target(self):
        agent = MagicMock()
        config = {"configurable": {"thread_id": "t1"}}

        with patch("app.routers.websocket._close_stopped_thread", new_callable=AsyncMock) as mock_close:
            await _close_stopped_run(agent, config)

        mock_close.assert_awaited_once_with(agent, config, False)


def _plan_subtasks():
    return [
        {"task": "a", "status": "completed", "agent": "rancher"},
        {"task": "b", "status": "in_progress", "agent": "fleet"},
        {"task": "c", "status": "pending", "agent": "rancher"},
    ]


def _cancelled_plan_event():
    # Subtasks keep their original status when plan is cancelled (no status modification)
    tasks = _plan_subtasks()
    return f'<plan>{json.dumps({"tasks": tasks, "approval": False, "status": "cancelled"})}</plan>'


class TestStoppedPlanEvent:
    @pytest.mark.asyncio
    async def test_sends_cancelled_plan(self):
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=MagicMock(values={"subtasks": _plan_subtasks()}))
        websocket = AsyncMock()

        await _send_stopped_plan_event(agent, {"configurable": {}}, websocket)

        websocket.send_text.assert_awaited_once_with(_cancelled_plan_event())

    @pytest.mark.asyncio
    async def test_no_plan_sends_nothing(self):
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=MagicMock(values={"messages": []}))
        websocket = AsyncMock()

        await _send_stopped_plan_event(agent, {"configurable": {}}, websocket)

        websocket.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_plan_stop_sends_cancelled_plan_before_reset(self):
        """Stopping a running plan sends the cancelled <plan> before the state is reset."""
        agent = _make_stream_agent([_text_event("partial")], stall_after=1)
        agent.aget_state = AsyncMock(return_value=MagicMock(values={"subtasks": _plan_subtasks()}))
        websocket = AsyncMock()
        websocket.receive_text = AsyncMock(return_value=_stop_payload())
        order = []
        websocket.send_text = AsyncMock(side_effect=lambda text: order.append(text))

        async def _close(*args, **kwargs):
            order.append("close")

        with patch("app.routers.websocket._close_stopped_run", side_effect=_close), \
             patch("app.routers.websocket._record_stopped_in_history", new_callable=AsyncMock):
            await _call_agent(agent=agent, input_data={"messages": []}, config={"configurable": {}}, websocket=websocket)

        plan_index = order.index(_cancelled_plan_event())
        assert plan_index < order.index("close") < order.index(_build_stop_reply(STOP_REASON_MESSAGE_MANUAL))


class TestCancelWaitingPlan:
    def _supervisor(self, values, interrupts=()):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=MagicMock(values=values, interrupts=interrupts))
        return SupervisorGraph(graph=graph, child_agents={})

    async def _cancel(self, agent):
        websocket = AsyncMock()
        with patch("app.routers.websocket._close_stopped_run", new_callable=AsyncMock) as mock_close, \
             patch("app.routers.websocket._record_stopped_in_history", new_callable=AsyncMock) as mock_record:
            result = await _cancel_waiting_plan(agent, {"configurable": {}}, websocket)
        return result, websocket, mock_close, mock_record

    @pytest.mark.asyncio
    async def test_awaiting_input_plan_is_cancelled(self):
        agent = self._supervisor({"subtasks": _plan_subtasks(), "awaiting_input": True})

        result, websocket, mock_close, mock_record = await self._cancel(agent)

        assert result is True
        sent = [c.args[0] for c in websocket.send_text.await_args_list]
        assert sent == ["<message>", _cancelled_plan_event(), _build_stop_reply(STOP_REASON_PLANNING_MANUAL)]
        mock_close.assert_awaited_once_with(agent, {"configurable": {}}, force=True)
        assert mock_record.call_args.args[3] == STOP_REASON_PLANNING_MANUAL

    @pytest.mark.asyncio
    async def test_plan_waiting_on_interrupt_is_cancelled(self):
        subtasks = [{**st, "status": "pending"} for st in _plan_subtasks()]
        agent = self._supervisor({"subtasks": subtasks, "awaiting_input": False}, interrupts=(MagicMock(),))

        result, websocket, mock_close, _ = await self._cancel(agent)

        assert result is True
        plan_event = websocket.send_text.await_args_list[1].args[0]
        payload = json.loads(plan_event[len("<plan>"):-len("</plan>")])
        assert payload["status"] == "cancelled"
        # Subtasks keep their original status (all pending in this case)
        assert all(t["status"] == "pending" for t in payload["tasks"])
        mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_idle_plan_is_untouched(self):
        agent = self._supervisor({"subtasks": [], "awaiting_input": False})

        result, websocket, mock_close, mock_record = await self._cancel(agent)

        assert result is False
        websocket.send_text.assert_not_awaited()
        mock_close.assert_not_awaited()
        mock_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_agent_is_untouched(self):
        agent = MagicMock()
        agent.aget_state = AsyncMock(return_value=MagicMock(values={"subtasks": _plan_subtasks()}, interrupts=(MagicMock(),)))

        result, websocket, mock_close, _ = await self._cancel(agent)

        assert result is False
        websocket.send_text.assert_not_awaited()
        mock_close.assert_not_awaited()
