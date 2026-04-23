import json
import pytest

from app.routers.websocket import websocket_endpoint, _build_hitl_resume, _extract_interrupt_value
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from fastapi import WebSocketDisconnect
from contextlib import asynccontextmanager

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
        self.client_state = "connected"
        self._receive_queue = messages or []
        self._send_queue = []

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

# TODO add more tests for different scenarios
@pytest.fixture
def mock_dependencies():
    with patch('app.routers.websocket.create_agent') as mock_create_agent, \
         patch('app.routers.websocket.stream_agent_response', new_callable=AsyncMock) as mock_stream_response:

        # Mock the create_agent to return an async context manager
        mock_agent = MagicMock()
        mock_agent.astream_events = AsyncMock(return_value=iter([]))
        mock_session = MagicMock()
        mock_client_ctx = MagicMock()
        
        @asynccontextmanager
        async def mock_create_agent_context(*args, **kwargs):
            yield MagicMock(agent=mock_agent, session=mock_session, client_ctx=mock_client_ctx)
        
        mock_create_agent.side_effect = mock_create_agent_context

        yield {
            "create_agent": mock_create_agent,
            "agent": mock_agent,
            "session": mock_session,
            "client_ctx": mock_client_ctx,
            "stream_agent_response": mock_stream_response,
        }

""" @pytest.mark.asyncio
async def test_websocket_endpoint(mock_dependencies):
    mock_ws = MockWebSocket(messages=["test message"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, mock_llm)

    assert mock_ws.accepted
    mock_dependencies["create_agent"].assert_called_once()
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert "messages" in call_kwargs['input_data']
    assert call_kwargs['input_data']["messages"][0]["role"] == "user"
    assert call_kwargs['input_data']["messages"][0]["content"] == "test message"
    assert call_kwargs['websocket'] == mock_ws
    assert call_kwargs['agent'] == mock_dependencies["agent"]
    
    # Verify cleanup was called
    mock_dependencies["session"].__aexit__.assert_awaited_once()
    mock_dependencies["client_ctx"].__aexit__.assert_awaited_once()
 """
""" @pytest.mark.asyncio
async def test_websocket_endpoint_context_message(mock_dependencies):
    mock_ws = MockWebSocket(messages=[
        '{"prompt": "show all pods", "context": { "namespace": "default", "cluster": "local"} }'
    ])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, mock_llm)

    mock_dependencies["create_agent"].assert_called_once()
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert "messages" in call_kwargs['input_data']
    assert call_kwargs['input_data']["messages"][0]["role"] == "user"
    assert "show all pods" in call_kwargs['input_data']["messages"][0]["content"]
    assert call_kwargs['websocket'] == mock_ws
"""


# ============================================================================
# _build_hitl_resume Tests
# ============================================================================

class TestBuildHitlResume:
    """Tests for _build_hitl_resume converting user responses to HITL decisions."""

    def _make_interrupt_value(self, num_actions=1):
        return {
            "action_requests": [
                {"name": "submit_plan", "args": {"goal": "Test", "steps": [{"title": f"Task {i}"}]}, "description": "plan"}
                for i in range(num_actions)
            ],
            "review_configs": [
                {"action_name": "submit_plan", "allowed_decisions": ["approve", "reject"]}
                for _ in range(num_actions)
            ],
        }

    @pytest.mark.parametrize("prompt", ["yes", "approve", "ok", "go ahead", "YES", " Approve "])
    def test_approve_responses(self, prompt):
        """Verify that approval keywords produce approve decisions."""
        iv = self._make_interrupt_value(1)
        result = _build_hitl_resume(prompt, iv)
        assert result == {"decisions": [{"type": "approve"}]}

    @pytest.mark.parametrize("prompt", ["no", "reject", "cancel", "stop", "NO", " Reject "])
    def test_reject_responses(self, prompt):
        """Verify that rejection keywords produce reject decisions."""
        iv = self._make_interrupt_value(1)
        result = _build_hitl_resume(prompt, iv)
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["type"] == "reject"

    def test_multiple_actions_approve(self):
        """Verify approve generates one decision per action request."""
        iv = self._make_interrupt_value(3)
        result = _build_hitl_resume("yes", iv)
        assert len(result["decisions"]) == 3
        assert all(d["type"] == "approve" for d in result["decisions"])

    def test_multiple_actions_reject(self):
        """Verify reject generates one decision per action request."""
        iv = self._make_interrupt_value(2)
        result = _build_hitl_resume("no", iv)
        assert len(result["decisions"]) == 2
        assert all(d["type"] == "reject" for d in result["decisions"])

    def test_raw_json_passthrough(self):
        """Verify that raw JSON with decisions key is passed through."""
        iv = self._make_interrupt_value(1)
        raw = json.dumps({"decisions": [{"type": "approve"}]})
        result = _build_hitl_resume(raw, iv)
        assert result == {"decisions": [{"type": "approve"}]}

    def test_freetext_becomes_rejection(self):
        """Verify that arbitrary text becomes a rejection with the text as message."""
        iv = self._make_interrupt_value(1)
        result = _build_hitl_resume("Please add a step for backups", iv)
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["type"] == "reject"
        assert result["decisions"][0]["message"] == "Please add a step for backups"

    def test_empty_actions_list(self):
        """Verify handling of empty action_requests."""
        iv = {"action_requests": [], "review_configs": []}
        result = _build_hitl_resume("yes", iv)
        assert result == {"decisions": []}


# ============================================================================
# _extract_interrupt_value HITL Tests
# ============================================================================

class TestExtractInterruptValueHitl:
    """Tests for _extract_interrupt_value handling of HITL interrupts."""

    def _make_stream(self, interrupt_value):
        """Build a minimal on_chain_stream event carrying an interrupt."""
        interrupt = MagicMock()
        interrupt.value = interrupt_value
        return {
            "event": "on_chain_stream",
            "data": {
                "chunk": ("updates", {"__interrupt__": [interrupt]})
            },
        }

    def test_hitl_interrupt_returns_description(self):
        """Verify that HITL interrupt extracts the description from action_requests."""
        hitl_value = {
            "action_requests": [
                {"name": "submit_plan", "args": {}, "description": "<plan-approval>{}</plan-approval>"}
            ],
            "review_configs": [],
        }
        stream = self._make_stream(hitl_value)
        result = _extract_interrupt_value(stream)
        assert result == "<plan-approval>{}</plan-approval>"

    def test_hitl_interrupt_falls_back_to_json(self):
        """Verify fallback to JSON when description is empty."""
        hitl_value = {
            "action_requests": [{"name": "submit_plan", "args": {}, "description": ""}],
            "review_configs": [],
        }
        stream = self._make_stream(hitl_value)
        result = _extract_interrupt_value(stream)
        assert result == json.dumps(hitl_value)

    def test_string_interrupt_unchanged(self):
        """Verify that plain string interrupts are returned as-is."""
        stream = self._make_stream("<confirmation-response>test</confirmation-response>")
        result = _extract_interrupt_value(stream)
        assert result == "<confirmation-response>test</confirmation-response>"