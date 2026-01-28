from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.loader import RANCHER_AGENT_PROMPT, AgentConfig, AuthenticationType
from app.services.llm import LLMManager
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from _pytest.monkeypatch import MonkeyPatch
from unittest.mock import AsyncMock

from tests.integration.test_multi_agent import FakeMessagesListChatModelWithTools

import time
import multiprocessing
import requests
import pytest

mock_mcp = FastMCP("mock")


@mock_mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers"""
    return f"sum is {a + b}"

def run_mock_mcp():
    """Runs the mock MCP server."""
    mock_mcp.run(transport="streamable-http")

client = TestClient(app)

@pytest.fixture(scope="module")
def module_monkeypatch(request):
    """
    A module-scoped version of the monkeypatch fixture.
    This fixture ensures that patches persist for the duration of the module,
    and cleanup happens only once at the end of the module.
    """
    mpatch = MonkeyPatch()

    yield mpatch

    mpatch.undo()

@pytest.fixture(scope="module", autouse=True)
def setup_mock_mcp_server(module_monkeypatch):
    """Sets up and tears down a mock MCP server for the duration of the test module."""
    module_monkeypatch.setenv("INSECURE_SKIP_TLS", "true")

    class MockMemoryManager:
        def get_checkpointer(self):
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    app.memory_manager = MockMemoryManager()
    
    module_monkeypatch.setattr("app.routers.websocket.get_user_id", AsyncMock(return_value="test-user-id"))
    
    mock_agent_config = AgentConfig(
        name="test-agent",
        description="Test agent for integration tests",
        system_prompt=RANCHER_AGENT_PROMPT,
        mcp_url="http://localhost:8000/mcp",
        authentication=AuthenticationType.NONE,
    )
    module_monkeypatch.setattr("app.services.agent.factory.load_agent_configs", lambda: [mock_agent_config])

    process = multiprocessing.Process(target=run_mock_mcp)
    process.start()

    # Wait for the mock server to be available before running tests.
    mcp_server_available = False

    while not mcp_server_available:
        try:
            requests.get("http://localhost:8000/mcp")
            mcp_server_available = True
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
       
    yield process

    process.terminate()

def test_websocket_single_prompt():
    """Tests a single prompt-response interaction."""
    prompts = ["fake prompt"]
    fake_llm_responses = [
        AIMessage(content="fake llm response"),
    ]
    expected_messages_send_to_websocket = ["<message>fake llm response</message>"]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []  # Reset call tracking
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 1, "Expected 1 LLM call"
        assert fake_llm.all_calls[0] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content="fake prompt"),
        ], "First call should have system prompt and user message"
    finally:
        LLMManager._instance = None

def test_websocket_multiple_prompts():
    """Tests multiple prompt-response interactions in sequence."""
    prompts = ["fake prompt 1", "fake prompt 2"]
    fake_llm_responses = [
        AIMessage(content="fake llm response 1"),
        AIMessage(content="fake llm response 2"),
    ]
    expected_messages_send_to_websocket = [
        "<message>fake llm response 1</message>",
        "<message>fake llm response 2</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []  # Reset call tracking
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 2, "Expected 2 LLM calls"
        assert fake_llm.all_calls[0] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content="fake prompt 1"),
        ], "First call should have system prompt and first user message"
        assert fake_llm.all_calls[1] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content="fake prompt 1"),
            AIMessage(content="fake llm response 1"),
            HumanMessage(content="fake prompt 2"),
        ], "Second call should include conversation history"
    finally:
        LLMManager._instance = None

def test_websocket_tool_call():
    """Tests agent interaction with tool calling."""
    prompts = ["sum 4 + 5"]
    fake_llm_responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "add",
                "args": {"a": 4, "b": 5}
            }]
        ),
        AIMessage(content="fake llm response"),
    ]
    expected_messages_send_to_websocket = ["<message>fake llm response</message>"]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []  # Reset call tracking
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 2, "Expected 2 LLM calls (initial + after tool)"
        assert fake_llm.all_calls[0] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content="sum 4 + 5"),
        ], "First call should have system prompt and user message"
        # Second call includes tool call and result
        second_call = fake_llm.all_calls[1]
        assert second_call[0] == SystemMessage(content=RANCHER_AGENT_PROMPT), "Second call should have system prompt"
        assert second_call[1] == HumanMessage(content="sum 4 + 5"), "Second call should have user message"
        assert isinstance(second_call[2], AIMessage) and second_call[2].tool_calls[0]["name"] == "add", "Second call should have AI message with tool call"
        assert isinstance(second_call[3], ToolMessage) and second_call[3].content == "sum is 9", "Second call should have tool result"
    finally:
        LLMManager._instance = None

def test_summary():
    """Tests that conversation history is summarized after reaching the threshold."""
    fake_prompt_1 = "fake prompt 1"
    fake_prompt_2 = "fake prompt 2"
    fake_prompt_3 = "fake prompt 3"
    fake_prompt_4 = "fake prompt 4"
    fake_prompt_5 = "fake prompt 5"

    fake_llm_response_1 = "fake llm response 1"
    fake_llm_response_2 = "fake llm response 2"
    fake_llm_response_3 = "fake llm response 3"
    fake_llm_response_4 = "fake llm response 4"
    fake_llm_response_5 = "fake llm response 5"
    fake_summary_response = "This is a summary of the conversation so far."

    prompts = [fake_prompt_1, fake_prompt_2, fake_prompt_3, fake_prompt_4, fake_prompt_5]
    
    fake_llm_responses = [
        AIMessage(content=fake_llm_response_1),
        AIMessage(content=fake_llm_response_2),
        AIMessage(content=fake_llm_response_3),
        AIMessage(content=fake_llm_response_4),
        AIMessage(content=fake_summary_response),
        AIMessage(content=fake_llm_response_5),
    ]
    expected_messages_send_to_websocket = [
        "<message>fake llm response 1</message>",
        "<message>fake llm response 2</message>",
        "<message>fake llm response 3</message>",
        "<message>fake llm response 4</message>",
        "<message>fake llm response 5</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []  # Reset call tracking
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 6, "Expected 6 LLM calls (5 prompts + 1 summary)"
        
        # First call - just prompt 1
        assert fake_llm.all_calls[0] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content=fake_prompt_1),
        ], "First call should have system prompt and first user message"
        
        # Second call - prompt 1 + response 1 + prompt 2
        assert fake_llm.all_calls[1] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
        ], "Second call should include conversation history"
        
        # Third call - full history up to prompt 3
        assert fake_llm.all_calls[2] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
        ], "Third call should include full conversation history"
        
        # Fourth call - full history up to prompt 4
        assert fake_llm.all_calls[3] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
            AIMessage(content=fake_llm_response_3),
            HumanMessage(content=fake_prompt_4),
        ], "Fourth call should include full conversation history"
        
        # Fifth call - summary generation (no system message)
        assert fake_llm.all_calls[4] == [
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
            AIMessage(content=fake_llm_response_3),
            HumanMessage(content=fake_prompt_4),
            AIMessage(content=fake_llm_response_4),
            HumanMessage(content="Create a summary of the conversation above:")
        ], "Fifth call should be summary generation with full conversation history (no system message)"
        
        # Sixth call - after summary, messages replaced by summary + new prompt
        assert fake_llm.all_calls[5] == [
            SystemMessage(content=RANCHER_AGENT_PROMPT),
            AIMessage(content=fake_summary_response),
            HumanMessage(content=fake_prompt_5),
        ], "Sixth call should have summary replacing conversation history"
        
    finally:
        LLMManager._instance = None
