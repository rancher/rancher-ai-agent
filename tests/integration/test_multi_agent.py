from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.loader import AgentConfig, AuthenticationType
from app.services.agent.parent import SYSTEM_ROUTER_PROMPT
from app.services.llm import LLMManager
from langchain_core.language_models import FakeMessagesListChatModel
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from unittest.mock import AsyncMock

import time
import multiprocessing
import requests
import pytest

# Agent names used in the multi-agent configuration
MATH_AGENT_NAME = "math-agent"
CALCULATOR_AGENT_NAME = "calculator-agent"

MATH_AGENT_PROMPT = "You are a math agent that can add numbers."
CALCULATOR_AGENT_PROMPT = "You are a calculator agent that can multiply numbers."

mock_mcp_1 = FastMCP("mock1")
mock_mcp_2 = FastMCP("mock2")


@mock_mcp_1.tool()
def add(a: int, b: int) -> str:
    """Add two numbers"""
    return f"sum is {a + b}"


@mock_mcp_2.tool()
def multiply(a: int, b: int) -> str:
    """Multiply two numbers"""
    return f"product is {a * b}"


def run_mock_mcp_1():
    """Runs the first mock MCP server on port 8001."""
    import uvicorn
    uvicorn.run(mock_mcp_1.streamable_http_app(), host="0.0.0.0", port=8001, log_level="error")


def run_mock_mcp_2():
    """Runs the second mock MCP server on port 8002."""
    import uvicorn
    uvicorn.run(mock_mcp_2.streamable_http_app(), host="0.0.0.0", port=8002, log_level="error")


client = TestClient(app)


class FakeMessagesListChatModelWithTools(FakeMessagesListChatModel):
    """
    A fake chat model that extends FakeMessagesListChatModel to support tool binding
    and capture the messages sent to the LLM for inspection in tests.
    
    In multi-agent setup, the first LLM call is for routing (invoke) and subsequent
    calls are for the child agent (stream).
    
    Note: This class uses a shared response index for both invoke and stream to ensure
    consistent response ordering across the parent agent (invoke) and child agent (stream).
    """
    tools: list[BaseTool] = None
    all_calls: list[LanguageModelInput] = []

    def bind_tools(self, tools):
        self.tools = tools
        return self
    
    def invoke(self, input, config = None, *, stop = None, **kwargs):
        # Capture the input messages before invoking the parent method.
        messages_send_to_llm = remove_message_ids(input)
        self.all_calls.append(messages_send_to_llm)
        # Use i counter from parent class (FakeMessagesListChatModel uses self.i)
        return super().invoke(input, config, stop=stop, **kwargs)
    
    def _stream(self, input, config = None, *, stop = None, **kwargs):
        """Override _stream to yield chunks from the response."""                
        # Use the same counter as invoke (self.i from parent class)
        if self.i < len(self.responses):
            response = self.responses[self.i]
            self.i += 1
            
            # Create a message chunk
            if hasattr(response, 'content') and response.content:
                chunk = AIMessageChunk(
                    content=response.content, 
                    tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else [],
                    id=response.id if hasattr(response, 'id') else None
                )
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                chunk = AIMessageChunk(
                    content="", 
                    tool_calls=response.tool_calls,
                    id=response.id if hasattr(response, 'id') else None
                )
            else:
                chunk = AIMessageChunk(content="")
            
            # Wrap in ChatGenerationChunk
            yield ChatGenerationChunk(message=chunk)


def remove_message_ids(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Creates a new list of BaseMessage objects with the 'id' field removed
    from each message.
    """
    new_messages = []
    
    for message in messages:
        # Create a copy of the message, explicitly setting the 'id' field to None for consistent comparison.
        if isinstance(message, BaseMessage):
            new_message = message.model_copy(update={
                "id": None,
                "additional_kwargs": {},
                "response_metadata": {}
            })
        else:
            new_message = message
        
        new_messages.append(new_message)
        
    return new_messages


def collect_messages(websocket, num_prompts: int) -> list[str]:
    """
    Collect messages from websocket until we get the expected final message markers.
    
    For multi-agent setup, each prompt results in one complete message that contains
    both the agent-metadata and the response content.
    """
    messages = []
    
    for _ in range(num_prompts):
        msg = ""
        while not msg.endswith("</message>"):
            msg += websocket.receive_text()
        messages.append(msg)
    
    return messages


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
def setup_mock_mcp_servers(module_monkeypatch):
    """Sets up and tears down multiple mock MCP servers for the duration of the test module."""
    module_monkeypatch.setenv("INSECURE_SKIP_TLS", "true")

    class MockMemoryManager:
        def get_checkpointer(self):
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    app.memory_manager = MockMemoryManager()
    
    module_monkeypatch.setattr("app.routers.websocket.get_user_id", AsyncMock(return_value="test-user-id"))
    
    # Create multiple agent configs for multi-agent setup
    mock_agent_config_1 = AgentConfig(
        name=MATH_AGENT_NAME,
        description="Agent that can perform addition operations",
        system_prompt=MATH_AGENT_PROMPT,
        mcp_url="http://localhost:8001/mcp",
        authentication=AuthenticationType.NONE,
    )
    
    mock_agent_config_2 = AgentConfig(
        name=CALCULATOR_AGENT_NAME,
        description="Agent that can perform multiplication operations",
        system_prompt=CALCULATOR_AGENT_PROMPT,
        mcp_url="http://localhost:8002/mcp",
        authentication=AuthenticationType.NONE,
    )
    
    module_monkeypatch.setattr(
        "app.services.agent.factory.load_agent_configs", 
        lambda: [mock_agent_config_1, mock_agent_config_2]
    )

    # Start both MCP servers
    process_1 = multiprocessing.Process(target=run_mock_mcp_1)
    process_2 = multiprocessing.Process(target=run_mock_mcp_2)
    process_1.start()
    process_2.start()

    # Wait for both mock servers to be available before running tests.
    for port in [8001, 8002]:
        mcp_server_available = False
        while not mcp_server_available:
            try:
                requests.get(f"http://localhost:{port}/mcp")
                mcp_server_available = True
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
       
    yield (process_1, process_2)

    process_1.terminate()
    process_2.terminate()


def test_single_prompt():
    """Tests a single prompt-response interaction with multiple agents.
    
    In multi-agent setup:
    1. Parent agent routes request to a child agent (first LLM call via invoke)
    2. Child agent processes the request (subsequent LLM calls via stream)
    """
    fake_prompt = "fake prompt"
    prompts = [fake_prompt]
    
    fake_llm_responses = [
        # First response: Parent agent routes to math-agent
        AIMessage(content=MATH_AGENT_NAME),
        # Second response: math-agent responds to the user
        AIMessage(content="fake llm response from math agent"),
    ]
    # Agent-metadata and response are inside the same <message>...</message> block
    expected_messages_send_to_websocket = [
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>fake llm response from math agent</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []  # Reset call tracking
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
            
            messages = collect_messages(websocket, len(prompts))
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 2, "Expected 2 LLM calls (routing + child agent)"
        assert fake_llm.all_calls[0] == [SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt)), HumanMessage(content=fake_prompt)], "First call should be routing call with prompt"
        assert fake_llm.all_calls[1] == [SystemMessage(content=MATH_AGENT_PROMPT), HumanMessage(content=fake_prompt)], "Second call should be child agent call with prompt"

    finally:
        LLMManager._instance = None


def test_multiple_prompts():
    """Tests multiple prompt-response interactions in sequence with multiple agents."""
    fake_prompt_1 = "fake prompt 1"
    fake_prompt_2 = "fake prompt 2"
    fake_llm_response_1 = "fake llm response 1"
    fake_llm_response_2 = "fake llm response 2"

    prompts = [fake_prompt_1, fake_prompt_2]
    
    fake_llm_responses = [
        # First prompt: route to math-agent
        AIMessage(content=MATH_AGENT_NAME),
        AIMessage(content=fake_llm_response_1),
        # Second prompt: route to calculator-agent
        AIMessage(content=CALCULATOR_AGENT_NAME),
        AIMessage(content=fake_llm_response_2),
    ]
    expected_messages_send_to_websocket = [
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>fake llm response 1</message>",
        f"<message><agent-metadata>{{\"agentName\": \"{CALCULATOR_AGENT_NAME}\"}}</agent-metadata>fake llm response 2</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            # Send prompts one at a time and collect responses
            messages = []
            for prompt in prompts:
                websocket.send_text(prompt)
                msgs = collect_messages(websocket, 1)
                messages.extend(msgs)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 4, "Expected 4 LLM calls (2 routing + 2 child agent)"
        # First routing call (only has the first prompt)
        assert fake_llm.all_calls[0] == [SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_1)), HumanMessage(content=fake_prompt_1)], "First call should be routing call with prompt"
        # First child agent call
        assert fake_llm.all_calls[1] == [SystemMessage(content=MATH_AGENT_PROMPT), HumanMessage(content=fake_prompt_1)], "Second call should be child agent call with prompt"
        # Second routing call (includes conversation history)
        assert fake_llm.all_calls[2] == [
            SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_2)), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2)
        ], "Third call should be routing call with conversation history"
        # Second child agent call (also includes conversation history)
        assert fake_llm.all_calls[3] == [
            SystemMessage(content=CALCULATOR_AGENT_PROMPT), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2)
        ], "Fourth call should be child agent call with conversation history"

    finally:
        LLMManager._instance = None


def test_delegate_to_child_agent_with_tool():
    """Tests that the parent agent can delegate to a child agent that uses a tool."""
    fake_prompt = "add 4 and 5"
    prompts = [fake_prompt]
    
    fake_llm_responses = [
        # Parent agent routes to math-agent
        AIMessage(content=MATH_AGENT_NAME),
        # Child agent (math-agent) calls the add tool
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_add_1",
                "name": "add",
                "args": {"a": 4, "b": 5}
            }]
        ),
        # Child agent responds with the result
        AIMessage(content="The sum of 4 and 5 is 9."),
    ]
    expected_messages_send_to_websocket = [
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>The sum of 4 and 5 is 9.</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
            
            messages = collect_messages(websocket, len(prompts))
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 3, "Expected 3 LLM calls (1 routing + 2 child agent)"
        # First routing call
        assert fake_llm.all_calls[0] == [SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt)), HumanMessage(content=fake_prompt)], "First call should be routing call with prompt"
        # First child agent call (requests tool)
        assert fake_llm.all_calls[1] == [SystemMessage(content=MATH_AGENT_PROMPT), HumanMessage(content=fake_prompt)], "Second call should be child agent call with prompt"
        # Second child agent call (after tool execution, includes conversation history with tool result)
        third_call = fake_llm.all_calls[2]
        assert third_call[0] == SystemMessage(content=MATH_AGENT_PROMPT), "Third call should have math agent system prompt"
        assert third_call[1] == HumanMessage(content=fake_prompt), "Third call should have original prompt"
        assert isinstance(third_call[2], AIMessage) and third_call[2].tool_calls[0]["name"] == "add", "Third call should have AI message with add tool call"
        assert isinstance(third_call[3], ToolMessage) and third_call[3].content == "sum is 9", "Third call should have tool result with sum"

    finally:
        LLMManager._instance = None

def test_summary():
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
        AIMessage(content=MATH_AGENT_NAME),
        AIMessage(content=fake_llm_response_1),
        AIMessage(content=CALCULATOR_AGENT_NAME),
        AIMessage(content=fake_llm_response_2),
        AIMessage(content=MATH_AGENT_NAME),
        AIMessage(content=fake_llm_response_3),
        AIMessage(content=CALCULATOR_AGENT_NAME),
        AIMessage(content=fake_llm_response_4),
        AIMessage(content=fake_summary_response),
        AIMessage(content=MATH_AGENT_NAME),
        AIMessage(content=fake_llm_response_5),
    ]
    expected_messages_send_to_websocket = [
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>fake llm response 1</message>",
        f"<message><agent-metadata>{{\"agentName\": \"{CALCULATOR_AGENT_NAME}\"}}</agent-metadata>fake llm response 2</message>",
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>fake llm response 3</message>",
        f"<message><agent-metadata>{{\"agentName\": \"{CALCULATOR_AGENT_NAME}\"}}</agent-metadata>fake llm response 4</message>",
        f"<message><agent-metadata>{{\"agentName\": \"{MATH_AGENT_NAME}\"}}</agent-metadata>fake llm response 5</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            # Send prompts one at a time and collect responses
            messages = []
            for prompt in prompts:
                websocket.send_text(prompt)
                msgs = collect_messages(websocket, 1)
                messages.extend(msgs)
            
        assert messages == expected_messages_send_to_websocket
        assert len(fake_llm.all_calls) == 11, "Expected 11 LLM calls (5 routing + 5 child agent + 1 summary)"
        assert fake_llm.all_calls[0] == [SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_1)), HumanMessage(content=fake_prompt_1)], "First call should be routing call with prompt"
        assert fake_llm.all_calls[1] == [SystemMessage(content=MATH_AGENT_PROMPT), HumanMessage(content=fake_prompt_1)], "Second call should be child agent call with prompt"
        assert fake_llm.all_calls[2] == [
            SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_2)), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2)
        ], "Third call should be routing call with conversation history"
        assert fake_llm.all_calls[3] == [
            SystemMessage(content=CALCULATOR_AGENT_PROMPT), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2)
        ], "Fourth call should be child agent call with conversation history"
        assert fake_llm.all_calls[4] == [
            SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_3)), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3)
        ], "Fifth call should be routing call with conversation history"
        assert fake_llm.all_calls[5] == [
            SystemMessage(content=MATH_AGENT_PROMPT), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3)
        ], "Sixth call should be child agent call with conversation history"
        assert fake_llm.all_calls[6] == [
            SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_4)), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
            AIMessage(content=fake_llm_response_3),
            HumanMessage(content=fake_prompt_4)
        ], "Seventh call should be routing call with conversation history"
        assert fake_llm.all_calls[7] == [
            SystemMessage(content=CALCULATOR_AGENT_PROMPT), 
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
            AIMessage(content=fake_llm_response_3),
            HumanMessage(content=fake_prompt_4)
        ], "Eighth call should be child agent call with conversation history"
        assert fake_llm.all_calls[8] == [
            HumanMessage(content=fake_prompt_1),
            AIMessage(content=fake_llm_response_1),
            HumanMessage(content=fake_prompt_2),
            AIMessage(content=fake_llm_response_2),
            HumanMessage(content=fake_prompt_3),
            AIMessage(content=fake_llm_response_3),
            HumanMessage(content=fake_prompt_4),
            AIMessage(content=fake_llm_response_4),
            HumanMessage(content="Create a summary of the conversation above:")
        ], "Ninth call should be summary generation with full conversation history"
        assert fake_llm.all_calls[9] == [
            SystemMessage(content=_build_router_prompt(SYSTEM_ROUTER_PROMPT, fake_prompt_5)),
            SystemMessage(content=f"Conversation summary: {fake_summary_response}"),
            HumanMessage(content=fake_prompt_5)
        ], "Tenth call should be routing call with summary (messages replaced by summary)"
        assert fake_llm.all_calls[10] == [
            SystemMessage(content=MATH_AGENT_PROMPT),
            SystemMessage(content=f"Conversation summary: {fake_summary_response}"),
            HumanMessage(content=fake_prompt_5)
        ], "Eleventh call should be child agent call with summary"

    finally:
        LLMManager._instance = None

def _build_router_prompt(system_router_prompt: str, user_request: str) -> str:
    """
    Builds the full router prompt by appending the available child agents
    and their descriptions to the base system router prompt.
    """
    router_prompt = system_router_prompt + "AVAILABLE CHILD AGENTS:\n"
    router_prompt += f"- {MATH_AGENT_NAME}: Agent that can perform addition operations\n"
    router_prompt += f"- {CALCULATOR_AGENT_NAME}: Agent that can perform multiplication operations\n"
    router_prompt += f"\nUSER'S REQUEST: {user_request}\n"
    return router_prompt
