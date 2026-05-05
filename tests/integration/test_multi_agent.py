from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.loader import AgentConfig, AuthenticationType
from app.services.agent.supervisor import SUPERVISOR_PROMPT
from app.services.llm import LLMManager
from app.services.memory import StorageType
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
    
    In the supervisor multi-agent setup:
    - The supervisor's model node uses ainvoke -> _astream -> _stream (async path)
    - The child agent's call_model_node uses invoke -> _generate (sync path)
    
    Both paths share the same response index (self.i) ensuring consistent ordering.
    
    Note: We capture calls in invoke (for child agent sync calls) and _stream
    (for supervisor async calls). We use a flag to prevent double-counting when
    invoke internally routes through _stream due to v2 streaming protocol.
    """
    tools: list[BaseTool] = None
    all_calls: list[LanguageModelInput] = []
    _in_invoke: bool = False

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        return self
    
    def invoke(self, input, config=None, *, stop=None, **kwargs):
        # Capture the input messages before invoking the parent method.
        messages_send_to_llm = remove_message_ids(input)
        self.all_calls.append(messages_send_to_llm)
        # Set flag to prevent _stream from double-capturing
        self._in_invoke = True
        try:
            return super().invoke(input, config, stop=stop, **kwargs)
        finally:
            self._in_invoke = False
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Override _stream to yield chunks from the response.
        
        This is called by the supervisor's model node (via ainvoke -> _astream -> _stream).
        When called from within invoke (v2 streaming protocol), we skip capturing
        since invoke already captured the messages.
        """
        if not self._in_invoke:
            messages_send_to_llm = remove_message_ids(messages)
            self.all_calls.append(messages_send_to_llm)
        
        if self.i < len(self.responses):
            response = self.responses[self.i]
            self.i += 1
            
            chunk = AIMessageChunk(
                content=response.content if hasattr(response, 'content') else "",
                tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else [],
                id=response.id if hasattr(response, 'id') else None
            )
            yield ChatGenerationChunk(message=chunk)


def remove_message_ids(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Creates a new list of BaseMessage objects with the 'id' field removed
    from each message.
    """
    new_messages = []
    
    for message in messages:
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
    
    Each prompt results in one complete message wrapped in <message>...</message>.
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
        def __init__(self):
            self.storage_type = StorageType.IN_MEMORY
        
        def get_checkpointer(self):
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    app.memory_manager = MockMemoryManager()
    
    module_monkeypatch.setattr("app.routers.websocket.get_user_id", AsyncMock(return_value="test-user-id"))
    
    # Create multiple agent configs for multi-agent setup
    mock_agent_config_1 = AgentConfig(
        name=MATH_AGENT_NAME,
        displayName="Math Agent",
        description="Agent that can perform addition operations",
        system_prompt=MATH_AGENT_PROMPT,
        mcp_url="http://localhost:8001/mcp",
        authentication=AuthenticationType.NONE,
    )
    
    mock_agent_config_2 = AgentConfig(
        name=CALCULATOR_AGENT_NAME,
        displayName="Calculator Agent",
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
    """Tests a single prompt-response interaction with supervisor architecture.
    
    In the supervisor multi-agent setup:
    1. Supervisor LLM decides to call the math-agent tool (via _stream)
    2. Child agent (math-agent) processes the request (via invoke)
    3. Supervisor LLM produces final response based on tool result (via _stream)
    """
    fake_prompt = "fake prompt"
    prompts = [fake_prompt]
    
    fake_llm_responses = [
        # Supervisor decides to call math-agent tool
        AIMessage(content="", tool_calls=[{
            "id": "call_1",
            "name": MATH_AGENT_NAME,
            "args": {"query": fake_prompt}
        }]),
        # Child agent (math-agent) responds
        AIMessage(content="fake llm response from math agent"),
        # Supervisor produces final response
        AIMessage(content="The math agent says the answer is ready."),
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
            
            messages = collect_messages(websocket, len(prompts))
            
        # The websocket message should contain custom events from monitor_tool middleware
        # and the final streamed response from the supervisor
        full_message = messages[0]
        assert full_message.startswith("<message>"), "Message should start with <message>"
        assert full_message.endswith("</message>"), "Message should end with </message>"
        
        # The supervisor dispatches a subagent_call custom event
        assert f"Supervisor is calling agent '{MATH_AGENT_NAME}'" in full_message, \
            "Should contain supervisor's subagent call notification"
        
        # The supervisor's final response should be streamed
        assert "The math agent says the answer is ready." in full_message, \
            "Should contain supervisor's final response"
        
        # Verify LLM calls: supervisor (stream) + child (invoke) + supervisor (stream) = 3
        assert len(fake_llm.all_calls) == 3, \
            f"Expected 3 LLM calls (supervisor + child + supervisor), got {len(fake_llm.all_calls)}"
        
        # First call: supervisor with system prompt + user message
        first_call = fake_llm.all_calls[0]
        assert first_call[0] == SystemMessage(content=SUPERVISOR_PROMPT), \
            "First call (supervisor) should have SUPERVISOR_PROMPT"
        assert first_call[-1] == HumanMessage(content=fake_prompt), \
            "First call should end with user's prompt"
        
        # Second call: child agent with its own system prompt + query from tool
        second_call = fake_llm.all_calls[1]
        assert second_call[0] == SystemMessage(content=MATH_AGENT_PROMPT), \
            "Second call (child) should have MATH_AGENT_PROMPT"
        assert second_call[1] == HumanMessage(content=fake_prompt), \
            "Second call should have the query as user message"
        
        # Third call: supervisor with full history (including tool result)
        third_call = fake_llm.all_calls[2]
        assert third_call[0] == SystemMessage(content=SUPERVISOR_PROMPT), \
            "Third call (supervisor) should have SUPERVISOR_PROMPT"

    finally:
        LLMManager._instance = None


def test_multiple_prompts():
    """Tests multiple prompt-response interactions in sequence with the supervisor."""
    fake_prompt_1 = "fake prompt 1"
    fake_prompt_2 = "fake prompt 2"
    fake_llm_response_1 = "response from math agent"
    fake_llm_response_2 = "response from calculator agent"
    fake_supervisor_response_1 = "Supervisor final response 1"
    fake_supervisor_response_2 = "Supervisor final response 2"

    prompts = [fake_prompt_1, fake_prompt_2]
    
    fake_llm_responses = [
        # First prompt: supervisor routes to math-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_1",
            "name": MATH_AGENT_NAME,
            "args": {"query": fake_prompt_1}
        }]),
        # math-agent responds
        AIMessage(content=fake_llm_response_1),
        # Supervisor final response for first prompt
        AIMessage(content=fake_supervisor_response_1),
        # Second prompt: supervisor routes to calculator-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_2",
            "name": CALCULATOR_AGENT_NAME,
            "args": {"query": fake_prompt_2}
        }]),
        # calculator-agent responds
        AIMessage(content=fake_llm_response_2),
        # Supervisor final response for second prompt
        AIMessage(content=fake_supervisor_response_2),
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []
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
            
        # Verify first message
        assert f"Supervisor is calling agent '{MATH_AGENT_NAME}'" in messages[0]
        assert fake_supervisor_response_1 in messages[0]
        
        # Verify second message
        assert f"Supervisor is calling agent '{CALCULATOR_AGENT_NAME}'" in messages[1]
        assert fake_supervisor_response_2 in messages[1]
        
        # 6 LLM calls: 2 supervisor + 1 child per prompt = 3 * 2 = 6
        assert len(fake_llm.all_calls) == 6, \
            f"Expected 6 LLM calls (3 per prompt), got {len(fake_llm.all_calls)}"
        
        # First supervisor call has just the first prompt
        first_call = fake_llm.all_calls[0]
        assert first_call[0] == SystemMessage(content=SUPERVISOR_PROMPT)
        assert first_call[-1] == HumanMessage(content=fake_prompt_1)
        
        # Fourth call (second supervisor call) should include conversation history
        fourth_call = fake_llm.all_calls[3]
        assert fourth_call[0] == SystemMessage(content=SUPERVISOR_PROMPT)
        assert fourth_call[-1] == HumanMessage(content=fake_prompt_2)
        # Should include previous messages in history
        assert len(fourth_call) > 3, "Second supervisor call should have conversation history"

    finally:
        LLMManager._instance = None


def test_delegate_to_child_agent_with_tool():
    """Tests that the supervisor delegates to a child agent which then uses an MCP tool."""
    fake_prompt = "add 4 and 5"
    prompts = [fake_prompt]
    
    fake_llm_responses = [
        # Supervisor calls math-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_supervisor_1",
            "name": MATH_AGENT_NAME,
            "args": {"query": fake_prompt}
        }]),
        # Child agent (math-agent) calls the add MCP tool
        AIMessage(content="", tool_calls=[{
            "id": "call_add_1",
            "name": "add",
            "args": {"a": 4, "b": 5}
        }]),
        # Child agent responds with the result after getting tool output
        AIMessage(content="The sum of 4 and 5 is 9."),
        # Supervisor produces final response
        AIMessage(content="The math agent calculated that 4 + 5 = 9."),
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm
    
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            for prompt in prompts:
                websocket.send_text(prompt)
            
            messages = collect_messages(websocket, len(prompts))
            
        full_message = messages[0]
        
        # Verify supervisor's subagent call notification
        assert f"Supervisor is calling agent '{MATH_AGENT_NAME}'" in full_message
        
        # Verify supervisor's final response
        assert "The math agent calculated that 4 + 5 = 9." in full_message
        
        # 4 LLM calls: supervisor + child (tool call) + child (after tool) + supervisor (final)
        assert len(fake_llm.all_calls) == 4, \
            f"Expected 4 LLM calls, got {len(fake_llm.all_calls)}"
        
        # First call: supervisor decides to call math-agent
        assert fake_llm.all_calls[0][0] == SystemMessage(content=SUPERVISOR_PROMPT)
        
        # Second call: child agent (first invocation, decides to call add tool)
        assert fake_llm.all_calls[1][0] == SystemMessage(content=MATH_AGENT_PROMPT)
        assert fake_llm.all_calls[1][1] == HumanMessage(content=fake_prompt)
        
        # Third call: child agent (after tool result, includes tool message)
        third_call = fake_llm.all_calls[2]
        assert third_call[0] == SystemMessage(content=MATH_AGENT_PROMPT)
        assert third_call[1] == HumanMessage(content=fake_prompt)
        # Should include tool call message and tool result
        assert isinstance(third_call[2], AIMessage) and third_call[2].tool_calls[0]["name"] == "add"
        assert isinstance(third_call[3], ToolMessage) and third_call[3].content == "sum is 9"
        
        # Fourth call: supervisor final response
        assert fake_llm.all_calls[3][0] == SystemMessage(content=SUPERVISOR_PROMPT)

    finally:
        LLMManager._instance = None


def test_delegate_to_child_agent_with_ui_tools():
    """Tests that child agents work with UI tools and dispatch <processing-ui-tools/> message."""
    from app.services.ui_tools.models import UITool, UIToolSchema, UIToolsConfig, UIToolsConfigData
    from unittest.mock import patch, MagicMock
    import json

    # Create a mock UI tool
    schema = UIToolSchema(
        type="object",
        properties={"name": {"type": "string", "description": "Pod name"}},
        required=["name"]
    )
    ui_tool = UITool(
        name="show-yaml",
        description="Display resource in YAML format",
        prompt="Show resource YAML",
        category="viewer",
        schema=schema,
        metadata={},
        enabled=True
    )

    fake_prompt = "show the pod"
    prompts = [fake_prompt]

    fake_llm_responses = [
        # Supervisor calls math-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_sup_1",
            "name": MATH_AGENT_NAME,
            "args": {"query": fake_prompt}
        }]),
        # Child agent responds with pod information
        AIMessage(content="The pod is running successfully with 3 replicas."),
        # Supervisor final response
        AIMessage(content="The pod is running successfully with 3 replicas."),
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm

    try:
        # Create agent configs with UI tools enabled
        math_config = AgentConfig(
            name=MATH_AGENT_NAME,
            displayName="Math Agent",
            description="Agent that can perform addition operations",
            system_prompt=MATH_AGENT_PROMPT,
            mcp_url="http://localhost:8001/mcp",
            authentication=AuthenticationType.NONE,
            ui_tools_selectors=["show-yaml"]
        )
        
        calc_config = AgentConfig(
            name=CALCULATOR_AGENT_NAME,
            displayName="Calculator Agent",
            description="Agent that can perform multiplication operations",
            system_prompt=CALCULATOR_AGENT_PROMPT,
            mcp_url="http://localhost:8002/mcp",
            authentication=AuthenticationType.NONE,
        )
        
        # Patch the load_agent_configs to return our configured agents
        with patch('app.services.agent.factory.load_agent_configs') as mock_load:
            mock_load.return_value = [math_config, calc_config]
            
            # Patch load_ui_tools_from_configmap to return our UI tools config
            with patch('app.services.agent.child.load_ui_tools_from_configmap') as mock_load_configmap:
                ui_tools_config = UIToolsConfigData(
                    tools=[ui_tool],
                    config=UIToolsConfig(enabled=True, max_tools=5, system_prompt="Select relevant UI tools")
                )
                mock_load_configmap.return_value = ui_tools_config
                
                # Mock the UI tools selector
                with patch('app.services.agent.child.create_ui_tools_selector') as mock_selector_factory:
                    mock_selector = MagicMock()
                    mock_selector_factory.return_value = mock_selector
                    mock_selector.select_tools.return_value = [
                        {
                            "toolName": "show-yaml",
                            "description": "Display resource in YAML format",
                            "prompt": "Show resource YAML",
                        }
                    ]
                    
                    messages = []
                    with client.websocket_connect("/v1/ws/messages") as websocket:
                        # Consume any initial messages from the server
                        websocket.receive_text()

                        for prompt in prompts:
                            # Send JSON request with tools configuration
                            request = json.dumps({
                                "prompt": prompt,
                                "tools": {"name": "default", "tools": ["show-yaml"]}
                            })
                            websocket.send_text(request)
                        
                        # Collect ALL messages from the websocket
                        msg = ""
                        while True:
                            chunk = websocket.receive_text()
                            messages.append(chunk)
                            msg += chunk
                            if msg.endswith("</message>"):
                                break
                    
                    full_stream = "".join(messages)
                    
                    # Verify the response contains the supervisor's final answer
                    assert "The pod is running successfully with 3 replicas." in full_stream, \
                        "Response should contain the final answer"
                    
                    # Verify UI tools dispatch: processing message sent
                    assert "<processing-ui-tools/>" in full_stream, \
                        "Missing <processing-ui-tools/> dispatch message"
                    
                    # Verify UI tools were dispatched
                    assert "<ui-tools>" in full_stream, "Missing <ui-tools> dispatch message"
                    
                    mock_load_configmap.assert_called()
                    mock_selector_factory.assert_called()
                    mock_selector.select_tools.assert_called()

    finally:
        LLMManager._instance = None


def test_conversation_history():
    """Tests that the supervisor maintains conversation history across multiple prompts."""
    fake_prompt_1 = "fake prompt 1"
    fake_prompt_2 = "fake prompt 2"
    fake_prompt_3 = "fake prompt 3"
    
    fake_supervisor_response_1 = "supervisor response 1"
    fake_supervisor_response_2 = "supervisor response 2"
    fake_supervisor_response_3 = "supervisor response 3"

    prompts = [fake_prompt_1, fake_prompt_2, fake_prompt_3]
    
    fake_llm_responses = [
        # Prompt 1: supervisor calls math-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_1", "name": MATH_AGENT_NAME, "args": {"query": fake_prompt_1}
        }]),
        AIMessage(content="child response 1"),
        AIMessage(content=fake_supervisor_response_1),
        # Prompt 2: supervisor calls calculator-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_2", "name": CALCULATOR_AGENT_NAME, "args": {"query": fake_prompt_2}
        }]),
        AIMessage(content="child response 2"),
        AIMessage(content=fake_supervisor_response_2),
        # Prompt 3: supervisor calls math-agent
        AIMessage(content="", tool_calls=[{
            "id": "call_3", "name": MATH_AGENT_NAME, "args": {"query": fake_prompt_3}
        }]),
        AIMessage(content="child response 3"),
        AIMessage(content=fake_supervisor_response_3),
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []
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
            
        # Verify all responses received
        assert fake_supervisor_response_1 in messages[0]
        assert fake_supervisor_response_2 in messages[1]
        assert fake_supervisor_response_3 in messages[2]
        
        # 9 LLM calls total: 3 per prompt (supervisor + child + supervisor)
        assert len(fake_llm.all_calls) == 9, \
            f"Expected 9 LLM calls (3 per prompt), got {len(fake_llm.all_calls)}"
        
        # First supervisor call: just the first prompt
        first_supervisor_call = fake_llm.all_calls[0]
        assert first_supervisor_call[0] == SystemMessage(content=SUPERVISOR_PROMPT)
        assert first_supervisor_call[-1] == HumanMessage(content=fake_prompt_1)
        assert len(first_supervisor_call) == 2, "First supervisor call should only have system prompt + user message"
        
        # Fourth call (second supervisor call): should include conversation history
        fourth_call = fake_llm.all_calls[3]
        assert fourth_call[0] == SystemMessage(content=SUPERVISOR_PROMPT)
        assert fourth_call[-1] == HumanMessage(content=fake_prompt_2)
        # History should include: system + human1 + AI(tool_call) + tool_result + AI(final) + human2
        assert len(fourth_call) > 3, "Second supervisor call should accumulate history"
        
        # Seventh call (third supervisor call): even more history
        seventh_call = fake_llm.all_calls[6]
        assert seventh_call[0] == SystemMessage(content=SUPERVISOR_PROMPT)
        assert seventh_call[-1] == HumanMessage(content=fake_prompt_3)
        # Should have accumulated even more history
        assert len(seventh_call) > len(fourth_call), \
            "Third supervisor call should have more history than second"

    finally:
        LLMManager._instance = None
