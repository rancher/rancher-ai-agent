"""Shared helpers for the websocket integration tests."""

from typing import Callable
from unittest.mock import AsyncMock

import time
import multiprocessing
import requests

from _pytest.monkeypatch import MonkeyPatch
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import BaseTool
from mcp.server.fastmcp import FastMCP

from app.main import app
from app.services.agent.child import CHILD_TOOL_USE_INSTRUCTIONS as _CHILD_TOOL_USE_INSTRUCTIONS
from app.services.agent.loader import AgentConfig
from app.services.agent.system_prompts import SEQUENTIAL_TOOL_CALLS
from app.services.memory import StorageType

# Child agents build their system prompt as: system_prompt + CHILD_TOOL_USE_INSTRUCTIONS + SEQUENTIAL_TOOL_CALLS
CHILD_TOOL_USE_INSTRUCTIONS = _CHILD_TOOL_USE_INSTRUCTIONS + SEQUENTIAL_TOOL_CALLS


def add(a: int, b: int) -> str:
    """Add two numbers"""
    return f"sum is {a + b}"


def multiply(a: int, b: int) -> str:
    """Multiply two numbers"""
    return f"product is {a * b}"


def serve_mock_mcp(name: str, port: int, tools: list[Callable]):
    """Runs a mock MCP server exposing ``tools`` on ``port``.

    The server is built inside the child process so the process target stays picklable
    (required by the ``forkserver``/``spawn`` multiprocessing start methods).
    """
    import uvicorn

    mcp = FastMCP(name)
    for tool in tools:
        mcp.add_tool(tool)
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port, log_level="error")


def start_mock_mcp_servers(servers: dict[int, tuple[str, list[Callable]]]) -> list[multiprocessing.Process]:
    """Starts one mock MCP server per port and waits until all of them are available."""
    processes = [
        multiprocessing.Process(target=serve_mock_mcp, args=(name, port, tools))
        for port, (name, tools) in servers.items()
    ]
    for process in processes:
        process.start()

    # Wait for the mock servers to be available before running tests.
    for port in servers:
        mcp_server_available = False
        while not mcp_server_available:
            try:
                requests.get(f"http://localhost:{port}/mcp")
                mcp_server_available = True
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)

    return processes


class MockMemoryManager:
    def __init__(self):
        self.storage_type = StorageType.IN_MEMORY

    def get_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def setup_agent_environment(monkeypatch: MonkeyPatch, agent_configs: list[AgentConfig]):
    """Patches the app so the websocket builds agents from ``agent_configs`` without external services."""
    monkeypatch.setenv("INSECURE_SKIP_TLS", "true")

    app.memory_manager = MockMemoryManager()

    monkeypatch.setattr("app.routers.websocket.get_user_id_from_token", AsyncMock(return_value="test-user-id"))
    # RBAC is covered by unit tests; disable it here so build_agent doesn't reach
    # the Rancher/K8s API (SubjectAccessReview) during these flow tests.
    monkeypatch.setattr("app.services.agent.factory.rbac_enabled", lambda: False)
    monkeypatch.setattr("app.services.agent.factory.load_agent_configs", lambda: agent_configs)


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
                "name": None,
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
