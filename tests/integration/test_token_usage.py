"""Integration tests for the chat token usage endpoint.

Runs real websocket conversations against a supervisor with two child agents, backed by
an in-memory checkpointer, then reads the recorded usage back through the REST API.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.dependencies import get_llm
from app.main import app
from app.services.agent.loader import AgentConfig, AuthenticationType
from app.services.llm import LLMManager
from app.services.memory import MemoryManager
from tests.integration.common import (
    FakeMessagesListChatModelWithTools,
    add,
    collect_messages,
    multiply,
    setup_agent_environment,
    start_mock_mcp_servers,
)

MATH_AGENT_NAME = "math-agent"
CALCULATOR_AGENT_NAME = "calculator-agent"
USER_ID = "test-user-id"
FAKE_MODEL = "fake-model"

client = TestClient(app)


class FakeUsageChatModel(FakeMessagesListChatModelWithTools):
    """Fake model whose streamed chunks carry the usage and model name of the scripted response."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.i] if self.i < len(self.responses) else None
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            chunk.message.usage_metadata = getattr(response, "usage_metadata", None)
            chunk.message.response_metadata = {"model_name": FAKE_MODEL}
            yield chunk


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _delegate(agent: str, query: str, call_id: str, usage: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": agent, "args": {"query": query}}],
        usage_metadata=usage,
    )


@pytest.fixture(scope="module", autouse=True)
def setup_environment(module_monkeypatch):
    setup_agent_environment(module_monkeypatch, [
        AgentConfig(
            name=MATH_AGENT_NAME,
            displayName="Math Agent",
            description="Agent that can perform addition operations",
            system_prompt="You are a math agent that can add numbers.",
            mcp_url="http://localhost:8007/mcp",
            authentication=AuthenticationType.NONE,
        ),
        AgentConfig(
            name=CALCULATOR_AGENT_NAME,
            displayName="Calculator Agent",
            description="Agent that can perform multiplication operations",
            system_prompt="You are a calculator agent that can multiply numbers.",
            mcp_url="http://localhost:8008/mcp",
            authentication=AuthenticationType.NONE,
        ),
    ])
    module_monkeypatch.setenv("ACTIVE_LLM", "openai")
    module_monkeypatch.setenv("DB_ENABLED", "false")
    module_monkeypatch.setattr("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=USER_ID))

    processes = start_mock_mcp_servers({
        8007: ("mock-usage-1", [add]),
        8008: ("mock-usage-2", [multiply]),
    })

    yield processes

    for process in processes:
        process.terminate()


@pytest.fixture(autouse=True)
def memory_manager():
    """A real MemoryManager per test, so the chat and usage endpoints read what the websocket wrote."""
    manager = MemoryManager()
    asyncio.run(manager.initialize())
    app.memory_manager = manager
    yield manager


def _run_chat(prompts: list[str], responses: list[AIMessage]) -> str:
    """Runs the prompts over a new websocket chat and returns its chat id."""
    LLMManager._instance = FakeUsageChatModel(responses=responses)
    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            metadata = websocket.receive_text()
            chat_id = metadata.split('"chatId": "')[1].split('"')[0]
            for prompt in prompts:
                websocket.send_text(prompt)
                collect_messages(websocket, 1)
        return chat_id
    finally:
        LLMManager._instance = None


def test_chat_usage_split_between_supervisor_and_child_agents():
    chat_id = _run_chat(
        ["add 1 and 2", "multiply 3 and 4"],
        [
            _delegate(MATH_AGENT_NAME, "add 1 and 2", "call_1", _usage(100, 10)),
            AIMessage(content="sum is 3", usage_metadata=_usage(50, 5)),
            AIMessage(content="The sum is 3.", usage_metadata=_usage(200, 20)),
            _delegate(CALCULATOR_AGENT_NAME, "multiply 3 and 4", "call_2", _usage(300, 30)),
            AIMessage(content="product is 12", usage_metadata=_usage(60, 6)),
            AIMessage(content="The product is 12.", usage_metadata=_usage(400, 40)),
        ],
    )

    response = client.get(f"/v1/api/chats/{chat_id}/usage")

    assert response.status_code == 200
    usage = response.json()
    assert usage["chatId"] == chat_id
    assert usage["llmCalls"] == 6
    assert usage["inputTokens"] == 100 + 50 + 200 + 300 + 60 + 400
    assert usage["outputTokens"] == 10 + 5 + 20 + 30 + 6 + 40
    assert usage["totalTokens"] == usage["inputTokens"] + usage["outputTokens"]

    by_agent = {name: (u["llmCalls"], u["totalTokens"]) for name, u in usage["byAgent"].items()}
    assert by_agent == {
        "supervisor": (4, 110 + 220 + 330 + 440),
        MATH_AGENT_NAME: (1, 55),
        CALCULATOR_AGENT_NAME: (1, 66),
    }
    assert set(usage["bySource"]) == {"agent"}
    assert usage["byModel"] == {"openai": {FAKE_MODEL: usage["bySource"]["agent"]}}


def test_chat_usage_without_llm_usage_metadata():
    chat_id = _run_chat(
        ["hello"],
        [
            _delegate(MATH_AGENT_NAME, "hello", "call_1", None),
            AIMessage(content="hi"),
            AIMessage(content="Hello!"),
        ],
    )

    usage = client.get(f"/v1/api/chats/{chat_id}/usage").json()

    assert usage["llmCalls"] == 0
    assert usage["totalTokens"] == 0
    assert usage["byAgent"] == {}


def test_chat_usage_unknown_chat_returns_404():
    assert client.get("/v1/api/chats/does-not-exist/usage").status_code == 404


def test_chat_usage_of_another_user_returns_404(module_monkeypatch):
    chat_id = _run_chat(
        ["hello"],
        [
            _delegate(MATH_AGENT_NAME, "hello", "call_1", _usage(10, 1)),
            AIMessage(content="hi", usage_metadata=_usage(10, 1)),
            AIMessage(content="Hello!", usage_metadata=_usage(10, 1)),
        ],
    )

    with module_monkeypatch.context() as mp:
        mp.setattr("app.routers.chat.get_user_id_from_token", AsyncMock(return_value="someone-else"))
        assert client.get(f"/v1/api/chats/{chat_id}/usage").status_code == 404


def test_chat_usage_requires_authentication(module_monkeypatch):
    with module_monkeypatch.context() as mp:
        mp.setattr("app.routers.chat.get_user_id_from_token", AsyncMock(return_value=None))
        assert client.get("/v1/api/chats/any/usage").status_code == 401


def _select_ui_tools(module_monkeypatch, chat_id: str | None, usage: dict | None):
    selector = MagicMock()
    selector.select_tools = AsyncMock(return_value=[])
    selector.last_response = AIMessage(content="", id="selector-call", usage_metadata=usage)
    body = {"prompt": "p", "tools": {"name": "tools"}, "context": []}
    if chat_id is not None:
        body["chatId"] = chat_id
    with module_monkeypatch.context() as mp:
        mp.setattr("app.routers.llm.get_user_id_from_token", AsyncMock(return_value=USER_ID))
        mp.setattr("app.routers.llm.create_ui_tools_selector", lambda **_: selector)
        app.dependency_overrides[get_llm] = lambda: MagicMock()
        try:
            return client.post("/v1/api/complete/ui-tools", json=body)
        finally:
            app.dependency_overrides.pop(get_llm, None)


def test_complete_ui_tools_usage_is_recorded_on_the_chat(module_monkeypatch):
    chat_id = _run_chat(
        ["hello"],
        [
            _delegate(MATH_AGENT_NAME, "hello", "call_1", _usage(10, 1)),
            AIMessage(content="hi", usage_metadata=_usage(10, 1)),
            AIMessage(content="Hello!", usage_metadata=_usage(10, 1)),
        ],
    )

    assert _select_ui_tools(module_monkeypatch, chat_id, _usage(40, 2)).status_code == 200

    usage = client.get(f"/v1/api/chats/{chat_id}/usage").json()
    assert usage["llmCalls"] == 4
    assert usage["bySource"]["ui-tools"]["totalTokens"] == 42
    assert [chat["id"] for chat in client.get("/v1/api/chats").json()] == [chat_id]


def test_complete_ui_tools_requires_chat_id(module_monkeypatch):
    assert _select_ui_tools(module_monkeypatch, None, _usage(40, 2)).status_code == 422


def test_complete_ui_tools_before_the_chat_is_saved(module_monkeypatch):
    """The UI may select tools for a chat the websocket hasn't saved yet."""
    assert _select_ui_tools(module_monkeypatch, "new-chat", _usage(40, 2)).status_code == 200

    chat_usage = client.get("/v1/api/chats/new-chat/usage")
    assert chat_usage.status_code == 200
    assert chat_usage.json()["totalTokens"] == 42


def test_complete_ui_tools_usage_never_counts_for_the_chat_owner(module_monkeypatch):
    chat_id = _run_chat(
        ["hello"],
        [
            _delegate(MATH_AGENT_NAME, "hello", "call_1", _usage(10, 1)),
            AIMessage(content="hi", usage_metadata=_usage(10, 1)),
            AIMessage(content="Hello!", usage_metadata=_usage(10, 1)),
        ],
    )

    with module_monkeypatch.context() as mp:
        mp.setattr("app.routers.llm.get_user_id_from_token", AsyncMock(return_value="someone-else"))
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        selector.last_response = AIMessage(content="", id="foreign", usage_metadata=_usage(500, 0))
        mp.setattr("app.routers.llm.create_ui_tools_selector", lambda **_: selector)
        app.dependency_overrides[get_llm] = lambda: MagicMock()
        try:
            body = {"prompt": "p", "tools": {"name": "tools"}, "context": [], "chatId": chat_id}
            assert client.post("/v1/api/complete/ui-tools", json=body).status_code == 200
        finally:
            app.dependency_overrides.pop(get_llm, None)

    assert client.get(f"/v1/api/chats/{chat_id}/usage").json()["totalTokens"] == 33
