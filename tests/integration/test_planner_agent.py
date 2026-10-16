from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.loader import AgentConfig, AuthenticationType
from app.services.agent.planner import (
    PLANNER_PROMPT,
    REDUCER_SYSTEM_PROMPT,
    SUBTASK_EVALUATION_SYSTEM_PROMPT,
)
from app.services.agent.system_prompts import PLANNER_SUBTASK_INSTRUCTIONS
from app.services.llm import LLMManager
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.integration.common import (
    CHILD_TOOL_USE_INSTRUCTIONS,
    FakeMessagesListChatModelWithTools,
    add,
    collect_messages,
    multiply,
    setup_agent_environment,
    start_mock_mcp_servers,
)

import json
import re
import pytest

MATH_AGENT_NAME = "math-agent"
CALCULATOR_AGENT_NAME = "calculator-agent"

MATH_AGENT_DESCRIPTION = "Agent that can perform addition operations"
CALCULATOR_AGENT_DESCRIPTION = "Agent that can perform multiplication operations"

MATH_AGENT_PROMPT = "You are a math agent that can add numbers."
CALCULATOR_AGENT_PROMPT = "You are a calculator agent that can multiply numbers."

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_planner(module_monkeypatch):
    """Enables the planner and sets up two child agents backed by mock MCP servers."""
    module_monkeypatch.setenv("PLAN_ENABLED", "true")
    module_monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "false")

    math_config = AgentConfig(
        name=MATH_AGENT_NAME,
        displayName="Math Agent",
        description=MATH_AGENT_DESCRIPTION,
        system_prompt=MATH_AGENT_PROMPT,
        mcp_url="http://localhost:8003/mcp",
        authentication=AuthenticationType.NONE,
    )
    calculator_config = AgentConfig(
        name=CALCULATOR_AGENT_NAME,
        displayName="Calculator Agent",
        description=CALCULATOR_AGENT_DESCRIPTION,
        system_prompt=CALCULATOR_AGENT_PROMPT,
        mcp_url="http://localhost:8004/mcp",
        authentication=AuthenticationType.NONE,
    )
    setup_agent_environment(module_monkeypatch, [math_config, calculator_config])

    processes = start_mock_mcp_servers({
        8003: ("mock-planner-1", [add]),
        8004: ("mock-planner-2", [multiply]),
    })

    yield processes

    for process in processes:
        process.terminate()


def test_plan_executes_subtasks_and_reduces():
    """Tests that the planner builds a plan, runs each subtask in its child agent and reduces the results.

    LLM call order:
    1. Planner generates the plan (structured output)
    2. math-agent runs the first subtask
    3. Evaluator judges the first subtask as completed
    4. calculator-agent runs the second subtask, with the first result as context
    5. Evaluator judges the second subtask as completed
    6. Reducer combines both results into the final answer
    """
    fake_prompt = "add 2 and 3, then multiply by 4"
    first_task = "add 2 and 3"
    second_task = "multiply 5 by 4"
    final_answer = "The final result is 20."

    fake_llm_responses = [
        AIMessage(content="", tool_calls=[{
            "id": "plan_1",
            "name": "Plan",
            "args": {"subtasks": [
                {"task": first_task, "agent": MATH_AGENT_NAME},
                {"task": second_task, "agent": CALCULATOR_AGENT_NAME},
            ]},
        }]),
        AIMessage(content="sum is 5"),
        AIMessage(content="yes"),
        AIMessage(content="product is 20"),
        AIMessage(content="yes"),
        AIMessage(content=final_answer),
    ]

    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm

    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            # Consume any initial messages from the server (chat-metadata, etc.)
            websocket.receive_text()

            websocket.send_text(fake_prompt)
            full_message = collect_messages(websocket, 1)[0]

        # The plan progress is streamed; the last update has every subtask completed.
        plans = [json.loads(p) for p in re.findall(r"<plan>(.*?)</plan>", full_message)]
        assert plans, "Should contain plan progress events"
        assert plans[-1]["approval"] is False
        assert [(t["task"], t["agent"], t["status"]) for t in plans[-1]["tasks"]] == [
            (first_task, MATH_AGENT_NAME, "completed"),
            (second_task, CALCULATOR_AGENT_NAME, "completed"),
        ]

        assert final_answer in full_message, "Should contain the reducer's final answer"

        assert len(fake_llm.all_calls) == 6, \
            f"Expected 6 LLM calls (plan + 2 * (child + evaluation) + reduce), got {len(fake_llm.all_calls)}"

        # Call 0 — planner: system prompt listing the available agents + user request
        agents_description = (
            f"- {MATH_AGENT_NAME}: {MATH_AGENT_DESCRIPTION}\n"
            f"- {CALCULATOR_AGENT_NAME}: {CALCULATOR_AGENT_DESCRIPTION}"
        )
        assert fake_llm.all_calls[0] == [
            SystemMessage(content=PLANNER_PROMPT.format(agents=agents_description)),
            HumanMessage(content=fake_prompt),
        ]

        # Call 1 — math-agent: runs as a planner subtask
        math_call = fake_llm.all_calls[1]
        assert math_call[0] == SystemMessage(
            content=MATH_AGENT_PROMPT + CHILD_TOOL_USE_INSTRUCTIONS + PLANNER_SUBTASK_INSTRUCTIONS
        )
        assert math_call[1] == HumanMessage(content=first_task)

        # Call 3 — calculator-agent: receives the previous result as context
        calculator_call = fake_llm.all_calls[3]
        assert calculator_call[0] == SystemMessage(
            content=CALCULATOR_AGENT_PROMPT + CHILD_TOOL_USE_INSTRUCTIONS + PLANNER_SUBTASK_INSTRUCTIONS
        )
        assert "sum is 5" in calculator_call[1].content
        assert calculator_call[1].content.endswith(f"Your task:\n{second_task}")

        # Calls 2 and 4 — evaluator: judges each child's response
        for evaluation_call, task, response in [
            (fake_llm.all_calls[2], first_task, "sum is 5"),
            (fake_llm.all_calls[4], second_task, "product is 20"),
        ]:
            assert evaluation_call[0] == SystemMessage(content=SUBTASK_EVALUATION_SYSTEM_PROMPT)
            assert task in evaluation_call[1].content
            assert response in evaluation_call[1].content

        # Call 5 — reducer: combines the original request and both results
        reducer_call = fake_llm.all_calls[5]
        assert reducer_call[0] == SystemMessage(content=REDUCER_SYSTEM_PROMPT)
        assert fake_prompt in reducer_call[1].content
        assert "sum is 5" in reducer_call[1].content
        assert "product is 20" in reducer_call[1].content

    finally:
        LLMManager._instance = None


def test_plan_with_unknown_agent_fails():
    """Tests that a plan assigning a subtask to an agent that does not exist is stopped.

    LLM call order:
    1. Planner generates the plan (structured output)

    The first subtask fails right away because no such agent is available, so the
    plan stops: no child agent runs, no evaluation or reduction happens.
    """
    fake_prompt = "add 2 and 3, then multiply by 4"
    first_task = "add 2 and 3"
    second_task = "multiply 5 by 4"
    unknown_agent_name = "ghost-agent"

    fake_llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content="", tool_calls=[{
            "id": "plan_1",
            "name": "Plan",
            "args": {"subtasks": [
                {"task": first_task, "agent": unknown_agent_name},
                {"task": second_task, "agent": CALCULATOR_AGENT_NAME},
            ]},
        }]),
    ])
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm

    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            websocket.receive_text()

            websocket.send_text(fake_prompt)
            full_message = collect_messages(websocket, 1)[0]

        plans = [json.loads(p) for p in re.findall(r"<plan>(.*?)</plan>", full_message)]
        assert plans, "Should contain plan progress events"
        assert [(t["task"], t["agent"], t["status"]) for t in plans[-1]["tasks"]] == [
            (first_task, unknown_agent_name, "failed"),
            (second_task, CALCULATOR_AGENT_NAME, "pending"),
        ]

        assert len(fake_llm.all_calls) == 1, \
            f"Expected only the planner LLM call, got {len(fake_llm.all_calls)}"

    finally:
        LLMManager._instance = None


def test_plan_approval_rejected(monkeypatch):
    """Tests that rejecting a plan awaiting approval cancels it without running any subtask.

    LLM call order:
    1. Planner generates the plan (structured output)

    The plan is then sent to the user for approval; answering "no" ends the run, so no
    child agent, evaluation or reducer call follows.
    """
    monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "true")

    fake_prompt = "add 2 and 3, then multiply by 4"
    first_task = "add 2 and 3"
    second_task = "multiply 5 by 4"

    fake_llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content="", tool_calls=[{
            "id": "plan_1",
            "name": "Plan",
            "args": {"subtasks": [
                {"task": first_task, "agent": MATH_AGENT_NAME},
                {"task": second_task, "agent": CALCULATOR_AGENT_NAME},
            ]},
        }]),
    ])
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm

    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            websocket.receive_text()

            websocket.send_text(fake_prompt)
            approval_message = collect_messages(websocket, 1)[0]

            websocket.send_text("no")
            rejection_message = collect_messages(websocket, 1)[0]

        # The plan is presented for approval, with every subtask still pending.
        plans = [json.loads(p) for p in re.findall(r"<plan>(.*?)</plan>", approval_message)]
        assert len(plans) == 1, "Should contain exactly the plan awaiting approval"
        assert plans[0]["approval"] is True
        assert [(t["task"], t["agent"], t["status"]) for t in plans[0]["tasks"]] == [
            (first_task, MATH_AGENT_NAME, "pending"),
            (second_task, CALCULATOR_AGENT_NAME, "pending"),
        ]

        # The rejection ends the run with a cancelled plan event (tasks keep their original pending status).
        rejection_plans = [json.loads(p) for p in re.findall(r"<plan>(.*?)</plan>", rejection_message)]
        assert len(rejection_plans) == 1, "Should contain exactly the cancelled plan"
        assert rejection_plans[0]["approval"] is False
        assert rejection_plans[0]["status"] == "cancelled"
        assert [(t["task"], t["agent"], t["status"]) for t in rejection_plans[0]["tasks"]] == [
            (first_task, MATH_AGENT_NAME, "pending"),
            (second_task, CALCULATOR_AGENT_NAME, "pending"),
        ]

        assert len(fake_llm.all_calls) == 1, \
            f"Expected only the planner LLM call, got {len(fake_llm.all_calls)}"

    finally:
        LLMManager._instance = None


def _receive_until(websocket, *suffixes: str) -> str:
    """Receives text frames until the accumulated text ends with one of ``suffixes``."""
    text = ""
    while not text.endswith(suffixes):
        text += websocket.receive_text()
    return text


def _run_plan_with_oauth2_agents(monkeypatch, oauth_agent_names: list[str]):
    """Runs the two-subtask plan with the given agents protected by OAuth2.

    Both agents are built without tools at connection time (their MCP servers answer 401
    until a bearer token is sent). The OAuth2 flow is simulated: every
    ``<authentication>`` request is answered with ``authentication_confirmed`` and the
    token the callback would have stored is already in the OAuth store.

    Returns the text of every frame group received: the authentication requests, in
    order, and the final message.
    """
    from unittest.mock import AsyncMock, MagicMock
    from app.services.oauth2.cookies import get_oauth_cookie_names
    from app.services.oauth2.store import oauth_store

    first_task = "add 2 and 3"
    second_task = "multiply 5 by 4"
    final_answer = "The final result is 20."

    def agent_config(name, description, prompt, port, tool_name):
        return AgentConfig(
            name=name,
            displayName=name,
            description=description,
            system_prompt=prompt,
            mcp_url=f"http://localhost:{port}/mcp",
            authentication=AuthenticationType.OAUTH2 if name in oauth_agent_names else AuthenticationType.NONE,
        )

    configs = [
        agent_config(MATH_AGENT_NAME, MATH_AGENT_DESCRIPTION, MATH_AGENT_PROMPT, 8005, "add"),
        agent_config(CALCULATOR_AGENT_NAME, CALCULATOR_AGENT_DESCRIPTION, CALCULATOR_AGENT_PROMPT, 8006, "multiply"),
    ]
    setup_agent_environment(monkeypatch, configs)
    monkeypatch.delenv("RANCHER_API_TOKEN", raising=False)

    oauth_client = MagicMock()
    oauth_client.create_authorization_url = AsyncMock(
        return_value={"state": "state", "url": "https://idp.example/authorize", "code_verifier": "verifier"}
    )
    oauth_manager = MagicMock()
    oauth_manager.get_client.return_value = oauth_client
    monkeypatch.setattr(
        "app.services.oauth2.handler.OAuthClientManager",
        MagicMock(get_instance=MagicMock(return_value=oauth_manager)),
    )
    for name in oauth_agent_names:
        oauth_store.set_token(get_oauth_cookie_names(name)["access_token"], f"token-{name}", "")

    fake_llm = FakeMessagesListChatModelWithTools(responses=[
        AIMessage(content="", tool_calls=[{
            "id": "plan_1",
            "name": "Plan",
            "args": {"subtasks": [
                {"task": first_task, "agent": MATH_AGENT_NAME},
                {"task": second_task, "agent": CALCULATOR_AGENT_NAME},
            ]},
        }]),
        AIMessage(content="sum is 5"),
        AIMessage(content="yes"),
        AIMessage(content="product is 20"),
        AIMessage(content="yes"),
        AIMessage(content=final_answer),
    ])
    fake_llm.all_calls = []
    LLMManager._instance = fake_llm

    processes = start_mock_mcp_servers({
        8005: ("mock-oauth-1", [add], MATH_AGENT_NAME in oauth_agent_names),
        8006: ("mock-oauth-2", [multiply], CALCULATOR_AGENT_NAME in oauth_agent_names),
    })

    try:
        with client.websocket_connect("/v1/ws/messages") as websocket:
            websocket.receive_text()

            websocket.send_text("add 2 and 3, then multiply by 4")
            authentication_requests = []
            while True:
                text = _receive_until(websocket, "</authentication>", "</message>")
                if not text.endswith("</authentication>"):
                    break
                authentication_requests.append(text)
                websocket.send_text("authentication_confirmed")

        return authentication_requests, text, fake_llm
    finally:
        LLMManager._instance = None
        for process in processes:
            process.terminate()
        # Wait for the ports to be released before another test reuses them.
        for process in processes:
            process.join(timeout=5)


def _authenticated_agents(authentication_requests: list[str]) -> list[str]:
    return [
        json.loads(re.search(r"<authentication>(.*?)</authentication>", request).group(1))["agent"]
        for request in authentication_requests
    ]


def test_plan_resumes_after_oauth2_authentication(monkeypatch):
    """Tests that a plan stops to authenticate the one agent that needs OAuth2, then resumes and finishes.

    math-agent needs no authentication and runs first; calculator-agent triggers the
    OAuth2 flow when its subtask is reached, and the plan continues where it stopped.
    """
    authentication_requests, final_message, fake_llm = _run_plan_with_oauth2_agents(
        monkeypatch, [CALCULATOR_AGENT_NAME]
    )

    assert _authenticated_agents(authentication_requests) == [CALCULATOR_AGENT_NAME]
    assert "<error>" not in final_message
    assert "The final result is 20." in final_message
    assert len(fake_llm.all_calls) == 6, \
        f"Expected 6 LLM calls (plan + 2 * (child + evaluation) + reduce), got {len(fake_llm.all_calls)}"


def test_plan_resumes_after_oauth2_authentication_of_two_agents(monkeypatch):
    """Tests that a plan authenticates two different OAuth2 agents, one after the other, and finishes.

    Each agent triggers its own OAuth2 flow when its subtask is reached, so the plan is
    interrupted twice and resumes after each authentication.
    """
    authentication_requests, final_message, fake_llm = _run_plan_with_oauth2_agents(
        monkeypatch, [MATH_AGENT_NAME, CALCULATOR_AGENT_NAME]
    )

    assert _authenticated_agents(authentication_requests) == [MATH_AGENT_NAME, CALCULATOR_AGENT_NAME]
    assert "<error>" not in final_message
    assert "The final result is 20." in final_message
    assert len(fake_llm.all_calls) == 6, \
        f"Expected 6 LLM calls (plan + 2 * (child + evaluation) + reduce), got {len(fake_llm.all_calls)}"
