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
