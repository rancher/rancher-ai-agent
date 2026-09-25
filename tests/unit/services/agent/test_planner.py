"""Unit tests for the planner agent's models and helper functions."""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from app.constants import INTERRUPT_CANCEL_REPLY
from app.services.agent._constants import INTERRUPT_CANCEL_MESSAGE
from app.services.agent.planner import (
    CANCEL_PLAN_REQUEST,
    PLAN_CANCELLED_REPLY,
    PLAN_FAILED_PREFIX,
    REQUEST_FAILURE_DETAILS,
    RESTART_PLAN_REQUEST,
    RETRY_SUBTASK_REQUEST,
    Plan,
    PlannerState,
    SubTask,
    _build_child_config,
    _build_task_message,
    _create_plan,
    _evaluate_subtask,
    _extract_text,
    _fail_plan,
    _format_plan,
    _handle_failure_actions,
    _is_cancelled,
    _is_direct_handoff,
    _last_plan_failure_details,
    _last_user_request,
    _next_subtask_index,
    _parse_plan_from_raw,
    _plan_approval_enabled,
    _retry_all_subtasks,
    _retry_failed_subtasks,
    _route_after_approval,
    _route_after_plan,
    _route_next,
    _run_pending_subtask,
    create_planner_agent,
)
from app.services.agent.supervisor import ChildAgent, _AgentCallCounter

PARENT_CONFIG = {"configurable": {"thread_id": "parent"}}
FAILURE_ACTIONS = [
    RETRY_SUBTASK_REQUEST,
    RESTART_PLAN_REQUEST,
    REQUEST_FAILURE_DETAILS,
    CANCEL_PLAN_REQUEST,
]


def _mock_llm(response=None, structured_response=None) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=structured_response)
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def _mock_child(name: str, result=None, interrupts_after=()) -> ChildAgent:
    """Build a ChildAgent whose graph returns ``result`` and reports the given interrupts.

    ``interrupts_after`` is the interrupts reported by the state check that follows the
    child's invocation; the state check before invocation reports none.
    """
    config = MagicMock()
    config.name = name
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value=result)
    agent.aget_state = AsyncMock(
        side_effect=[
            SimpleNamespace(interrupts=[]),
            SimpleNamespace(interrupts=list(interrupts_after)),
        ]
    )
    return ChildAgent(config=config, agent=agent)


class TestTaskMessages:
    def test_returns_task_unchanged_without_previous_results(self):
        assert _build_task_message("do the thing", []) == "do the thing"

    def test_includes_previous_results_before_task(self):
        message = _build_task_message("do the thing", ["first result", "second result"])

        assert message.index("first result") < message.index("second result")
        assert message.index("second result") < message.index("Your task:\ndo the thing")


class TestPlanParsing:
    def test_extract_text_from_string_and_message(self):
        assert _extract_text("plain text") == "plain text"
        assert _extract_text(AIMessage(content="message text")) == "message text"

    def test_extract_text_keeps_only_text_blocks(self):
        content = [
            {"type": "reasoning_content", "reasoning_content": {"text": "hidden"}},
            {"type": "text", "text": "visible "},
            "suffix",
            {"type": "text", "text": "answer"},
        ]

        assert _extract_text(content) == "visible suffixanswer"

    def test_parse_plan_recovers_json_surrounded_by_markdown(self):
        raw = AIMessage(
            content='```json\n{"subtasks":[{"task":"List clusters","agent":"rancher"}]}\n```'
        )

        plan = _parse_plan_from_raw(raw)

        assert plan == Plan(subtasks=[SubTask(task="List clusters", agent="rancher")])

    @pytest.mark.parametrize(
        "raw",
        [
            AIMessage(content="not json"),
            AIMessage(content="{invalid json}"),
            AIMessage(content='{"subtasks": [{"task": "missing agent"}]}'),
        ],
    )
    def test_parse_plan_returns_none_for_unusable_content(self, raw):
        assert _parse_plan_from_raw(raw) is None

    def test_format_plan_returns_json(self):
        subtasks = [{"task": "List clusters", "agent": "rancher", "status": "pending"}]

        assert json.loads(_format_plan(subtasks)) == subtasks


class TestFailureReporting:
    def test_fail_plan_marks_task_failed_and_returns_user_message(self):
        subtasks = [
            {"task": "step one", "agent": "a", "status": "in_progress"},
            {"task": "step two", "agent": "b", "status": "pending"},
        ]
        results = ["an earlier result"]

        with patch("app.services.agent.planner.dispatch_custom_event") as dispatch:
            update = _fail_plan(
                subtasks,
                results,
                index=0,
                task="step one",
                error=ValueError("boom"),
                emit_plan=False,
            )

        assert subtasks[0]["status"] == "failed"
        assert subtasks[0]["actions"] == FAILURE_ACTIONS
        assert subtasks[1]["status"] == "pending"
        assert "actions" not in subtasks[1]
        assert update["subtasks"] is subtasks
        assert update["results"] is results
        assert update["cancelled"] is True
        assert update["messages"][0].content.startswith(PLAN_FAILED_PREFIX)
        assert "step one" in update["messages"][0].content
        assert "boom" in update["messages"][0].content
        dispatch.assert_not_called()

    def test_fail_plan_emits_updated_plan_when_requested(self):
        subtasks = [{"task": "step", "agent": "a", "status": "in_progress"}]

        with patch("app.services.agent.planner.dispatch_custom_event") as dispatch:
            _fail_plan(subtasks, [], 0, "step", "failed", emit_plan=True)

        event_name, payload = dispatch.call_args.args
        assert event_name == "planner-plan-created"
        assert json.loads(payload.removeprefix("<plan>").removesuffix("</plan>")) == {
            "tasks": subtasks,
            "approval": False,
        }


class TestRouting:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ({"cancelled": True, "subtasks": []}, "end"),
            ({"cancelled": False, "subtasks": [{"status": "pending"}]}, "execute"),
            ({"cancelled": False, "subtasks": [{"status": "completed"}]}, "end"),
            (
                {
                    "cancelled": False,
                    "subtasks": [{"status": "completed"}, {"status": "completed"}],
                },
                "reduce",
            ),
        ],
    )
    def test_route_next(self, state, expected):
        assert _route_next(cast(PlannerState, state)) == expected

    def test_route_next_resumes_in_progress_subtask(self):
        state = cast(
            PlannerState,
            {"cancelled": False, "subtasks": [{"status": "completed"}, {"status": "in_progress"}]},
        )
        assert _route_next(state) == "execute"

    def test_route_next_ends_while_awaiting_input(self):
        state = cast(
            PlannerState,
            {
                "cancelled": False,
                "awaiting_input": True,
                "subtasks": [{"status": "in_progress"}, {"status": "pending"}],
            },
        )
        assert _route_next(state) == "end"

    def test_route_after_plan_ends_on_empty_plan(self):
        assert _route_after_plan(cast(PlannerState, {"subtasks": []})) == "end"

    def test_route_after_plan_executes_direct_handoff(self, monkeypatch):
        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "true")
        state = cast(PlannerState, {"subtasks": [{"status": "pending"}]})
        assert _route_after_plan(state) == "execute"

    def test_route_after_plan_skips_approval_on_retry(self, monkeypatch):
        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "true")
        state = cast(
            PlannerState,
            {"subtasks": [{"status": "pending"}, {"status": "pending"}], "retry": True},
        )
        assert _route_after_plan(state) == "execute"

    def test_route_after_plan_skips_approval_when_awaiting_input(self, monkeypatch):
        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "true")
        state = cast(
            PlannerState,
            {"subtasks": [{"status": "in_progress"}, {"status": "pending"}], "awaiting_input": True},
        )
        assert _route_after_plan(state) == "execute"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, False), ("false", False), ("true", True), ("TRUE", True)],
    )
    def test_plan_approval_enabled(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("PLAN_APPROVAL_ENABLED", raising=False)
        else:
            monkeypatch.setenv("PLAN_APPROVAL_ENABLED", value)
        assert _plan_approval_enabled() is expected

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ({"cancelled": True, "feedback": []}, "end"),
            ({"cancelled": False, "feedback": ["split the second step"]}, "plan"),
            ({"cancelled": False, "feedback": []}, "execute"),
        ],
    )
    def test_route_after_approval(self, state, expected):
        assert _route_after_approval(cast(PlannerState, state)) == expected

    def test_direct_handoff_requires_exactly_one_subtask(self):
        assert not _is_direct_handoff([])
        assert _is_direct_handoff([{"status": "pending"}])
        assert not _is_direct_handoff([{"status": "pending"}, {"status": "pending"}])

    def test_next_subtask_index_prefers_in_progress_over_pending(self):
        subtasks = [{"status": "completed"}, {"status": "pending"}, {"status": "in_progress"}]
        assert _next_subtask_index(subtasks) == 2

    def test_next_subtask_index_returns_first_pending(self):
        subtasks = [{"status": "completed"}, {"status": "pending"}, {"status": "pending"}]
        assert _next_subtask_index(subtasks) == 1

    def test_next_subtask_index_raises_when_nothing_to_run(self):
        with pytest.raises(ValueError):
            _next_subtask_index([{"status": "completed"}, {"status": "failed"}])


class TestRetryHelpers:
    SUBTASKS = [
        {"task": "one", "agent": "a", "status": "completed"},
        {"task": "two", "agent": "b", "status": "failed", "actions": FAILURE_ACTIONS},
        {"task": "three", "agent": "a", "status": "pending"},
    ]

    @pytest.mark.parametrize("helper", [_retry_failed_subtasks, _retry_all_subtasks])
    def test_returns_none_without_previous_plan(self, helper):
        assert helper(cast(PlannerState, {"subtasks": []})) is None
        assert helper(cast(PlannerState, {})) is None

    def test_retry_failed_subtasks_resets_only_failed(self):
        plan = _retry_failed_subtasks(cast(PlannerState, {"subtasks": self.SUBTASKS}))

        assert plan is not None
        assert [st.status for st in plan.subtasks] == ["completed", "pending", "pending"]
        assert [st.task for st in plan.subtasks] == ["one", "two", "three"]

    def test_retry_all_subtasks_resets_every_subtask(self):
        plan = _retry_all_subtasks(cast(PlannerState, {"subtasks": self.SUBTASKS}))

        assert plan is not None
        assert [st.status for st in plan.subtasks] == ["pending", "pending", "pending"]
        assert [st.agent for st in plan.subtasks] == ["a", "b", "a"]


class TestMessageStateHelpers:
    def test_is_cancelled_checks_most_recent_tool_message(self):
        cancelled = ToolMessage(content=INTERRUPT_CANCEL_MESSAGE, tool_call_id="cancel")

        assert _is_cancelled({"messages": [AIMessage(content="ignored"), cancelled]})
        assert not _is_cancelled({"messages": [AIMessage(content=INTERRUPT_CANCEL_MESSAGE)]})

    def test_last_user_request_skips_non_human_messages(self):
        state = cast(
            PlannerState,
            {
                "messages": [
                    HumanMessage(content="first request"),
                    AIMessage(content="answer"),
                    HumanMessage(content="latest request"),
                    AIMessage(content="later answer"),
                ]
            },
        )

        assert _last_user_request(state) == "latest request"

    def test_last_plan_failure_details_requires_adjacent_failure(self):
        failure = AIMessage(content=f"{PLAN_FAILED_PREFIX} could not run a step")
        request = HumanMessage(content="try again")

        adjacent = cast(PlannerState, {"messages": [failure, request]})
        non_adjacent = cast(
            PlannerState,
            {"messages": [failure, AIMessage(content="other"), request]},
        )

        assert _last_plan_failure_details(adjacent) == failure.content
        assert _last_plan_failure_details(non_adjacent) is None
        assert _last_plan_failure_details(cast(PlannerState, {"messages": [request]})) is None


class TestChildConfiguration:
    def test_build_child_config_namespaces_thread_and_drops_parent_callbacks(self):
        with patch(
            "app.services.agent.planner.ensure_config",
            return_value={
                "configurable": {"thread_id": "parent", "request_id": "request"},
                "callbacks": ["parent callback"],
            },
        ):
            config = _build_child_config("rancher")

        assert config.get("configurable") == {"thread_id": "parent::planner::rancher"}
        assert config.get("callbacks") == []

    def test_build_child_config_flags_planner_subtask(self):
        with patch("app.services.agent.planner.ensure_config", return_value=PARENT_CONFIG):
            config = _build_child_config("rancher", planner_subtask=True)

        assert config.get("configurable") == {
            "thread_id": "parent::planner::rancher",
            "planner_subtask": True,
        }

    def test_build_child_config_requires_parent_thread_id(self):
        with patch("app.services.agent.planner.ensure_config", return_value={"configurable": {}}):
            with pytest.raises(ValueError, match="thread_id is required"):
                _build_child_config("rancher")


class TestHandleFailureActions:
    @pytest.mark.asyncio
    async def test_request_details_asks_llm_about_failure(self):
        reply = AIMessage(content="more details")
        llm = _mock_llm(response=reply)

        update = await _handle_failure_actions(REQUEST_FAILURE_DETAILS.upper(), "PLAN FAILED: boom", llm)

        assert update is not None
        assert update["messages"] == [reply]
        assert update["cancelled"] is False
        assert update["subtasks"] == []
        assert "PLAN FAILED: boom" in llm.ainvoke.call_args.kwargs["input"]

    @pytest.mark.asyncio
    async def test_cancel_ends_plan(self):
        llm = _mock_llm()

        update = await _handle_failure_actions(CANCEL_PLAN_REQUEST, "PLAN FAILED: boom", llm)

        assert update is not None
        assert update["cancelled"] is True
        assert update["subtasks"] == []
        assert update["messages"][0].content == PLAN_CANCELLED_REPLY
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_requests_are_not_handled(self):
        llm = _mock_llm()

        assert await _handle_failure_actions("do something else", "PLAN FAILED: boom", llm) is None
        llm.ainvoke.assert_not_called()


class TestCreatePlan:
    FAILED_STATE_SUBTASKS = [
        {"task": "one", "agent": "a", "status": "completed"},
        {"task": "two", "agent": "b", "status": "failed"},
    ]

    @staticmethod
    def _state(request: str, subtasks: list[dict] | None = None) -> PlannerState:
        return cast(
            PlannerState,
            {"messages": [HumanMessage(content=request)], "subtasks": subtasks or []},
        )

    @staticmethod
    def _human_prompt(llm: MagicMock) -> str:
        messages = llm.with_structured_output.return_value.ainvoke.call_args.args[0]
        return messages[1].content

    @pytest.mark.asyncio
    async def test_returns_parsed_plan(self):
        expected = Plan(subtasks=[SubTask(task="List clusters", agent="rancher")])
        llm = _mock_llm(structured_response={"raw": None, "parsed": expected})

        plan, retry = await _create_plan(llm, "- rancher: desc", self._state("list clusters"))

        assert plan == expected
        assert retry is False
        assert self._human_prompt(llm) == "list clusters"
        system = llm.with_structured_output.return_value.ainvoke.call_args.args[0][0]
        assert "- rancher: desc" in system.content

    @pytest.mark.asyncio
    async def test_recovers_plan_from_raw_message(self):
        raw = AIMessage(content='{"subtasks":[{"task":"List clusters","agent":"rancher"}]}')
        llm = _mock_llm(structured_response={"raw": raw, "parsed": None})

        plan, retry = await _create_plan(llm, "", self._state("list clusters"))

        assert plan == Plan(subtasks=[SubTask(task="List clusters", agent="rancher")])
        assert retry is False

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_fails(self):
        llm = _mock_llm()
        llm.with_structured_output.return_value.ainvoke.side_effect = RuntimeError("bad output")

        assert await _create_plan(llm, "", self._state("list clusters")) == (None, False)

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_is_parseable(self):
        llm = _mock_llm(structured_response={"raw": AIMessage(content="no plan"), "parsed": None})

        assert await _create_plan(llm, "", self._state("list clusters")) == (None, False)

    @pytest.mark.asyncio
    async def test_includes_feedback_in_prompt(self):
        expected = Plan(subtasks=[SubTask(task="t", agent="a")])
        llm = _mock_llm(structured_response={"raw": None, "parsed": expected})

        await _create_plan(llm, "", self._state("list clusters"), feedback=["first", "second"])

        prompt = self._human_prompt(llm)
        assert prompt.startswith("New user message:\nlist clusters")
        assert prompt.index("- first") < prompt.index("- second")
        assert "previous plan attempt failed" not in prompt

    @pytest.mark.asyncio
    async def test_includes_previous_failure_in_prompt(self):
        expected = Plan(subtasks=[SubTask(task="t", agent="a")])
        llm = _mock_llm(structured_response={"raw": None, "parsed": expected})

        await _create_plan(
            llm, "", self._state("use namespace foo"), previous_plan_failure_message="PLAN FAILED: boom"
        )

        prompt = self._human_prompt(llm)
        assert prompt.startswith("New user message:\nuse namespace foo")
        assert "PLAN FAILED: boom" in prompt

    @pytest.mark.asyncio
    async def test_retry_quick_action_reuses_plan_without_llm(self):
        llm = _mock_llm()
        state = self._state(RETRY_SUBTASK_REQUEST.lower(), self.FAILED_STATE_SUBTASKS)

        plan, retry = await _create_plan(llm, "", state, previous_plan_failure_message="PLAN FAILED: x")

        assert retry is True
        assert plan is not None
        assert [st.status for st in plan.subtasks] == ["completed", "pending"]
        llm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_quick_action_resets_plan_without_llm(self):
        llm = _mock_llm()
        state = self._state(RESTART_PLAN_REQUEST, self.FAILED_STATE_SUBTASKS)

        plan, retry = await _create_plan(llm, "", state, previous_plan_failure_message="PLAN FAILED: x")

        assert retry is True
        assert plan is not None
        assert [st.status for st in plan.subtasks] == ["pending", "pending"]
        llm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_quick_actions_are_ignored_without_previous_failure(self):
        expected = Plan(subtasks=[SubTask(task="t", agent="a")])
        llm = _mock_llm(structured_response={"raw": None, "parsed": expected})
        state = self._state(RETRY_SUBTASK_REQUEST, self.FAILED_STATE_SUBTASKS)

        plan, retry = await _create_plan(llm, "", state)

        assert plan == expected
        assert retry is False


class TestEvaluateSubtask:
    @pytest.mark.asyncio
    async def test_empty_response_fails_without_llm_call(self):
        llm = _mock_llm()

        assert await _evaluate_subtask(llm, "task", "   ") == "failed"
        llm.ainvoke.assert_not_called()

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("yes", "completed"),
            ("Yes.", "completed"),
            ("no", "failed"),
            ("No, it did not.", "failed"),
            ("input", "needs_input"),
            ("Input.", "needs_input"),
        ],
    )
    @pytest.mark.asyncio
    async def test_interprets_llm_answer(self, answer, expected):
        llm = _mock_llm(response=AIMessage(content=answer))

        assert await _evaluate_subtask(llm, "list clusters", "Here are the clusters") == expected
        prompt = llm.ainvoke.call_args.args[0][1].content
        assert "list clusters" in prompt
        assert "Here are the clusters" in prompt

    @pytest.mark.asyncio
    async def test_assumes_completed_when_evaluation_fails(self):
        llm = _mock_llm()
        llm.ainvoke.side_effect = RuntimeError("unavailable")

        assert await _evaluate_subtask(llm, "task", "response") == "completed"


class TestRunPendingSubtask:
    @pytest.fixture(autouse=True)
    def parent_config(self):
        with patch("app.services.agent.planner.ensure_config", return_value=PARENT_CONFIG):
            yield

    @pytest.fixture
    def dispatch(self):
        with patch("app.services.agent.planner.dispatch_custom_event") as dispatch:
            yield dispatch

    @staticmethod
    def _plan(*agents: str) -> list[dict]:
        return [{"task": f"task {i}", "agent": agent, "status": "pending"} for i, agent in enumerate(agents)]

    @pytest.mark.asyncio
    async def test_completes_subtask_and_records_result(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="done")]})
        subtasks = self._plan("rancher", "rancher")
        results = ["Task: earlier"]
        llm = _mock_llm(response=AIMessage(content="yes"))

        outcome = await _run_pending_subtask(
            llm, {"rancher": child}, subtasks, results, _AgentCallCounter(), emit_plan=True
        )

        assert outcome == "done"
        assert subtasks[0]["status"] == "completed"
        assert subtasks[1]["status"] == "pending"
        assert results[-1] == "Task: task 0\nAgent: rancher\nResult: done"
        sent = child.agent.ainvoke.call_args.args[0]["messages"][0].content
        assert "Task: earlier" in sent and "task 0" in sent
        config = child.agent.ainvoke.call_args.kwargs["config"]
        assert config["configurable"]["planner_subtask"] is True
        # The in-progress plan is emitted before the child runs.
        assert dispatch.call_args_list[0].args[0] == "planner-plan-created"

    @pytest.mark.asyncio
    async def test_direct_handoff_does_not_flag_subtask_or_emit_plan(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="done")]})
        llm = _mock_llm(response=AIMessage(content="yes"))

        outcome = await _run_pending_subtask(
            llm, {"rancher": child}, self._plan("rancher"), [], _AgentCallCounter(), emit_plan=False
        )

        assert outcome == "done"
        config = child.agent.ainvoke.call_args.kwargs["config"]
        assert "planner_subtask" not in config["configurable"]
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_agent_fails_plan(self, dispatch):
        subtasks = self._plan("missing", "rancher")

        outcome = await _run_pending_subtask(
            _mock_llm(), {}, subtasks, [], _AgentCallCounter(), emit_plan=True
        )

        assert isinstance(outcome, dict)
        assert outcome["cancelled"] is True
        assert subtasks[0]["status"] == "failed"
        assert "No agent named 'missing'" in outcome["messages"][0].content

    @pytest.mark.asyncio
    async def test_child_error_fails_multi_subtask_plan(self, dispatch):
        child = _mock_child("rancher")
        child.agent.ainvoke.side_effect = RuntimeError("boom")
        subtasks = self._plan("rancher", "rancher")

        outcome = await _run_pending_subtask(
            _mock_llm(), {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=True
        )

        assert isinstance(outcome, dict)
        assert outcome["cancelled"] is True
        assert subtasks[0]["status"] == "failed"
        assert "boom" in outcome["messages"][0].content

    @pytest.mark.asyncio
    async def test_child_error_propagates_on_direct_handoff(self, dispatch):
        child = _mock_child("rancher")
        child.agent.ainvoke.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await _run_pending_subtask(
                _mock_llm(), {"rancher": child}, self._plan("rancher"), [], _AgentCallCounter(), emit_plan=False
            )

    @pytest.mark.asyncio
    async def test_new_child_interrupt_leaves_subtask_in_progress(self, dispatch):
        child = _mock_child("rancher", {"messages": []}, interrupts_after=[SimpleNamespace(value="confirm?")])
        subtasks = self._plan("rancher", "rancher")
        llm = _mock_llm()

        outcome = await _run_pending_subtask(
            llm, {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=True
        )

        assert outcome == {"subtasks": subtasks, "results": []}
        assert subtasks[0]["status"] == "in_progress"
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_resumes_paused_child_with_user_response(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="done")]})
        child.agent.aget_state.side_effect = [
            SimpleNamespace(interrupts=[SimpleNamespace(value="confirm?")]),
            SimpleNamespace(interrupts=[]),
        ]
        subtasks = self._plan("rancher", "rancher")
        subtasks[0]["status"] = "in_progress"
        llm = _mock_llm(response=AIMessage(content="yes"))

        with patch("langgraph.types.interrupt", return_value="yes") as interrupt:
            outcome = await _run_pending_subtask(
                llm, {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=True
            )

        interrupt.assert_called_once_with("confirm?")
        assert child.agent.ainvoke.call_args.args[0].resume == "yes"
        assert outcome == "done"
        assert subtasks[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_declined_confirmation_cancels_plan(self, dispatch):
        cancelled = {"messages": [ToolMessage(content=INTERRUPT_CANCEL_MESSAGE, tool_call_id="c")]}
        child = _mock_child("rancher", cancelled)
        child.agent.aget_state.side_effect = [SimpleNamespace(interrupts=[SimpleNamespace(value="confirm?")])]
        subtasks = self._plan("rancher", "rancher")
        subtasks[0]["status"] = "in_progress"

        with patch("langgraph.types.interrupt", return_value="no"):
            outcome = await _run_pending_subtask(
                _mock_llm(), {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=True
            )

        assert isinstance(outcome, dict)
        assert outcome["cancelled"] is True
        assert outcome["messages"][0].content == INTERRUPT_CANCEL_REPLY
        assert subtasks[0]["status"] == "cancelled"
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_child_error_during_resume_fails_plan(self, dispatch):
        child = _mock_child("rancher")
        child.agent.ainvoke.side_effect = RuntimeError("resume failed")
        child.agent.aget_state.side_effect = [SimpleNamespace(interrupts=[SimpleNamespace(value="confirm?")])]
        subtasks = self._plan("rancher", "rancher")
        subtasks[0]["status"] = "in_progress"

        with patch("langgraph.types.interrupt", return_value="yes"):
            outcome = await _run_pending_subtask(
                _mock_llm(), {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=True
            )

        assert isinstance(outcome, dict)
        assert subtasks[0]["status"] == "failed"
        assert "resume failed" in outcome["messages"][0].content

    @pytest.mark.asyncio
    async def test_incomplete_subtask_fails_plan(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="I need a cluster name")]})
        subtasks = self._plan("rancher", "rancher")
        results: list[str] = []

        outcome = await _run_pending_subtask(
            _mock_llm(response=AIMessage(content="no")),
            {"rancher": child}, subtasks, results, _AgentCallCounter(), emit_plan=True,
        )

        assert isinstance(outcome, dict)
        assert outcome["cancelled"] is True
        assert subtasks[0]["status"] == "failed"
        assert results == []
        assert "did not complete the task" in outcome["messages"][0].content

    @pytest.mark.asyncio
    async def test_child_asking_for_input_pauses_plan(self, dispatch):
        question = "What is the name of the namespace?"
        child = _mock_child("rancher", {"messages": [AIMessage(content=question)]})
        subtasks = self._plan("rancher", "rancher")
        results: list[str] = []

        outcome = await _run_pending_subtask(
            _mock_llm(response=AIMessage(content="input")),
            {"rancher": child}, subtasks, results, _AgentCallCounter(), emit_plan=True,
        )

        assert isinstance(outcome, dict)
        assert outcome["awaiting_input"] is True
        assert "cancelled" not in outcome
        assert outcome["messages"][0].content == question
        assert subtasks[0]["status"] == "in_progress"
        assert "actions" not in subtasks[0]
        assert results == []
        # Only the in-progress plan is emitted; no failed plan follows.
        dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_forwards_user_reply_to_waiting_child(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="Namespace test-ns created")]})
        subtasks = self._plan("rancher", "rancher")
        subtasks[0]["status"] = "in_progress"
        results: list[str] = []

        outcome = await _run_pending_subtask(
            _mock_llm(response=AIMessage(content="yes")),
            {"rancher": child}, subtasks, results, _AgentCallCounter(), emit_plan=True,
            user_reply="test-ns",
        )

        assert outcome == "Namespace test-ns created"
        sent = child.agent.ainvoke.call_args.args[0]["messages"]
        assert len(sent) == 1 and sent[0].content == "test-ns"
        assert subtasks[0]["status"] == "completed"
        assert results == ["Task: task 0\nAgent: rancher\nResult: Namespace test-ns created"]
        # The subtask was already shown as in progress, so it is not emitted again.
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_direct_handoff_child_asking_for_input_pauses(self, dispatch):
        child = _mock_child("rancher", {"messages": [AIMessage(content="Which cluster?")]})
        subtasks = self._plan("rancher")

        outcome = await _run_pending_subtask(
            _mock_llm(response=AIMessage(content="input")),
            {"rancher": child}, subtasks, [], _AgentCallCounter(), emit_plan=False,
        )

        assert isinstance(outcome, dict)
        assert outcome["awaiting_input"] is True
        assert subtasks[0]["status"] == "in_progress"
        dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_recommends_agent_after_five_consecutive_subtasks(self, dispatch):
        counter = _AgentCallCounter()
        counter.record("rancher")
        counter.count = 4
        child = _mock_child("rancher", {"messages": [AIMessage(content="done")]})

        await _run_pending_subtask(
            _mock_llm(response=AIMessage(content="yes")),
            {"rancher": child}, self._plan("rancher", "rancher"), [], counter, emit_plan=False,
        )

        dispatch.assert_called_once()
        event_name, payload = dispatch.call_args.args
        assert event_name == "subagent_choice_event"
        assert '"recommended": "rancher"' in payload
        assert counter.count == 0


class TestAwaitingUserInput:
    """End-to-end planner graph runs where a child asks the user for more information."""

    @staticmethod
    def _graph(plan: Plan, evaluations: list[str], child_replies: list[str]):
        llm = _mock_llm(
            structured_response={"raw": None, "parsed": plan, "parsing_error": None},
        )
        llm.ainvoke.side_effect = [AIMessage(content=answer) for answer in evaluations]
        config = MagicMock()
        config.name = "rancher"
        config.description = "Rancher agent"
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            side_effect=[{"messages": [AIMessage(content=reply)]} for reply in child_replies]
        )
        agent.aget_state = AsyncMock(return_value=SimpleNamespace(interrupts=[]))
        graph = create_planner_agent(llm, [ChildAgent(config=config, agent=agent)], InMemorySaver())
        return graph, llm, agent

    @pytest.mark.asyncio
    async def test_direct_handoff_resumes_child_with_user_answer(self):
        plan = Plan(subtasks=[SubTask(task="Create a namespace.", agent="rancher")])
        graph, llm, agent = self._graph(
            plan, evaluations=["input", "yes"], child_replies=["What is the name?", "Namespace test-ns created"]
        )
        config = {"configurable": {"thread_id": "t"}}

        state = await graph.ainvoke({"messages": [HumanMessage(content="create a namespace")]}, config)

        assert state["awaiting_input"] is True
        assert state["subtasks"][0]["status"] == "in_progress"
        assert state["messages"][-1].content == "What is the name?"

        state = await graph.ainvoke({"messages": [HumanMessage(content="test-ns")]}, config)

        assert state["awaiting_input"] is False
        assert state["subtasks"][0]["status"] == "completed"
        assert state["messages"][-1].content == "Namespace test-ns created"
        # The answer is forwarded to the child instead of generating a new plan.
        llm.with_structured_output.return_value.ainvoke.assert_awaited_once()
        assert agent.ainvoke.call_args.args[0]["messages"][0].content == "test-ns"

    @pytest.mark.asyncio
    async def test_plan_continues_with_next_subtask_after_user_answer(self, monkeypatch):
        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "false")
        plan = Plan(
            subtasks=[
                SubTask(task="Create a namespace.", agent="rancher"),
                SubTask(task="Create a pod in the namespace.", agent="rancher"),
            ]
        )
        graph, llm, agent = self._graph(
            plan,
            evaluations=["input", "yes", "input"],
            child_replies=["What is the name?", "Namespace test-ns created", "Which image?"],
        )
        config = {"configurable": {"thread_id": "t"}}

        with patch("app.services.agent.planner.dispatch_custom_event"):
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content="create a namespace and a pod")]}, config
            )

            assert state["awaiting_input"] is True
            assert [st["status"] for st in state["subtasks"]] == ["in_progress", "pending"]
            assert state["results"] == []

            state = await graph.ainvoke({"messages": [HumanMessage(content="test-ns")]}, config)

        # The first subtask completes with the user's answer, then the second one runs
        # and pauses on its own question.
        assert state["awaiting_input"] is True
        assert [st["status"] for st in state["subtasks"]] == ["completed", "in_progress"]
        assert state["results"] == [
            "Task: Create a namespace.\nAgent: rancher\nResult: Namespace test-ns created"
        ]
        sent = [call.args[0]["messages"][0].content for call in agent.ainvoke.call_args_list]
        assert sent[1] == "test-ns"
        assert "Namespace test-ns created" in sent[2]
        assert "Create a pod in the namespace." in sent[2]
        llm.with_structured_output.return_value.ainvoke.assert_awaited_once()
