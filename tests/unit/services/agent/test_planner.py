"""Unit tests for the planner agent's models and helper functions."""

import json
from typing import cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.constants import INTERRUPT_CANCEL_REPLY
from app.services.agent._constants import INTERRUPT_CANCEL_MESSAGE
from app.services.agent.planner import (
    PLAN_FAILED_PREFIX,
    Plan,
    PlannerState,
    SubTask,
    _build_child_config,
    _build_task_message,
    _extract_text,
    _fail_plan,
    _format_plan,
    _is_cancelled,
    _is_direct_handoff,
    _last_plan_failure_details,
    _last_user_request,
    _parse_plan_from_raw,
    _plan_approval_enabled,
    _route_after_approval,
    _route_after_plan,
    _route_next,
)


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
        assert subtasks[1]["status"] == "pending"
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

    def test_route_after_plan_executes_direct_handoff(self):
        state = cast(PlannerState, {"subtasks": [{"status": "pending"}]})
        assert _route_after_plan(state) == "execute"

    def test_route_after_plan_uses_approval_setting(self, monkeypatch):
        state = cast(
            PlannerState,
            {"subtasks": [{"status": "pending"}, {"status": "pending"}]},
        )
        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "true")
        assert _route_after_plan(state) == "approval"

        monkeypatch.setenv("PLAN_APPROVAL_ENABLED", "false")
        assert _route_after_plan(state) == "execute"

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
        assert _last_plan_failure_details(
            non_adjacent
        ) is None


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

    def test_build_child_config_requires_parent_thread_id(self):
        with patch("app.services.agent.planner.ensure_config", return_value={"configurable": {}}):
            with pytest.raises(ValueError, match="thread_id is required"):
                _build_child_config("rancher")
