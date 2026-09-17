"""
Unit tests for planner helpers that detect and report subtask failures.

These cover the pure helpers used to fail a plan when a child agent signals that
it could not complete its subtask, rather than silently marking it completed.
"""
from unittest.mock import patch

from app.services.agent._constants import SUBTASK_FAILED_MARKER
from app.services.agent.planner import (
    _build_task_message,
    _fail_plan,
    _subtask_failure_reason,
)


class TestSubtaskFailureReason:
    def test_marker_prefix_returns_reason(self):
        content = f"{SUBTASK_FAILED_MARKER}: missing the cluster name"
        assert _subtask_failure_reason(content) == "missing the cluster name"

    def test_marker_prefix_after_whitespace(self):
        content = f"\n  {SUBTASK_FAILED_MARKER}: no permissions\n"
        assert _subtask_failure_reason(content) == "no permissions"

    def test_marker_without_reason_uses_default(self):
        assert _subtask_failure_reason(f"{SUBTASK_FAILED_MARKER}:") == (
            "The agent reported it could not complete the task."
        )

    def test_no_marker_returns_none(self):
        assert _subtask_failure_reason("Here is your answer: 42.") is None

    def test_list_content_blocks_with_reasoning(self):
        # Some models return content as a list of typed blocks (text + reasoning);
        # the marker lives in a text block and must still be detected.
        content = [
            {"type": "text", "text": "", "index": 0},
            {"type": "reasoning_content", "reasoning_content": {"text": "thinking"}, "index": 1},
            {"type": "text", "text": f"{SUBTASK_FAILED_MARKER}: no cluster specified", "index": 2},
        ]
        assert _subtask_failure_reason(content) == "no cluster specified"

    def test_list_content_without_marker_returns_none(self):
        content = [
            {"type": "reasoning_content", "reasoning_content": {"text": "thinking"}, "index": 0},
            {"type": "text", "text": "Here is your answer.", "index": 1},
        ]
        assert _subtask_failure_reason(content) is None

    def test_marker_mid_text_is_not_flagged(self):
        # A prefix-only match avoids false positives when the child merely mentions
        # the marker while actually succeeding.
        content = f"The task succeeded. I did not need to use {SUBTASK_FAILED_MARKER}:."
        assert _subtask_failure_reason(content) is None


class TestBuildTaskMessage:
    def test_includes_failure_instruction_without_previous_results(self):
        message = _build_task_message("do the thing", [])
        assert message.startswith("do the thing")
        assert f"{SUBTASK_FAILED_MARKER}:" in message

    def test_includes_failure_instruction_with_previous_results(self):
        message = _build_task_message("do the thing", ["earlier result"])
        assert "earlier result" in message
        assert "Your task:\ndo the thing" in message
        assert f"{SUBTASK_FAILED_MARKER}:" in message


class TestFailPlan:
    def test_marks_subtask_failed_and_cancels_with_string_reason(self):
        subtasks = [
            {"task": "step one", "agent": "a", "status": "in_progress"},
            {"task": "step two", "agent": "b", "status": "pending"},
        ]
        results: list[str] = []

        with patch("app.services.agent.planner.dispatch_custom_event") as dispatch:
            update = _fail_plan(
                subtasks,
                results,
                index=0,
                task="step one",
                agent_name="a",
                error="missing information",
                emit_plan=False,
            )

        assert subtasks[0]["status"] == "failed"
        # The remaining subtask is left untouched so the plan stops here.
        assert subtasks[1]["status"] == "pending"
        assert update["cancelled"] is True
        assert update["subtasks"] is subtasks
        assert update["results"] is results
        reply = update["messages"][0].content
        assert "step one" in reply
        assert "missing information" in reply
        # The failure reply is also emitted as a custom event so the client shows it,
        # since child output is suppressed (no-stream).
        dispatch.assert_any_call("planner-message", reply)

    def test_accepts_exception_reason(self):
        subtasks = [{"task": "t", "agent": "a", "status": "in_progress"}]
        with patch("app.services.agent.planner.dispatch_custom_event"):
            update = _fail_plan(
                subtasks,
                [],
                index=0,
                task="t",
                agent_name="a",
                error=ValueError("boom"),
                emit_plan=False,
            )
        assert subtasks[0]["status"] == "failed"
        assert "boom" in update["messages"][0].content
