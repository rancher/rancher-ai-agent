"""
Subtask execution for the planner agent.

Runs the next pending subtask in its assigned child agent, surfaces child interrupts,
asks the LLM to evaluate the child's response, and stops the plan when a subtask fails
or is cancelled.
"""

import json
import logging
from typing import Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.callbacks.manager import dispatch_custom_event
import langgraph.types
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from ..supervisor import ChildAgent, _AgentCallCounter, _build_agent_metadata, _extract_last_message
from .._constants import INTERRUPT_CANCEL_MESSAGE
from .prompts import (
    CANCEL_PLAN_REQUEST,
    PLAN_SUBTASK_CANCELLED_REPLY,
    PLAN_SUBTASK_FAILED_REPLY,
    REQUEST_FAILURE_DETAILS,
    RESTART_PLAN_REQUEST,
    RETRY_SUBTASK_REQUEST,
    SUBTASK_EVALUATION_PROMPT,
    SUBTASK_EVALUATION_SYSTEM_PROMPT,
)


async def _run_pending_subtask(
    llm: BaseChatModel,
    agents_by_name: dict[str, ChildAgent],
    subtasks: list[dict],
    results: list[str],
    call_counter: _AgentCallCounter,
    emit_plan: bool,
    user_reply: str | None = None,
) -> dict | str:
    """Run the next pending subtask in its assigned child agent.

    If the child previously asked the user for more information, ``user_reply`` holds
    the user's answer and is sent to the child (whose thread already holds the task and
    its question) instead of the task message.

    Handles human-in-the-loop interrupts: if a child agent pauses for confirmation
    or to ask the user for more data, the subtask is left ``in_progress`` and the node
    returns so the graph routes back to execute. The next run surfaces the interrupt at
    the planner level (at most one planner interrupt per node run) and resumes the child
    once the user responds. On completion the subtask is
    marked completed and its result is appended to ``results``. When ``emit_plan`` is
    True, plan-progress events are dispatched to the client.

    If the child agent raises an error the plan is stopped rather than silently
    continuing with the remaining subtasks: the failure is reported to the user via
    ``_fail_plan``.

    Returns a state-update dict when the user cancels the plan, a subtask fails, or the
    child is paused on an interrupt, otherwise the child agent's final message content.
    """
    index = _next_subtask_index(subtasks)
    subtask = subtasks[index]
    agent_name = subtask["agent"]
    task = subtask["task"]

    child = agents_by_name.get(agent_name)
    if child is None:
        reason = f"No agent named '{agent_name}' is available to run this task."
        logging.error(reason)
        return _fail_plan(subtasks, results, index, task, reason, emit_plan)
    else:
        # Children running one step of a multi-subtask plan must only return their result,
        # without offering follow-ups; a direct hand-off talks to the user as usual.
        child_config = _build_child_config(agent_name, planner_subtask=not _is_direct_handoff(subtasks))
        child_state = await child.agent.aget_state(config=child_config)

        if child_state and child_state.interrupts:
            # The child is paused on a previous interrupt. Surface it at the planner
            # level to collect the user's decision, then resume the child.
            resume_value = langgraph.types.interrupt(child_state.interrupts[0].value)
            try:
                result = await child.agent.ainvoke(Command(resume=resume_value), config=child_config)
            except GraphBubbleUp:
                # LangGraph internal signal (e.g. a new interrupt) must propagate so
                # the planner runtime can handle it.
                raise
            except Exception as e:
                logging.exception(f"Subtask agent '{agent_name}' failed during resume: {e}")
                return _fail_plan(subtasks, results, index, task, e, emit_plan)

            # The user declined the confirmation: cancel the whole plan instead of
            # continuing with the remaining subtasks.
            if _is_cancelled(result):
                logging.debug("Planner subtask for agent '%s' cancelled by the user", agent_name)
                return _cancel_plan(subtasks, results, index, task, emit_plan)
        else:
            if user_reply is not None:
                # The subtask is already in progress: forward the user's answer.
                message = user_reply
            else:
                message = _build_task_message(task, results)
                subtasks[index]["status"] = "in_progress"
                if emit_plan:
                    dispatch_custom_event(
                        "planner-plan-created",
                        f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
                    )
            try:
                result = await child.agent.ainvoke(
                    {"messages": [HumanMessage(content=message)]},
                    config=child_config,
                )
            except GraphBubbleUp:
                raise
            except Exception as e:
                if _is_direct_handoff(subtasks):
                    raise e  # Let the exception bubble up so the error is displayed to the user.

                logging.exception(f"Subtask agent '{agent_name}' failed: {e}")
                return _fail_plan(subtasks, results, index, task, e, emit_plan)

        # ainvoke() suppresses a GraphInterrupt raised inside the child, returning
        # normally. If the child paused on a new interrupt, finish this node run with the
        # subtask still in progress and route back to execute: the next run surfaces the
        # interrupt from the resume branch above with a fresh resume index. Calling
        # interrupt() a second time here would be matched to a stale resume value on
        # replay, so a third interrupt would silently be skipped.
        child_state = await child.agent.aget_state(config=child_config)
        if child_state and child_state.interrupts:
            return {"subtasks": subtasks, "results": results}

        content = _extract_last_message(result)

    # The child returned without raising, but it may not have actually completed the
    # task (missing information, an error, a refusal, ...). Ask the LLM to judge the
    # child's response so the plan is stopped instead of marking the subtask completed
    # and continuing with the remaining subtasks. If the child is asking the user for more
    # information, pause the plan with the subtask still in progress: the user's next
    # message is forwarded to the child as its answer.
    evaluation = await _evaluate_subtask(llm, task, content)
    if evaluation == "needs_input":
        logging.debug("Planner subtask for agent '%s' is waiting for user input", agent_name)
        return {
            "subtasks": subtasks,
            "results": results,
            "awaiting_input": True,
            "messages": [AIMessage(content=content)],
        }
    if evaluation == "failed":
        logging.debug("Planner subtask for agent '%s' evaluated as failed", agent_name)
        reason = "The agent did not complete the task."
        return _fail_plan(subtasks, results, index, task, reason, emit_plan)

    # Recommend switching to single-agent selection if the same agent completes 5
    # consecutive subtasks in a row.
    agent_selected_count = call_counter.record(agent_name)
    if agent_selected_count >= 5:
        recommended_field = f', "recommended": "{agent_name}"'
        dispatch_custom_event(
            "subagent_choice_event",
            _build_agent_metadata(agent_name, "auto", recommended_field),
        )
        call_counter.count = 0

    subtasks[index]["status"] = "completed"
    results.append(f"Task: {task}\nAgent: {agent_name}\nResult: {content}")
    return content


async def _evaluate_subtask(
    llm: BaseChatModel, task: str, content: str | list
) -> Literal["completed", "needs_input", "failed"]:
    """Judge, using the LLM, whether the child agent completed the subtask.

    The child's reply is streamed to the client, so a marker-based signal is not
    reliably detectable. Instead, run a separate, non-streamed LLM call that receives
    the subtask and the child's response and answers "yes", "input" (the child is asking
    the user for more information) or "no".
    """
    response_text = _extract_text(content).strip()
    if not response_text:
        return "failed"

    messages = [
        SystemMessage(content=SUBTASK_EVALUATION_SYSTEM_PROMPT),
        HumanMessage(
            content=SUBTASK_EVALUATION_PROMPT.format(task=task, response=response_text)
        ),
    ]

    try:
        response = await llm.ainvoke(messages, config={"tags": ["no-stream"]})
    except Exception:  # noqa: BLE001
        logging.warning("Planner subtask evaluation failed", exc_info=True)
        return "completed"

    answer = _extract_text(response).strip().lower()
    if answer.startswith("input"):
        return "needs_input"
    if answer.startswith("no"):
        return "failed"
    return "completed"


def _fail_plan(
    subtasks: list[dict],
    results: list[str],
    index: int,
    task: str,
    error: Exception | str,
    emit_plan: bool,
) -> dict:
    """Stop the plan after a subtask fails and report the failure to the user.

    Marks the failed subtask, emits a plan-progress event when running a multi-subtask
    plan, and returns a state update that routes the graph to END (via ``cancelled``)
    with a user-facing explanation instead of silently marking the subtask completed and
    continuing with the remaining subtasks.

    ``error`` may be the exception that was raised or a plain reason string (e.g. when the
    child agent signalled failure via the marker or no agent was available).
    """
    subtasks[index]["status"] = "failed"
    subtasks[index]["actions"] = [
        RETRY_SUBTASK_REQUEST,
        RESTART_PLAN_REQUEST,
        REQUEST_FAILURE_DETAILS,
        CANCEL_PLAN_REQUEST,
    ]
    if emit_plan:
        dispatch_custom_event(
            "planner-plan-created",
            f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
        )
    return {
        "subtasks": subtasks,
        "results": results,
        "cancelled": True,
        "messages": [
            AIMessage(
                content=PLAN_SUBTASK_FAILED_REPLY.format(
                    task=task, error=error, plan=_format_plan(subtasks)
                )
            )
        ],
    }


def _cancel_plan(
    subtasks: list[dict],
    results: list[str],
    index: int,
    task: str,
    emit_plan: bool,
) -> dict:
    """Stop the plan after the user cancels a subtask and report it to the user.

    Mirrors ``_fail_plan``: marks the cancelled subtask, offers the retry/restart recovery
    actions, emits a plan-progress event when running a multi-subtask plan, and returns a
    state update that routes the graph to END (via ``cancelled``) with a user-facing
    explanation instead of continuing with the remaining subtasks.
    """
    subtasks[index]["status"] = "cancelled"
    subtasks[index]["actions"] = [
        RETRY_SUBTASK_REQUEST,
        RESTART_PLAN_REQUEST,
    ]
    if emit_plan:
        dispatch_custom_event(
            "planner-plan-created",
            f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
        )
    return {
        "subtasks": subtasks,
        "results": results,
        "cancelled": True,
        "messages": [
            AIMessage(
                content=PLAN_SUBTASK_CANCELLED_REPLY.format(
                    task=task, plan=_format_plan(subtasks)
                )
            )
        ],
    }


def _build_task_message(task: str, previous_results: list[str]) -> str:
    """Build the message sent to a child agent, including prior subtask outcomes.

    A subtask may depend on the results of the subtasks that ran before it, so the
    accumulated results are prepended as context ahead of the current task.
    """
    if not previous_results:
        return task
    joined = "\n\n".join(previous_results)
    return (
        "Results of the previous subtasks in the plan (use them as needed to complete "
        f"your task):\n{joined}\n\nYour task:\n{task}"
    )


def _format_plan(subtasks: list[dict]) -> str:
    """Render the plan's subtasks as JSON, matching the ``SubTask`` schema."""
    return json.dumps(subtasks)


def _next_subtask_index(subtasks: list[dict]) -> int:
    """Return the in-progress subtask (child paused on an interrupt), else the first pending one."""
    for status in ("in_progress", "pending"):
        for i, st in enumerate(subtasks):
            if st["status"] == status:
                return i
    raise ValueError("No in-progress or pending subtask to run.")


def _is_cancelled(result: dict) -> bool:
    """Return True if the last ToolMessage in a child agent result is the cancellation marker."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg.content == INTERRUPT_CANCEL_MESSAGE
    return False


def _is_direct_handoff(subtasks: list) -> bool:
    """Return True when the plan is a single subtask delegated straight to its agent.

    A one-subtask plan is treated as a plain delegation rather than an orchestrated plan:
    it is never confirmed with or exposed to the client, emits no plan-progress events,
    and its child agent's answer is returned as-is without going through the reducer.
    Centralizing this check keeps the plan, execute, and routing stages in agreement on
    what counts as a direct hand-off.
    """
    return len(subtasks) == 1


def _extract_text(raw: object) -> str:
    """Concatenate the textual content of an AIMessage, ignoring reasoning blocks.
    Message content can be a plain string or a list of typed blocks (e.g. ``text`` and
    ``reasoning_content``). Only ``text`` blocks are kept.
    """
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _build_child_config(agent_name: str, planner_subtask: bool = False) -> RunnableConfig:
    """Build a namespaced run-config so each child agent checkpoints independently.

    When ``planner_subtask`` is True the child is flagged as running a planner subtask,
    so its system prompt is extended to reply only with the final result.
    """
    parent_configurable = ensure_config().get("configurable", {})
    parent_thread_id = parent_configurable.get("thread_id", "")
    if not parent_thread_id:
        raise ValueError("thread_id is required in configurable but was not provided")

    child_configurable = {
        "thread_id": f"{parent_thread_id}::planner::{agent_name}",
    }
    if planner_subtask:
        child_configurable["planner_subtask"] = True
    return RunnableConfig(configurable=child_configurable, callbacks=[])
