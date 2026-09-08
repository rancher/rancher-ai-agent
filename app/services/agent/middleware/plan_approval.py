import json
import logging
import os
from collections.abc import Callable
from datetime import datetime

import langgraph.types
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from .._constants import INTERRUPT_CANCEL_MESSAGE

# Name of the tool provided by ``TodoListMiddleware`` that writes/updates the plan.
_WRITE_TODOS_TOOL = "write_todos"


def plan_approval_enabled() -> bool:
    """Whether plan confirmation is enabled via the ``PLAN_APPROVAL_ENABLED`` environment variable."""
    return os.environ.get("PLAN_APPROVAL_ENABLED", "false").lower() == "true"


def has_active_plan(todos) -> bool:
    """Whether a proposed ``write_todos`` call is a status update on an approved plan.

    ``TodoListMiddleware`` replaces the whole ``todos`` list on every ``write_todos`` call
    and requires previously ``completed`` todos to be carried forward unchanged. So any
    status update (or revision) of an already-approved plan always includes at least one
    ``completed`` todo, whereas a brand-new plan has none — the initial layout, or the
    first plan after the previous one finished or was stopped, hasn't completed anything.

    The decision is based on the *proposed* todos (the ``write_todos`` args), not the
    persisted ``todos`` state. That state is never cleared, so a plan that was stopped
    before completing would leave stale non-completed todos behind; keying off it would
    make the next request's fresh plan look like an in-progress plan and skip approval.
    A brand-new plan has no completed todos regardless of that stale state, so it is
    gated for approval again.
    """
    return any(todo.get("status") == "completed" for todo in todos)


def plan_approval_middleware():
    """``@wrap_tool_call`` middleware that gates every new plan behind human approval.

    Plan approval is opt-in and controlled by the ``PLAN_APPROVAL_ENABLED`` environment
    variable. This middleware is only registered when plan approval is enabled (see
    ``plan_approval_enabled``), so when disabled it is never added to the agent.

    ``TodoListMiddleware`` exposes a ``write_todos`` tool the agent uses to lay out a
    multi-step plan. When the agent proposes a brand-new plan, this middleware pauses the
    graph via ``langgraph.types.interrupt()`` and surfaces the proposed todo list to the
    client so the user can accept, reject, or revise it before any work starts.

    - ``"yes"``: the ``write_todos`` tool executes normally and the agent proceeds.
    - ``"no"``: a ``ToolMessage`` with ``INTERRUPT_CANCEL_MESSAGE`` is returned so
      ``cancel_human_validation_middleware`` ends the graph gracefully.
    - any other text: treated as feedback. The plan is not written and a ``ToolMessage``
      relaying the feedback is returned so the agent revises the plan and calls
      ``write_todos`` again — which is gated by this middleware once more.

    Status updates on an already-approved plan (``write_todos`` calls that carry the plan's
    completed todos forward) are not gated, so the agent can mark todos
    in-progress/completed without re-prompting. Every fresh plan — including the ones that
    follow a finished or stopped plan — is gated again (see ``has_active_plan``).
    """

    @wrap_tool_call
    async def plan_approval(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call

        # Only gate the plan-writing tool; every other tool passes straight through.
        if tool_call["name"] != _WRITE_TODOS_TOOL:
            return await handler(request)

        todos = tool_call.get("args", {}).get("todos", [])

        # Only ask for approval when a brand-new plan is proposed. A status update or
        # revision of an already-approved plan carries its completed todos forward, so it
        # is not re-confirmed. A fresh plan (initial layout, or the first plan after the
        # previous one finished or was stopped) has no completed todos and is gated again.
        if has_active_plan(todos):
            return await handler(request)

        additional_kwargs: dict = {"created_at": datetime.now().isoformat()}

        response = langgraph.types.interrupt(
            {
                "message": f"<planning-approval>{json.dumps(todos)}</planning-approval>",
                "todos": todos,
            }
        )

        normalized = response.strip().lower() if isinstance(response, str) else response

        if normalized == "no":
            logging.debug("User rejected the proposed plan")
            additional_kwargs["confirmation"] = False
            return ToolMessage(
                content=INTERRUPT_CANCEL_MESSAGE,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs=additional_kwargs,
            )

        if normalized != "yes":
            # Any answer other than yes/no is treated as feedback: the plan is not
            # written and the agent is asked to revise it. Since write_todos never
            # executes, the `todos` state is unchanged (no active plan) and the revised
            # plan is gated by this middleware again.
            additional_kwargs["confirmation"] = False
            return ToolMessage(
                content=(
                    "The user requested changes to this plan. Immediately call "
                    "write_todos again with the revised plan that incorporates the "
                    "feedback below. Do NOT reply with text, do NOT ask the user to "
                    "confirm, and do NOT wait for further input — the updated plan will "
                    "be presented to the user for approval automatically.\n"
                    f"User feedback: {response}"
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs=additional_kwargs,
            )

        additional_kwargs["confirmation"] = True
        result = await handler(request)
        if isinstance(result, ToolMessage):
            result.additional_kwargs = {**result.additional_kwargs, **additional_kwargs}
        return result

    return plan_approval
