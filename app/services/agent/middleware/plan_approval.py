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


def _plan_approval_enabled() -> bool:
    """Whether plan confirmation is enabled via the ``PLAN_ENABLED`` environment variable."""
    return os.environ.get("PLAN_ENABLED", "false").lower() == "true"


def plan_approval_middleware():
    """``@wrap_tool_call`` middleware that gates the initial plan behind human approval.

    Plan approval is opt-in and controlled by the ``PLAN_ENABLED`` environment variable.
    When it is not enabled, this middleware is a no-op and every tool call passes through.

    ``TodoListMiddleware`` exposes a ``write_todos`` tool the agent uses to lay out a
    multi-step plan. When the agent first creates that plan, this middleware pauses the
    graph via ``langgraph.types.interrupt()`` and surfaces the proposed todo list to the
    client so the user can accept, reject, or revise it before any work starts.

    - ``"yes"``: the ``write_todos`` tool executes normally and the agent proceeds.
    - ``"no"``: a ``ToolMessage`` with ``INTERRUPT_CANCEL_MESSAGE`` is returned so
      ``cancel_human_validation_middleware`` ends the graph gracefully.
    - any other text: treated as feedback. The plan is not written and a ``ToolMessage``
      relaying the feedback is returned so the agent revises the plan and calls
      ``write_todos`` again — which is gated by this middleware once more.

    Subsequent ``write_todos`` calls (status updates on an already-approved plan) are not
    gated, so the agent can mark todos in-progress/completed without re-prompting.
    """

    @wrap_tool_call
    async def plan_approval(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call

        # Plan confirmation is opt-in. When ``PLAN_ENABLED`` is not enabled, every
        # tool call — including ``write_todos`` — passes straight through.
        if not _plan_approval_enabled():
            return await handler(request)

        # Only gate the plan-writing tool; every other tool passes straight through.
        if tool_call["name"] != _WRITE_TODOS_TOOL:
            return await handler(request)

        # Only ask for approval when the plan is first created. Once todos exist in
        # state the plan was already approved, so status updates are not re-confirmed.
        if request.state.get("todos"):
            return await handler(request)

        todos = tool_call.get("args", {}).get("todos", [])
        additional_kwargs: dict = {"created_at": datetime.now().isoformat()}

        response = langgraph.types.interrupt(
            {
                "message": f"<plan-approval>{json.dumps({'todos': todos})}</plan-approval>",
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
            # executes, the `todos` state stays empty and the revised plan is gated
            # by this middleware again.
            logging.debug("User requested changes to the proposed plan")
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

        logging.debug("User approved the proposed plan")
        additional_kwargs["confirmation"] = True
        result = await handler(request)
        if isinstance(result, ToolMessage):
            result.additional_kwargs = {**result.additional_kwargs, **additional_kwargs}
        return result

    return plan_approval
