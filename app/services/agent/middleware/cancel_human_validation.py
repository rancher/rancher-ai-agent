from typing import Any
from datetime import datetime

from langchain.agents.middleware import AgentState, before_model
from langchain.messages import AIMessage, ToolMessage
from langchain_core.callbacks.manager import dispatch_custom_event
from langgraph.runtime import Runtime

from .._constants import INTERRUPT_CANCEL_MESSAGE
from ....constants import (
    INTERRUPT_CANCEL_REPLY,
    PLAN_CANCEL_LLM_REPLY,
    PLAN_CANCEL_REPLY,
)


def _active_plan_task(state: AgentState) -> str | None:
    """Return the in-progress todo description if a plan is being executed, else ``None``.

    ``TodoListMiddleware`` keeps the current plan in ``state["todos"]`` and the agent is
    mandated to keep exactly one todo ``in_progress`` while work remains. So an
    ``in_progress`` todo means the canceled tool was part of an active plan, and its
    ``content`` identifies the step that was rejected.
    """
    todos = state.get("todos") or []
    for todo in todos:
        if todo.get("status") == "in_progress":
            return todo.get("content") or None
    return None


def cancel_human_validation_middleware():
    """Before-model middleware: skip LLM call if the last tool was cancelled."""

    @before_model(can_jump_to=["end"])
    def cancel_human_validation(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.content == INTERRUPT_CANCEL_MESSAGE:
            # When the cancellation happened inside an active plan, surface a plan-specific
            # reply that offers to revise the plan. Unlike INTERRUPT_CANCEL_REPLY, this
            # reply is kept in the message history so a follow-up "yes" gives the LLM the
            # context it needs to propose a revised plan.
            active_task = _active_plan_task(state)
            if active_task is not None:
                # The graph jumps straight to end without a model call, so this reply is
                # never streamed as model tokens. Dispatch the user-facing text as a custom
                # event so the websocket forwards it to the client as visible text.
                dispatch_custom_event("plan_cancel", PLAN_CANCEL_REPLY)

                # The AIMessage stored in history carries the LLM-facing guidance (naming
                # the canceled step so the model avoids it on the next turn), while the
                # user still only sees PLAN_CANCEL_REPLY — live via the custom event above
                # and on reload via the ``display_message`` kwarg.
                task = active_task or getattr(last, "name", "") or "the canceled step"
                reply = AIMessage(
                    PLAN_CANCEL_LLM_REPLY.format(task=task),
                    additional_kwargs={
                        "display_message": PLAN_CANCEL_REPLY,
                        "created_at": datetime.now().isoformat(),
                    },
                )
                return {
                    "messages": [reply],
                    # Clear the current plan so it is actually cancelled and not left with a
                    # stale in-progress todo.
                    "todos": [],
                    "jump_to": "end",
                }
            return {
                "messages": [AIMessage(INTERRUPT_CANCEL_REPLY)],
                "jump_to": "end",
            }
        return None

    return cancel_human_validation
