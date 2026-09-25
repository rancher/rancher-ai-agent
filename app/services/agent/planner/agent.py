
"""
Planner agent implementation.

The planner first builds a plan: a list of subtasks, each assigned to one of the
available child agents. The subtasks are then executed sequentially, each child
agent notifying when it has finished. Once every subtask is completed, a reducer
node synthesizes the individual results into a single final answer for the user.
"""

import json
import logging
import os
from typing import Annotated, Literal, TypedDict, cast
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import dispatch_custom_event
import langgraph.types
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, Checkpointer

from ..supervisor import ChildAgent, _AgentCallCounter
from ..middleware import (
    MessagesHistoryMiddleware,
    inject_additional_kwargs_middleware,
    ui_tools_middleware,
)
from .prompts import (
    CANCEL_PLAN_REQUEST,
    PLAN_CANCELLED_PREFIX,
    PLAN_CANCELLED_REPLY,
    PLAN_FAILED_PREFIX,
    PLAN_FAILED_REPLY,
    PLANNER_CANCEL_SUFFIX,
    PLANNER_FEEDBACK_SUFFIX,
    PLANNER_PROMPT,
    PLANNER_RETRY_SUFFIX,
    REDUCER_PROMPT,
    REDUCER_SYSTEM_PROMPT,
    REQUEST_FAILURE_DETAILS,
    RESTART_PLAN_REQUEST,
    RETRY_SUBTASK_REQUEST,
)
from .subtasks import (
    _build_child_config,
    _extract_text,
    _is_direct_handoff,
    _run_pending_subtask,
)


class SubTask(BaseModel):
    """A single unit of work assigned to a child agent."""

    task: str = Field(description="The task to be performed.")
    status: Literal["pending", "in_progress", "completed", "cancelled", "failed"] = Field(
        default="pending", description="The current status of the subtask."
    )
    agent: str = Field(description="The name of the child agent chosen to run this task.")


class Plan(BaseModel):
    """The full plan produced by the planner: an ordered list of subtasks."""

    subtasks: list[SubTask] = Field(description="Ordered list of subtasks to execute.")


class PlannerState(TypedDict):
    """State shared across the planner graph nodes."""

    messages: Annotated[list, add_messages]
    messages_history: Annotated[list, add_messages]
    subtasks: list[dict]
    results: list[str]
    cancelled: bool
    feedback: list[str]
    retry: bool
    awaiting_input: bool



def create_planner_agent(
    llm: BaseChatModel,
    child_agents: list[ChildAgent],
    checkpointer: Checkpointer,
) -> CompiledStateGraph:
    """Create a planner agent that plans, delegates, and reduces child agent work.

    Args:
        llm: The language model to use for planning and reducing.
        child_agents: A list of child agents that the planner can delegate tasks to.
        checkpointer: A checkpointer to manage state persistence.

    Returns:
        A compiled state graph representing the planner agent.
    """
    agents_by_name = {child.config.name: child for child in child_agents}
    agents_description = "\n".join(
        f"- {child.config.name}: {child.config.description}" for child in child_agents
    )
    call_counter = _AgentCallCounter()

    # TODO check middleware here!
    reducer_agent = create_agent(
        llm,
        tools=[],
        system_prompt=REDUCER_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name="planner-reducer",
        middleware=[
            MessagesHistoryMiddleware(),
            inject_additional_kwargs_middleware(),
            ui_tools_middleware(llm),
            SummarizationMiddleware(model=llm, trigger=[("messages", 30), ("tokens", 30000)], keep=("messages", 15)),
        ],
    )

    async def plan_node(state: PlannerState) -> dict:
        """Generate the list of subtasks from the user's request.

        The generated plan is written to state so that, when the plan needs user
        approval, the separate approval node can interrupt and later resume without
        re-running this node. Re-running the node would call the LLM again and produce a
        different plan, so the plan shown to the user (``<plan-approval>``) would not
        match the plan that gets executed (``<plan>``). Any accumulated feedback from
        rejected plans is read from state so successive rounds of feedback all apply.
        """
        feedback = list(state.get("feedback") or [])

        request = _last_user_request(state)

        # A subtask asked the user for more information: the new message is the user's
        # answer, so keep the current plan and let execute_node forward it to the child.
        if state.get("awaiting_input") and any(
            st["status"] == "in_progress" for st in state.get("subtasks") or []
        ):
            return {"messages_history": [HumanMessage(content=request)]}

        previous_plan_failure_message = _last_plan_failure_details(state)

        if previous_plan_failure_message:
            failure_action_result = await _handle_failure_actions(request, previous_plan_failure_message, llm)
            if failure_action_result is not None:
                return failure_action_result

        previous_plan_cancellation_message = _last_plan_cancellation_details(state)

        plan, retry = await _create_plan(
            llm,
            agents_description,
            state,
            feedback,
            previous_plan_failure_message,
            previous_plan_cancellation_message,
        )
        if plan is None or plan.subtasks is None or not plan.subtasks:
            logging.error("Planner failed to produce a valid plan.")
            return {
                "subtasks": [],
                "results": [],
                "cancelled": False,
                "feedback": [],
                "messages": [AIMessage(content=PLAN_FAILED_REPLY)],
            }

        subtasks = [subtask.model_dump() for subtask in plan.subtasks]
        return {
            "subtasks": subtasks, 
            "results": [], 
            "cancelled": False, 
            "feedback": feedback, 
            "retry": retry, 
            "messages_history": [HumanMessage(content=request)],
        }

    async def approval_node(state: PlannerState) -> dict:
        """Ask the user to approve the plan that ``plan_node`` produced.

        This node only interrupts and interprets the user's response; it never
        regenerates the plan. On resume LangGraph re-runs the node from the top, but the
        plan already lives in state, so the approved plan (``<plan-approval>``) is exactly
        the one that gets executed. A "yes" proceeds to execution, a "no" cancels, and any
        other response is accumulated as feedback and routed back to ``plan_node`` for a
        fresh plan that the user reviews again.
        """
        subtasks = state["subtasks"]
        response = langgraph.types.interrupt(
            f"<plan>{json.dumps({'tasks': subtasks, 'approval': True})}</plan>"
        )
        normalized = response.strip().lower() if isinstance(response, str) else response

        if normalized == "yes":
            return {"feedback": []}

        if normalized == "no":
            logging.debug("Planner plan was rejected by the user.")
            return {
                "subtasks": [],
                "results": [],
                "cancelled": True,
                "feedback": [],
                "messages": [AIMessage(content="Plan was not approved by the user.")],
            }

        # Any other response is treated as feedback: accumulate it so successive rounds of
        # feedback all apply, then route back to plan_node to regenerate the plan.
        logging.debug("Planner regenerating plan from user feedback.")
        return {"feedback": [*(state.get("feedback") or []), response]}

    async def execute_node(state: PlannerState) -> dict:
        """Run the next pending subtask in its assigned child agent."""
        subtasks = state["subtasks"]
        results = list(state.get("results", []))
        # When the child asked for more information, the latest user message is its answer.
        user_reply = _last_user_request(state) if state.get("awaiting_input") else None

        # A single-subtask plan is handed off directly to the child agent: run it with no
        # plan-progress events and return its answer as-is; _route_next then sends the
        # result straight to END, skipping the reducer.
        if _is_direct_handoff(subtasks):
            outcome = await _run_pending_subtask(
                llm, agents_by_name, subtasks, results, call_counter, emit_plan=False, user_reply=user_reply
            )
            if isinstance(outcome, dict):
                return {"awaiting_input": False, **outcome}
            return {
                "subtasks": subtasks,
                "results": results,
                "awaiting_input": False,
                "messages": [AIMessage(content=outcome)],
            }

        outcome = await _run_pending_subtask(
            llm, agents_by_name, subtasks, results, call_counter, emit_plan=True, user_reply=user_reply
        )
        if isinstance(outcome, dict):
            return {"awaiting_input": False, **outcome}

        # The child agent notifies that it has finished its subtask.
        dispatch_custom_event(
            "planner-plan-finished",
            f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
        )
        return {"subtasks": subtasks, "results": results, "awaiting_input": False}

    async def reduce_node(state: PlannerState) -> dict:
        """Combine all subtask results into a single final answer."""
        request = _last_user_request(state)
        joined = "\n\n".join(state.get("results", [])) or "No subtasks were executed."
        prompt = REDUCER_PROMPT.format(request=request, results=joined)
        config = _build_child_config("reducer")
        result = await reducer_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )
        return {
            "messages": [result["messages"][-1]], 
            "subtasks":[], 
            "results": [],
            "messages_history": [result["messages"][-1]]
        }

    graph = StateGraph(PlannerState)
    # "plan" node: Analyzes the user's request and generates a list of sequential subtasks,
    # mapping each subtask to the most suitable child agent.
    graph.add_node("plan", plan_node)
    # "approval" node: Blocks execution to show the generated plan to the user,
    # waiting for approval ("yes"), rejection ("no"), or refinement feedback.
    graph.add_node("approval", approval_node)
    # "execute" node: Sequentially triggers child agents to complete pending subtasks
    # and logs their respective execution results.
    graph.add_node("execute", execute_node)
    # "reduce" node: Synthesizes the results of all executed subtasks
    # into a final summarized answer for the user.
    graph.add_node("reduce", reduce_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", _route_after_plan, {"approval": "approval", "execute": "execute", "end": END}
    )
    graph.add_conditional_edges(
        "approval", _route_after_approval, {"plan": "plan", "execute": "execute", "end": END}
    )
    graph.add_conditional_edges("execute", _route_next, {"execute": "execute", "reduce": "reduce", "end": END})
    graph.add_edge("reduce", END)

    return graph.compile(checkpointer=checkpointer)


async def _create_plan(
    llm: BaseChatModel,
    agents_description: str,
    state: PlannerState,
    feedback: list[str] | None = None,
    previous_plan_failure_message: str | None = None,
    previous_plan_cancellation_message: str | None = None,
) -> tuple[Plan | None, bool]:
    """Generate a plan from the user's request using the LLM.

    When ``feedback`` is provided, the user rejected one or more previous plans and
    asked for changes; every round of feedback is appended to the prompt, in order,
    so the new plan reflects all of their requests.

    When the user's latest message immediately follows a failed plan, the retry
    suffix is appended instead, instructing the LLM to build a brand-new, complete
    plan for the whole request rather than resuming from where the previous attempt
    stopped.
    """
    request = _last_user_request(state)

    # Check harcoded messages from quick actions
    if previous_plan_failure_message or previous_plan_cancellation_message:
        normalized_request = request.lower()
        if normalized_request == RETRY_SUBTASK_REQUEST.lower():
            # Re-run the plan that just stopped: keep the existing subtasks and only
            # reset the failed/cancelled ones to pending so execution resumes from where
            # it stopped, instead of asking the LLM for a brand-new plan.
            return _retry_failed_subtasks(state), True
        if normalized_request == RESTART_PLAN_REQUEST.lower():
            # Restart the whole plan from the beginning: reset every subtask to
            # pending instead of asking the LLM for a brand-new plan.
            return _retry_all_subtasks(state), True

    if previous_plan_failure_message or previous_plan_cancellation_message or feedback:
        # Label the request so it is not confused with the appended failure/cancellation/
        # feedback context that follows it.
        human_content = f"New user message:\n{request}"
        if previous_plan_failure_message:
            human_content += PLANNER_RETRY_SUFFIX.format(details=previous_plan_failure_message)
        if previous_plan_cancellation_message:
            human_content += PLANNER_CANCEL_SUFFIX.format(details=previous_plan_cancellation_message)
        if feedback:
            joined = "\n".join(f"- {item}" for item in feedback)
            human_content += PLANNER_FEEDBACK_SUFFIX.format(feedback=joined)
    else:
        human_content = request

    messages = [
        SystemMessage(content=PLANNER_PROMPT.format(agents=agents_description)),
        HumanMessage(content=human_content),
    ]

    plan: Plan | None = None
    try:
        response = await llm.with_structured_output(
            Plan, include_raw=True
        ).ainvoke(messages, config={"tags": ["no-stream"]})
    except Exception:  # noqa: BLE001 - small models can emit unparsable output
        logging.warning("Planner structured output failed", exc_info=True)
        response = None

    if response is not None:
        # include_raw=True returns {"raw": AIMessage, "parsed": Plan|None,
        # "parsing_error": Exception|None}.
        response = cast(dict, response)
        candidate = response.get("parsed")
        if isinstance(candidate, Plan) and candidate.subtasks:
            plan = candidate
        else:
            # Some models (e.g. gpt-oss-20b on bedrock_converse) emit the plan as
            # plain text/JSON instead of a tool call, so tool-call-based structured
            # output yields parsed=None. Recover the JSON from the raw message text.
            candidate = _parse_plan_from_raw(response.get("raw"))
            if candidate is not None and candidate.subtasks:
                logging.debug("Planner recovered plan from raw message text.")
                plan = candidate

    return plan, False


async def _handle_failure_actions(
    request: str,
    previous_plan_failure_message: str,
    llm: BaseChatModel,
) -> dict | None:
    """Handle quick actions like requesting details or cancelling when a plan fails."""
    if request.lower() == REQUEST_FAILURE_DETAILS.lower():
        response = await llm.ainvoke(input=f"Requesting more details about the failure: {previous_plan_failure_message}")
        return {
            "subtasks": [],
            "results": [],
            "cancelled": False,
            "feedback": [],
            "messages": [response],
        }
    if request.lower() == CANCEL_PLAN_REQUEST.lower():
        return {
            "subtasks": [],
            "results": [],
            "cancelled": True,
            "feedback": [],
            "messages": [AIMessage(content=PLAN_CANCELLED_REPLY)],
        }
    return None


def _retry_failed_subtasks(state: PlannerState) -> Plan | None:
    """Rebuild the previous plan with its failed or cancelled subtasks reset to pending.

    Used when the user asks to retry the subtask that stopped the plan: the existing plan
    is reused as-is and only the ``failed``/``cancelled`` subtasks are set back to
    ``pending`` so execution resumes from where it stopped, instead of generating a
    brand-new plan. Returns ``None`` when there is no plan to retry.
    """
    subtasks = state.get("subtasks") or []
    if not subtasks:
        return None
    retried = [
        {**st, "status": "pending" if st.get("status") in ("failed", "cancelled") else st.get("status")}
        for st in subtasks
    ]
    return Plan(subtasks=[SubTask(**st) for st in retried])

def _retry_all_subtasks(state: PlannerState) -> Plan | None:
    """Rebuild the previous plan with all subtasks reset to pending.

    Used when the user asks to retry all subtasks: the existing plan is reused as-is
    and all subtasks are set back to ``pending`` so execution resumes from
    where it stopped, instead of generating a brand-new plan. Returns ``None`` when there
    is no plan to retry.
    """
    subtasks = state.get("subtasks") or []
    if not subtasks:
        return None
    retried = [
        {**st, "status": "pending"}
        for st in subtasks
    ]
    return Plan(subtasks=[SubTask(**st) for st in retried])


def _parse_plan_from_raw(raw: object) -> Plan | None:
    """Best-effort recovery of a Plan from a raw message when tool-calling parsing fails.

    Small models sometimes return the plan as JSON text instead of a tool call. Extract
    the first ``{...}`` JSON object from the message text and validate it against Plan.
    """
    text = _extract_text(raw)
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        logging.warning("Planner could not JSON-decode recovered snippet: %r", snippet)
        return None

    try:
        return Plan.model_validate(data)
    except Exception:  # noqa: BLE001 - validation failure just means no usable plan
        logging.warning("Planner recovered JSON did not match Plan schema: %r", data)
        return None


def _route_next(state: PlannerState) -> str:
    """Route to execute while in-progress or pending subtasks remain, otherwise reduce."""
    if state.get("cancelled"):
        return "end"
    # The child asked the user for more information: end the run and wait for the answer.
    if state.get("awaiting_input"):
        return "end"
    subtasks = state.get("subtasks", [])
    # An in-progress subtask has a child paused on an interrupt that must be resumed.
    if any(st["status"] in ("in_progress", "pending") for st in subtasks):
        return "execute"
    # A single-subtask plan is handed off directly to the child agent, so skip the
    # reducer and end with the child's answer.
    if _is_direct_handoff(subtasks):
        return "end"
    return "reduce"


def _route_after_plan(state: PlannerState) -> str:
    """Route a freshly generated plan to approval, direct execution, or end.

    An empty plan means generation failed: the failure reply is already in state, so end.
    A single-subtask plan is a plain delegation that is never confirmed with or exposed to
    the client, so it goes straight to execution. When plan approval is disabled the plan
    is executed without asking the user. Any larger plan with approval enabled must be
    approved by the user first.
    """
    subtasks = state.get("subtasks", [])
    if not subtasks:
        return "end"
    # The user answered a subtask's question: resume the plan that is already running.
    if state.get("awaiting_input"):
        return "execute"
    if _is_direct_handoff(subtasks) or not _plan_approval_enabled() or state.get("retry"):
        return "execute"
    return "approval"


def _route_after_approval(state: PlannerState) -> str:
    """Route the user's approval decision.

    A rejection cancels the run, pending feedback sends the plan back to ``plan_node`` for
    regeneration, and an approval proceeds to execution of the approved plan.
    """
    if state.get("cancelled"):
        return "end"
    if state.get("feedback"):
        return "plan"
    return "execute"


def _plan_approval_enabled() -> bool:
    """Return True when the user must approve a multi-subtask plan before it runs.

    Controlled by the ``PLAN_APPROVAL_ENABLED`` environment variable (default disabled),
    which is surfaced through the chart's ``planApproval.enabled`` value.
    """
    return os.environ.get("PLAN_APPROVAL_ENABLED", "false").lower() == "true"


def _last_user_request(state: PlannerState) -> str:
    """Return the content of the most recent human message."""
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None)
        if content and (isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human"):
            return content
    return ""


def _last_plan_failure_details(state: PlannerState) -> str | None:
    """Return the failed-plan message content if the latest request follows one.

    After a subtask fails, the planner reports a message beginning with the
    ``PLAN_FAILED_PREFIX`` marker and ends the run. If the message right before the
    user's latest request is such a message, return its content so the planner can be
    instructed to build a brand-new, complete plan instead of assuming stale progress
    from the failed attempt.
    """
    messages = state.get("messages", [])
    if len(messages) < 2:
        return None
    text = _extract_text(messages[-2])
    if text.startswith(PLAN_FAILED_PREFIX):
        return text
    return None


def _last_plan_cancellation_details(state: PlannerState) -> str | None:
    """Return the cancelled-plan message content if the latest request follows one.

    Mirrors ``_last_plan_failure_details``: after the user cancels a subtask, the planner
    reports a message beginning with the ``PLAN_CANCELLED_PREFIX`` marker and ends the run.
    If the message right before the user's latest request is such a message, return its
    content so the planner can be instructed to build a brand-new, complete plan instead of
    assuming stale progress from the cancelled attempt.
    """
    messages = state.get("messages", [])
    if len(messages) < 2:
        return None
    text = _extract_text(messages[-2])
    if text.startswith(PLAN_CANCELLED_PREFIX):
        return text
    return None
