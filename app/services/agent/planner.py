
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.callbacks.manager import dispatch_custom_event
import langgraph.types
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, Checkpointer

from .supervisor import ChildAgent, _extract_last_message
from ._constants import INTERRUPT_CANCEL_MESSAGE
from ...constants import INTERRUPT_CANCEL_REPLY
from .middleware import (
    MessagesHistoryMiddleware,
    inject_additional_kwargs_middleware,
    ui_tools_middleware,
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
    subtasks: list[dict]
    results: list[str]
    cancelled: bool
    feedback: list[str]
    retry: bool


PLANNER_PROMPT = """\
You are a planning agent. Break the user's request into an ordered list of subtasks.
Assign each subtask to exactly one of the available agents (use the agent name).

Available agents:
{agents}

Return a plan where each subtask has a clear, self-contained task description and the
name of the agent best suited to perform it.

Rules you MUST follow:
- Always return at least one subtask. If the request is simple, return exactly one
  subtask that covers the whole request.
- If the request is complex, break it down into as many subtasks as needed so each one
  covers a single, self-contained piece of work. Do not bundle unrelated or multi-step
  work into a single subtask when it can reasonably be split,
- The "agent" field of every subtask MUST be one of the agent names listed above,
  copied exactly (case-sensitive). Do not invent new agent names.
- Respond with a single, valid JSON object only. Do not add explanations, comments,
  markdown code fences, or any text before or after the JSON.
"""

REDUCER_PROMPT = """\
You are summarizing the outcome of a multi-step plan for the user.

Original request:
{request}

Results of each subtask:
{results}

Combine these into a single, coherent final answer for the user.
"""

REDUCER_SYSTEM_PROMPT = """\
You are the reducer of a planner agent. You are given the original user request and the
results of each subtask that was executed. Combine them into a single, coherent final
answer for the user.
"""

SUBTASK_EVALUATION_SYSTEM_PROMPT = """\
You are the evaluator of a planner agent. You are given a subtask that was assigned to a
child agent and the response that agent produced. Judge whether the agent actually
completed the task. Answer with a single word: "yes" if it completed the task, or "no" if
it did not. Do not add any other text.
"""

SUBTASK_EVALUATION_PROMPT = """\
Did the agent complete the assigned subtask?

Subtask:
{task}

Agent's response:
{response}

The task was NOT completed if the agent reports it is missing information, hit an error,
lacks the capability or permissions, refused, or otherwise did not accomplish what was
asked. If the agent accomplished the task and produced a useful result, it was completed.

Answer with a single word: "yes" if it completed the task, or "no" if it did not.
"""

PLANNER_FEEDBACK_SUFFIX = """\

The user reviewed previous versions of this plan and requested changes. Produce a new
plan that takes ALL of the following feedback into account, in order (later feedback
refines or overrides earlier feedback):
{feedback}
"""

PLANNER_RETRY_SUFFIX = """\

---
The previous plan attempt failed and execution was stopped early:
{details}

Treat the new user message above as the user's chosen way to recover, and produce a plan
accordingly:
- If the user wants to retry the step that failed, produce a plan that reattempts that
  failed step (incorporating any new or corrected information the user provided), followed
  by the remaining steps needed to satisfy the original request.
- If the user wants to start from the beginning, produce a plan that re-runs the ENTIRE
  original request from the first step, without assuming any step of the previous attempt
  is still valid.
- If the user wants to create a new plan, treat their message as new or corrected
  information and produce a brand-new, complete plan that covers the entire original
  request.
"""

PLAN_FAILED_PREFIX = "PLAN FAILED:"

# User-facing option, sent back as the request, to re-run the failed subtask using the
# existing plan instead of asking the LLM to generate a brand-new one.
RETRY_SUBTASK_REQUEST = "Retry executing the failed subtask"

# User-facing option, sent back as the request, to restart the entire plan from its
# first subtask using the existing plan instead of asking the LLM to generate a
# brand-new one.
RESTART_PLAN_REQUEST = "Restart the execution of the entire plan"

# User-facing option, sent back as the request, to ask for more details about why the subtask failed.
REQUEST_FAILURE_DETAILS = "Request more details about the failure"

# User-facing option, sent back as the request, to cancel the current plan.
CANCEL_PLAN_REQUEST = "Cancel the plan"

PLAN_CANCELLED_REPLY = (
    "The current plan has been cancelled as per your request."
)
PLAN_FAILED_REPLY = (
    "There was a problem generating a plan for your request. Please try again with a different prompt."
)

PLAN_SUBTASK_FAILED_REPLY = (
    PLAN_FAILED_PREFIX + ' I couldn\'t complete the step "{task}": {error}\n\n'
    "Here is the plan that was being executed:\n{plan}\n\n"
    "I've stopped the plan here instead of continuing with the remaining steps, so the "
    "results stay consistent. Please provide any missing information or adjust your "
    "request, and I'll create a new plan."
)

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
        previous_plan_failure_message = _last_plan_failure_details(state)

        if previous_plan_failure_message:
            failure_action_result = await _handle_failure_actions(request, previous_plan_failure_message, llm)
            if failure_action_result is not None:
                return failure_action_result

        plan, retry = await _create_plan(llm, agents_description, state, feedback, previous_plan_failure_message)
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
        return {"subtasks": subtasks, "results": [], "cancelled": False, "feedback": feedback, "retry": retry}

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

        # A single-subtask plan is handed off directly to the child agent: run it with no
        # plan-progress events and return its answer as-is; _route_next then sends the
        # result straight to END, skipping the reducer.
        if _is_direct_handoff(subtasks):
            outcome = await _run_pending_subtask(llm, agents_by_name, subtasks, results, emit_plan=False)
            if isinstance(outcome, dict):
                return outcome
            return {
                "subtasks": subtasks,
                "results": results,
                "messages": [AIMessage(content=outcome)],
            }

        outcome = await _run_pending_subtask(llm, agents_by_name, subtasks, results, emit_plan=True)
        if isinstance(outcome, dict):
            return outcome

        # The child agent notifies that it has finished its subtask.
        dispatch_custom_event(
            "planner-plan-finished",
            f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
        )
        return {"subtasks": subtasks, "results": results}

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
        return {"messages": [result["messages"][-1]], "subtasks":[], "results": []}

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


async def _run_pending_subtask(
    llm: BaseChatModel,
    agents_by_name: dict[str, ChildAgent],
    subtasks: list[dict],
    results: list[str],
    emit_plan: bool,
) -> dict | str:
    """Run the next pending subtask in its assigned child agent.

    Handles human-in-the-loop interrupts: if a child agent pauses for confirmation
    or to ask the user for more data, the interrupt is surfaced at the planner level
    and the child is resumed once the user responds. On completion the subtask is
    marked completed and its result is appended to ``results``. When ``emit_plan`` is
    True, plan-progress events are dispatched to the client.

    If the child agent raises an error the plan is stopped rather than silently
    continuing with the remaining subtasks: the failure is reported to the user via
    ``_fail_plan``.

    Returns a state-update dict when the user cancels the plan or a subtask fails,
    otherwise the child agent's final message content.
    """
    index = next(i for i, st in enumerate(subtasks) if st["status"] == "pending")
    subtask = subtasks[index]
    agent_name = subtask["agent"]
    task = subtask["task"]

    child = agents_by_name.get(agent_name)
    if child is None:
        reason = f"No agent named '{agent_name}' is available to run this task."
        logging.error(reason)
        return _fail_plan(subtasks, results, index, task, reason, emit_plan)
    else:
        child_config = _build_child_config(agent_name)
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
                subtasks[index]["status"] = "cancelled"
                if emit_plan:
                    dispatch_custom_event(
                        "planner-plan-created",
                        f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
                    )
                return {
                    "subtasks": subtasks,
                    "results": results,
                    "cancelled": True,
                    "messages": [AIMessage(content=INTERRUPT_CANCEL_REPLY)],
                }
        else:
            subtasks[index]["status"] = "in_progress"
            if emit_plan:
                dispatch_custom_event(
                    "planner-plan-created",
                    f"<plan>{json.dumps({'tasks': subtasks, 'approval': False})}</plan>",
                )
            try:
                result = await child.agent.ainvoke(
                    {"messages": [HumanMessage(content=_build_task_message(task, results))]},
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
        # normally. Re-trigger any new interrupt at the planner level so the client
        # receives the confirmation prompt; the node re-runs on resume.
        child_state = await child.agent.aget_state(config=child_config)
        if child_state and child_state.interrupts:
            langgraph.types.interrupt(child_state.interrupts[0].value)

        content = _extract_last_message(result)

    # The child returned without raising, but it may not have actually completed the
    # task (missing information, an error, a refusal, ...). Ask the LLM to judge the
    # child's response so the plan is stopped instead of marking the subtask completed
    # and continuing with the remaining subtasks.
    if not await _evaluate_subtask(llm, task, content):
        logging.debug("Planner subtask for agent '%s' evaluated as failed", agent_name)
        reason = "The agent did not complete the task."
        return _fail_plan(subtasks, results, index, task, reason, emit_plan)

    subtasks[index]["status"] = "completed"
    results.append(f"Task: {task}\nAgent: {agent_name}\nResult: {content}")
    return content


async def _evaluate_subtask(llm: BaseChatModel, task: str, content: str | list) -> bool:
    """Return True if the child agent completed the subtask, using the LLM.

    The child's reply is streamed to the client, so a marker-based signal is not
    reliably detectable. Instead, run a separate, non-streamed LLM call that receives
    the subtask and the child's response and answers "yes" or "no".
    """
    response_text = _extract_text(content).strip()
    if not response_text:
        return False

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
        return True

    answer = _extract_text(response).strip().lower()
    return not answer.startswith("no")


async def _create_plan(
    llm: BaseChatModel,
    agents_description: str,
    state: PlannerState,
    feedback: list[str] | None = None,
    previous_plan_failure_message: str | None = None,
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
    if previous_plan_failure_message:
        normalized_request = request.lower()
        if normalized_request == RETRY_SUBTASK_REQUEST.lower():
            # Re-run the plan that just failed: keep the existing subtasks and only
            # reset the failed ones to pending so execution resumes from where it
            # stopped, instead of asking the LLM for a brand-new plan.
            return _retry_failed_subtasks(state), True
        if normalized_request == RESTART_PLAN_REQUEST.lower():
            # Restart the whole plan from the beginning: reset every subtask to
            # pending instead of asking the LLM for a brand-new plan.
            return _retry_all_subtasks(state), True

    if previous_plan_failure_message or feedback:
        # Label the request so it is not confused with the appended failure/feedback
        # context that follows it.
        human_content = f"New user message:\n{request}"
        if previous_plan_failure_message:
            human_content += PLANNER_RETRY_SUFFIX.format(details=previous_plan_failure_message)
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


def _retry_failed_subtasks(state: PlannerState) -> Plan | None:
    """Rebuild the previous plan with its failed subtasks reset to pending.

    Used when the user asks to retry the failed subtask: the existing plan is reused as-is
    and only the ``failed`` subtasks are set back to ``pending`` so execution resumes from
    where it stopped, instead of generating a brand-new plan. Returns ``None`` when there
    is no plan to retry.
    """
    subtasks = state.get("subtasks") or []
    if not subtasks:
        return None
    retried = [
        {**st, "status": "pending" if st.get("status") == "failed" else st.get("status")}
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
    """Route to execute while pending subtasks remain, otherwise reduce."""
    if state.get("cancelled"):
        return "end"
    subtasks = state.get("subtasks", [])
    if any(st["status"] == "pending" for st in subtasks):
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


def _is_direct_handoff(subtasks: list) -> bool:
    """Return True when the plan is a single subtask delegated straight to its agent.

    A one-subtask plan is treated as a plain delegation rather than an orchestrated plan:
    it is never confirmed with or exposed to the client, emits no plan-progress events,
    and its child agent's answer is returned as-is without going through the reducer.
    Centralizing this check keeps the plan, execute, and routing stages in agreement on
    what counts as a direct hand-off.
    """
    return len(subtasks) == 1


def _plan_approval_enabled() -> bool:
    """Return True when the user must approve a multi-subtask plan before it runs.

    Controlled by the ``PLAN_APPROVAL_ENABLED`` environment variable (default disabled),
    which is surfaced through the chart's ``planApproval.enabled`` value.
    """
    return os.environ.get("PLAN_APPROVAL_ENABLED", "false").lower() == "true"


def _is_cancelled(result: dict) -> bool:
    """Return True if the last ToolMessage in a child agent result is the cancellation marker."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg.content == INTERRUPT_CANCEL_MESSAGE
    return False


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


def _build_child_config(agent_name: str) -> RunnableConfig:
    """Build a namespaced run-config so each child agent checkpoints independently."""
    parent_configurable = ensure_config().get("configurable", {})
    parent_thread_id = parent_configurable.get("thread_id", "")
    if not parent_thread_id:
        raise ValueError("thread_id is required in configurable but was not provided")

    child_configurable = {
        "thread_id": f"{parent_thread_id}::planner::{agent_name}",
    }
    return RunnableConfig(configurable=child_configurable, callbacks=[])
