
"""
Planner agent implementation.

The planner first builds a plan: a list of subtasks, each assigned to one of the
available child agents. The subtasks are then executed sequentially, each child
agent notifying when it has finished. Once every subtask is completed, a reducer
node synthesizes the individual results into a single final answer for the user.
"""

import json
import logging
from typing import Annotated, Literal, TypedDict, cast

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.callbacks.manager import dispatch_custom_event
import langgraph.types
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, Checkpointer

from .supervisor import ChildAgent, _extract_last_message
from ._constants import INTERRUPT_CANCEL_MESSAGE, SUBTASK_FAILED_MARKER
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
  work into a single subtask when it can reasonably be split, even if multiple
  subtasks end up assigned to the same agent.
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

Treat the new user message above as new or corrected information. Create a brand-new,
complete plan that covers the ENTIRE original request from the beginning, with every
step needed to fully satisfy it. Do not resume from where the previous attempt stopped
and do not assume any of its steps are still valid.
"""

PLAN_FAILED_PREFIX = "PLAN FAILED:"

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
        dispatch_custom_event("planner-plan-created", f"<plan>{json.dumps(subtasks)}</plan>")
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
        """Generate the list of subtasks from the user's request."""
        needs_confirmation = True #TODO replace with env var
        feedback: list[str] = []

        while True:
            plan = await _create_plan(state, feedback)
            if plan is None or plan.subtasks is None or not plan.subtasks:
                logging.error("Planner failed to produce a valid plan.")
                return {
                    "subtasks": [],
                    "results": [],
                    "cancelled": False,
                    "messages": [AIMessage(content=PLAN_FAILED_REPLY)],
                }
            subtasks = [subtask.model_dump() for subtask in plan.subtasks]
            
            # With a single subtask, hand off directly to the subagent without exposing the
            # plan to the client.
            if len(plan.subtasks) == 1:
                break

            if not needs_confirmation:
                break

            response = langgraph.types.interrupt(
                f"<plan-approval>{json.dumps([st.model_dump() for st in plan.subtasks])}</plan-approval>"
            )
            normalized = response.strip().lower() if isinstance(response, str) else response

            if normalized == "yes":
                break

            if normalized == "no":
                logging.debug("Planner plan was rejected by the user.")
                return {
                    "subtasks": [],
                    "results": [],
                    "cancelled": True,
                    "messages": [AIMessage(content="Plan was not approved by the user.")],
                }

            # Any other response is treated as feedback: accumulate it and regenerate
            # the plan so successive rounds of feedback all apply, then ask the user to
            # review the revised version.
            logging.debug("Planner regenerating plan from user feedback.")
            feedback.append(response)

            # TODO do we need this? dispatch_custom_event("planner-plan-created",  f"<plan>{json.dumps(subtasks)}</plan>")
        
        return {"subtasks": subtasks, "results": [], "cancelled": False}

    async def _run_pending_subtask(
        subtasks: list[dict], results: list[str], emit_plan: bool
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
                            "planner-plan-created", f"<plan>{json.dumps(subtasks)}</plan>"
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
                    dispatch_custom_event("planner-plan-created", f"<plan>{json.dumps(subtasks)}</plan>")
                try:
                    result = await child.agent.ainvoke(
                        {"messages": [HumanMessage(content=_build_task_message(task, results))]},
                        config=child_config,
                    )
                except GraphBubbleUp:
                    raise
                except Exception as e:
                    logging.exception(f"Subtask agent '{agent_name}' failed: {e}")
                    return _fail_plan(subtasks, results, index, task, e, emit_plan)

            # ainvoke() suppresses a GraphInterrupt raised inside the child, returning
            # normally. Re-trigger any new interrupt at the planner level so the client
            # receives the confirmation prompt; the node re-runs on resume.
            child_state = await child.agent.aget_state(config=child_config)
            if child_state and child_state.interrupts:
                langgraph.types.interrupt(child_state.interrupts[0].value)

            content = _extract_last_message(result)

        # The child returned without raising, but it may have signalled that it could not
        # complete the task via the failure marker. Stop the plan instead of marking the
        # subtask completed and continuing with the remaining subtasks.
        reason = _subtask_failure_reason(content)
        if reason is not None:
            logging.debug("Planner subtask for agent '%s' reported failure: %s", agent_name, reason)
            return _fail_plan(subtasks, results, index, task, reason, emit_plan)

        subtasks[index]["status"] = "completed"
        results.append(f"Task: {task}\nAgent: {agent_name}\nResult: {content}")
        return content

    async def _create_plan(state: PlannerState, feedback: list[str] | None = None) -> Plan | None:
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
        retry_details = _last_plan_failure_details(state)
        if retry_details or feedback:
            # Label the request so it is not confused with the appended failure/feedback
            # context that follows it.
            human_content = f"New user message:\n{request}"
            if retry_details:
                human_content += PLANNER_RETRY_SUFFIX.format(details=retry_details)
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
        
        return plan

    async def execute_node(state: PlannerState) -> dict:
        """Run the next pending subtask in its assigned child agent."""
        subtasks = state["subtasks"]
        results = list(state.get("results", []))

        single_subtask = len(subtasks) == 1

        # A single-subtask plan is handed off directly to the subagent without exposing
        # the plan to the client: execute it and exit here, returning its answer as-is;
        # _route_next sends it to END, skipping the reducer.
        if single_subtask:
            outcome = await _run_pending_subtask(subtasks, results, emit_plan=False)
            if isinstance(outcome, dict):
                return outcome
            return {
                "subtasks": subtasks,
                "results": results,
                "messages": [AIMessage(content=outcome)],
            }

        outcome = await _run_pending_subtask(subtasks, results, emit_plan=True)
        if isinstance(outcome, dict):
            return outcome

        # The child agent notifies that it has finished its subtask.
        dispatch_custom_event("planner-plan-finished", f"<plan>{json.dumps(subtasks)}</plan>")
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
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reduce", reduce_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_next, {"execute": "execute", "reduce": "reduce", "end": END})
    graph.add_conditional_edges("execute", _route_next, {"execute": "execute", "reduce": "reduce", "end": END})
    graph.add_edge("reduce", END)

    return graph.compile(checkpointer=checkpointer)


_SUBTASK_FAILURE_INSTRUCTION = (
    f"\n\nIf you cannot complete this task (missing information, an error, or you lack the "
    f"capability), respond with a message that begins with '{SUBTASK_FAILED_MARKER}:' "
    "followed by a short reason. Otherwise, complete the task normally and do not use that "
    "prefix."
)


def _build_task_message(task: str, previous_results: list[str]) -> str:
    """Build the message sent to a child agent, including prior subtask outcomes.

    A subtask may depend on the results of the subtasks that ran before it, so the
    accumulated results are prepended as context ahead of the current task. The child is
    also instructed to signal failure with the ``SUBTASK_FAILED`` marker so the planner
    can stop the plan instead of continuing on an unsuccessful step.
    """
    if not previous_results:
        return task + _SUBTASK_FAILURE_INSTRUCTION
    joined = "\n\n".join(previous_results)
    return (
        "Results of the previous subtasks in the plan (use them as needed to complete "
        f"your task):\n{joined}\n\nYour task:\n{task}" + _SUBTASK_FAILURE_INSTRUCTION
    )


def _format_plan(subtasks: list[dict]) -> str:
    """Render the plan's subtasks as JSON, matching the ``SubTask`` schema."""
    return json.dumps(subtasks)


#TODO remove?
def _describe_agent(child: ChildAgent) -> str:
    """Render an agent's name, description, and available tools for the planner prompt."""
    description = child.config.description or "Specialized agent"
    lines = [f"- {child.config.name}: {description}"]
    if child.tools:
        tool_lines = "\n".join(
            f"    - {tool.name}: {tool.description or 'No description'}"
            for tool in child.tools
        )
        lines.append(f"  Tools:\n{tool_lines}")
    return "\n".join(lines)


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
    # A single-subtask plan is handed off directly to the subagent, so skip the reducer
    # and end with the child's answer.
    if len(subtasks) <= 1:
        return "end"
    return "reduce"


def _subtask_failure_reason(content: str | list) -> str | None:
    """Return the failure reason if the child signalled failure, else None.

    Child agents are instructed to begin their reply with the ``SUBTASK_FAILED:`` marker
    when they cannot complete a task. A prefix match on the stripped content avoids false
    positives from the marker merely appearing mid-text.

    Message content may be a plain string or a list of typed blocks (e.g. ``text`` and
    ``reasoning_content``); ``_extract_text`` normalizes both to the concatenated text,
    dropping reasoning so the marker is detected against the visible reply.
    """
    stripped = _extract_text(content).strip()
    prefix = f"{SUBTASK_FAILED_MARKER}:"
    if stripped.startswith(prefix):
        return stripped[len(prefix):].strip() or "The agent reported it could not complete the task."
    return None


def _is_cancelled(result: dict) -> bool:
    """Return True if a child agent result contains the user-cancellation marker."""
    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "content", None) == INTERRUPT_CANCEL_MESSAGE:
            return True
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
