
"""
Planner agent implementation.

The planner first builds a plan: a list of subtasks, each assigned to one of the
available child agents. The subtasks are then executed sequentially, each child
agent notifying when it has finished. Once every subtask is completed, a reducer
node synthesizes the individual results into a single final answer for the user.
"""

import json
import logging
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
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
    status: Literal["pending", "in_progress", "completed", "cancelled"] = Field(
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

User request:
{request}

Return a plan where each subtask has a clear, self-contained task description and the
name of the agent best suited to perform it. Keep the number of subtasks minimal.
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
        _describe_agent(child) for child in child_agents
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
        request = _last_user_request(state)
        prompt = PLANNER_PROMPT.format(agents=agents_description, request=request)
        plan = await llm.with_structured_output(Plan).ainvoke(
            prompt, config={"tags": ["no-stream"]}
        )
        assert isinstance(plan, Plan)

        subtasks = [subtask.model_dump() for subtask in plan.subtasks]
        logging.info("Planner created %d subtask(s)", len(subtasks))

        dispatch_custom_event("planner-plan-created",  f"<plan>{json.dumps(subtasks)}</plan>")
        return {"subtasks": subtasks, "results": [], "cancelled": False}

    async def execute_node(state: PlannerState) -> dict:
        """Run the next pending subtask in its assigned child agent.

        Handles human-in-the-loop interrupts: if a child agent pauses for confirmation,
        the interrupt is surfaced at the planner level and the child is resumed once the
        user responds.
        """
        subtasks = state["subtasks"]
        results = list(state.get("results", []))

        index = next(i for i, st in enumerate(subtasks) if st["status"] == "pending")
        subtask = subtasks[index]
        agent_name = subtask["agent"]
        task = subtask["task"]

        child = agents_by_name.get(agent_name)
        if child is None:
            content = f"No agent named '{agent_name}' is available to run this task."
            logging.error(content)
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
                    result = {"messages": []}

                # The user declined the confirmation: cancel the whole plan instead of
                # continuing with the remaining subtasks.
                if _is_cancelled(result):
                    logging.info("Planner subtask for agent '%s' cancelled by the user", agent_name)
                    subtasks[index]["status"] = "cancelled"
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
                dispatch_custom_event("planner-plan-created", f"<plan>{json.dumps(subtasks)}</plan>")
                try:
                    result = await child.agent.ainvoke(
                        {"messages": [HumanMessage(content=task)]},
                        config=child_config,
                    )
                except GraphBubbleUp:
                    raise
                except Exception as e:
                    logging.exception(f"Subtask agent '{agent_name}' failed: {e}")
                    result = {"messages": []}

            # ainvoke() suppresses a GraphInterrupt raised inside the child, returning
            # normally. Re-trigger any new interrupt at the planner level so the client
            # receives the confirmation prompt; the node re-runs on resume.
            child_state = await child.agent.aget_state(config=child_config)
            if child_state and child_state.interrupts:
                langgraph.types.interrupt(child_state.interrupts[0].value)

            content = _extract_last_message(result)

        subtasks[index]["status"] = "completed"
        results.append(f"Task: {task}\nAgent: {agent_name}\nResult: {content}")

        # The child agent notifies that it has finished its subtask.
        dispatch_custom_event("planner-plan-created", f"<plan>{json.dumps(subtasks)}</plan>")
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


def _route_next(state: PlannerState) -> str:
    """Route to execute while pending subtasks remain, otherwise reduce."""
    if state.get("cancelled"):
        return "end"
    if any(st["status"] == "pending" for st in state.get("subtasks", [])):
        return "execute"
    return "reduce"


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
