"""
Parent agent implementation using deepagents for intelligent routing to specialized child agents.

This module provides a parent agent that uses create_deep_agent to route user requests
to the most appropriate specialized child agent based on the request content.
The agent always creates a plan using submit_plan before execution, and the plan
is presented to the user for approval via a LangGraph interrupt.
"""
import json
import logging
from typing import Annotated

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import InterruptOnConfig
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in a plan."""
    title: str = Field(description="Short description of the step")
    description: str = Field(default="", description="Optional details about what this step involves")


class Plan(BaseModel):
    """A plan that must be approved by the user before execution."""
    goal: str = Field(description="The overall goal of this plan")
    steps: list[PlanStep] = Field(description="The ordered list of steps to execute")


PLANNING_SYSTEM_PROMPT = """## MANDATORY Planning Requirement — READ THIS FIRST

You have a `submit_plan` tool. You MUST call it before doing ANY work — no exceptions.

**EVERY user request requires a plan, no matter how simple.** Even a single-step task
like "list pods" must have a plan submitted and approved before you proceed.

Your workflow for EVERY request:
1. Read the user's request
2. Immediately call `submit_plan` with a goal and steps
3. STOP and wait — the user will approve or reject
4. Only after approval, execute the steps (delegate to subagents, call tools, etc.)

NEVER skip submit_plan. NEVER call a subagent or any other tool before submit_plan.
NEVER start working before the plan is approved. This is a hard requirement.
"""


def _format_plan_description(tool_call, state, runtime):
    """Format the submit_plan interrupt description as a plan approval request.

    This callable is used by HumanInTheLoopMiddleware to generate a
    human-readable description of the plan for the client. The output
    is wrapped in a <plan-approval> tag so the client can detect it
    and render an approval UI.

    Args:
        tool_call: The ToolCall dict containing name, args, id.
        state: The current agent state.
        runtime: The agent runtime context.

    Returns:
        A string wrapped in <plan-approval> tags containing the plan JSON.
    """
    args = tool_call.get("args", {})
    plan_data = json.dumps(args)
    return f"<plan-approval>{plan_data}</plan-approval>"


@tool(response_format="content")
def submit_plan(
    goal: str,
    steps: list[PlanStep],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Submit a plan for user approval. You MUST call this tool before doing any work.

    Every user request — no matter how simple — requires a plan. Call this tool
    with a goal and ordered steps, then wait for user approval before proceeding.

    Args:
        goal: The overall goal of this plan.
        steps: The ordered list of steps to execute.
    """
    steps_summary = ", ".join(s.title for s in steps)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Plan APPROVED by the user. Goal: {goal}, Steps: [{steps_summary}]. "
                        "Proceed immediately with execution — delegate to the appropriate subagent now."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


def create_parent_agent(
    llm: BaseChatModel,
    subagents: list[SubAgent],
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """
    Factory function to create a parent agent with routing capabilities using deepagents.

    Uses create_deep_agent to build a multi-agent system where the parent agent
    automatically routes user requests to the most appropriate specialized subagent.
    The agent is configured to always create a plan (via submit_plan) and interrupt
    for user approval before proceeding with execution.

    Args:
        llm: Language model for routing decisions and agent reasoning
        subagents: List of SubAgent specifications defining child agents
        checkpointer: Checkpointer for state persistence

    Returns:
        Compiled parent agent ready to route requests to child agents
    """
    agent_names = [s["name"] for s in subagents]
    logging.info(f"Creating deep agent with {len(subagents)} subagent(s): {agent_names}")

    return create_deep_agent(
        model=llm,
        subagents=subagents,
        checkpointer=checkpointer,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        tools=[submit_plan],
        interrupt_on={
            "submit_plan": InterruptOnConfig(
                allowed_decisions=["approve", "reject"],
                description=_format_plan_description,
            ),
        },
    )
