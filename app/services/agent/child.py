"""
Agent builder for creating LangGraph agents with tool execution and human-in-the-loop validation.

Provides create_child_agent which constructs agents for use under a supervisor (multi-agent)
or standalone (single-agent) setups using the langchain create_agent factory with middleware.
Each agent has an LLM-driven reasoning loop with tool execution, human validation gates,
and automatic retry on malformed tool calls.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import Checkpointer, CompiledStateGraph

from .loader import AgentConfig
from .middleware import (
    MessagesHistoryMiddleware,
    ui_tools_middleware,
    inject_additional_kwargs_middleware,
    identity_preamble_middleware,
    human_validation_middleware,
    cancel_human_validation_middleware,
)

INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE = "tool execution cancelled because previous tool call failed"

CHILD_TOOL_USE_INSTRUCTIONS = """

## CRITICAL — TOOL EXECUTION BEHAVIOR
* You MUST call tools immediately to fulfill requests. NEVER respond with only a text message acknowledging or describing what you intend to do.
* A text-only response (without a tool call) is treated as your FINAL answer. If you respond with "I will proceed to create X" without calling a tool, the task ends incomplete.
* When asked to create, update, delete, or inspect a resource: call the appropriate tool in your FIRST response. Do NOT announce your intent — just do it.
"""


def create_child_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Checkpointer,
    agent_config: AgentConfig,
) -> CompiledStateGraph:
    """Create and compile a child agent graph using langchain create_agent with middleware.

    The agent uses the same create_agent factory as the supervisor, with middleware
    implementing: human-in-the-loop validation, metadata injection, UI tools dispatch,
    and tool execution error handling.
    """
    planning_tools = [t for t in tools if t.name.endswith("Plan")]
    execution_tools = [t for t in tools if not t.name.endswith("Plan")]
    planning_tools_by_name = {t.name: t for t in planning_tools}

    middleware = [
        MessagesHistoryMiddleware(),
        human_validation_middleware(planning_tools_by_name, agent_config),
        identity_preamble_middleware(),
        cancel_human_validation_middleware(),
        inject_additional_kwargs_middleware(),
        ui_tools_middleware(llm, only_when_direct=True),
        SummarizationMiddleware(model=llm, trigger=[("messages", 30), ("tokens", 30000)], keep=("messages", 15)),
    ]

    return create_agent(
        llm,
        tools=execution_tools,
        system_prompt=system_prompt + CHILD_TOOL_USE_INSTRUCTIONS,
        checkpointer=checkpointer,
        middleware=middleware,
    )
