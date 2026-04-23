"""
Parent agent implementation using deepagents for intelligent routing to specialized child agents.

This module provides a parent agent that uses create_deep_agent to route user requests
to the most appropriate specialized child agent based on the request content.
"""
import logging

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.language_models.chat_models import BaseChatModel


def create_parent_agent(
    llm: BaseChatModel,
    subagents: list[SubAgent],
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """
    Factory function to create a parent agent with routing capabilities using deepagents.

    Uses create_deep_agent to build a multi-agent system where the parent agent
    automatically routes user requests to the most appropriate specialized subagent.

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
    )
