"""
Parent agent implementation using LangGraph Command primitive for routing to specialized child agents.

This module provides a parent agent that uses an LLM to intelligently route user requests
to the most appropriate specialized child agent based on the request content.
"""
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass

from .loader import DEFAULT_AGENT_NAME, AgentConfig
from .base import BaseAgentBuilder, AgentState, build_agent_metadata

def build_router_prompt(agent_names: list[str]) -> str:
    """Build the system router prompt with the list of valid agent names.
    
    Args:
        agent_names: List of valid agent names that can be selected
        
    Returns:
        System prompt for routing requests to child agents
    """
    agents_list = ", ".join(f"'{name}'" for name in agent_names)
    return f"""You are a routing supervisor for a multi-agent system. Your job is to analyze the user's request and select the most appropriate child agent to handle it.

INSTRUCTIONS:
1. Carefully analyze the user's request and intent
2. Match the request to the child agent whose description best aligns with the user's needs
3. Consider the full conversation context if available
4. Respond with ONLY the exact name of the selected child agent (no explanations or extra text). Valid agents are: {agents_list}
5. You MUST choose exactly one agent - never return multiple names or explanations
6. If the request is ambiguous or could match multiple agents, default to 'rancher'

"""

@dataclass
class ChildAgent:
    """
    Represents a specialized child agent that can handle specific types of requests.
    
    Attributes:
        name: Unique identifier for the child agent
        description: Description of the child agent's capabilities and use cases
        agent: The compiled LangGraph agent that handles the actual work
    """
    config: AgentConfig
    agent: CompiledStateGraph


class ParentAgentBuilder(BaseAgentBuilder):
    """
    Builder class for creating a parent agent that routes requests to specialized child agents.
    
    The parent agent uses an LLM to analyze incoming requests and route them to the
    most appropriate child agent using LangGraph's Command primitive for navigation.
    """
    old_agent_selected: str = ""
    agent_selected_count: int = 0
    
    def __init__(self, llm: BaseChatModel, child_agents: list[ChildAgent], checkpointer: Checkpointer):
        """
        Initialize the parent agent builder.
        
        Args:
            llm: Language model used for routing decisions
            child_agents: List of available specialized child agents
            checkpointer: Checkpointer for persisting agent state
        """
        super().__init__(llm=llm, tools=[], system_prompt="", checkpointer=checkpointer, agent_config=None)
        self.child_agents = child_agents
    
    def choose_child_agent(self, state: AgentState, config: RunnableConfig):
        """
        Route the user request to the most appropriate child agent.
        
        Uses the LLM to analyze the user's request and select which child agent
        should handle it based on the child agent descriptions.
        
        Args:
            state: Current agent state containing messages
            config: Runtime configuration
            
        Returns:
            Command object directing workflow to the selected child agent
        """

        # UI override to force a specific child agent
        agent_override = config.get("configurable", {}).get("agent", "")
        if agent_override:
            self.old_agent_selected = ""
            self.agent_selected_count = 0
            self.agent_selected = agent_override

            dispatch_custom_event(
                "subagent_choice_event",
                build_agent_metadata(agent_override, "manual"),
            )

            return Command(
                goto=agent_override,
                update={
                    "selected_agent": {
                        "name": agent_override,
                        "mode": "manual"
                    }
                }
            )

        messages = state["messages"]

        # Build routing prompt with available child agents and their descriptions
        agent_names = [child.config.name for child in self.child_agents]
        router_prompt = build_router_prompt(agent_names) + "AVAILABLE CHILD AGENTS:\n"
        for child in self.child_agents:
            router_prompt += f"- {child.config.name}: {child.config.description}\n"
        router_prompt += f"\nUSER'S REQUEST: {messages[-1].content}"
        
        user_and_ai_messages = [msg for msg in self._get_messages_from_last_summary(state) if isinstance(msg, (HumanMessage, AIMessage, SystemMessage))]

        # Use LLM to select the appropriate child agent
        child_agent = self.llm.invoke([SystemMessage(content=router_prompt)] + user_and_ai_messages).text.strip()
        if child_agent not in [child.config.name for child in self.child_agents]:
            # Fallback to default agent if the agent selection from LLM is invalid
            logging.warning(f"LLM selected invalid agent '{child_agent}', defaulting to '{DEFAULT_AGENT_NAME}'")
            child_agent = DEFAULT_AGENT_NAME

        self.agent_selected = child_agent

        if child_agent == self.old_agent_selected:
            self.agent_selected_count += 1
        else:
            self.old_agent_selected = child_agent
            self.agent_selected_count = 1

        recommended_field = f', "recommended": "{child_agent}"' if self.agent_selected_count >= 3 else ""
        dispatch_custom_event(
            "subagent_choice_event",
            build_agent_metadata(child_agent, "auto", recommended_field),
        )

        # Return Command to navigate to the selected child agent
        return Command(
            goto=child_agent,
            update={
                "selected_agent": {
                    "name": child_agent,
                    "mode": "auto"
                }
            }
        )

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the parent agent workflow graph.
        
        Creates a LangGraph workflow with a routing node and nodes for each child agent.
        The workflow starts with routing and navigates to the appropriate child agent.
        
        Returns:
            Compiled state graph ready for execution
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("choose_child_agent", self.choose_child_agent)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)

        # Add a node for each child agent
        for child in self.child_agents:
            workflow.add_node(child.config.name, child.agent)
            workflow.add_conditional_edges(
                child.config.name,
                self.should_summarize_conversation,
                {
                    "summarize_conversation": "summarize_conversation",
                    "end": END,
                },
            )

        
        # Set the routing node as the entry point
        workflow.set_entry_point("choose_child_agent")

        return workflow.compile(checkpointer=self.checkpointer)

def create_parent_agent(llm: BaseChatModel, child_agents: list[ChildAgent], checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Factory function to create a parent agent with routing capabilities.
    
    Args:
        llm: Language model for routing decisions
        child_agents: List of specialized child agents to route between
        checkpointer: Checkpointer for state persistence
        
    Returns:
        Compiled parent agent ready to route requests to child agents
    """
    builder = ParentAgentBuilder(llm, child_agents, checkpointer)

    return builder.build()
