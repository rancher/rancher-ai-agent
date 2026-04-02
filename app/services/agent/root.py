"""
Root agent implementation for independent agents that run without a parent.
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from .loader import AgentConfig
from .base import BaseAgentBuilder, AgentState


class RootAgentBuilder(BaseAgentBuilder):
    """Builder for root-level agents that run independently without a parent."""

    def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the LangGraph root agent.
        
        Workflow paths:
        - With tool_calls: agent -> tools (loop) -> agent
        - No tool_calls: agent -> ui_tools -> should_summarize_conversation
          - If should summarize: -> summarize_conversation -> END
          - If not: -> END
        - Tool rejection: tools -> END (skip ui_tools)
        
        Key: should_summarize_conversation is called after ui_tools

        Returns:
            A compiled LangGraph StateGraph ready to be invoked.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("ui_tools", self.ui_tools_node)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)
        
        # Agent stage: only check for tool calls, route to ui_tools on completion
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": "ui_tools",  # No tool calls -> go to ui_tools
            },
        )
        
        workflow.add_conditional_edges(
            "tools",
            self.should_continue_after_interrupt,
            {
                "continue": "agent",
                # Rejected human validation: END immediately, skip ui_tools
                "end": END,
            },
        )
        
        # After ui_tools dispatch, check if summarization is needed
        workflow.add_conditional_edges(
            "ui_tools",
            self.should_summarize_conversation,
            {
                "summarize_conversation": "summarize_conversation",
                "end": END,
            },
        )
        
        workflow.add_edge("summarize_conversation", END)
        workflow.set_entry_point("agent")

        return workflow.compile(checkpointer=self.checkpointer)


def create_root_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer, agent_config: AgentConfig) -> CompiledStateGraph:
    """
    Creates a LangGraph root agent capable of interacting with Rancher and Kubernetes resources.
    
    This factory function instantiates the RootAgentBuilder, builds the agent graph,
    and returns the compiled agent.
    
    Args:
        llm: The language model to use for the agent's decisions.
        tools: A list of tools the agent can use (e.g., to interact with K8s).
        system_prompt: The initial system-level instructions for the agent.
        checkpointer: The checkpointer for persisting agent state.
    
    Returns:
        A compiled LangGraph StateGraph ready to be invoked.
    """
    builder = RootAgentBuilder(llm, tools, system_prompt, checkpointer, agent_config)

    return builder.build()
