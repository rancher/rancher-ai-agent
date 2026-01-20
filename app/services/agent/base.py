"""
Base agent builder with shared logic for all agent types.
"""

import json
import logging
import langgraph.types

from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph.state import Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from langchain_core.callbacks.manager import dispatch_custom_event
from .loader import AgentConfig, HumanValidationTool
from .types import AgentState

INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"

class BaseAgentBuilder:
    """Base class for agent builders with shared logic."""
    
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer, agent_config: AgentConfig):
        """
        Initializes the BaseAgentBuilder.

        Args:
            llm: The language model to use for the agent's decisions.
            tools: A list of tools the agent can use.
            system_prompt: The initial system-level instructions for the agent.
            checkpointer: The checkpointer for persisting agent state.
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.agent_config = agent_config
    
    def summarize_conversation_node(self, state: AgentState):
        """
        Summarizes the conversation history.

        This node is invoked when the conversation becomes too long. It asks the LLM
        to create or extend a summary of the conversation, then replaces the
        previous messages with the new summary to keep the context concise.

        Args:
            state: The current state of the agent, containing messages and an optional summary.

        Returns:
            A dictionary with the updated summary and a condensed list of messages."""
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n"
                "Extend the summary by taking into account the new messages above:" )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm.invoke(messages)

        # Mark this response explicitly as a summary so that it can be filtered
        # out from the memory endpoints results.
        # Summary are used to condense the conversation but should not appear
        # as part of the conversation history.
        if response.additional_kwargs is None:
            response.additional_kwargs = {}
        response.additional_kwargs["is_summary"] = True

        new_messages = [RemoveMessage(id=m.id) for m in messages[:-1]]
        new_messages = new_messages + [response]

        logging.debug("summarizing conversation")
        
        return {"summary": response.content, "messages": new_messages}

    def _invoke_llm_with_retry(self, messages: list, config: RunnableConfig):
        """
        Invokes the LLM, with a single retry for a tool call parsing error.
        Models can sometimes hallucinate and produce malformed JSON for tool calls.
        This method attempts the LLM call and retries if it fails with a specific
        "error parsing tool" """
        # 1 initial attempt + 1 retry
        for attempt in range(2):
            try:
                return self.llm_with_tools.invoke(messages, config)
            except ResponseError as e:
                if "error parsing tool call:" in str(e.error) and attempt < 1:
                    logging.warning(f"retrying due to tool call parsing error: {e.error}")
                    continue
                raise e

    def call_model_node(self, state: AgentState, config: RunnableConfig):
        """
        Invokes the language model with the current state and context.

        This node prepares the messages for the LLM, including the system prompt,
        any contextual information (like current cluster), and the conversation history.
        It then calls the LLM to get the next response.

        Args:
            state: The current state of the agent.
            config: The runnable configuration.

        Returns:
            A dictionary containing the LLM's response message."""
        
        logging.debug("calling model")
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self._invoke_llm_with_retry(messages, config)
        
        response.additional_kwargs["request_id"] = config["configurable"]["request_id"]

        logging.debug("model call finished")

        return {"messages": [response]}

    async def tool_node(self, state: AgentState, config: RunnableConfig):
        """
        Executes tools based on the LLM's request.

        This node processes tool calls from the last message, handling user
        confirmation for sensitive operations. It invokes the appropriate tool
        and returns the results as ToolMessage objects.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing a list of ToolMessage objects with the tool results,
            or an error message if a tool fails or is cancelled."""
        outputs = []
        
        request_id = config["configurable"]["request_id"]

        for tool_call in getattr(state["messages"][-1], "tool_calls", []):
            should_continue, interrupt_message = handle_interrupt(getattr(self.agent_config, "human_validation_tools", []), tool_call)

            additional_kwargs = { "request_id": request_id }

            if interrupt_message:
                additional_kwargs["interrupt_message"] = interrupt_message
                additional_kwargs["confirmation"] = True

            if not should_continue:
                additional_kwargs["confirmation"] = False
                return {
                    "messages": [ToolMessage(
                        content=INTERRUPT_CANCEL_MESSAGE,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs=additional_kwargs
                    )]
                }
            
            try:
                logging.debug("calling tool")
                tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                logging.debug("tool call finished")

                processed_result, mcp_response = process_tool_result(tool_result, state)

                if mcp_response:
                    additional_kwargs["mcp_response"] = mcp_response

                outputs.append(
                    ToolMessage(
                        content=processed_result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs=additional_kwargs
                    )
                )
            except ToolException as e:
                return {
                    "messages": [ToolMessage(
                        content=str(e),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs=additional_kwargs
                    )]
                }
            except Exception as e:
                logging.error(f"unexpected error during tool call: {e}")
                return {
                    "messages": [ToolMessage(
                        content=f"unexpected error during tool call: {e}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        additional_kwargs=additional_kwargs
                    )]
                }

        return {"messages": outputs}
    
    def should_summarize_conversation(self, state: AgentState):
        """
        Determines the next step in the agent's workflow.

        This conditional edge checks the last message in the state to decide whether to
        continue with a tool call, summarize the conversation, or end the execution.

        Args:
            state: The current state of the agent.

        Returns:
            A string indicating the next node to transition to: "continue",
            "summarize_conversation", or "end"."""
        messages = state["messages"]
        last_message = messages[-1]
        if not getattr(last_message, "tool_calls", []):
            if len(messages) > 7:
                return "summarize_conversation"
            return "end"
        else:
            return "continue"
        
    def should_continue(self, state: AgentState):
        """Check if agent should continue based on tool calls."""
        last_message = state['messages'][-1]
        if not getattr(last_message, "tool_calls", []):
            return "end"
        return "continue"
        
    def should_continue_after_interrupt(self, state: AgentState):
        """
        Determines whether to continue execution after a tool interruption.

        This conditional edge checks if the last message indicates that a tool
        execution was cancelled by the user. If so, it ends the workflow;
        otherwise, it continues back to the agent node.

        Args:
            state: The current state of the agent.

        Returns:
            A string indicating the next node: "end" if the user cancelled,
            or "continue" to proceed with the agent."""
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage) and last_message.content == INTERRUPT_CANCEL_MESSAGE:
            return "end"
        
        return "continue"


def create_confirmation_response(payload: str, type: str, name: str, kind: str, cluster: str, namespace: str):
    """
    Creates a structured confirmation response for the UI.

    This function formats a JSON payload that the UI can use to prompt the user
    for confirmation before executing a sensitive operation.

    Args:
        payload: The data for the operation (e.g., a patch or a resource definition).
        type: The type of operation (e.g., "patch").
        name: The name of the resource.
        kind: The kind of the resource (e.g., "Deployment").
        cluster: The target cluster.
        namespace: The target namespace.
    """
    payload_data = {
        "payload": payload,
        "type": type,
        "resource": {
            "name": name,
            "kind": kind,
            "cluster": cluster,
            "namespace": namespace
        }
    }

    json_payload = json.dumps(payload_data)

    return f'<confirmation-response>{json_payload}</confirmation-response>'


def should_interrupt(human_validation_tools: list[HumanValidationTool], tool_call: any) -> str:
    """
    Checks if a tool call requires user confirmation and generates an interrupt message.

    Args:
        tool_call: The tool call dictionary from the LLM.

    Returns:
        A formatted string to trigger a langgraph.types.interrupt, or an empty string
        if no interruption is needed.
    """
    for tools in human_validation_tools:
        if tools.name == tool_call["name"]:
            if tools.type == "CREATE":
                payload = tool_call['args']['resource']
            elif tools.type == "UPDATE":
                payload = tool_call['args']['patch']
            else:
                logging.error(f"unknown human validation tool type: {tools.type}")
                return ""
            return create_confirmation_response(payload, tools.type.lower(), tool_call['args']['name'], tool_call['args']['kind'], tool_call['args']['cluster'], tool_call['args']['namespace'])

    return ""

    
def handle_interrupt(human_validation_tools: list[HumanValidationTool], tool_call: dict) -> tuple[bool, str | None]:
    """Handles the user confirmation interrupt for a tool call.
    
    Returns:
        A tuple of (should_continue, interrupt_message) where:
        - should_continue: True if execution should continue, False if cancelled
        - interrupt_message: The interrupt message if one was triggered, None otherwise
    """
    if interrupt_message := should_interrupt(human_validation_tools, tool_call):
        response = langgraph.types.interrupt(interrupt_message)
        if response != "yes":
            return False, interrupt_message
        return True, interrupt_message
          
    return True, None 


def process_tool_result(tool_result: str | list, state: AgentState) -> tuple[str, str | None]:
    """Processes the raw tool result, handling JSON and streaming UI context if necessary.
       MCP returns example: {"uiContext":{}, "llm": {}}
       
       Returns:
           A tuple of (processed_result, mcp_response) where mcp_response is None if no uiContext.
    """
    mcp_response = None
    try:
        # Handle list format: [{"type": "text", "text": "tool response", "id": "..."}]
        if isinstance(tool_result, list) and len(tool_result) > 0:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if "uiContext" in json_result:
            mcp_response = f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>"
            dispatch_custom_event("ui_context",mcp_response)
        if "docLinks" in json_result:
            for link in json_result['docLinks']:
                dispatch_custom_event(
                "dock_link",
                f"<mcp-doclink>{link}</mcp-doclink>")

        # Return the value for the LLM, or the full object if 'llm' key is not present
        return convert_to_string_if_needed(json_result.get("llm", json_result)), mcp_response
    except (json.JSONDecodeError, TypeError):
        # If it's not a valid JSON, return the raw string result
        return tool_result, mcp_response


def convert_to_string_if_needed(var):
    """
    Converts a variable to a JSON formatted string only if it is a dict or list
    
    Args:
        var: The variable to process.
        
    Returns:
        The JSON string representation if it was a dict/list,
        otherwise the original variable.
    """
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    else:
        return var
