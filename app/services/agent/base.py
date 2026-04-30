"""
Base agent builder with shared logic for all agent types.
"""

import json
import logging
from langchain.messages import AIMessage
import langgraph.types
import yaml

from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph.state import Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from langchain_core.callbacks.manager import dispatch_custom_event
from .loader import AgentConfig
from .state import AgentState
from ..ui_tools.loader import load_ui_tools_from_configmap
from ..ui_tools.selector import create_ui_tools_selector, filter_tool

INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"
INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE = "tool execution cancelled because previous tool call failed"
MAX_CONSECUTIVE_TOOL_CALLS = 5


class InterruptToolException(Exception):
    """Exception raised when invoking a tool that should trigger an interrupt."""
    pass

class BaseAgentBuilder:
    """Base class for agent builders with shared logic."""
    
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer, agent_config: AgentConfig, all_children_agents: list[AgentConfig] = []):
        """
        Initializes the BaseAgentBuilder.

        Args:
            llm: The language model to use for the agent's decisions.
            tools: A list of all tools the agent can use. Tools whose names end with
                'Plan' are automatically assigned to planning_tools; the rest to tools.
            system_prompt: The initial system-level instructions for the agent.
            checkpointer: The checkpointer for persisting agent state.
            agent_config: Configuration for the agent's behavior and settings.
            all_children_agents: List of all child agent configurations in the system.
        """
        self.llm = llm
        self.planning_tools = [tool for tool in tools if tool.name.endswith("Plan")]
        self.tools = [tool for tool in tools if not tool.name.endswith("Plan")]
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.planning_tools_by_name = {tool.name: tool for tool in self.planning_tools}
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.agent_config = agent_config
        self.all_children_agents = all_children_agents
        
    def _get_messages_from_last_summary(self, state: AgentState) -> list:
        """
        Combines the current summary (if any), and the relevant messages since the last summary.
        """
        messages = []

        summary = state.get("summary", {})
        
        summary_text = summary.get("text", "")
        msg_count = summary.get("msg_count", 0)
        
        if summary_text:
            messages.append(SystemMessage(content=f"Conversation summary: {summary_text}"))
            # Get messages since last summary only + current messages window
            messages += state["messages"][msg_count:]
        else:
            messages += state["messages"]

        return messages
    
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
        summary = state.get("summary", {})
        summary_text = summary.get("text", "")
        
        messages = self._get_messages_from_last_summary(state)
        
        summary_prompt = "Create a summary of the conversation above:"
        if summary_text:
            summary_prompt = (
                "Extend the current summary by incorporating the new messages above. "
                "Maintain a concise but complete overview of the entire conversation."
            )

        messages.append(HumanMessage(content=summary_prompt))
        response = self.llm.invoke(messages)

        logging.debug(f"Conversation summarized. New history window will start from index {len(state['messages'])}")

        return {
            "summary": {
                "text": response.content,
                "msg_count": len(state["messages"])
            }
        }

    def _extract_context_for_tool_selection(self, state: AgentState, config: RunnableConfig) -> tuple[str, str, str, str]:
        """
        Extracts and formats context from conversation for UI tool selection.
        
        Collects the most recent user message, assistant message, and MCP response data
        to provide context for tool selection decisions.
        
        Args:
            state: The current agent state containing messages
            config: The runtime configuration containing request_id
            
        Returns:
            Tuple of (user_message, ai_message, mcp_response, mcp_data)
        """
        request_id = config["configurable"]["request_id"]
        user_message = ''
        ai_message = ''
        mcp_data = ''
        
        # Collect recent messages (user, assistant, tool results) for current request
        for msg in reversed(state["messages"][-10:]):
            additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}

            # Only consider messages from the current request context
            if request_id != additional_kwargs.get("request_id", ""):
                break

            # All messages collected, early exit
            if user_message and ai_message and mcp_data:
                break

            if hasattr(msg, "text") and isinstance(msg.text, str):
                if isinstance(msg, HumanMessage) and not user_message:
                    if "request_metadata" in additional_kwargs:
                        request_metadata = additional_kwargs["request_metadata"]
                        user_input = request_metadata.get("user_input", "")
                        context = request_metadata.get("context", {})
                        user_message += f"\n[User Message]: {user_input} - ui context: {context}"
                    if not user_message:
                        user_message += f"\n[User Message]: {msg.text}"
                elif isinstance(msg, AIMessage) and not ai_message:
                    ai_message += f"\n[Assistant Message]: {msg.text}"
                
                # Collect the mcp result's payloads
                elif isinstance(msg, ToolMessage) and not mcp_data:
                    mcp_data = self._convert_tool_message_to_context(msg.content)

        # Collect MCP response resources from all messages
        mcp_response = self._extract_mcp_responses(state)
        
        return user_message, ai_message, mcp_response, mcp_data

    def _convert_tool_message_to_context(self, content: str) -> str:
        """
        Converts a tool message content to formatted context (YAML format).
        
        Args:
            content: The raw tool message content (usually JSON)
            
        Returns:
            Formatted content string with MCP result payloads label
        """
        try:
            parsed = json.loads(content)
            # Handle both single objects and arrays of objects
            if isinstance(parsed, list):
                # Convert array of objects to YAML with document separators
                yaml_parts = []
                for item in parsed:
                    yaml_parts.append(yaml.dump(item, default_flow_style=False, sort_keys=False))
                content = "---\n".join(yaml_parts)
                logging.debug(f"Converted ToolMessage array content to YAML format ({len(parsed)} items)")
            elif isinstance(parsed, dict):
                # Convert single object to YAML
                content = yaml.dump(parsed, default_flow_style=False, sort_keys=False)
                logging.debug(f"Converted ToolMessage dict content to YAML format")
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, use original content
            pass
        
        return f"\n[MCP result payloads]: {content}"

    def _extract_mcp_responses(self, state: AgentState) -> str:
        """
        Extracts MCP response resources from all messages in state.
        
        Args:
            state: The current agent state
            
        Returns:
            Formatted MCP response resources string, or empty string if none found
        """
        mcp_response = ''
        for msg in reversed(state["messages"]):
            additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
            
            # Include MCP response data if present (from tool execution)
            if "mcp_response" in additional_kwargs:
                mcp_response += f"\n{additional_kwargs['mcp_response'].strip('<mcp-response></mcp-response>')}"
                
        if mcp_response:
            mcp_response = '\n[MCP result resources]: ' + mcp_response
            
        return mcp_response

    def _dispatch_ui_tools_event(self, state: AgentState, config: RunnableConfig) -> list[dict]:
        """
        Helper method to dispatch UI tools event.
        
        Selects and dispatches UI tools based on the current state.
        This can be called from various nodes, including when a confirmation-response
        is being sent, to ensure UI tools are available to the client.
        
        Args:
            state: The current agent state
            config: The runtime configuration containing ui_tools_config
            
        Returns:
            List of selected UI tools, or empty list if dispatch was skipped
        """
        try:
            # Extract the UI tools configuration from request metadata
            request_metadata = config.get("configurable", {}).get("request_metadata", {})
            ui_tools_config = request_metadata.get("ui_tools", {})
            
            logging.debug(f"_dispatch_ui_tools_event: config={ui_tools_config}")
            
            name = ui_tools_config.get("name", "")
            tool_filters = ui_tools_config.get("tools", [])

            if not name:
                logging.debug("UI tools config name is missing, skipping ui tools dispatch")
                return []
            
            if not tool_filters or len(tool_filters) == 0:
                logging.debug("UI tools list is empty, skipping ui tools dispatch")
                return []
            
            ui_tools_config_data = load_ui_tools_from_configmap(name)
            
            if not ui_tools_config_data or not ui_tools_config_data.config:
                logging.debug(f"UI tools config {name} not found, skipping ui tools dispatch")
                return []
            
            if not ui_tools_config_data.config.enabled:
                logging.debug(f"UI tools config {name} are disabled, skipping ui tools dispatch")
                return []

            # Filter tools: keep only enabled tools that are in the tool_filters list
            filtered_tools = []
            for tool in ui_tools_config_data.tools:
                if filter_tool(tool, tool_filters):
                    filtered_tools.append(tool)
            logging.debug(f"Filtered UI tools: {[t.name for t in filtered_tools]} based on filters: {tool_filters}")

            # Skip if filtered tools list is empty
            if len(filtered_tools) == 0:
                logging.debug("No UI tools available after filtering, skipping ui tools dispatch")
                return []
            
            # Get the selected agent context
            selected_agent = state.get("selected_agent", {}).get("name", "")
            agent_config = None
            for child in self.child_agents:
                if child.config.name == selected_agent:
                    agent_config = child.config
                    break

            # Extract context for tool selection
            user_message, ai_message, mcp_response, mcp_data = self._extract_context_for_tool_selection(state, config)

            # Get the system prompt and max_tools from the config
            system_prompt = ui_tools_config_data.config.system_prompt
            max_tools = ui_tools_config_data.config.max_tools
            
            # Create a UI tools selector using the same LLM and the system prompt
            selector = create_ui_tools_selector(self.llm, system_prompt=system_prompt, max_tools=max_tools)
            
            # Dispatch processing message to notify the ui-tools selection is in progress
            # This should become a standard pattern, potentially implemented as a LangGraph event, to notify the UI the status of operations
            dispatch_custom_event("notify_processing", "<processing-ui-tools/>")
            
            # select_tools already returns sanitized and validated ui tools
            ui_tools_list = selector.select_tools(
                agent_config=agent_config,
                context=user_message + ai_message,
                mcp_response=mcp_response + mcp_data,
                available_tools=filtered_tools,
            )
            
            # Dispatch custom event with all tools
            self._dispatch_ui_tools(ui_tools_list)
            
            return ui_tools_list
        except Exception as e:
            logging.error(f"Error dispatching UI tools event: {e}", exc_info=True)
            return []
            
    def _dispatch_preprocessed_ui_tools(self, state: AgentState, config: RunnableConfig, tools: list[dict]) -> None:
        """
        Helper method to dispatch preprocessed UI tools.

        This method can be used to dispatch UI tools that have been preprocessed,
        for example based on the MCP response or other intermediate computations.
        It ensures the tools are sent to the UI in a consistent format.

        Args:
            state: The current state of the agent, containing the full conversation.
            config: The runtime configuration containing ui_tools_config name.
            tools: A list of dictionaries representing the preprocessed UI tools to dispatch.
        """
        # Extract the UI tools configuration from request metadata
        request_metadata = config.get("configurable", {}).get("request_metadata", {})
        ui_tools_config = request_metadata.get("ui_tools", {})
        
        logging.debug(f"_dispatch_preprocessed_ui_tools: config={ui_tools_config}")
        
        name = ui_tools_config.get("name", "")

        if not name:
            logging.debug("UI tools config name is missing, skipping ui tools dispatch")
            return
        
        self._dispatch_ui_tools(tools)

    def _dispatch_ui_tools(self, tools: list[dict]) -> None:
        """
        Helper method to dispatch UI tools event with a given list of tools.
        
        Args:
            tools: A list of dictionaries representing the selected UI tools to dispatch.
        """
        try:
            ui_tools_json = json.dumps(tools)
            ui_tools_event = f"<ui-tools>{ui_tools_json}</ui-tools>"
            dispatch_custom_event("ui_tools", ui_tools_event)
            logging.debug(f"Dispatched {len(tools)} UI tool(s): {[t['toolName'] for t in tools]}")
        except Exception as e:
            logging.error(f"Error dispatching UI tools: {e}", exc_info=True)

    def ui_tools_node(self, state: AgentState, config: RunnableConfig):
        """
        Select appropriate UI tools for the agent's response.
        
        This node is the deterministic step that ALL execution paths converge to
        before ending or summarizing. Therefore, it runs exactly once per request.
        
        When UI tools are disabled or the filtered tools list is empty, this node
        returns early without executing the selector and without dispatching the 
        ui-tools event to the UI.
        
        Args:
            state: The current state of the agent, containing the full conversation.
            config: The runtime configuration containing ui_tools_config name.
        
        Returns:
            A dictionary with the updated AIMessage containing ui_tools in additional_kwargs (empty if skipped).
        """
        # Dispatch UI tools using the helper method
        ui_tools_list = self._dispatch_ui_tools_event(state, config)
        
        # Save UI tools to the AIMessage's additional_kwargs for this request_id
        if ui_tools_list:
            request_id = config["configurable"]["request_id"]
            
            # Find the AIMessage with matching request_id
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
                    if additional_kwargs.get("request_id") == request_id:
                        # Add ui_tools to the AIMessage's additional_kwargs
                        additional_kwargs["ui_tools"] = ui_tools_list
                        msg.additional_kwargs = additional_kwargs
                        return {"messages": [msg]}
            
            logging.warning(f"Could not find AIMessage with request_id {request_id} to attach ui_tools")
        
        return {"messages": []}

    def _count_consecutive_tool_rounds(self, state: AgentState) -> int:
        """Count the number of consecutive tool call rounds since the last HumanMessage.
        
        A tool call round is counted for each AIMessage that contains tool_calls.
        Counting stops when a HumanMessage is encountered.
        
        Args:
            state: The current state of the agent.

        Returns:
            The number of consecutive tool call rounds.
        """
        count = 0
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                break
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                count += 1
        return count

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

        base_messages = self._get_messages_from_last_summary(state)
        
        # Add the System Prompt - it should be used only for user requests
        messages = []
        if self.system_prompt.strip():
            selected_agent = state.get("selected_agent", {})
            if selected_agent and self.all_children_agents:
                # Build list of available child agents (excluding the currently selected one)
                available_children = [
                    f"- {child.name}: {child.description}\n"
                    for child in self.all_children_agents
                    if child.name != selected_agent.get("name")
                ]
                
                if available_children:
                    children_description = "\n".join(available_children)
                    system_prompt_with_children = f"""{self.system_prompt}

You are a highly specialized Assistant. Your primary goal is to provide accurate information within your domain. To maintain accuracy, you must never guess. If a user's request falls outside your expertise, you are required to direct them to the appropriate specialized agent from the following list of available child agents:

{children_description}"""
                    messages.append(SystemMessage(content=system_prompt_with_children))
                else:
                    messages.append(SystemMessage(content=self.system_prompt))
            else:
                messages.append(SystemMessage(content=self.system_prompt))
        
        messages.extend(base_messages)

        # Check if consecutive tool call limit has been reached
        consecutive_rounds = self._count_consecutive_tool_rounds(state)
        if consecutive_rounds >= MAX_CONSECUTIVE_TOOL_CALLS:
            logging.warning(
                f"Reached maximum consecutive tool calls ({MAX_CONSECUTIVE_TOOL_CALLS}). "
                "Forcing LLM to respond without tools."
            )
            forced_answer_messages = messages + [HumanMessage(
                content=(
                    "You have reached the maximum number of consecutive tool calls. "
                    "You MUST now provide a final answer based on the information gathered so far. "
                    "Do NOT request any more tool calls. "
                    "If you need more information, ask the user to provide it."
                )
            )]
            response = self.llm.invoke(forced_answer_messages, config)
        else:
            response = self._invoke_llm_with_retry(messages, config)

        response.additional_kwargs["request_id"] = config["configurable"]["request_id"]
        response.additional_kwargs["selected_agent"] = state.get("selected_agent", {})

        if response.invalid_tool_calls:
            logging.error(f"model response contained invalid tool calls: {response.invalid_tool_calls}")
            raise ValueError(f"model response contained invalid tool calls: {response.invalid_tool_calls}")

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

        tool_calls = getattr(state["messages"][-1], "tool_calls", [])
        human_validation_tools = getattr(self.agent_config, "human_validation_tools", [])

        # Phase 1: Resolve all interrupt decisions before executing any tools.
        # langgraph replays the entire node on resume after an interrupt, so if
        # interrupts and tool executions are interleaved, tools that ran before a
        # later interrupt would be re-invoked on replay.  Collecting all interrupt
        # responses first ensures every tool is executed exactly once.
        interrupt_messages = {}
        for idx, tool_call in enumerate(tool_calls):
            should_continue, interrupt_message, ui_tools_list = None, None, []
            
            try:
                should_continue, interrupt_message, ui_tools_list = await self.handle_interrupt(human_validation_tools, tool_call, state, config)
            except InterruptToolException as e:
                logging.error(f"Error during confirmation interrupt: {e}")
                interrupt_messages[tool_call["id"]] = {
                    "message": f"{e}",
                    "ui_tools": [],
                    "interrupt-error": True
                }
                break

            if not should_continue:
                # Cancel ALL tool calls: previously approved ones, the rejected one,
                # and any remaining unevaluated ones — no tools will be executed.
                outputs = self._cancel_remaining_tool_calls(tool_calls[:idx], request_id, state, INTERRUPT_CANCEL_MESSAGE)
                cancel_kwargs = {
                    "request_id": request_id,
                    "selected_agent": state.get("selected_agent", {}),
                    "interrupt_message": interrupt_message,
                    "ui_tools": ui_tools_list,
                    "confirmation": False
                }
                outputs.append(ToolMessage(
                    content=INTERRUPT_CANCEL_MESSAGE,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=cancel_kwargs
                ))
                outputs.extend(self._cancel_remaining_tool_calls(tool_calls[idx + 1:], request_id, state, INTERRUPT_CANCEL_MESSAGE))
                return {"messages": outputs}
            if interrupt_message:
                interrupt_messages[tool_call["id"]] = {
                    "message": interrupt_message,
                    "ui_tools": ui_tools_list
                }

        # Phase 2: Execute tools (all interrupts were approved OR interrupt errors occurred if we reach here).
        for idx, tool_call in enumerate(tool_calls):
            additional_kwargs = {
                "request_id": request_id,
                "selected_agent": state.get("selected_agent", {})
            }

            interrupt_message = interrupt_messages.get(tool_call["id"])
            if interrupt_message:
                additional_kwargs["interrupt_message"] = interrupt_message["message"]
                additional_kwargs["confirmation"] = True
                additional_kwargs["ui_tools"] = interrupt_message["ui_tools"]
            
            try:
                # Skip execution and return the exception as information for the next nodes to handle
                if interrupt_message and interrupt_message.get("interrupt-error", False):
                    raise ToolException(interrupt_message['message'])

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
            except Exception as e:
                logging.error(f"unexpected error during tool call: {e}")
                outputs.append(ToolMessage(
                    content=f"unexpected error during tool call: {e}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs
                ))
                outputs.extend(self._cancel_remaining_tool_calls(tool_calls[idx + 1:], request_id, state, INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE))
                return {"messages": outputs}

        return {"messages": outputs}
    
    def _cancel_remaining_tool_calls(self, remaining_tool_calls: list[dict], request_id: str, state: AgentState, message: str) -> list[ToolMessage]:
        """Creates cancel ToolMessages for tool calls that will not be executed.

        When a tool call is cancelled or fails, LangGraph still expects a
        ToolMessage for every tool_call_id in the AIMessage.  This helper
        produces the missing messages for any tool calls that come after the
        one that stopped execution.
        """
        messages = []
        for tc in remaining_tool_calls:
            messages.append(ToolMessage(
                content=message,
                name=tc["name"],
                tool_call_id=tc["id"],
                additional_kwargs={
                    "request_id": request_id,
                    "selected_agent": state.get("selected_agent", {}),
                    "confirmation": False
                }
            ))
        return messages

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

        if getattr(last_message, "tool_calls", []):
            return "continue"

        summary = state.get("summary", {})
        
        # Summarize batches of 7 messages - this threshold is arbitrary
        last_count = summary.get("msg_count", 0) if summary else 0
        if len(messages) - last_count >= 7:
            return "summarize_conversation"

        return "end"
        
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
    
    async def should_interrupt(self, human_validation_tools: list[str], tool_call: any) -> str:
        """
        Checks if a tool call requires user confirmation and generates an interrupt message.

        Args:
            tool_call: The tool call dictionary from the LLM.

        Returns:
            A formatted string to trigger a langgraph.types.interrupt, or an empty string
            if no interruption is needed.
        """
        for tool_name in human_validation_tools:
            if tool_name == tool_call["name"]:
                plan_tool_name = tool_call["name"] + "Plan"
                plan_tool = self.planning_tools_by_name.get(plan_tool_name)
                if plan_tool is None:
                    raise ValueError(f"planning tool '{plan_tool_name}' not found for tool '{tool_call['name']}'")
                try:
                    plan_response = await plan_tool.ainvoke(tool_call["args"])
                except Exception as e:
                    logging.error(f"error invoking planning tool '{plan_tool_name}' for interrupt: {e}")
                    raise InterruptToolException(e)

                # Normalize list response from MCP tools: [{"type": "text", "text": "..."}]
                if isinstance(plan_response, list) and len(plan_response) > 0:
                    if isinstance(plan_response[0], dict) and "text" in plan_response[0]:
                        plan_response = plan_response[0]["text"]
                
                try:
                    safe_response = json.dumps(json.loads(plan_response))
                except (json.JSONDecodeError, TypeError):
                    safe_response = json.dumps(plan_response)

                return safe_response

        return ""

    async def handle_interrupt(self, human_validation_tools: list[str], tool_call: dict, state: AgentState, config: RunnableConfig = None) -> tuple[bool, str | None, list[dict]]:
        """Handles the user confirmation interrupt for a tool call.
        
        Args:
            human_validation_tools: List of tool names that require human validation
            tool_call: The tool call dictionary
            state: The current agent state
            config: The runtime configuration (optional, needed to dispatch UI tools)
        
        Returns:
            A tuple of (should_continue, interrupt_message, ui_tools_list) where:
            - should_continue: True if execution should continue, False if cancelled
            - interrupt_message: The interrupt message if one was triggered, None otherwise
            - ui_tools_list: List of UI tools for this confirmation, empty if none
        """
        interrupt_message = None

        try:
            interrupt_message = await self.should_interrupt(human_validation_tools, tool_call)
        except InterruptToolException as e:
            logging.error(f"Error during interrupt message generation: {e}")
            raise e

        if interrupt_message:
            logging.info(f"Confirmation interrupt triggered for tool '{tool_call.get('name')}', config={'present' if config else 'missing'}")
            
            ui_tools_list = []
            
            # Dispatch UI tools before the interrupt, so they're available to the client
            if config is not None:
                try:
                    data = json.loads(interrupt_message)
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                        
                    # Build ui tool
                    resource = data.get("resource", {})
                    input = {
                        "resourceKind": resource.get("kind"),
                        "resourceName": resource.get("name"),
                        "resourceNamespace": resource.get("namespace"),
                    }
                    ui_tool_name = "show-yaml"

                    if data.get("type") == "create":
                        input["yaml"] = data.get("payload", {})
                    else:
                        ui_tool_name = "show-yaml-diff"
                        input["original"] = data.get("payload", {}).get("original")
                        input["patched"] = data.get("payload", {}).get("patched")

                    input = {k: v for k, v in input.items() if v is not None}
                    
                    ui_tools_list = [{
                        "toolName": ui_tool_name,
                        "input": input,
                    }]
                    self._dispatch_preprocessed_ui_tools(state, config, ui_tools_list)
                except Exception as e:
                    logging.debug(f"Could not extract precomputed fields from interrupt message and dispatch UI tools: {e}")

            else:
                logging.warning("config is None, cannot dispatch UI tools before confirmation")
            
            interrupt_message_result = build_interrupt_message_result(interrupt_message)

            response = langgraph.types.interrupt(interrupt_message_result)
            if response != "yes":
                return False, interrupt_message_result, ui_tools_list
            
            selected_agent = state.get("selected_agent", {})
            if selected_agent:
                dispatch_custom_event(
                    "subagent_choice_event",
                    build_agent_metadata(selected_agent.get("name"), selected_agent.get("mode")),
                )
            return True, interrupt_message_result, ui_tools_list
            
        return True, None, []



def build_agent_metadata(agent_name: str, selection_mode: str, extra_metadata : str = "") -> str:
    """Builds a structured agent metadata string for custom events."""
    return f'<agent-metadata>{{"agentName": "{agent_name}", "selectionMode": "{selection_mode}"{extra_metadata}}}</agent-metadata>'


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
            dispatch_custom_event("ui_context", mcp_response)
        if "docLinks" in json_result:
            for link in json_result['docLinks']:
                dispatch_custom_event(
                "dock_link",
                f"<mcp-doclink>{link}</mcp-doclink>")

        # Return the value for the LLM, or the full object if 'llm' key is not present
        llm_result = json_result.get("llm", json_result) if isinstance(json_result, dict) else json_result
        return convert_to_string_if_needed(llm_result), mcp_response
    except (json.JSONDecodeError, TypeError):
        # If it's not a valid JSON, return the raw string result
        return tool_result, mcp_response
    
def build_interrupt_message_result(interrupt_message) -> str:
    """Builds the interrupt message content based on the tool call and its planning response."""
    return f"<confirmation-response>{interrupt_message}</confirmation-response>"


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
