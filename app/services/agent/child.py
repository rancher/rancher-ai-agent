"""
Agent builder for creating LangGraph agents with tool execution and human-in-the-loop validation.

Provides ChildAgentBuilder which constructs agents for use under a supervisor (multi-agent)
or standalone (single-agent) setups. Each agent has an LLM-driven reasoning loop with
tool execution, human validation gates, and automatic retry on malformed tool calls.
"""

import json
import logging

import langgraph.types
import yaml
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Checkpointer, CompiledStateGraph
from ollama import ResponseError

from .loader import AgentConfig
from .state import AgentState
from ..ui_tools.loader import load_ui_tools_from_configmap
from ..ui_tools.selector import create_ui_tools_selector, filter_tool

INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"
INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE = "tool execution cancelled because previous tool call failed"
MAX_CONSECUTIVE_TOOL_CALLS = 10


class ChildAgentBuilder:
    """
    Builds a LangGraph agent with tool execution and human-in-the-loop validation.

    The agent graph follows a simple loop::

        agent (LLM call) -> tools -> agent -> ... -> END

    Features:
    - Planning tools (names ending with ``Plan``) are separated for interrupt handling
    - Human validation via LangGraph interrupts for sensitive operations
    - Consecutive tool call limiting to prevent infinite loops
    - LLM retry on malformed tool call responses
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str,
        checkpointer: Checkpointer,
        agent_config: AgentConfig,
    ):
        self.llm = llm
        self.planning_tools = [t for t in tools if t.name.endswith("Plan")]
        self.tools = [t for t in tools if not t.name.endswith("Plan")]
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.planning_tools_by_name = {t.name: t for t in self.planning_tools}
        self.tools_by_name = {t.name: t for t in self.tools}
        self.agent_config = agent_config

    def build(self) -> CompiledStateGraph:
        """Build and compile the agent graph.

        Workflow paths:
        - With tool_calls: agent -> tools (loop) -> agent
        - No tool_calls: agent -> ui_tools -> END
        - Tool rejection: tools -> END (skip ui_tools)
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("ui_tools", self.ui_tools_node)

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {"continue": "tools", "end": "ui_tools"},
        )
        workflow.add_conditional_edges(
            "tools",
            self.should_continue_after_interrupt,
            {"continue": "agent", "end": END},
        )
        workflow.add_edge("ui_tools", END)
        workflow.set_entry_point("agent")

        return workflow.compile(checkpointer=self.checkpointer)

    # -- Graph nodes -----------------------------------------------------------

    def call_model_node(self, state: AgentState, config: RunnableConfig):
        """Invoke the LLM with the system prompt and conversation history."""
        logging.debug("calling model")

        messages = [
            *([SystemMessage(content=self.system_prompt)] if self.system_prompt.strip() else []),
            *state["messages"],
        ]

        response = self._invoke_llm_with_retry(messages, config)

        response.additional_kwargs["request_id"] = config["configurable"]["request_id"]
        response.additional_kwargs["selected_agent"] = state.get("selected_agent", {})

        if response.invalid_tool_calls:
            logging.error(f"model response contained invalid tool calls: {response.invalid_tool_calls}")
            raise ValueError(f"model response contained invalid tool calls: {response.invalid_tool_calls}")

        logging.debug("model call finished")
        return {"messages": [response]}

    async def tool_node(self, state: AgentState, config: RunnableConfig):
        """Execute tools requested by the LLM, with human validation when configured."""
        outputs: list[ToolMessage] = []
        request_id = config["configurable"]["request_id"]
        tool_calls = getattr(state["messages"][-1], "tool_calls", [])
        human_validation_tools = getattr(self.agent_config, "human_validation_tools", [])

        # Phase 1: Resolve all interrupt decisions before executing any tools.
        # LangGraph replays the entire node on resume after an interrupt, so
        # collecting all interrupt responses first ensures every tool runs exactly once.
        interrupt_messages: dict[str, dict] = {}
        for idx, tool_call in enumerate(tool_calls):
            should_continue, interrupt_message, ui_tools_list = await self.handle_interrupt(
                human_validation_tools, tool_call, state, config
            )
            if not should_continue:
                # Cancel ALL tool calls (previously approved, rejected, and remaining).
                outputs = self._cancel_remaining_tool_calls(
                    tool_calls[:idx], request_id, state, INTERRUPT_CANCEL_MESSAGE
                )
                outputs.append(ToolMessage(
                    content=INTERRUPT_CANCEL_MESSAGE,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs={
                        "request_id": request_id,
                        "selected_agent": state.get("selected_agent", {}),
                        "interrupt_message": interrupt_message,
                        "ui_tools": ui_tools_list,
                        "confirmation": False,
                    },
                ))
                outputs.extend(self._cancel_remaining_tool_calls(
                    tool_calls[idx + 1:], request_id, state, INTERRUPT_CANCEL_MESSAGE
                ))
                return {"messages": outputs}
            if interrupt_message:
                interrupt_messages[tool_call["id"]] = {
                    "message": interrupt_message,
                    "ui_tools": ui_tools_list,
                }

        # Phase 2: Execute tools (all interrupts were approved if we reach here).
        for idx, tool_call in enumerate(tool_calls):
            additional_kwargs: dict = {
                "request_id": request_id,
                "selected_agent": state.get("selected_agent", {}),
            }
            interrupt_info = interrupt_messages.get(tool_call["id"])
            if interrupt_info:
                additional_kwargs["interrupt_message"] = interrupt_info["message"]
                additional_kwargs["confirmation"] = True
                additional_kwargs["ui_tools"] = interrupt_info["ui_tools"]

            try:
                logging.debug("calling tool")
                tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                logging.debug("tool call finished")

                processed_result, mcp_response = process_tool_result(tool_result, state)
                if mcp_response:
                    additional_kwargs["mcp_response"] = mcp_response

                outputs.append(ToolMessage(
                    content=processed_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs,
                ))
            except Exception as e:
                logging.error(f"unexpected error during tool call: {e}")
                outputs.append(ToolMessage(
                    content=f"unexpected error during tool call: {e}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=additional_kwargs,
                ))
                outputs.extend(self._cancel_remaining_tool_calls(
                    tool_calls[idx + 1:], request_id, state, INTERRUPT_PREVIOUS_TOOL_FAILED_MESSAGE
                ))
                return {"messages": outputs}

        return {"messages": outputs}

    # -- Edge conditions -------------------------------------------------------

    def should_continue(self, state: AgentState) -> str:
        """Route to 'tools' if the LLM requested tool calls, otherwise 'end'."""
        if not getattr(state["messages"][-1], "tool_calls", []):
            return "end"
        return "continue"

    def should_continue_after_interrupt(self, state: AgentState) -> str:
        """Route to 'end' if the last tool was cancelled by the user, otherwise 'continue'."""
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.content == INTERRUPT_CANCEL_MESSAGE:
            return "end"
        return "continue"

    # -- UI tools --------------------------------------------------------------

    def ui_tools_node(self, state: AgentState, config: RunnableConfig):
        """Select and dispatch appropriate UI tools for the agent's response.

        This node runs after the agent produces its final answer (no tool calls)
        and before the graph ends.  It dispatches a ``ui_tools`` custom event so
        the client can render relevant UI components.
        """
        ui_tools_list = self._dispatch_ui_tools_event(state, config)

        if ui_tools_list:
            request_id = config["configurable"]["request_id"]
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
                    if additional_kwargs.get("request_id") == request_id:
                        additional_kwargs["ui_tools"] = ui_tools_list
                        msg.additional_kwargs = additional_kwargs
                        return {"messages": [msg]}

            logging.warning(f"Could not find AIMessage with request_id {request_id} to attach ui_tools")

        return {"messages": []}

    def _dispatch_ui_tools_event(self, state: AgentState, config: RunnableConfig) -> list[dict]:
        """Select and dispatch UI tools based on the current conversation state.

        Returns:
            List of selected UI tools, or empty list if dispatch was skipped.
        """
        try:
            request_metadata = config.get("configurable", {}).get("request_metadata", {})
            ui_tools_config = request_metadata.get("ui_tools", {})

            logging.debug(f"_dispatch_ui_tools_event: config={ui_tools_config}")

            name = ui_tools_config.get("name", "")
            tool_filters = ui_tools_config.get("tools", [])

            if not name:
                logging.debug("UI tools config name is missing, skipping ui tools dispatch")
                return []

            if not tool_filters:
                logging.debug("UI tools list is empty, skipping ui tools dispatch")
                return []

            ui_tools_config_data = load_ui_tools_from_configmap(name)

            if not ui_tools_config_data or not ui_tools_config_data.config:
                logging.debug(f"UI tools config {name} not found, skipping ui tools dispatch")
                return []

            if not ui_tools_config_data.config.enabled:
                logging.debug(f"UI tools config {name} are disabled, skipping ui tools dispatch")
                return []

            filtered_tools = [t for t in ui_tools_config_data.tools if filter_tool(t, tool_filters)]
            logging.debug(
                f"Filtered UI tools: {[t.name for t in filtered_tools]} "
                f"based on filters: {tool_filters}"
            )

            if not filtered_tools:
                logging.debug("No UI tools available after filtering, skipping ui tools dispatch")
                return []

            # Get the selected agent context
            selected_agent = state.get("selected_agent", {}).get("name", "")
            agent_config = None
            for child in self.child_agents:
                if child.config.name == selected_agent:
                    agent_config = child.config
                    break

            user_message, ai_message, mcp_response, mcp_data = self._extract_context_for_tool_selection(
                state, config
            )

            system_prompt = ui_tools_config_data.config.system_prompt
            max_tools = ui_tools_config_data.config.max_tools

            selector = create_ui_tools_selector(self.llm, system_prompt=system_prompt, max_tools=max_tools)

            dispatch_custom_event("notify_processing", "<processing-ui-tools/>")

            ui_tools_list = selector.select_tools(
                agent_config=agent_config,
                context=user_message + ai_message,
                mcp_response=mcp_response + mcp_data,
                available_tools=filtered_tools,
            )

            self._dispatch_ui_tools(ui_tools_list)
            return ui_tools_list
        except Exception as e:
            logging.error(f"Error dispatching UI tools event: {e}", exc_info=True)
            return []

    def _extract_context_for_tool_selection(
        self, state: AgentState, config: RunnableConfig,
    ) -> tuple[str, str, str, str]:
        """Extract context from the conversation for UI tool selection.

        Returns:
            ``(user_message, ai_message, mcp_response, mcp_data)``
        """
        request_id = config["configurable"]["request_id"]
        user_message = ""
        ai_message = ""
        mcp_data = ""

        for msg in reversed(state["messages"][-10:]):
            additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}

            if request_id != additional_kwargs.get("request_id", ""):
                break

            if user_message and ai_message and mcp_data:
                break

            if hasattr(msg, "text") and isinstance(msg.text, str):
                if isinstance(msg, HumanMessage) and not user_message:
                    if "request_metadata" in additional_kwargs:
                        rm = additional_kwargs["request_metadata"]
                        user_input = rm.get("user_input", "")
                        context = rm.get("context", {})
                        user_message += f"\n[User Message]: {user_input} - ui context: {context}"
                    if not user_message:
                        user_message += f"\n[User Message]: {msg.text}"
                elif isinstance(msg, AIMessage) and not ai_message:
                    ai_message += f"\n[Assistant Message]: {msg.text}"
                elif isinstance(msg, ToolMessage) and not mcp_data:
                    mcp_data = self._convert_tool_message_to_context(msg.content)

        mcp_response = self._extract_mcp_responses(state)
        return user_message, ai_message, mcp_response, mcp_data

    @staticmethod
    def _convert_tool_message_to_context(content: str) -> str:
        """Convert tool message content to YAML-formatted context."""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                yaml_parts = [yaml.dump(item, default_flow_style=False, sort_keys=False) for item in parsed]
                content = "---\n".join(yaml_parts)
            elif isinstance(parsed, dict):
                content = yaml.dump(parsed, default_flow_style=False, sort_keys=False)
        except (json.JSONDecodeError, TypeError):
            pass
        return f"\n[MCP result payloads]: {content}"

    @staticmethod
    def _extract_mcp_responses(state: AgentState) -> str:
        """Extract MCP response resources from all messages."""
        mcp_response = ""
        for msg in reversed(state["messages"]):
            additional_kwargs = msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}
            if "mcp_response" in additional_kwargs:
                mcp_response += f"\n{additional_kwargs['mcp_response'].strip('<mcp-response></mcp-response>')}"
        if mcp_response:
            mcp_response = "\n[MCP result resources]: " + mcp_response
        return mcp_response

    # -- Interrupt / human validation ------------------------------------------

    async def should_interrupt(self, human_validation_tools: list[str], tool_call: dict) -> str:
        """Return a confirmation prompt if *tool_call* requires human validation, else ``''``."""
        for tool_name in human_validation_tools:
            if tool_name == tool_call["name"]:
                plan_tool_name = tool_call["name"] + "Plan"
                plan_tool = self.planning_tools_by_name.get(plan_tool_name)
                if plan_tool is None:
                    raise ValueError(
                        f"planning tool '{plan_tool_name}' not found for tool '{tool_call['name']}'"
                    )
                plan_response = await plan_tool.ainvoke(tool_call["args"])

                # Normalize list response: [{"type": "text", "text": "..."}]
                if isinstance(plan_response, list) and plan_response:
                    if isinstance(plan_response[0], dict) and "text" in plan_response[0]:
                        plan_response = plan_response[0]["text"]

                try:
                    safe_response = json.dumps(json.loads(plan_response))
                except (json.JSONDecodeError, TypeError):
                    safe_response = json.dumps(plan_response)
                return f"<confirmation-response>{safe_response}</confirmation-response>"
        return ""

    async def handle_interrupt(
        self, human_validation_tools: list[str], tool_call: dict, state: AgentState,
        config: RunnableConfig = None,
    ) -> tuple[bool, str | None, list[dict]]:
        """Handle user confirmation for a tool call.

        Returns:
            A tuple of ``(should_continue, interrupt_message, ui_tools_list)``:

            - *should_continue*: ``False`` when the user rejected the action.
            - *interrupt_message*: the prompt shown to the user (``None`` when
              no interrupt was needed).
            - *ui_tools_list*: preprocessed UI tools for this confirmation
              (empty list when none).
        """
        ui_tools_list: list[dict] = []

        interrupt_message = await self.should_interrupt(human_validation_tools, tool_call)
        if not interrupt_message:
            return True, None, ui_tools_list

        logging.info(f"Confirmation interrupt triggered for tool '{tool_call.get('name')}'")

        # Build preprocessed UI tools from the interrupt payload and dispatch
        # them *before* the interrupt so the client can render them immediately.
        if config is not None:
            try:
                data = json.loads(
                    interrupt_message.strip("<confirmation-response></confirmation-response>")
                )
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]

                resource = data.get("resource", {})
                tool_input: dict = {
                    "resourceKind": resource.get("kind"),
                    "resourceName": resource.get("name"),
                    "resourceNamespace": resource.get("namespace"),
                }
                ui_tool_name = "show-yaml"

                if data.get("type") == "create":
                    tool_input["yaml"] = data.get("payload", {})
                else:
                    ui_tool_name = "show-yaml-diff"
                    tool_input["original"] = data.get("payload", {}).get("original")
                    tool_input["patched"] = data.get("payload", {}).get("patched")

                tool_input = {k: v for k, v in tool_input.items() if v is not None}

                ui_tools_list = [{"toolName": ui_tool_name, "input": tool_input}]
                self._dispatch_preprocessed_ui_tools(state, config, ui_tools_list)
            except Exception as e:
                logging.debug(
                    f"Could not extract precomputed fields from interrupt message "
                    f"and dispatch UI tools: {e}"
                )

        response = langgraph.types.interrupt(interrupt_message)
        if response != "yes":
            return False, interrupt_message, ui_tools_list

        selected_agent = state.get("selected_agent", {})
        if selected_agent:
            dispatch_custom_event(
                "subagent_choice_event",
                build_agent_metadata(selected_agent.get("name"), selected_agent.get("mode")),
            )
        return True, interrupt_message, ui_tools_list

    # -- Private helpers -------------------------------------------------------

    def _invoke_llm_with_retry(self, messages: list, config: RunnableConfig):
        """Invoke the LLM with a single retry for tool-call parsing errors."""
        for attempt in range(2):
            try:
                return self.llm_with_tools.invoke(messages, config)
            except ResponseError as e:
                if "error parsing tool call:" in str(e.error) and attempt < 1:
                    logging.warning(f"retrying due to tool call parsing error: {e.error}")
                    continue
                raise

    def _count_consecutive_tool_rounds(self, state: AgentState) -> int:
        """Count consecutive AI messages with tool_calls since the last HumanMessage."""
        count = 0
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                break
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                count += 1
        return count

    def _cancel_remaining_tool_calls(
        self,
        remaining: list[dict],
        request_id: str,
        state: AgentState,
        message: str,
    ) -> list[ToolMessage]:
        """Create cancel ToolMessages for tool calls that will not be executed."""
        return [
            ToolMessage(
                content=message,
                name=tc["name"],
                tool_call_id=tc["id"],
                additional_kwargs={
                    "request_id": request_id,
                    "selected_agent": state.get("selected_agent", {}),
                    "confirmation": False,
                },
            )
            for tc in remaining
        ]

    def _dispatch_preprocessed_ui_tools(
        self, state: AgentState, config: RunnableConfig, tools: list[dict],
    ) -> None:
        """Dispatch preprocessed UI tools (e.g. show-yaml, show-yaml-diff).

        Checks that the request metadata contains a UI tools config name before
        dispatching.
        """
        request_metadata = config.get("configurable", {}).get("request_metadata", {})
        ui_tools_config = request_metadata.get("ui_tools", {})
        logging.debug(f"_dispatch_preprocessed_ui_tools: config={ui_tools_config}")

        name = ui_tools_config.get("name", "")
        if not name:
            logging.debug("UI tools config name is missing, skipping ui tools dispatch")
            return

        self._dispatch_ui_tools(tools)

    def _dispatch_ui_tools(self, tools: list[dict]) -> None:
        """Dispatch a ``ui_tools`` custom event with the given tools list."""
        try:
            ui_tools_json = json.dumps(tools)
            ui_tools_event = f"<ui-tools>{ui_tools_json}</ui-tools>"
            dispatch_custom_event("ui_tools", ui_tools_event)
            logging.debug(f"Dispatched {len(tools)} UI tool(s): {[t['toolName'] for t in tools]}")
        except Exception as e:
            logging.error(f"Error dispatching UI tools: {e}", exc_info=True)


# Backward-compatible alias (base.py used to export this name)
BaseAgentBuilder = ChildAgentBuilder


def create_child_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Checkpointer,
    agent_config: AgentConfig,
) -> CompiledStateGraph:
    """Create and compile a child agent graph."""
    builder = ChildAgentBuilder(
        llm, tools, system_prompt, checkpointer, agent_config,
    )
    return builder.build()


# -- Utility functions ---------------------------------------------------------


def build_agent_metadata(agent_name: str, selection_mode: str, extra_metadata: str = "") -> str:
    """Build a structured agent metadata string for custom events."""
    return (
        f'<agent-metadata>{{"agentName": "{agent_name}", '
        f'"selectionMode": "{selection_mode}"{extra_metadata}}}</agent-metadata>'
    )


def process_tool_result(tool_result: str | list, state: AgentState) -> tuple[str, str | None]:
    """Process the raw tool result, extracting UI context and doc links if present.

    Returns:
        ``(processed_result, mcp_response)`` where *mcp_response* is ``None`` if no uiContext.
    """
    mcp_response = None
    try:
        # Handle list format: [{"type": "text", "text": "..."}]
        if isinstance(tool_result, list) and tool_result:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if "uiContext" in json_result:
            mcp_response = f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>"
            dispatch_custom_event("ui_context", mcp_response)
        if "docLinks" in json_result:
            for link in json_result["docLinks"]:
                dispatch_custom_event("dock_link", f"<mcp-doclink>{link}</mcp-doclink>")

        llm_result = json_result.get("llm", json_result) if isinstance(json_result, dict) else json_result
        return convert_to_string_if_needed(llm_result), mcp_response
    except (json.JSONDecodeError, TypeError):
        return tool_result, mcp_response


def convert_to_string_if_needed(var):
    """Convert dicts and lists to JSON strings; pass through everything else."""
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    return var
