"""
UI Tools Agent selector
This module implements a generic UI tools selector that works with the LLM provider.
"""

import json
import logging
from typing import Optional, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.tools import Tool
from pydantic import create_model, Field

from .registry import UITool, UIToolCall, get_ui_tools_registry


def filter_tool(ui_tool: UITool, ui_tools_selectors: list[str]) -> bool:
    """
    Checks if a ui_tool is enabled and the tool name is included in the ui_tools_selectors list.
    Args:
        ui_tool: The UI tool to check.
        ui_tools_selectors: List of tool names to select from the registry.
    Returns:
        True if the tool is enabled and matches the selectors, False otherwise.
    """
    if not ui_tool.enabled:
        logging.debug(f"Tool '{ui_tool.name}' is disabled, skipping")
        return False

    if not ui_tools_selectors or len(ui_tools_selectors) == 0:
        logging.debug(f"No UI tool selectors provided, including all enabled tools. Tool '{ui_tool.name}' is enabled and will be included.")
        return False
    
    for tool_name in ui_tools_selectors:
        if ui_tool.name == tool_name:
            logging.debug(f"Tool '{ui_tool.name}' matches selector '{tool_name}' and is enabled, including it.")
            return True
    
    return False


def ui_tool_to_langchain_tool(ui_tool: UITool) -> Tool:
    """
    Convert a UITool to a LangChain Tool with dynamic input schema.
    
    Args:
        ui_tool: The UI tool to convert
        
    Returns:
        A LangChain Tool ready to be bound to an LLM
    """
    # Type mapping from JSON schema types to Python types
    type_map = {
        'string': str,
        'number': float,
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
    }
    
    # Extract properties from the tool's schema
    properties = ui_tool.schema.properties
    
    # Build field definitions for the dynamic model
    field_definitions = {}
    for field_name, field_schema in properties.items():
        field_type = field_schema.get('type', 'string')
        python_type = type_map.get(field_type, str)
        field_description = field_schema.get('description', '')
        is_required = field_schema.get('required', False)
        
        # Create Field with description and default if not required
        if is_required:
            field_definitions[field_name] = (python_type, Field(..., description=field_description))
        else:
            field_definitions[field_name] = (python_type, Field(None, description=field_description))
    
    # Create dynamic Pydantic model for the tool's input schema
    if field_definitions:
        input_model = create_model(
            f"{ui_tool.name}Input",
            **field_definitions
        )
    else:
        # Empty model if no properties defined
        input_model = create_model(f"{ui_tool.name}Input")
    
    # Create and return the LangChain Tool
    return Tool(
        name=ui_tool.name,
        description=ui_tool.prompt,
        args_schema=input_model,
        func=lambda **kwargs: kwargs,  # Placeholder function - we only care about schema validation
    )


class UIToolsSelector:
    """
    Generic selector for UI tools that works with any LLM provider
    Added as step in the LangGraph workflow
    """
    
    def __init__(self, llm: BaseLanguageModel, system_prompt: str, max_tools: int = 5):
        """
        Initialize the UI Tools Selector
        
        Args:
            llm: llm provider to use for selection
            system_prompt: System prompt to guide the LLM in selecting tools
            max_tools: Maximum number of UI tools to select per response (0 = unlimited)
        """
        self.llm = llm
        self.registry = get_ui_tools_registry()
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_tools = max_tools if max_tools > 0 else None  # None means unlimited
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the selector"""
        return """You are a UI component selector. Your role is to analyze the current response 
and select the most appropriate UI tools from the available list to enhance the user experience.

When selecting UI tools:
- Analyze the context and the information being presented
- Choose tools that will best visualize or present the information
- You can recommend multiple tools if they complement each other
- Match the complexity of the tool to the task."""
    
    def select_tools(
        self,
        context: str,  # Current response/context from agent
        available_tools: Optional[List[UITool]] = None,
        conversation_context: Optional[str] = None,
    ) -> List[UIToolCall]:
        """
        Use any LLM to select appropriate UI tools using bind_tools for structured output.
        
        Args:
            context: The response/context to enhance with UI tools
            available_tools: List of available tools (uses all if not specified)
            conversation_context: Additional context for selection
            
        Returns:
            List of recommended UI tool calls
        """
        if not available_tools:
            logging.warning("No UI tools available for selection")
            return []
        
        try:
            # Convert UITools to LangChain Tool objects with proper schemas
            langchain_tools = [ui_tool_to_langchain_tool(tool) for tool in available_tools]
            
            # Bind tools to the LLM for structured tool calling
            llm_with_tools = self.llm.bind_tools(langchain_tools)
            
            logging.debug(f"Calling LLM for UI tool selection with bind_tools. Available tools: {[t.name for t in available_tools]}")
            
            # Build the prompt
            max_tools_instruction = ""
            if self.max_tools:
                max_tools_instruction = f"\n\nIMPORTANT LIMIT: You can select at most {self.max_tools} different UI tool(s) by unique name (you may call the same tool with different parameters). If more are appropriate, select only the {self.max_tools} most important unique tool(s)."
            
            prompt_text = f"""Analyze this context and select appropriate UI tools to enhance the response.

CONTEXT:
{context}

{f'CONVERSATION CONTEXT:{chr(10)}{conversation_context}' if conversation_context else ''}

Based on the context, invoke the most appropriate UI tools to enhance this response.{max_tools_instruction}

If no tools are appropriate, do not invoke any tools."""
            
            user_msg = HumanMessage(content=prompt_text)
            system_msg = SystemMessage(content=self.system_prompt)
            
            # Call the LLM with bound tools
            response = llm_with_tools.invoke([system_msg, user_msg])
            
            # Extract tool calls
            ui_tool_calls = self._extract_tool_calls_from_response(response, available_tools)
            logging.debug(f"UI tool selection result: {len(ui_tool_calls)} UI tools selected before validation")
            
            # Sanitize and validate tool calls against schema
            ui_tools_list = self._sanitize_ui_tools(ui_tool_calls)
            logging.debug(f"UI tool selection result after validation: {len(ui_tools_list)} valid UI tools")
            return ui_tools_list
            
        except Exception as e:
            logging.error(f"Error selecting UI tools with bind_tools: {e}")
            return []
    
    def _extract_tool_calls_from_response(self, response: Any, available_tools: List[UITool]) -> List[UIToolCall]:
        """
        Extract tool calls from the LLM response when using bind_tools.
        
        Args:
            response: The response from the LLM (AIMessage)
            available_tools: List of available tools for validation
            
        Returns:
            List of UIToolCall objects extracted from the response
        """
        ui_tool_calls = []
        
        try:
            # Check if response has tool_calls attribute (AIMessage from bind_tools)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name') or tool_call.get('tool_name')
                    tool_input = tool_call.get('args', tool_call.get('input', {}))
                    
                    # Validate tool exists in available tools
                    if any(t.name == tool_name for t in available_tools):
                        logging.debug(f"Extracted tool call: {tool_name} with input: {tool_input}")
                        ui_tool_calls.append(
                            UIToolCall(
                                tool_name=tool_name,
                                input=tool_input if isinstance(tool_input, dict) else {},
                            )
                        )
                    else:
                        logging.debug(f"Tool '{tool_name}' not found in available tools: {[t.name for t in available_tools]}")
            else:
                logging.debug("No tool_calls found in response, trying fallback JSON extraction from content")
                # Fallback to content extraction if tools are not structured
                if hasattr(response, 'content'):
                    ui_tool_calls = self._extract_ui_tool_calls(response.content, available_tools)
        
        except Exception as e:
            logging.error(f"Error extracting tool calls from response: {e}")
        
        return ui_tool_calls

    def _sanitize_ui_tools(self, ui_tool_calls: List[UIToolCall]) -> list:
        """
        Sanitize UI tool calls by deduplicating and capping to max_tools.
        
        Deduplication Logic:
        - Removes duplicate tool calls based on exact match of {toolName, input}
        - If the LLM returned the same tool with identical parameters twice, only the first occurrence is kept
        - This ensures no redundant tool calls are sent to the frontend
        
        Capping Logic:
        - If max_tools is set, limits to that many unique tool names
        - Same tool with different parameters still counts as the same unique tool
        
        Args:
            ui_tool_calls: List of tool calls from the selector
            
        Returns:
            List of deduplicated and capped tool calls in state format
        """        
        # Phase 1: Basic sanity checks (tool_name exists and is not empty)
        validated_tools = []
        validation_removed_count = 0
        
        for idx, call in enumerate(ui_tool_calls):
            # Sanity check: tool_name exists and is not empty
            if not isinstance(call.tool_name, str) or not call.tool_name.strip():
                logging.warning(f"UI tool call at index {idx} has invalid/empty tool_name: {call.tool_name}")
                validation_removed_count += 1
                continue
            
            # Sanity check: input is a dict
            if call.input is None:
                logging.debug(f"UI tool call '{call.tool_name}' at index {idx} has None input, using empty dict")
                call.input = {}
            elif not isinstance(call.input, dict):
                logging.warning(f"UI tool call '{call.tool_name}' at index {idx} has non-dict input: {type(call.input)}")
                validation_removed_count += 1
                continue
            
            validated_tools.append({
                "toolName": call.tool_name.strip(),
                "input": call.input,
            })
        
        if validation_removed_count > 0:
            logging.debug(f"Sanity checks removed {validation_removed_count} invalid UI tool call(s), {len(validated_tools)} valid UI tool(s) remaining")
        
        # Phase 2: Deduplicate by {toolName, input}
        seen = set()
        deduplicated_tools = []
        deduplication_removed_count = 0
        
        for tool_call in validated_tools:
            tool_name = tool_call["toolName"]
            input_json = json.dumps(tool_call["input"], sort_keys=True)
            
            # Create unique key based on toolName and input
            dedup_key = (tool_name, input_json)
            
            if dedup_key in seen:
                logging.debug(f"Duplicate removed: toolName='{tool_name}', input={input_json[:100]}...")
                deduplication_removed_count += 1
                continue
            
            seen.add(dedup_key)
            deduplicated_tools.append(tool_call)
        
        if deduplication_removed_count > 0:
            logging.debug(f"Deduplication removed {deduplication_removed_count} duplicate(s), {len(deduplicated_tools)} unique UI tool call(s) remaining")
        
        # Phase 3: Cap the results to maxTools if specified (by unique tool names)
        if self.max_tools:
            # Count unique tool names in the list
            unique_tool_names = set()
            capped_list = []
            
            for tool_call in deduplicated_tools:
                tool_name = tool_call["toolName"]
                # If we haven't seen this tool name yet and we have room, add it
                if tool_name not in unique_tool_names and len(unique_tool_names) < self.max_tools:
                    unique_tool_names.add(tool_name)
                    capped_list.append(tool_call)
                # If we've seen this tool name before, we can still add it (same tool, different params)
                elif tool_name in unique_tool_names:
                    capped_list.append(tool_call)
            
            if len(unique_tool_names) > self.max_tools or len(capped_list) != len(deduplicated_tools):
                logging.debug(f"Capping applied: {len(deduplicated_tools)} total selections reduced to {len(capped_list)} calls, unique tools limited to {len(unique_tool_names)}/{self.max_tools}")
            
            return capped_list
        
        return deduplicated_tools


def create_ui_tools_selector(llm: BaseLanguageModel, system_prompt: str, max_tools: int = 5) -> UIToolsSelector:
    """Factory function to create a UI Tools Selector with any LLM"""
    return UIToolsSelector(llm, system_prompt, max_tools)
