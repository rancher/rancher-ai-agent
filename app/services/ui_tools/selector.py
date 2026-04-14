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
        
        # Patch ui_tool description for enum fields
        is_enum = field_type == 'string' and 'enum' in field_schema
        if is_enum:       
            # Get enum description from metadata if available
            enum_fields = ui_tool.metadata.get('enum', {}).get(field_name) if ui_tool.metadata else None
            if enum_fields and isinstance(enum_fields, dict):
                field_description += " Options:"
                for enum_name, enum_schema in enum_fields.items():
                    requires = enum_schema.get('requires')
                    requires_description = f" - requires: {requires}" if requires else ""
                    field_description += f" '{enum_name}' ({enum_schema.get('description', '')}{requires_description}),"
                field_description += ". ONLY these options are valid for this field. DO NOT allow any values other than these."
        
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
        
        self.max_tools = max_tools if max_tools > 0 else None
        
        # Merge system_prompt with default if both are provided
        default_prompt = self._get_default_system_prompt()
        if system_prompt and system_prompt.strip():
            self.system_prompt = f"{system_prompt}\n\n{default_prompt}"
        else:
            self.system_prompt = default_prompt
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the selector"""
        prompt = """You are a UI component selector. Your role is to analyze the current response 
and select the most appropriate UI tools from the available list to enhance the user experience.

GENERAL GUIDELINES:

When selecting UI tools:
- Analyze the content and the mcp response (if available) to understand what information is being presented and what the user might need to interact with it effectively
- Choose tools that will best visualize or present the information
- You can recommend multiple tools if they complement each other
- Match the complexity of the tool to the task
- If the assistant message contains additional requests for the user (e.g. "Please provide more details / cluster name / namespace, etc."), prioritize tools that enable that interaction and DO NOT provide tools that anticipate information that the assistant is explicitly asking the user to provide in follow-up messages.
  - For example, if the assistant message is "Please provide the cluster name to view more details", do NOT provide a show YAML tool that shows a resource YAML content

IMPORTANT RULES:

Some tools are designed for specific resource requests (e.g. a tool for viewing a single Kubernetes resource) vs other tools are designed for viewing lists of resources.
  - When the user asks for a list of resources or the assistant message presents a list of resources, DO NOT create individual tool calls for each resource.
    - For example, if the user message is "Show me the pods in the cluster", do NOT create a separate tool call for each pod. Instead, provide a single tool that can help the user to display the list of pods and their details.
    - For example, if the assistant message is "Here are the pods in the cluster: pod1, pod2, pod3", do NOT create 3 separate tool calls for each pod. Instead, provide a single tool that can help to display the list of pods and their details.
  - When the user specifically asks for a single resource or the assistant message presents information about a single resource, it is appropriate to create a tool call for that specific resource.
    - For example, if the user message is "Show me the details of pod1", it is appropriate to create a tool call for pod1 that shows its YAML or details.
When passing YAML content to UI tools:
  - Maintain proper indentation and line breaks from the original YAML
  - Do not convert YAML to JSON or any other format
  - Do not modify or re-indent the YAML content"""
  
        if self.max_tools:
            prompt += f"\n\nIMPORTANT LIMIT: You can select at most {self.max_tools} different UI tool(s) by unique name (you may call the same tool with different parameters). If more are appropriate, select only the {self.max_tools} most important unique tool(s)."
            
        return prompt

    def select_tools(
        self,
        context: str,  # Current response/context from agent
        mcp_response: Optional[str] = None,  # Raw MCP response if available for better tool selection
        available_tools: Optional[List[UITool]] = None,
    ) -> List[UIToolCall]:
        """
        Use any LLM to select appropriate UI tools using bind_tools for structured output.
        
        Args:
            context: The response/context to enhance with UI tools
            mcp_response: Raw MCP response if available for better tool selection
            available_tools: List of available tools (uses all if not specified)
            
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
            
            # Build the prompt with context and MCP response if available            
            prompt_text = f"""Analyze this context + mcp response (if available) and select appropriate UI tools to enhance the response.

CONTEXT:
{context}

{f'MCP RESPONSE:{chr(10)}{mcp_response}' if mcp_response else ''}

If no tools are appropriate, do not invoke any tools."""
            
            user_msg = HumanMessage(content=prompt_text)
            system_msg = SystemMessage(content=self.system_prompt)
            
            # Call the LLM with bound tools
            response = llm_with_tools.invoke([system_msg, user_msg])
            
            # Extract tool calls
            ui_tool_calls = self._extract_tool_calls_from_response(response, available_tools)
            logging.debug(f"UI tool selection result: {len(ui_tool_calls)} UI tools selected before validation")
            
            # Sanitize and validate tool calls against schema
            ui_tools_list = self._sanitize_ui_tools(ui_tool_calls, available_tools)
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
                logging.debug("No tool_calls found in response")
        
        except Exception as e:
            logging.error(f"Error extracting tool calls from response: {e}")
        
        return ui_tool_calls

    def _is_tool_call_valid(self, call: UIToolCall, available_tools: List[UITool]) -> bool:
        """
        Validates a single tool call against its schema.
        
        Args:
            call: The tool call to validate
            available_tools: List of available tools for schema lookup
            
        Returns:
            True if the tool call is valid, False otherwise
        """
        # Sanity check: tool_name exists and is not empty
        if not isinstance(call.tool_name, str) or not call.tool_name.strip():
            logging.warning(f"Invalid/empty tool_name: {call.tool_name}")
            return False
        
        # Sanity check: input is a dict
        if call.input is None:
            call.input = {}
        elif not isinstance(call.input, dict):
            logging.warning(f"UI tool call '{call.tool_name}' has non-dict input: {type(call.input)}")
            return False
        
        # Schema validation: check for required fields and empty values
        tool_schemas = {tool.name: tool.schema for tool in available_tools}
        schema = tool_schemas.get(call.tool_name)
        
        if schema and hasattr(schema, 'properties'):
            properties = schema.properties
            
            # Check each property for required flag
            for field_name, field_schema in properties.items():
                # Check if this field is marked as required
                is_required = field_schema.get('required', False) if isinstance(field_schema, dict) else False
                
                if is_required:
                    # Check if field is present and not empty
                    if field_name not in call.input:
                        logging.warning(f"UI tool call '{call.tool_name}' is missing required field: {field_name}")
                        return False
                    elif call.input[field_name] == "" or call.input[field_name] is None:
                        logging.warning(f"UI tool call '{call.tool_name}' has empty required field: {field_name}")
                        return False
        
        return True

    def _sanitize_ui_tools(self, ui_tool_calls: List[UIToolCall], available_tools: List[UITool]) -> list:
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
            available_tools: List of available tools for schema validation
            
        Returns:
            List of deduplicated and capped tool calls in state format
        """
        
        # Phase 1: Validate tool calls against schema and required fields
        validated_tools = []
        for call in ui_tool_calls:
            if self._is_tool_call_valid(call, available_tools):
                validated_tools.append({
                    "toolName": call.tool_name.strip(),
                    "input": call.input,
                })
                
        logging.debug(f"Phase 1: Validated {len(validated_tools)} tool calls out of {len(ui_tool_calls)}")
                
        # Phase 2: Deduplicate by {toolName, input}
        seen = set()
        deduplicated_tools = []
        
        for tool_call in validated_tools:
            tool_name = tool_call["toolName"]
            input_json = json.dumps(tool_call["input"], sort_keys=True)
            dedup_key = (tool_name, input_json)
            
            if dedup_key not in seen:
                seen.add(dedup_key)
                deduplicated_tools.append(tool_call)
        
        logging.debug(f"Phase 2: Deduplicated {len(validated_tools)} tool calls to {len(deduplicated_tools)}")
        
        # Phase 3: Cap the results to maxTools if specified (by unique tool names)
        if self.max_tools:
            unique_tool_names = set()
            capped_list = []
            
            for tool_call in deduplicated_tools:
                tool_name = tool_call["toolName"]
                if tool_name not in unique_tool_names and len(unique_tool_names) < self.max_tools:
                    unique_tool_names.add(tool_name)
                    capped_list.append(tool_call)
                elif tool_name in unique_tool_names:
                    capped_list.append(tool_call)
            
            logging.debug(f"Phase 3: Capped to {len(capped_list)} tool calls, {len(unique_tool_names)} unique tools")
            return capped_list
        
        return deduplicated_tools


def create_ui_tools_selector(llm: BaseLanguageModel, system_prompt: str, max_tools: int = 5) -> UIToolsSelector:
    """Factory function to create a UI Tools Selector with any LLM"""
    return UIToolsSelector(llm, system_prompt, max_tools)
