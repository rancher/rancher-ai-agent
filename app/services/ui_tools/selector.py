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

from ..agent.loader import AgentConfig
from .models import UITool, UIToolCall
from .validator import create_ui_tools_validator

def filter_tool(ui_tool: UITool, ui_tools_selectors: list[str]) -> bool:
    """
    Checks if a ui_tool is enabled and the tool name is included in the ui_tools_selectors list.
    Args:
        ui_tool: The UI tool to check.
        ui_tools_selectors: List of tool names to select from the available UI tools list.
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
        
        # Handle array types: extract the item type from the items schema
        if field_type == 'array':
            items_schema = field_schema.get('items', {})
            items_type = items_schema.get('type', 'string')
            items_python_type = type_map.get(items_type, str)
            python_type = list[items_python_type]  # Use generic list syntax for array of items
        
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
        
        self.max_tools = max_tools if max_tools > 0 else None
        
        self._build_system_prompt(system_prompt)
    
    def _build_default_system_prompt(self) -> str:
        """Get the default system prompt for the selector"""
        prompt = """You are a UI component selector. Your role is to select appropriate UI tools to enhance responses.

KEY PRINCIPLE:
Read each tool's description carefully - it contains critical USE ONLY WHEN and NOT FOR guidance that determines when to use that tool.
Tools are designed for specific scenarios: some for SINGLE resources, some for LISTS/COLLECTIONS. Mismatching tool to scenario breaks UX.

CORE RULES:
1. SINGLE RESOURCE context (e.g., 'pod-xyz', 'deployment-web-app'): Use tools marked for single resources
2. MULTIPLE RESOURCES context (e.g., 'list of pods', 'show all deployments'): Use tools marked for lists/collections, NOT single-resource tools
3. DETECTION: If tempted to call the same tool 2+ times with different resource names → you're in a list scenario, use a list tool instead
4. YAML CONTENT HANDLING (CRITICAL): When passing YAML to UI tools, you MUST:
   - YAML content MUST be fetched from the context or [MCP result payloads] - NEVER make the LLM generate YAML on its own, it will be error-prone and break the UI
   - Preserve EXACT indentation and line breaks from the original YAML
   - Keep all whitespace exactly as it appears
   - Do NOT convert YAML to JSON, TOML, or any other format
   - Do NOT reformat, re-indent, or normalize the YAML
   - Do NOT add/remove/modify any lines
   
   EXAMPLE - DO THIS:
     yaml: |
       apiVersion: v1
       kind: Pod
       metadata:
         name: my-pod
         namespace: default
   
   EXAMPLE - DON'T DO THIS:
     ✗ Converting to JSON: {"apiVersion":"v1","kind":"Pod",...}
     ✗ Re-indenting: (different spacing than source)
     ✗ Minifying: entire YAML on one line
     ✗ Line additions: adding comments or extra metadata

5. TASK-TOOL COMPLEXITY MATCHING: Select tools whose complexity and scope match the task's needs.
   - For simple clarifications: use minimal/lightweight tools that provide focused information
   - For complex investigations: use more sophisticated tools capable of deeper analysis
   - Avoid cognitive overkill: don't burden the user with powerful tools for trivial questions
   - Avoid underfitting: don't use basic tools when the task demands comprehensive visualization or advanced capabilities

6. RESPECT EXPLICIT USER INTERACTION FLOWS: If the assistant message explicitly requests user input (e.g., 'Please provide the cluster name...', 'Please specify the namespace...', 'Need more details about...'):
   - DO use tools that facilatate or offers assistance in providing the requested information (e.g., search/discovery tools, input forms, selection inputs, guided wizards)
   - DO NOT provide tools that would bypass or anticipate this requested information

   
   EXAMPLE - DON'T DO THIS:
     ✗ Assistant says: 'Please provide the cluster name to show more details'
     ✗ You provide: 'list-all-clusters' tool to bypass the ask
   
   EXAMPLE - DO THIS:
     ✓ Assistant says: 'Please provide the cluster name'
     ✓ You either: provide NO tools (let user respond naturally)
     ✓ Or provide: a search/discovery tool to help them find and specify the cluster name

7. REAL VALUES ONLY - NEVER PLACEHOLDER TEXT (CRITICAL): When calling ANY tool with parameters:
   - EVERY parameter value MUST be extracted from the context, MCP results, or assistant message
   - NEVER use placeholder values like 'option1', 'option2', 'option3', 'name1', 'value1', 'resource'
   - NEVER use template defaults, field descriptions, or generic text as actual values
   
   EXAMPLE - DON'T DO THIS:
     ✗ Assistant asks: 'Which pod do you want to debug?'
     ✗ You call the tool with: 'option1', 'option2', 'option3'
   
   EXAMPLE - DO THIS:
     ✓ Assistant asks: 'Which pod do you want to debug?'
     ✓ From MCP data, you have: pods=['nginx-pod', 'web-server-pod', 'api-pod']
     ✓ You call the tool with: 'nginx-pod', 'web-server-pod', 'api-pod'

Each tool's description explains its scope. Follow it strictly."""
  
        if self.max_tools:
            prompt += f"\n\nIMPORTANT LIMIT: You can select at most {self.max_tools} different UI tool(s) by unique name (you may call the same tool with different parameters). If more are appropriate, select only the {self.max_tools} most important unique tool(s)."
            
        return prompt
    
    def _build_system_prompt(self, system_prompt: str) -> str:
        """Get the system prompt for the selector, merging default guidance with any custom prompt provided"""
        default_prompt = self._build_default_system_prompt()
        if system_prompt and system_prompt.strip():
            self.system_prompt = f"{system_prompt}\n\n{default_prompt}"
        else:
            self.system_prompt = default_prompt
            
    def _build_text_prompt(self, agent_config: AgentConfig, context: str, mcp_response: Optional[str]) -> str:
        """Build the text prompt for the LLM based on the agent context and MCP response if available"""
        return f"""Analyze this CONTEXT + MCP RESPONSE (if available) + the SELECTED AGENT used to perform the task, and select appropriate UI tools to enhance the response.

SELECTED AGENT: name: {agent_config.displayName}, description: "{agent_config.description}"

CONTEXT:
{context}

{f'MCP RESPONSE:{chr(10)}{mcp_response}' if mcp_response else ''}

If no tools are appropriate, do not invoke any tools."""

    def select_tools(
        self,
        agent_config: AgentConfig, # The agent used to permorm the user request, used to scope tool selection and for better prompt guidance
        context: str,  # Current response/context from agent
        mcp_response: Optional[str] = None,  # Raw MCP response if available for better tool selection
        available_tools: Optional[List[UITool]] = None,
    ) -> List[UIToolCall]:
        """
        Use any LLM to select appropriate UI tools using bind_tools for structured output.
        
        Args:
            agent_config: The agent configuration using this selector
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
            
            # Build the prompt with the agent, context and MCP response if available            
            text_prompt = self._build_text_prompt(agent_config, context, mcp_response)
            
            user_msg = HumanMessage(content=text_prompt)
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
        
        # Validate tool calls
        validator = create_ui_tools_validator()

        validated_tools = validator.validate_tool_calls(ui_tool_calls, available_tools)
        logging.debug(f"Validated {len(validated_tools)} tool calls out of {len(ui_tool_calls)}")
                
        # Deduplicate by {toolName, input}
        seen = set()
        deduplicated_tools = []
        
        for tool_call in validated_tools:
            tool_name = tool_call["toolName"]
            input_json = json.dumps(tool_call["input"], sort_keys=True)
            dedup_key = (tool_name, input_json)
            
            if dedup_key not in seen:
                seen.add(dedup_key)
                deduplicated_tools.append(tool_call)
        
        logging.debug(f"Deduplicated {len(validated_tools)} tool calls to {len(deduplicated_tools)}")
        
        # Cap the results to maxTools if specified (by unique tool names)
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
            
            logging.debug(f"Capped to {len(capped_list)} tool calls, {len(unique_tool_names)} unique tools")
            return capped_list
        
        return deduplicated_tools


def create_ui_tools_selector(llm: BaseLanguageModel, system_prompt: str, max_tools: int = 5) -> UIToolsSelector:
    """Factory function to create a UI Tools Selector with any LLM"""
    return UIToolsSelector(llm, system_prompt, max_tools)
