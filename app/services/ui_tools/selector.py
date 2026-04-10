"""
UI Tools Agent selector
This module implements a generic UI tools selector that works with the LLM provider.
"""

import json
import logging
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.llms import BaseLanguageModel

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
        Use any LLM to select appropriate UI tools
        
        Args:
            context: The response/context to enhance with UI tools
            available_tools: List of available tools (uses all if not specified)
            conversation_context: Additional context for selection
            
        Returns:
            List of recommended UI tool calls
        """
        if available_tools is None:
            # Get all available tools (no config scoping at selector level)
            available_tools = self.registry.get_all_tools()
        
        if not available_tools:
            logging.warning("No UI tools available for selection")
            return []
        
        # Build tools description
        tools_description = self._format_available_tools(available_tools)
        
        # Build messages for the LLM
        system_msg = SystemMessage(
            content=f"{self.system_prompt}\n\n{tools_description}"
        )
        
        max_tools_instruction = ""
        if self.max_tools:
            max_tools_instruction = f"\n\nIMPORTANT LIMIT: You can select at most {self.max_tools} different UI tool(s) by unique name (you may call the same tool with different parameters). If more are appropriate, select only the {self.max_tools} most important unique tool(s)."
        
        prompt_text = f"""Analyze this context and select appropriate UI tools.

CONTEXT:
{context}

{f'CONVERSATION CONTEXT:{chr(10)}{conversation_context}' if conversation_context else ''}

Based on the context, select which UI tools would best enhance this response.{max_tools_instruction}

IMPORTANT INSTRUCTIONS:
1. Only select tools from the available list above
2. For each tool, provide input parameters that EXACTLY match the tool's schema
3. Use the field names shown in the "Expected Input Format" section - do NOT use different field names
4. Include all REQUIRED fields marked with (REQUIRED) in your input
5. Optional fields can be omitted if not needed
6. Output a valid JSON array at the very end with this format:
[
  {{"toolName": "tool_name", "input": {{"field1": "value1", "field2": "value2"}}}},
  {{"toolName": "another_tool", "input": {{"requiredField": "value"}}}}
]

If no tools are appropriate, output an empty array: []

Output the JSON array on its own line at the end."""
        
        user_msg = HumanMessage(content=prompt_text)
        
        try:
            # Call the LLM (works with any provider)
            logging.debug(f"Calling LLM for UI tool selection. Available tools: {[t.name for t in available_tools]}")
            response = self.llm.invoke([system_msg, user_msg])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract tool calls
            ui_tool_calls = self._extract_ui_tool_calls(response_text, available_tools)
            logging.debug(f"UI tool selection result: {len(ui_tool_calls)} UI tools selected before validation")
            
            # Sanitize and validate tool calls against schema
            ui_tools_list = self.sanitize_ui_tools(ui_tool_calls, available_tools)
            logging.debug(f"UI tool selection result after validation: {len(ui_tools_list)} valid UI tools")
            return ui_tools_list
            
        except Exception as e:
            logging.error(f"Error selecting UI tools: {e}")
            return []
    
    def _format_available_tools(self, tools: List[UITool]) -> str:
        """Format available tools for the system prompt with schema details"""
        tools_list = []
        for tool in tools:
            tool_info = f"- **{tool.name}** ({tool.category}): {tool.prompt}"
            
            # Extract properties from tool.schema object
            properties = tool.schema.properties
            
            if properties:
                props_desc = []
                required_fields = []
                
                # Build properties description and collect required fields
                for prop_name, prop_schema in properties.items():
                    prop_type = prop_schema.get('type', 'unknown')
                    prop_desc = prop_schema.get('description', '')
                    is_required = prop_schema.get('required', False)
                    if is_required:
                        required_fields.append(prop_name)
                    required_mark = " (REQUIRED)" if is_required else " (optional)"
                    props_desc.append(f"    - {prop_name} ({prop_type}){required_mark}: {prop_desc}")
                
                if props_desc:
                    tool_info += "\n  Input Schema:\n" + "\n".join(props_desc)
                
                # Add JSON example showing expected input format
                if required_fields or any(properties.values()):
                    tool_info += "\n  Expected Input Format (JSON):\n  {"
                    
                    # Show required fields first with example values
                    if required_fields:
                        for field_name in required_fields:
                            field_type = properties.get(field_name, {}).get('type', 'string')
                            if field_type == 'string':
                                tool_info += f'\n    "{field_name}": "value",'
                            elif field_type == 'boolean':
                                tool_info += f'\n    "{field_name}": true,'
                            elif field_type == 'number' or field_type == 'integer':
                                tool_info += f'\n    "{field_name}": 0,'
                            elif field_type == 'array':
                                tool_info += f'\n    "{field_name}": [],'
                            elif field_type == 'object':
                                tool_info += f'\n    "{field_name}": {{}},'
                        
                        # Show optional fields as comment
                        optional_fields = [name for name in properties.keys() if name not in required_fields]
                        if optional_fields:
                            tool_info += f'\n    // optional: {", ".join(optional_fields)}'
                    
                    tool_info = tool_info.rstrip(',') + "\n  }"
            
            tools_list.append(tool_info)
        
        return f"\n\nAvailable UI Tools:\n" + "\n".join(tools_list)
    
    def _extract_ui_tool_calls(
        self,
        response_text: str,
        available_tools: List[UITool],
    ) -> List[UIToolCall]:
        """
        Extract UI tool calls from the LLM response
        Looks for JSON-formatted tool calls with robust parsing.
        If full array parsing fails, attempts to extract individual valid tool objects.
        """
        ui_tool_calls: List[UIToolCall] = []

        logging.debug(f"LLM response for UI tool extraction: {response_text[:500]}...")
        
        try:
            import re
            
            # Strategy 1: Look for JSON array with [ ] markers (target arrays like [{...}, {...}])
            bracket_pattern = r'\[\s*\{.*?\}\s*\]'
            bracket_matches = re.findall(bracket_pattern, response_text, re.DOTALL)
            
            for json_str in bracket_matches:
                try:
                    tool_calls_data = json.loads(json_str)
                    if isinstance(tool_calls_data, list):
                        for call_data in tool_calls_data:
                            if isinstance(call_data, dict) and "toolName" in call_data:
                                tool_name = call_data.get("toolName")
                                # Validate the tool exists
                                if any(t.name == tool_name for t in available_tools):
                                    logging.debug(f"Extracted UI tool call: {tool_name}")
                                    ui_tool_calls.append(
                                        UIToolCall(
                                            tool_name=tool_name,
                                            input=call_data.get("input", {}),
                                        )
                                    )
                                else:
                                    logging.debug(f"UI tool '{tool_name}' not found in available tools: {[t.name for t in available_tools]}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON array: {json_str[:100]}... Error: {e}")
                    logging.debug("Attempting to extract individual tool objects from the malformed array")
                    # Try to extract individual tool objects even if the full array is malformed
                    individual_matches = re.findall(r'\{"toolName"[^}]*\}', json_str)
                    if individual_matches:
                        for obj_str in individual_matches:
                            try:
                                call_data = json.loads(obj_str)
                                if isinstance(call_data, dict) and "toolName" in call_data:
                                    tool_name = call_data.get("toolName")
                                    if any(t.name == tool_name for t in available_tools):
                                        logging.debug(f"Extracted individual UI tool object from malformed array: {tool_name}")
                                        ui_tool_calls.append(
                                            UIToolCall(
                                                tool_name=tool_name,
                                                input=call_data.get("input", {}),
                                            )
                                        )
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to parse individual UI tool object: {obj_str[:100]}... Error: {e}")
                                continue
                    continue
            
            # Strategy 2: If no arrays found, look for individual objects with toolName
            if not ui_tool_calls:
                logging.debug("No JSON arrays found, trying to find individual UI tool objects")
                object_pattern = r'\{"toolName"\s*:\s*"[^"]*"[^}]*"input"\s*:\s*\{[^}]*\}[^}]*\}'
                obj_matches = re.findall(object_pattern, response_text)
                
                for obj_str in obj_matches:
                    try:
                        call_data = json.loads(obj_str)
                        if isinstance(call_data, dict) and "toolName" in call_data:
                            tool_name = call_data.get("toolName")
                            if any(t.name == tool_name for t in available_tools):
                                logging.debug(f"Extracted UI tool object: {tool_name}")
                                ui_tool_calls.append(
                                    UIToolCall(
                                        tool_name=tool_name,
                                        input=call_data.get("input", {}),
                                    )
                                )
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse individual UI tool object: {obj_str[:100]}... Error: {e}")
                        continue
            
            logging.debug(f"Total UI tools extracted: {len(ui_tool_calls)}")
            
        except Exception as e:
            logging.error(f"Error extracting UI tool calls: {e}")
        
        return ui_tool_calls

    def _validate_tool_input(self, tool_name: str, input_data: dict, available_tools: List[UITool]) -> bool:
        """
        Validate that the generated input matches the tool's schema definition.
        
        Validation Rules:
        - ALL required fields MUST be present in input_data
        - NO unexpected fields are allowed (only fields defined in properties)
        - All provided fields MUST match the expected type
        
        Args:
            tool_name: Name of the tool
            input_data: The input dict generated by the LLM
            available_tools: List of available tools with schemas
            
        Returns:
            True if input is valid according to tool schema, False otherwise
        """
        # Find the tool definition
        tool_def = next((t for t in available_tools if t.name == tool_name), None)
        if not tool_def:
            logging.warning(f"Tool '{tool_name}' definition not found")
            return False

        # Extract properties from tool.schema object
        properties = tool_def.schema.properties
        
        # Build required fields list by checking each property's required field
        required_fields = [name for name, prop in properties.items() if prop.get('required', False)]
        
        # Validation 1: Check ALL required fields are present
        missing_required = [field for field in required_fields if field not in input_data]
        if missing_required:
            logging.warning(
                f"Tool '{tool_name}' validation failed: missing required fields {missing_required}. "
                f"Expected: {required_fields}, got: {list(input_data.keys())}"
            )
            return False
        
        # Validation 2: Check NO unexpected fields are present
        unexpected_fields = [field for field in input_data.keys() if field not in properties]
        if unexpected_fields:
            logging.warning(
                f"Tool '{tool_name}' validation failed: unexpected fields {unexpected_fields}. "
                f"Valid fields are: {list(properties.keys())}"
            )
            return False
        
        # Validation 3: Check type of each provided field matches schema
        for field_name, field_value in input_data.items():
            expected_type = properties[field_name].get('type', 'string')
            
            # Map JSON schema types to Python types
            type_mapping = {
                'string': str,
                'number': (int, float),
                'integer': int,
                'boolean': bool,
                'object': dict,
                'array': list,
            }
            
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type and not isinstance(field_value, expected_python_type):
                logging.warning(
                    f"Tool '{tool_name}' validation failed: field '{field_name}' has wrong type. "
                    f"Expected {expected_type} but got {type(field_value).__name__} with value: {field_value}"
                )
                return False
        
        logging.debug(f"Tool '{tool_name}' input validation passed: {list(input_data.keys())}")
        return True

    def sanitize_ui_tools(self, ui_tool_calls: List[UIToolCall], available_tools: Optional[List[UITool]] = None) -> list:
        """
        Sanitize UI tool calls by validating, deduplicating and converting to state format.
        
        Deduplication Logic:
        - Removes duplicate tool calls based on exact match of {toolName, input}
        - If the LLM returned the same tool with identical parameters twice, only the first occurrence is kept
        - This ensures no redundant tool calls are sent to the frontend
        
        Performs sanity checks:
        - Validates tool_name is not empty and is a string
        - Validates input is a dict (not None)
        - Validates input matches the tool's schema definition
        
        Args:
            ui_tool_calls: List of tool calls from the selector
            available_tools: List of tool definitions for schema validation
            
        Returns:
            List of validated and deduplicated tool calls in state format
        """
        if available_tools is None:
            # Get all available tools (no config scoping at sanitize level)
            available_tools = self.registry.get_all_tools()
        
        # Phase 1: Validate each tool call
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
                logging.warning(f"UI tool call '{call.tool_name}' at index {idx} has None input, using empty dict")
                call.input = {}
            elif not isinstance(call.input, dict):
                logging.warning(f"UI tool call '{call.tool_name}' at index {idx} has non-dict input: {type(call.input)}")
                validation_removed_count += 1
                continue
            
            # Schema validation: check if input matches tool definition
            if not self._validate_tool_input(call.tool_name, call.input, available_tools):
                logging.warning(f"UI tool call '{call.tool_name}' at index {idx} has invalid input according to schema")
                validation_removed_count += 1
                continue
            
            validated_tools.append({
                "toolName": call.tool_name.strip(),
                "input": call.input,
            })
        
        if validation_removed_count > 0:
            logging.debug(f"Validation removed {validation_removed_count} invalid UI tool call(s), {len(validated_tools)} valid UI tool(s) remaining")
        
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
