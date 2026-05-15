"""
UI tools validation logic for the Rancher AI Agent.
This module defines the UIToolCallValidator class,
which provides methods to validate UI tool calls based on their defined schemas and categories.
"""
import logging
from typing import List, Optional

from .models import UITool, UIToolCall, UIToolSchema, UIToolCategory


class UIToolCallValidator:
    """
    Validator class for UIToolCall objects.
    """
    
    def validate_tool_calls(self, calls: List[UIToolCall], available_tools: List[UITool]) -> List[dict]:
        """
        Validates a list of tool calls against the available tools and their schemas.
        
        Args:
            calls: List of UIToolCall objects to validate
            available_tools: List of available UITool objects for schema reference
        Returns:
            List of valid UIToolCall objects (invalid calls are filtered out)
        """
        valid_calls = []
        for call in calls:
            if self._validate_tool(call, available_tools):
                valid_calls.append({
                    "toolName": call.tool_name.strip(),
                    "input": call.input,
                })
            else:
                logging.warning(f"Rejected invalid UI tool call: {call.tool_name} with input {call.input}")

        return valid_calls

    def _validate_tool(self, call: UIToolCall, available_tools: List[UITool]) -> bool:
        """
        Validates a single tool call.
        
        Args:
            call: The tool call to validate
            available_tools: List of available tools for schema lookup
            
        Returns:
            True if the tool call is valid, False otherwise
        """
        # Check: tool_name exists and is not empty
        if not isinstance(call.tool_name, str) or not call.tool_name.strip():
            logging.warning(f"Invalid/empty tool_name: {call.tool_name}")
            return False
        
        # Check: input is a dict
        if call.input is None:
            call.input = {}
        elif not isinstance(call.input, dict):
            logging.warning(f"UI tool call '{call.tool_name}' has non-dict input: {type(call.input)}")
            return False
        
        # Get tool definition
        tool = next((t for t in available_tools if t.name == call.tool_name), None)
        if not tool:
            logging.warning(f"UI tool call '{call.tool_name}' refers to an unavailable tool.")
            return False
        
        schema_validator = SchemaValidator(tool)
        if not schema_validator.validate(call):
            return False
        
        category_validator = CategoryValidator(tool)
        if not category_validator.validate(call):
            return False

        return True


class SchemaValidator:
    """
    Validates tool call inputs against their defined schemas.
    """
    def __init__(self, tool: UITool):
        self.tool = tool
    
    def validate(self, call: UIToolCall) -> bool:
        """
        Validate a tool call's input against its schema.
        
        Args:
            call: The tool call to validate
            
        Returns:
            True if the input is valid according to the schema, False otherwise
        """
        
        # Check required fields
        if not self._check_required_fields(call, self.tool.schema):
            return False
        
        # Check enum fields
        if not self._check_enum_fields(call, self.tool.schema):
            return False
        
        return True
    
    def _check_required_fields(self, call: UIToolCall, schema: Optional[UIToolSchema] = None) -> bool:
        """
        Check if required fields are present and not empty in the tool call input.
        """
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

    def _check_enum_fields(self, call: UIToolCall, schema: Optional[UIToolSchema] = None) -> bool:
        """
        Check if enum field values are valid according to metadata definitions.
        """
        enum_metadata = self.tool.metadata.get('enum', {})
        
        if not enum_metadata:
            return True

        if schema and hasattr(schema, 'properties'):
            properties = schema.properties
        
            # Check each property for enum type
            for field_name, field_schema in properties.items():
                field_type = field_schema.get('type', 'string') if isinstance(field_schema, dict) else None
                is_enum = field_type == 'string' and 'enum' in field_schema if isinstance(field_schema, dict) else False
                
                # Skip if not an enum field or field not in input
                if not is_enum or field_name not in call.input:
                    continue
                
                # Get the enum definition from metadata
                enum_definition = enum_metadata.get(field_name)
                if not enum_definition or not isinstance(enum_definition, dict):
                    continue
                
                field_value = call.input[field_name]
                if not isinstance(field_value, str):
                    logging.warning(f"UI tool call '{call.tool_name}' enum field '{field_name}' must be a string, got {type(field_value)}")
                    return False
                
                # Check if value is one of the valid enum options
                valid_options = list(enum_definition.keys())
                if field_value not in valid_options:
                    valid_list = ", ".join([f"'{opt}'" for opt in valid_options])
                    logging.warning(f"UI tool call '{call.tool_name}' enum field '{field_name}' has invalid value '{field_value}'. Valid options are: {valid_list}")
                    return False
                
                # Check if the selected enum option has required fields
                selected_option = enum_definition[field_value]
                if isinstance(selected_option, dict):
                    required_fields = selected_option.get('requires', [])
                    if required_fields and isinstance(required_fields, list):
                        for required_field in required_fields:
                            if required_field not in call.input:
                                logging.warning(f"UI tool call '{call.tool_name}' option '{field_value}' requires field '{required_field}', but it is missing from input")
                                return False
                            elif call.input[required_field] == "" or call.input[required_field] is None:
                                logging.warning(f"UI tool call '{call.tool_name}' option '{field_value}' requires field '{required_field}', but it is empty")
                                return False
        
        return True

class CategoryValidator:
    """
    Category-specific validation rules for UI tool calls.
    """
    def __init__(self, tool: UITool):
        self.tool = tool
    
    def validate(self, call: UIToolCall) -> bool:
        """
        Validate a tool call based on its category.
        
        Args:
            call: The tool call to validate
            
        Returns:
            True if the tool call is valid for the category, False otherwise
        """
        if self.tool.category == UIToolCategory.SELECTOR.value:
            # Additional category rules cab be implemented here.
            return self._check_duplicate_values(call, self.tool.schema)
        
        # Additional categories can go here
        
        return True
    
    def _check_duplicate_values(self, call: UIToolCall, schema: Optional[UIToolSchema] = None) -> bool:
        """
        Check if there are duplicate values in the tool call input after normalizing (removing final digit).
        """
        if schema and hasattr(schema, 'properties'):
            properties = schema.properties
            str_values = []

            for field_name in properties.keys():
                if field_name in call.input and isinstance(call.input[field_name], str):
                    str_values.append(call.input[field_name])

            # Normalize values by removing final digit and converting to lowercase
            normalized_values = []
            for value in str_values:
                # Remove last character if it's a digit
                if value and value[-1].isdigit():
                    normalized = value[:-1].lower().strip()
                else:
                    normalized = value.lower().strip()
                normalized_values.append(normalized)

            # Check if more than 1 option is a known placeholder
            known_placeholders = ['option', 'name', 'value', 'item', 'element', 'placeholder', 'field']

            placeholder_count = sum(1 for norm_val in normalized_values if norm_val in known_placeholders)
            if placeholder_count > 1:
                logging.warning(f"UI tool call '{call.tool_name}' has {placeholder_count} placeholder patterns: '{', '.join(normalized_values)}' (rejecting)")
                return False

            # Check if 2 or more options are the same
            if len(normalized_values) >= 2 and len(set(normalized_values)) < len(normalized_values):
                logging.warning(f"UI tool call '{call.tool_name}' has duplicate values after removing final digit: '{', '.join(set(normalized_values))}' (rejecting)")
                return False
            
        return True


def create_ui_tools_validator() -> UIToolCallValidator:
    """
    Factory function to create a UI tool call validator.
    """
    return UIToolCallValidator()