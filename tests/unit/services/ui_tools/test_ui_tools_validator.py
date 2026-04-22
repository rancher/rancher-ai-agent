"""
Unit tests for UI Tools Validator functionality.
Tests validation of tool calls against schemas, categories, enums, and required fields.
"""
import pytest
from app.services.ui_tools.models import (
    UITool,
    UIToolSchema,
    UIToolCall,
    UIToolCategory,
)
from app.services.ui_tools.validator import (
    UIToolCallValidator,
    SchemaValidator,
    CategoryValidator,
    create_ui_tools_validator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_ui_tool():
    """Create a simple UI tool for testing."""
    schema = UIToolSchema(
        type="object",
        properties={
            "name": {"type": "string", "description": "Resource name", "required": True},
            "namespace": {"type": "string", "description": "Kubernetes namespace"}
        },
        required=["name"]
    )
    return UITool(
        name="show-yaml",
        description="Display resource in YAML format",
        prompt="Show resource YAML",
        category="viewer",
        schema=schema,
        metadata={},
        enabled=True
    )


@pytest.fixture
def enum_ui_tool():
    """Create a UI tool with enum fields for testing."""
    schema = UIToolSchema(
        type="object",
        properties={
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["create", "update", "delete"]
            },
            "name": {
                "type": "string", 
                "description": "Resource name",
                "required": True
            }
        },
        required=["name"]
    )
    metadata = {
        "enum": {
            "action": {
                "create": {"description": "Create resource"},
                "update": {"description": "Update resource", "requires": ["version"]},
                "delete": {"description": "Delete resource"}
            }
        }
    }
    return UITool(
        name="manage-resource",
        description="Manage resources",
        prompt="Manage a resource",
        category="editor",
        schema=schema,
        metadata=metadata,
        enabled=True
    )


@pytest.fixture
def selector_ui_tool():
    """Create a UI tool in selector category for testing."""
    schema = UIToolSchema(
        type="object",
        properties={
            "resource": {"type": "string", "description": "Resource type"},
            "target": {"type": "string", "description": "Target"}
        },
        required=[]
    )
    return UITool(
        name="select-resource",
        description="Select resources",
        prompt="Select a resource",
        category=UIToolCategory.SELECTOR.value,
        schema=schema,
        metadata={},
        enabled=True
    )


@pytest.fixture
def validator():
    """Create a UI tool validator."""
    return UIToolCallValidator()


# ============================================================================
# UIToolCallValidator Tests
# ============================================================================

class TestUIToolCallValidator:
    """Test the main UIToolCallValidator class."""
    
    def test_validate_tool_calls_single_valid_call(self, validator, simple_ui_tool):
        """Test validating a single valid tool call."""
        call = UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 1
        assert result[0]["toolName"] == "show-yaml"
        assert result[0]["input"]["name"] == "nginx-pod"
    
    def test_validate_tool_calls_multiple_valid_calls(self, validator, simple_ui_tool):
        """Test validating multiple valid tool calls."""
        calls = [
            UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"}),
            UIToolCall(tool_name="show-yaml", input={"name": "apache-pod"}),
        ]
        
        result = validator.validate_tool_calls(calls, [simple_ui_tool])
        
        assert len(result) == 2
        assert all(call["toolName"] == "show-yaml" for call in result)
    
    def test_validate_tool_calls_invalid_tool_name(self, validator, simple_ui_tool):
        """Test validating call with non-existent tool."""
        call = UIToolCall(tool_name="unknown-tool", input={"name": "pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 0
    
    def test_validate_tool_calls_empty_tool_name(self, validator, simple_ui_tool):
        """Test validating call with empty tool name."""
        call = UIToolCall(tool_name="", input={"name": "pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 0
    
    def test_validate_tool_calls_invalid_input_type(self, validator, simple_ui_tool):
        """Test validating call with non-dict input."""
        call = UIToolCall(tool_name="show-yaml", input="not-a-dict")
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 0
    
    def test_validate_tool_calls_none_input(self, validator, simple_ui_tool):
        """Test validating call with None input (should be converted to empty dict)."""
        call = UIToolCall(tool_name="show-yaml", input=None)
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        # Should be valid with empty input if no required fields
        assert len(result) == 0  # Because "name" is required
    
    def test_validate_tool_calls_mixed_valid_invalid(self, validator, simple_ui_tool):
        """Test validating mixed valid and invalid calls."""
        calls = [
            UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"}),
            UIToolCall(tool_name="unknown-tool", input={"name": "pod"}),
            UIToolCall(tool_name="show-yaml", input={"name": "apache-pod"}),
        ]
        
        result = validator.validate_tool_calls(calls, [simple_ui_tool])
        
        assert len(result) == 2
        assert all(call["toolName"] == "show-yaml" for call in result)
    
    def test_validate_tool_calls_empty_list(self, validator, simple_ui_tool):
        """Test validating empty list of calls."""
        result = validator.validate_tool_calls([], [simple_ui_tool])
        
        assert len(result) == 0
    
    def test_validate_tool_calls_whitespace_tool_name(self, validator, simple_ui_tool):
        """Test validating call with whitespace-only tool name."""
        call = UIToolCall(tool_name="   ", input={"name": "pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 0
    
    def test_validate_tool_calls_strips_whitespace(self, validator, simple_ui_tool):
        """Test that whitespace in tool names prevents matching (not stripped on lookup)."""
        call = UIToolCall(tool_name="  show-yaml  ", input={"name": "pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        # Whitespace in tool name prevents finding the tool in available_tools
        assert len(result) == 0


# ============================================================================
# SchemaValidator Tests
# ============================================================================

class TestSchemaValidatorRequiredFields:
    """Test SchemaValidator._check_required_fields method."""
    
    def test_required_field_present_and_valid(self, simple_ui_tool):
        """Test validation passes when required field is present."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"})
        
        result = validator._check_required_fields(call, simple_ui_tool.schema)
        
        assert result is True
    
    def test_required_field_missing(self, simple_ui_tool):
        """Test validation fails when required field is missing."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"namespace": "default"})
        
        result = validator._check_required_fields(call, simple_ui_tool.schema)
        
        assert result is False
    
    def test_required_field_empty_string(self, simple_ui_tool):
        """Test validation fails when required field is empty string."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": ""})
        
        result = validator._check_required_fields(call, simple_ui_tool.schema)
        
        assert result is False
    
    def test_required_field_none_value(self, simple_ui_tool):
        """Test validation fails when required field is None."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": None})
        
        result = validator._check_required_fields(call, simple_ui_tool.schema)
        
        assert result is False
    
    def test_optional_field_missing(self, simple_ui_tool):
        """Test validation passes when optional field is missing."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"})
        
        result = validator._check_required_fields(call, simple_ui_tool.schema)
        
        assert result is True
    
    def test_multiple_required_fields(self):
        """Test validation with multiple required fields."""
        schema = UIToolSchema(
            type="object",
            properties={
                "name": {"type": "string", "required": True},
                "version": {"type": "string", "required": True},
                "description": {"type": "string"}
            }
        )
        tool = UITool(
            name="test-tool",
            description="Test",
            prompt="Test",
            category="viewer",
            schema=schema,
            metadata={},
            enabled=True
        )
        validator = SchemaValidator(tool)
        call = UIToolCall(tool_name="test-tool", input={"name": "test"})
        
        result = validator._check_required_fields(call, schema)
        
        assert result is False
    
    def test_no_required_fields(self):
        """Test validation passes when no fields are required."""
        schema = UIToolSchema(
            type="object",
            properties={
                "name": {"type": "string"},
                "version": {"type": "string"}
            }
        )
        tool = UITool(
            name="test-tool",
            description="Test",
            prompt="Test",
            category="viewer",
            schema=schema,
            metadata={},
            enabled=True
        )
        validator = SchemaValidator(tool)
        call = UIToolCall(tool_name="test-tool", input={})
        
        result = validator._check_required_fields(call, schema)
        
        assert result is True


class TestSchemaValidatorEnumFields:
    """Test SchemaValidator._check_enum_fields method."""
    
    def test_enum_valid_value(self, enum_ui_tool):
        """Test validation passes with valid enum value."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(tool_name="manage-resource", input={"action": "create", "name": "resource"})
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is True
    
    def test_enum_invalid_value(self, enum_ui_tool):
        """Test validation fails with invalid enum value."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(tool_name="manage-resource", input={"action": "invalid", "name": "resource"})
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is False
    
    def test_enum_non_string_value(self, enum_ui_tool):
        """Test validation fails when enum value is not string."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(tool_name="manage-resource", input={"action": 123, "name": "resource"})
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is False
    
    def test_enum_field_not_in_input(self, enum_ui_tool):
        """Test validation passes when enum field is not provided."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(tool_name="manage-resource", input={"name": "resource"})
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is True
    
    def test_enum_with_required_fields(self, enum_ui_tool):
        """Test enum validation with required dependent fields."""
        validator = SchemaValidator(enum_ui_tool)
        # "update" action requires "version" field
        call = UIToolCall(tool_name="manage-resource", input={"action": "update", "name": "resource"})
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is False
    
    def test_enum_with_required_fields_satisfied(self, enum_ui_tool):
        """Test enum validation when required dependent fields are satisfied."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(
            tool_name="manage-resource",
            input={"action": "update", "name": "resource", "version": "1.0"}
        )
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is True
    
    def test_enum_required_field_empty(self, enum_ui_tool):
        """Test enum validation fails when required dependent field is empty."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(
            tool_name="manage-resource",
            input={"action": "update", "name": "resource", "version": ""}
        )
        
        result = validator._check_enum_fields(call, enum_ui_tool.schema)
        
        assert result is False
    
    def test_no_enum_metadata(self, simple_ui_tool):
        """Test validation passes when no enum metadata exists."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": "pod"})
        
        result = validator._check_enum_fields(call, simple_ui_tool.schema)
        
        assert result is True


class TestSchemaValidatorValidateMethod:
    """Test SchemaValidator.validate method."""
    
    def test_validate_valid_call(self, simple_ui_tool):
        """Test validate method with valid call."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": "nginx-pod"})
        
        result = validator.validate(call)
        
        assert result is True
    
    def test_validate_invalid_required_field(self, simple_ui_tool):
        """Test validate method fails with invalid required field."""
        validator = SchemaValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"namespace": "default"})
        
        result = validator.validate(call)
        
        assert result is False
    
    def test_validate_invalid_enum_value(self, enum_ui_tool):
        """Test validate method fails with invalid enum value."""
        validator = SchemaValidator(enum_ui_tool)
        call = UIToolCall(tool_name="manage-resource", input={"action": "invalid", "name": "resource"})
        
        result = validator.validate(call)
        
        assert result is False


# ============================================================================
# CategoryValidator Tests
# ============================================================================

class TestCategoryValidatorDuplicateValues:
    """Test CategoryValidator._check_duplicate_values method."""
    
    def test_no_duplicates(self, selector_ui_tool):
        """Test validation passes when no duplicates."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "pod", "target": "node"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is True
    
    def test_exact_duplicate_values(self, selector_ui_tool):
        """Test validation fails with exact duplicate values."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "pod", "target": "pod"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is False
    
    def test_duplicate_after_normalization(self, selector_ui_tool):
        """Test validation fails when duplicates exist after removing final digit."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "pod1", "target": "pod2"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is False
    
    def test_multiple_placeholder_patterns(self, selector_ui_tool):
        """Test validation fails with multiple placeholder patterns."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "option1", "target": "option2"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is False
    
    def test_single_placeholder_allowed(self, selector_ui_tool):
        """Test validation passes with single placeholder pattern."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "option1", "target": "node1"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is True
    
    def test_case_insensitive_duplicates(self, selector_ui_tool):
        """Test validation detects duplicates case-insensitively."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "POD", "target": "pod"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is False
    
    def test_whitespace_normalization(self, selector_ui_tool):
        """Test that whitespace is normalized in duplicate detection."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": " pod ", "target": "pod"})
        
        result = validator._check_duplicate_values(call, selector_ui_tool.schema)
        
        assert result is False


class TestCategoryValidatorValidateMethod:
    """Test CategoryValidator.validate method."""
    
    def test_validate_selector_valid(self, selector_ui_tool):
        """Test validate method for selector category with valid input."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "pod", "target": "node"})
        
        result = validator.validate(call)
        
        assert result is True
    
    def test_validate_selector_invalid(self, selector_ui_tool):
        """Test validate method for selector category with invalid input."""
        validator = CategoryValidator(selector_ui_tool)
        call = UIToolCall(tool_name="select-resource", input={"resource": "pod1", "target": "pod2"})
        
        result = validator.validate(call)
        
        assert result is False
    
    def test_validate_non_selector_category(self, simple_ui_tool):
        """Test validate always passes for non-selector categories."""
        validator = CategoryValidator(simple_ui_tool)
        call = UIToolCall(tool_name="show-yaml", input={"name": "pod"})
        
        result = validator.validate(call)
        
        assert result is True


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateUIToolsValidator:
    """Test the factory function."""
    
    def test_create_validator_returns_instance(self):
        """Test factory creates a validator instance."""
        validator = create_ui_tools_validator()
        
        assert isinstance(validator, UIToolCallValidator)
    
    def test_create_validator_is_functional(self, simple_ui_tool):
        """Test the validator created by factory is functional."""
        validator = create_ui_tools_validator()
        call = UIToolCall(tool_name="show-yaml", input={"name": "pod"})
        
        result = validator.validate_tool_calls([call], [simple_ui_tool])
        
        assert len(result) == 1
    
    def test_multiple_factory_calls_create_new_instances(self):
        """Test factory creates new instances each time."""
        validator1 = create_ui_tools_validator()
        validator2 = create_ui_tools_validator()
        
        assert validator1 is not validator2
