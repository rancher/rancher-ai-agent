"""
Unit tests for UI Tools Selector functionality.
Tests tool filtering, conversion to LangChain tools, selector initialization,
prompt building, tool selection, and sanitization.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.ui_tools.selector import UIToolsSelector
from app.services.ui_tools.models import (
    UITool,
    UIToolSchema,
    UIToolCall,
)


@pytest.fixture
def sample_ui_tool():
    """Create a sample UI tool for testing."""
    schema = UIToolSchema(
        type="object",
        properties={
            "name": {"type": "string", "description": "Resource name"},
            "namespace": {"type": "string", "description": "Kubernetes namespace"},
            "replicas": {"type": "integer", "description": "Number of replicas"}
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
def disabled_ui_tool():
    """Create a disabled UI tool for testing."""
    schema = UIToolSchema(type="object", properties={}, required=[])
    return UITool(
        name="disabled-tool",
        description="Disabled tool",
        prompt="This is disabled",
        category="viewer",
        schema=schema,
        metadata={},
        enabled=False
    )


@pytest.fixture
def enum_ui_tool():
    """Create a UI tool with enum fields."""
    schema = UIToolSchema(
        type="object",
        properties={
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete"],
                "description": "Action to perform"
            }
        },
        required=["action"]
    )
    return UITool(
        name="enum-tool",
        description="Tool with enum field",
        prompt="Select action",
        category="selector",
        schema=schema,
        metadata={
            "enum": {
                "action": {
                    "create": {"description": "Create new resource", "requires": None},
                    "update": {"description": "Update existing resource", "requires": "name"},
                    "delete": {"description": "Delete resource", "requires": "name"}
                }
            }
        },
        enabled=True
    )


@pytest.fixture
def agent_config():
    """Create a mock agent config to avoid circular imports."""
    config = MagicMock()
    config.displayName = "Test Agent"
    config.description = "Test agent for UI tools"
    return config


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


# ============================================================================
# filter_tool Tests
# ============================================================================

class TestFilterTool:
    """Test filter_tool function."""
    
    def test_filter_tool_enabled_and_in_selectors(self, sample_ui_tool):
        """Test filtering when tool is enabled and in selectors."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(sample_ui_tool, ["show-yaml", "other-tool"])
        assert result is True
    
    def test_filter_tool_enabled_but_not_in_selectors(self, sample_ui_tool):
        """Test filtering when tool is enabled but not in selectors."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(sample_ui_tool, ["other-tool", "another-tool"])
        assert result is False
    
    def test_filter_tool_disabled(self, disabled_ui_tool):
        """Test filtering when tool is disabled."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(disabled_ui_tool, ["disabled-tool"])
        assert result is False
    
    def test_filter_tool_empty_selectors(self, sample_ui_tool):
        """Test filtering with empty selectors list."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(sample_ui_tool, [])
        assert result is False
    
    def test_filter_tool_none_selectors(self, sample_ui_tool):
        """Test filtering with None selectors."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(sample_ui_tool, None)
        assert result is False
    
    def test_filter_tool_case_sensitive(self, sample_ui_tool):
        """Test filtering is case-sensitive."""
        from app.services.ui_tools.selector import filter_tool
        result = filter_tool(sample_ui_tool, ["Show-YAML"])
        assert result is False


# ============================================================================
# ui_tool_to_langchain_tool Tests
# ============================================================================

class TestUIToolToLangchainTool:
    """Test ui_tool_to_langchain_tool function."""
    
    def test_convert_simple_tool(self, sample_ui_tool):
        """Test converting a simple UI tool to LangChain tool."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        langchain_tool = ui_tool_to_langchain_tool(sample_ui_tool)
        
        assert langchain_tool.name == "show-yaml"
        assert langchain_tool.description == "Show resource YAML"
        assert langchain_tool.args_schema is not None
    
    def test_convert_tool_with_string_field(self, sample_ui_tool):
        """Test converting tool with string field."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        langchain_tool = ui_tool_to_langchain_tool(sample_ui_tool)
        
        # Check schema has the fields
        schema = langchain_tool.args_schema.model_json_schema()
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
    
    def test_convert_tool_with_integer_field(self, sample_ui_tool):
        """Test converting tool with integer field."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        langchain_tool = ui_tool_to_langchain_tool(sample_ui_tool)
        
        schema = langchain_tool.args_schema.model_json_schema()
        assert "replicas" in schema["properties"]
        assert schema["properties"]["replicas"]["type"] == "integer"
    
    def test_convert_tool_with_required_fields(self, sample_ui_tool):
        """Test converting tool with required fields."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        langchain_tool = ui_tool_to_langchain_tool(sample_ui_tool)
        
        # The name field should exist in properties
        schema = langchain_tool.args_schema.model_json_schema()
        assert "name" in schema["properties"]
        # Verify the tool was created successfully
        assert langchain_tool is not None
        assert langchain_tool.name == "show-yaml"
    
    def test_convert_tool_with_enum_field(self, enum_ui_tool):
        """Test converting tool with enum field."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        langchain_tool = ui_tool_to_langchain_tool(enum_ui_tool)
        
        schema = langchain_tool.args_schema.model_json_schema()
        assert "action" in schema["properties"]
        # Check description includes options
        assert "Options:" in schema["properties"]["action"]["description"]
    
    def test_convert_tool_with_empty_schema(self):
        """Test converting tool with empty schema."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        schema = UIToolSchema(type="object", properties={}, required=[])
        tool = UITool(
            name="empty-tool",
            description="Empty tool",
            prompt="Empty",
            category="viewer",
            schema=schema,
            metadata={},
            enabled=True
        )
        
        langchain_tool = ui_tool_to_langchain_tool(tool)
        
        assert langchain_tool.name == "empty-tool"
        assert langchain_tool.args_schema is not None
    
    def test_convert_tool_with_different_types(self):
        """Test converting tool with different field types."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        schema = UIToolSchema(
            type="object",
            properties={
                "str_field": {"type": "string"},
                "int_field": {"type": "integer"},
                "float_field": {"type": "number"},
                "bool_field": {"type": "boolean"},
                "array_field": {"type": "array"},
                "obj_field": {"type": "object"}
            },
            required=[]
        )
        tool = UITool(
            name="types-tool",
            description="Tool with different types",
            prompt="Types",
            category="viewer",
            schema=schema,
            metadata={},
            enabled=True
        )
        
        langchain_tool = ui_tool_to_langchain_tool(tool)
        
        schema_json = langchain_tool.args_schema.model_json_schema()
        assert "str_field" in schema_json["properties"]
        assert "int_field" in schema_json["properties"]
        assert "float_field" in schema_json["properties"]
        assert "bool_field" in schema_json["properties"]
        assert "array_field" in schema_json["properties"]
        assert "obj_field" in schema_json["properties"]
    
    def test_convert_tool_with_array_of_strings(self):
        """Test converting tool with array of strings field (with items schema)."""
        from app.services.ui_tools.selector import ui_tool_to_langchain_tool
        schema = UIToolSchema(
            type="object",
            properties={
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords for autocomplete"
                }
            },
            required=["keywords"]
        )
        tool = UITool(
            name="autocomplete",
            description="Provides autocomplete suggestions",
            prompt="Select autocomplete keywords",
            category="selector",
            schema=schema,
            metadata={},
            enabled=True
        )
        
        langchain_tool = ui_tool_to_langchain_tool(tool)
        
        # Verify the tool was created with the array field
        assert langchain_tool.name == "autocomplete"
        schema_json = langchain_tool.args_schema.model_json_schema()
        assert "keywords" in schema_json["properties"]
        # Verify it's defined as an array type
        assert schema_json["properties"]["keywords"]["type"] == "array"


# ============================================================================
# UIToolsSelector Tests
# ============================================================================

class TestUIToolsSelectorInitialization:
    """Test UIToolsSelector initialization."""
    
    def test_selector_initialization(self, mock_llm):
        """Test selector initialization."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "Test prompt", max_tools=5)
        
        assert selector.llm == mock_llm
        assert selector.max_tools == 5
        assert selector.system_prompt is not None
        assert len(selector.system_prompt) > 0
    
    def test_selector_initialization_with_zero_max_tools(self, mock_llm):
        """Test selector with zero max_tools (unlimited)."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "Test prompt", max_tools=0)
        
        assert selector.max_tools is None
    
    def test_selector_initialization_with_custom_prompt(self, mock_llm):
        """Test selector with custom prompt."""
        from app.services.ui_tools.selector import UIToolsSelector
        custom_prompt = "Custom system prompt"
        selector = UIToolsSelector(mock_llm, custom_prompt, max_tools=5)
        
        assert custom_prompt in selector.system_prompt
        assert "UI component selector" in selector.system_prompt


class TestUIToolsSelectorPromptBuilding:
    """Test prompt building in UIToolsSelector."""
    
    def test_build_default_system_prompt(self, mock_llm):
        """Test default system prompt building."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        prompt = selector.system_prompt
        assert "UI component selector" in prompt
        assert "SINGLE RESOURCE context" in prompt
        assert "MULTIPLE RESOURCES context" in prompt
        assert "at most 5" in prompt
    
    def test_build_system_prompt_without_max_tools(self, mock_llm):
        """Test system prompt when max_tools is 0 (unlimited)."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=0)
        
        prompt = selector.system_prompt
        assert "at most" not in prompt
    
    def test_build_text_prompt(self, mock_llm, agent_config, sample_ui_tool):
        """Test text prompt building."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        context = "Showing pod nginx-pod"
        text_prompt = selector._build_text_prompt(agent_config, context, None)
        
        assert "Test Agent" in text_prompt
        assert "Showing pod nginx-pod" in text_prompt
        assert "CONTEXT:" in text_prompt
    
    def test_build_text_prompt_with_mcp_response(self, mock_llm, agent_config):
        """Test text prompt with MCP response."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        context = "Got resources"
        mcp_response = "MCP Response data"
        text_prompt = selector._build_text_prompt(agent_config, context, mcp_response)
        
        assert "MCP RESPONSE:" in text_prompt
        assert "MCP Response data" in text_prompt


class TestUIToolsSelectorToolSelection:
    """Test tool selection in UIToolsSelector."""
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_select_tools_success(
        self, mock_validator, mock_llm, agent_config, sample_ui_tool
    ):
        """Test successful tool selection."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        # Mock LLM response with tool calls
        response = MagicMock()
        response.tool_calls = [
            {"name": "show-yaml", "args": {"name": "my-pod", "namespace": "default"}}
        ]
        mock_llm.invoke.return_value = response
        
        # Mock validator
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "show-yaml", "input": {"name": "my-pod", "namespace": "default"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector.select_tools(agent_config, "test context", None, [sample_ui_tool])
        
        assert len(result) == 1
        assert result[0]["toolName"] == "show-yaml"
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_select_tools_no_available_tools(self, mock_validator, mock_llm, agent_config):
        """Test tool selection with no available tools."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        result = selector.select_tools(agent_config, "test context", None, [])
        
        assert result == []
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_select_tools_error_handling(self, mock_validator, mock_llm, agent_config, sample_ui_tool):
        """Test error handling in tool selection."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        result = selector.select_tools(agent_config, "test context", None, [sample_ui_tool])
        
        assert result == []
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_select_tools_no_tool_calls_in_response(
        self, mock_validator, mock_llm, agent_config, sample_ui_tool
    ):
        """Test when LLM response has no tool calls."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        response = MagicMock()
        response.tool_calls = []
        mock_llm.invoke.return_value = response
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = []
        mock_validator.return_value = mock_validator_instance
        
        result = selector.select_tools(agent_config, "test context", None, [sample_ui_tool])
        
        assert result == []


class TestExtractToolCalls:
    """Test tool call extraction from LLM response."""
    
    def test_extract_tool_calls_from_response(self, mock_llm, sample_ui_tool):
        """Test extracting tool calls from LLM response."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        response = MagicMock()
        response.tool_calls = [
            {"name": "show-yaml", "args": {"name": "pod-1", "namespace": "default"}}
        ]
        
        tool_calls = selector._extract_tool_calls_from_response(response, [sample_ui_tool])
        
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "show-yaml"
        assert tool_calls[0].input == {"name": "pod-1", "namespace": "default"}
    
    def test_extract_tool_calls_alternative_format(self, mock_llm, sample_ui_tool):
        """Test extracting tool calls with alternative format."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        response = MagicMock()
        response.tool_calls = [
            {"tool_name": "show-yaml", "input": {"name": "pod-1"}, "id": "call-1"}
        ]
        
        tool_calls = selector._extract_tool_calls_from_response(response, [sample_ui_tool])
        
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "show-yaml"
    
    def test_extract_tool_calls_filter_unknown_tools(self, mock_llm, sample_ui_tool):
        """Test filtering out unknown tools."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        response = MagicMock()
        response.tool_calls = [
            {"name": "show-yaml", "args": {"name": "pod-1"}},
            {"name": "unknown-tool", "args": {"param": "value"}}
        ]
        
        tool_calls = selector._extract_tool_calls_from_response(response, [sample_ui_tool])
        
        # Only valid tool should be returned
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "show-yaml"
    
    def test_extract_tool_calls_no_tool_calls(self, mock_llm, sample_ui_tool):
        """Test extracting when no tool calls present."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        response = MagicMock()
        response.tool_calls = None
        
        tool_calls = selector._extract_tool_calls_from_response(response, [sample_ui_tool])
        
        assert tool_calls == []


class TestSanitizeUITools:
    """Test sanitization of UI tool calls."""
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_sanitize_deduplicates_tools(self, mock_validator, mock_llm, sample_ui_tool):
        """Test deduplication of identical tool calls."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        tool_calls = [
            UIToolCall(tool_name="show-yaml", input={"name": "pod-1"}),
            UIToolCall(tool_name="show-yaml", input={"name": "pod-1"}),  # Duplicate
            UIToolCall(tool_name="show-yaml", input={"name": "pod-2"})
        ]
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "show-yaml", "input": {"name": "pod-1"}},
            {"toolName": "show-yaml", "input": {"name": "pod-1"}},  # Duplicate
            {"toolName": "show-yaml", "input": {"name": "pod-2"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector._sanitize_ui_tools(tool_calls, [sample_ui_tool])
        
        # Should have 2 items (duplicate removed)
        assert len(result) == 2
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_sanitize_caps_to_max_tools(self, mock_validator, mock_llm, sample_ui_tool):
        """Test capping to max_tools limit by unique tool names."""
        from app.services.ui_tools.selector import UIToolsSelector
        schema = UIToolSchema(type="object", properties={}, required=[])
        tool2 = UITool(
            name="edit-yaml", description="Edit YAML", prompt="Edit",
            category="editor", schema=schema, metadata={}, enabled=True
        )
        
        selector = UIToolsSelector(mock_llm, "", max_tools=2)
        
        # Create tools with 3 unique tool names (should be capped to 2)
        tools = [
            UIToolCall(tool_name="show-yaml", input={"name": "pod-1"}),
            UIToolCall(tool_name="edit-yaml", input={"name": "pod-2"}),
            UIToolCall(tool_name="describe-pod", input={"name": "pod-3"})
        ]
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "show-yaml", "input": {"name": "pod-1"}},
            {"toolName": "edit-yaml", "input": {"name": "pod-2"}},
            {"toolName": "describe-pod", "input": {"name": "pod-3"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector._sanitize_ui_tools(tools, [sample_ui_tool, tool2])
        
        # Should have 2 items (capped by 2 unique tool names)
        assert len(result) == 2
        assert result[0]["toolName"] == "show-yaml"
        assert result[1]["toolName"] == "edit-yaml"
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_sanitize_multiple_unique_tools_capped(self, mock_validator, mock_llm):
        """Test capping with multiple unique tool names."""
        from app.services.ui_tools.selector import UIToolsSelector
        schema = UIToolSchema(type="object", properties={}, required=[])
        tool1 = UITool(
            name="tool1", description="Tool 1", prompt="T1",
            category="viewer", schema=schema, metadata={}, enabled=True
        )
        tool2 = UITool(
            name="tool2", description="Tool 2", prompt="T2",
            category="viewer", schema=schema, metadata={}, enabled=True
        )
        
        selector = UIToolsSelector(mock_llm, "", max_tools=1)
        
        tools = [
            UIToolCall(tool_name="tool1", input={"param": "val1"}),
            UIToolCall(tool_name="tool1", input={"param": "val2"}),
            UIToolCall(tool_name="tool2", input={"param": "val3"})
        ]
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "tool1", "input": {"param": "val1"}},
            {"toolName": "tool1", "input": {"param": "val2"}},
            {"toolName": "tool2", "input": {"param": "val3"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector._sanitize_ui_tools(tools, [tool1, tool2])
        
        # Should have 2 items (tool1 with 2 different params), tool2 filtered out
        assert len(result) == 2
        assert all(item["toolName"] == "tool1" for item in result)
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_sanitize_empty_list(self, mock_validator, mock_llm, sample_ui_tool):
        """Test sanitizing empty tool list."""
        from app.services.ui_tools.selector import UIToolsSelector
        selector = UIToolsSelector(mock_llm, "", max_tools=5)
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = []
        mock_validator.return_value = mock_validator_instance
        
        result = selector._sanitize_ui_tools([], [sample_ui_tool])
        
        assert result == []


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateUIToolsSelector:
    """Test factory function for creating selector."""
    
    def test_create_selector_with_defaults(self, mock_llm):
        """Test creating selector with default parameters."""
        from app.services.ui_tools.selector import create_ui_tools_selector
        selector = create_ui_tools_selector(mock_llm, "Test prompt")
        
        assert isinstance(selector, UIToolsSelector)
        assert selector.llm == mock_llm
        assert selector.max_tools == 5
    
    def test_create_selector_with_custom_max_tools(self, mock_llm):
        """Test creating selector with custom max_tools."""
        from app.services.ui_tools.selector import create_ui_tools_selector
        selector = create_ui_tools_selector(mock_llm, "Test prompt", max_tools=10)
        
        assert selector.max_tools == 10
    
    def test_create_selector_with_unlimited_tools(self, mock_llm):
        """Test creating selector with unlimited tools."""
        from app.services.ui_tools.selector import create_ui_tools_selector
        selector = create_ui_tools_selector(mock_llm, "Test prompt", max_tools=0)
        
        assert selector.max_tools is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestUIToolsSelectorIntegration:
    """Integration tests for UIToolsSelector."""
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_full_workflow_single_tool_selection(
        self, mock_validator, mock_llm, agent_config, sample_ui_tool
    ):
        """Test full workflow of tool selection."""
        from app.services.ui_tools.selector import create_ui_tools_selector
        selector = create_ui_tools_selector(mock_llm, "Custom prompt", max_tools=5)
        
        # Mock LLM response
        response = MagicMock()
        response.tool_calls = [
            {"name": "show-yaml", "args": {"name": "nginx-pod", "namespace": "default"}}
        ]
        mock_llm.invoke.return_value = response
        
        # Mock validator
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "show-yaml", "input": {"name": "nginx-pod", "namespace": "default"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector.select_tools(
            agent_config,
            "Deploy nginx pod to default namespace",
            None,
            [sample_ui_tool]
        )
        
        assert len(result) == 1
        assert result[0]["toolName"] == "show-yaml"
        assert result[0]["input"]["name"] == "nginx-pod"
    
    @patch('app.services.ui_tools.selector.create_ui_tools_validator')
    def test_workflow_with_mcp_response(
        self, mock_validator, mock_llm, agent_config, sample_ui_tool
    ):
        """Test workflow with MCP response context."""
        from app.services.ui_tools.selector import create_ui_tools_selector
        selector = create_ui_tools_selector(mock_llm, "", max_tools=5)
        
        mcp_response = """
        Resources:
        - name: nginx-pod
          namespace: default
        - name: web-pod
          namespace: default
        """
        
        response = MagicMock()
        response.tool_calls = [
            {"name": "show-yaml", "args": {"name": "nginx-pod"}}
        ]
        mock_llm.invoke.return_value = response
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_tool_calls.return_value = [
            {"toolName": "show-yaml", "input": {"name": "nginx-pod"}}
        ]
        mock_validator.return_value = mock_validator_instance
        
        result = selector.select_tools(
            agent_config,
            "Show yaml for nginx pod",
            mcp_response,
            [sample_ui_tool]
        )
        
        assert len(result) == 1
        # Verify LLM was called (with mcp_response included)
        mock_llm.invoke.assert_called_once()
