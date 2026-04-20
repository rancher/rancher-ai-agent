"""
Unit tests for UI Tools Registry.
Tests the UIToolsRegistry class and related data structures.
"""
import pytest
from datetime import datetime
from app.services.ui_tools.registry import (
    UIToolsConfig,
    UIToolSchema,
    UITool,
    UIToolCall,
    UIToolCategory,
    UIToolsRegistry,
    get_ui_tools_registry,
)


@pytest.fixture
def sample_schema():
    """Create a sample UIToolSchema for testing."""
    return UIToolSchema(
        type="object",
        properties={
            "resource": {"type": "string", "description": "Resource name"},
            "action": {"type": "string", "description": "Action to perform"}
        },
        required=["resource"],
        description="Sample tool schema"
    )


@pytest.fixture
def sample_tool(sample_schema):
    """Create a sample UITool for testing."""
    return UITool(
        name="test-tool",
        description="A test tool",
        prompt="Test tool prompt",
        category="selector",
        schema=sample_schema,
        metadata={"key": "value"},
        revision=1,
        enabled=True
    )


@pytest.fixture
def sample_config():
    """Create a sample UIToolsConfig for testing."""
    return UIToolsConfig(
        enabled=True,
        revision=1,
        system_prompt="Test system prompt",
        max_tools=5
    )


class TestUIToolsConfig:
    """Test UIToolsConfig dataclass."""
    
    def test_create_default_config(self):
        """Test creating a config with default values."""
        config = UIToolsConfig()
        assert config.enabled is True
        assert config.revision == 0
        assert config.system_prompt is None
        assert config.max_tools == 5
    
    def test_create_custom_config(self, sample_config):
        """Test creating a config with custom values."""
        assert sample_config.enabled is True
        assert sample_config.revision == 1
        assert sample_config.system_prompt == "Test system prompt"
        assert sample_config.max_tools == 5
    
    def test_config_to_dict(self, sample_config):
        """Test converting config to dictionary."""
        config_dict = sample_config.to_dict()
        assert config_dict["enabled"] is True
        assert config_dict["revision"] == 1
        assert config_dict["system_prompt"] == "Test system prompt"
        assert config_dict["max_tools"] == 5


class TestUIToolSchema:
    """Test UIToolSchema dataclass."""
    
    def test_create_schema(self, sample_schema):
        """Test creating a schema."""
        assert sample_schema.type == "object"
        assert "resource" in sample_schema.properties
        assert "action" in sample_schema.properties
        assert sample_schema.required == ["resource"]
        assert sample_schema.description == "Sample tool schema"
    
    def test_schema_to_dict(self, sample_schema):
        """Test converting schema to dictionary."""
        schema_dict = sample_schema.to_dict()
        assert schema_dict["type"] == "object"
        assert "resource" in schema_dict["properties"]
        assert schema_dict["required"] == ["resource"]


class TestUITool:
    """Test UITool dataclass."""
    
    def test_create_tool(self, sample_tool):
        """Test creating a tool."""
        assert sample_tool.name == "test-tool"
        assert sample_tool.description == "A test tool"
        assert sample_tool.prompt == "Test tool prompt"
        assert sample_tool.category == "selector"
        assert sample_tool.revision == 1
        assert sample_tool.enabled is True
    
    def test_tool_to_dict(self, sample_tool):
        """Test converting tool to dictionary."""
        tool_dict = sample_tool.to_dict()
        assert tool_dict["name"] == "test-tool"
        assert tool_dict["description"] == "A test tool"
        assert tool_dict["prompt"] == "Test tool prompt"
        assert tool_dict["category"] == "selector"
        assert tool_dict["metadata"] == {"key": "value"}
        assert isinstance(tool_dict["schema"], dict)


class TestUIToolCall:
    """Test UIToolCall dataclass."""
    
    def test_create_tool_call(self):
        """Test creating a tool call."""
        call = UIToolCall(
            tool_name="test-tool",
            input={"resource": "my-resource"}
        )
        assert call.tool_name == "test-tool"
        assert call.input == {"resource": "my-resource"}
        assert call.tool_call_id is None
    
    def test_create_tool_call_with_id(self):
        """Test creating a tool call with ID."""
        call = UIToolCall(
            tool_name="test-tool",
            input={"resource": "my-resource"},
            tool_call_id="call_123"
        )
        assert call.tool_call_id == "call_123"
    
    def test_tool_call_to_dict(self):
        """Test converting tool call to dictionary."""
        call = UIToolCall(
            tool_name="test-tool",
            input={"resource": "my-resource"},
            tool_call_id="call_123"
        )
        call_dict = call.to_dict()
        assert call_dict["tool_name"] == "test-tool"
        assert call_dict["input"] == {"resource": "my-resource"}
        assert call_dict["tool_call_id"] == "call_123"


class TestUIToolCategory:
    """Test UIToolCategory enum."""
    
    def test_selector_category(self):
        """Test SELECTOR category exists."""
        assert UIToolCategory.SELECTOR.value == "selector"
    
    def test_category_to_string(self):
        """Test converting category to string."""
        assert str(UIToolCategory.SELECTOR) == "selector"


class TestUIToolsRegistry:
    """Test UIToolsRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return UIToolsRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initializes empty."""
        assert len(registry.configs_map) == 0
        assert isinstance(registry.last_updated, datetime)
    
    def test_register_tools_config_default_name(self, registry, sample_tool, sample_config):
        """Test registering tools with default config name."""
        registry.register_tools_config([sample_tool], sample_config)
        
        assert "default" in registry.configs_map
        config_data = registry.configs_map["default"]
        assert config_data.config == sample_config
        assert "test-tool" in config_data.tools
        assert config_data.tools["test-tool"] == sample_tool
    
    def test_register_tools_config_custom_name(self, registry, sample_tool, sample_config):
        """Test registering tools with custom config name."""
        registry.register_tools_config([sample_tool], sample_config, config_name="custom-config")
        
        assert "custom-config" in registry.configs_map
        config_data = registry.configs_map["custom-config"]
        assert "test-tool" in config_data.tools
    
    def test_register_multiple_tools(self, registry, sample_schema, sample_config):
        """Test registering multiple tools."""
        tool1 = UITool(
            name="tool-1",
            description="Tool 1",
            prompt="Prompt 1",
            category="selector",
            schema=sample_schema
        )
        tool2 = UITool(
            name="tool-2",
            description="Tool 2",
            prompt="Prompt 2",
            category="selector",
            schema=sample_schema
        )
        
        registry.register_tools_config([tool1, tool2], sample_config)
        
        config_data = registry.configs_map["default"]
        assert len(config_data.tools) == 2
        assert "tool-1" in config_data.tools
        assert "tool-2" in config_data.tools
    
    def test_get_tools_config_existing(self, registry, sample_tool, sample_config):
        """Test getting existing tools config."""
        registry.register_tools_config([sample_tool], sample_config, config_name="test")
        
        config_data = registry.get_tools_config("test")
        assert config_data is not None
        assert config_data.config == sample_config
    
    def test_get_tools_config_nonexistent(self, registry):
        """Test getting non-existent tools config."""
        config_data = registry.get_tools_config("nonexistent")
        assert config_data is None
    
    def test_get_all_tools_with_config_name(self, registry, sample_tool, sample_config):
        """Test getting all tools by config name."""
        registry.register_tools_config([sample_tool], sample_config, config_name="test")
        
        tools = registry.get_all_tools("test")
        assert len(tools) == 1
        assert tools[0] == sample_tool
    
    def test_get_all_tools_without_config_name(self, registry, sample_schema, sample_config):
        """Test getting all tools without config name."""
        tool1 = UITool(
            name="tool-1",
            description="Tool 1",
            prompt="Prompt 1",
            category="selector",
            schema=sample_schema
        )
        tool2 = UITool(
            name="tool-2",
            description="Tool 2",
            prompt="Prompt 2",
            category="selector",
            schema=sample_schema
        )
        
        registry.register_tools_config([tool1], sample_config, config_name="config1")
        registry.register_tools_config([tool2], sample_config, config_name="config2")
        
        all_tools = registry.get_all_tools(None)
        assert len(all_tools) == 2
    
    def test_get_tools_by_category(self, registry, sample_schema, sample_config):
        """Test getting tools by category."""
        tool1 = UITool(
            name="tool-1",
            description="Tool 1",
            prompt="Prompt 1",
            category="selector",
            schema=sample_schema
        )
        tool2 = UITool(
            name="tool-2",
            description="Tool 2",
            prompt="Prompt 2",
            category="custom",
            schema=sample_schema
        )
        
        registry.register_tools_config([tool1, tool2], sample_config)
        
        selector_tools = registry.get_tools_by_category("selector")
        assert len(selector_tools) == 1
        assert selector_tools[0].name == "tool-1"
        
        custom_tools = registry.get_tools_by_category("custom")
        assert len(custom_tools) == 1
        assert custom_tools[0].name == "tool-2"
    
    def test_clear_tools_specific_config(self, registry, sample_tool, sample_config):
        """Test clearing tools for a specific config."""
        registry.register_tools_config([sample_tool], sample_config, config_name="test")
        assert "test" in registry.configs_map
        
        registry.clear_tools(config_name="test")
        assert "test" not in registry.configs_map
    
    def test_clear_tools_all(self, registry, sample_tool, sample_config):
        """Test clearing all tools."""
        registry.register_tools_config([sample_tool], sample_config, config_name="config1")
        registry.register_tools_config([sample_tool], sample_config, config_name="config2")
        
        assert len(registry.configs_map) == 2
        registry.clear_tools()
        assert len(registry.configs_map) == 0
    
    def test_last_updated_changes(self, registry, sample_tool, sample_config):
        """Test that last_updated timestamp changes on modifications."""
        initial_time = registry.last_updated
        
        # Wait a tiny bit and register
        import time
        time.sleep(0.01)
        registry.register_tools_config([sample_tool], sample_config)
        
        assert registry.last_updated > initial_time


class TestGlobalRegistry:
    """Test the global registry singleton."""
    
    def test_get_ui_tools_registry_singleton(self):
        """Test that get_ui_tools_registry returns the same instance."""
        registry1 = get_ui_tools_registry()
        registry2 = get_ui_tools_registry()
        
        assert registry1 is registry2
    
    def test_registry_persistence(self):
        """Test that changes persist across get_ui_tools_registry calls."""
        # Clear global registry first
        import app.services.ui_tools.registry as registry_module
        registry_module._ui_tools_registry = None
        
        # Get fresh registry
        registry1 = get_ui_tools_registry()
        
        config = UIToolsConfig()
        schema = UIToolSchema(type="object", properties={})
        tool = UITool(
            name="persistent-tool",
            description="Test",
            prompt="Test",
            category="selector",
            schema=schema
        )
        registry1.register_tools_config([tool], config)
        
        # Get registry again
        registry2 = get_ui_tools_registry()
        tools = registry2.get_all_tools(None)
        
        assert len(tools) == 1
        assert tools[0].name == "persistent-tool"
