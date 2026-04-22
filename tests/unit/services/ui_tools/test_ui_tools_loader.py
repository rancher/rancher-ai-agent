"""
Unit tests for UI Tools Loader.
Tests loading and parsing UI tools from Kubernetes ConfigMaps.
"""
import pytest
import json
from app.services.ui_tools.loader import (
    _get_ui_tools_object,
    _load_ui_tools_data_from_config_map,
    _parse_ui_tool_config_definition,
    _parse_ui_tool_definition,
)
from app.services.ui_tools.models import UIToolsConfig


@pytest.fixture
def sample_config_map_data():
    """Create sample ConfigMap data for testing."""
    config_obj = {
        "config": {
            "enabled": True,
            "revision": 1,
            "systemPrompt": "Test system prompt",
            "maxTools": 5
        },
        "tools": [
            {
                "name": "test-tool",
                "description": "A test tool",
                "prompt": "Test prompt",
                "category": "selector",
                "schema": {
                    "type": "object",
                    "properties": {
                        "resource": {"type": "string", "description": "Resource name"}
                    },
                    "required": ["resource"],
                    "description": "Test schema"
                },
                "metadata": {
                    "enum": {
                        "resource": {
                            "option1": {"label": "Option 1"},
                            "option2": {"label": "Option 2", "requires": ["action"]}
                        }
                    }
                },
                "revision": 0,
                "enabled": True
            }
        ]
    }
    
    return {
        "config": json.dumps(config_obj)
    }


class TestGetUIToolsObject:
    """Test _get_ui_tools_object function."""
    
    def test_get_valid_ui_tools_object(self):
        """Test extracting valid UI tools object from ConfigMap data."""
        config_obj = {"key": "value"}
        data = {"config": json.dumps(config_obj)}
        
        result = _get_ui_tools_object(data)
        assert result == config_obj
    
    def test_get_ui_tools_object_missing_config(self):
        """Test when config key is missing."""
        data = {"other": "value"}
        result = _get_ui_tools_object(data)
        assert result is None
    
    def test_get_ui_tools_object_empty_config(self):
        """Test when config value is empty."""
        data = {"config": ""}
        result = _get_ui_tools_object(data)
        assert result is None
    
    def test_get_ui_tools_object_invalid_json(self):
        """Test when config is invalid JSON."""
        data = {"config": "not valid json"}
        result = _get_ui_tools_object(data)
        assert result is None
    
    def test_get_ui_tools_object_non_dict_json(self):
        """Test when config JSON is not a dictionary."""
        data = {"config": json.dumps(["array", "not", "dict"])}
        result = _get_ui_tools_object(data)
        assert result is None


class TestParseUIToolConfigDefinition:
    """Test _parse_ui_tool_config_definition function."""
    
    def test_parse_config_with_all_fields(self):
        """Test parsing config with all fields."""
        config_data = {
            "enabled": True,
            "revision": 2,
            "systemPrompt": "Custom prompt",
            "maxTools": 10
        }
        
        config = _parse_ui_tool_config_definition(config_data)
        
        assert config.enabled is True
        assert config.revision == 2
        assert config.system_prompt == "Custom prompt"
        assert config.max_tools == 10
    
    def test_parse_config_with_defaults(self):
        """Test parsing config with missing fields uses defaults."""
        config_data = {}
        
        config = _parse_ui_tool_config_definition(config_data)
        
        assert config.enabled is True
        assert config.revision == 0
        assert config.system_prompt is None
        assert config.max_tools == 5
    
    def test_parse_config_partial_fields(self):
        """Test parsing config with only some fields."""
        config_data = {
            "enabled": False,
            "maxTools": 15
        }
        
        config = _parse_ui_tool_config_definition(config_data)
        
        assert config.enabled is False
        assert config.revision == 0
        assert config.system_prompt is None
        assert config.max_tools == 15


class TestParseUIToolDefinition:
    """Test _parse_ui_tool_definition function."""
    
    def test_parse_valid_tool_definition(self):
        """Test parsing a valid tool definition."""
        tool_def = {
            "name": "my-tool",
            "description": "My Tool",
            "prompt": "Tool prompt",
            "category": "selector",
            "schema": {
                "type": "object",
                "properties": {"field": {"type": "string"}},
                "required": ["field"]
            },
            "metadata": {"key": "value"},
            "revision": 1,
            "enabled": True
        }
        
        tool = _parse_ui_tool_definition(tool_def)
        
        assert tool is not None
        assert tool.name == "my-tool"
        assert tool.description == "My Tool"
        assert tool.prompt == "Tool prompt"
        assert tool.category == "selector"
        assert tool.metadata == {"key": "value"}
        assert tool.revision == 1
        assert tool.enabled is True
    
    def test_parse_tool_missing_required_fields(self):
        """Test parsing tool with missing required fields."""
        tool_def = {
            "name": "tool",
            "description": "Description"
            # Missing prompt
        }
        
        tool = _parse_ui_tool_definition(tool_def)
        assert tool is None
    
    def test_parse_tool_missing_name(self):
        """Test parsing tool with missing name."""
        tool_def = {
            "description": "Description",
            "prompt": "Prompt",
            "category": "selector"
        }
        
        tool = _parse_ui_tool_definition(tool_def)
        assert tool is None
    
    def test_parse_tool_defaults(self):
        """Test parsing tool with minimal fields uses defaults."""
        tool_def = {
            "name": "minimal-tool",
            "description": "Description",
            "prompt": "Prompt"
        }
        
        tool = _parse_ui_tool_definition(tool_def)
        
        assert tool is not None
        assert tool.category == "custom"  # Default category
        assert tool.metadata == {}
        assert tool.revision == 0
        assert tool.enabled is True
    
    def test_parse_tool_empty_description(self):
        """Test parsing tool with empty description."""
        tool_def = {
            "name": "tool",
            "description": "",
            "prompt": "Prompt"
        }
        
        tool = _parse_ui_tool_definition(tool_def)
        assert tool is None


class TestLoadUIToolsDataFromConfigMap:
    """Test _load_ui_tools_data_from_config_map function."""
    
    def test_load_valid_configmap_data(self, sample_config_map_data):
        """Test loading valid ConfigMap data."""
        tools, config = _load_ui_tools_data_from_config_map(sample_config_map_data)
        
        assert len(tools) == 1
        assert tools[0].name == "test-tool"
        assert config.enabled is True
        assert config.revision == 1
    
    def test_load_configmap_data_none(self):
        """Test loading when data is None."""
        tools, config = _load_ui_tools_data_from_config_map(None)
        
        assert len(tools) == 0
        assert isinstance(config, UIToolsConfig)
    
    def test_load_configmap_empty_data(self):
        """Test loading when data is empty."""
        tools, config = _load_ui_tools_data_from_config_map({})
        
        assert len(tools) == 0
        assert isinstance(config, UIToolsConfig)
    
    def test_load_configmap_invalid_json_config(self):
        """Test loading when config field has invalid JSON."""
        data = {"config": "invalid json"}
        
        tools, config = _load_ui_tools_data_from_config_map(data)
        
        assert len(tools) == 0
        assert isinstance(config, UIToolsConfig)
    
    def test_load_configmap_multiple_tools(self):
        """Test loading ConfigMap with multiple tools."""
        config_obj = {
            "config": {
                "enabled": True,
                "revision": 1
            },
            "tools": [
                {
                    "name": "tool1",
                    "description": "Tool 1",
                    "prompt": "Prompt 1",
                    "category": "selector"
                },
                {
                    "name": "tool2",
                    "description": "Tool 2",
                    "prompt": "Prompt 2",
                    "category": "selector"
                }
            ]
        }
        
        data = {"config": json.dumps(config_obj)}
        tools, config = _load_ui_tools_data_from_config_map(data)
        
        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"
    
    def test_load_configmap_partial_tool_data(self):
        """Test loading ConfigMap when some tools are invalid."""
        config_obj = {
            "config": {"enabled": True},
            "tools": [
                {
                    "name": "valid-tool",
                    "description": "Valid",
                    "prompt": "Prompt",
                    "category": "selector"
                },
                {
                    "name": "invalid-tool"
                    # Missing required fields
                }
            ]
        }
        
        data = {"config": json.dumps(config_obj)}
        tools, config = _load_ui_tools_data_from_config_map(data)
        
        # Should only include valid tool
        assert len(tools) == 1
        assert tools[0].name == "valid-tool"
    
    def test_load_configmap_no_tools_array(self):
        """Test loading ConfigMap without tools array."""
        config_obj = {
            "config": {"enabled": True}
            # No tools field
        }
        
        data = {"config": json.dumps(config_obj)}
        tools, config = _load_ui_tools_data_from_config_map(data)
        
        assert len(tools) == 0
        assert config.enabled is True
