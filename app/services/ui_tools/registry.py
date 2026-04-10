"""
UI Tools Registry and definitions for managing available UI tools and their metadata for frontend rendering.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class UIToolsConfig:
    """UIToolsConfig spec-level configuration"""
    enabled: bool = True
    revision: int = 0
    system_prompt: Optional[str] = None
    max_tools: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "revision": self.revision,
            "system_prompt": self.system_prompt,
            "max_tools": self.max_tools,
        }

@dataclass
class UIToolSchema:
    """Schema definition for UI tool input validation"""
    type: str  # "object", "array", etc.
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UITool:
    """Definition of a UI tool that can be rendered by the frontend"""
    name: str
    description: str
    prompt: str
    category: str
    schema: UIToolSchema
    metadata: Dict[str, Any] = field(default_factory=dict)
    revision: int = 0
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "category": self.category,
            "schema": self.schema.to_dict(),
            "metadata": self.metadata,
            "revision": self.revision,
            "enabled": self.enabled,
        }


@dataclass
class UIToolCall:
    """Represents a UI tool invocation"""
    tool_name: str
    input: Dict[str, Any]
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UIToolsConfigData:
    """Container for tools and config"""
    config: Optional[UIToolsConfig] = None
    tools: Dict[str, UITool] = field(default_factory=dict)


class UIToolsRegistry:
    """Registry for managing available UI tools scoped by configuration"""
    
    def __init__(self):
        self.configs_map: Dict[str, UIToolsConfigData] = {}
        self.last_updated = datetime.now()
    
    def register_tools_config(self, tools: List[UITool], config: UIToolsConfig, config_name: str = None) -> None:
        """Register both tools and config, scoped by config name"""
        
        if config_name is None:
            config_name = "default"
        
        if config_name not in self.configs_map:
            self.configs_map[config_name] = UIToolsConfigData()
        
        # Register config
        self.configs_map[config_name].config = config
        
        # Register tools
        for tool in tools:
            self.configs_map[config_name].tools[tool.name] = tool

        self.last_updated = datetime.now()
        
    def get_tools_config(self, config_name: str) -> Optional[UIToolsConfigData]:
        """Get tools and config for a specific config name"""
        return self.configs_map.get(config_name)
    
    def get_all_tools(self, config_name: str) -> List[UITool]:
        """
        Get all registered tools, optionally filtered by config
        
        Args:
            config_name: Config name (returns all if None)
            
        Returns:
            List of UITool objects
        """
        if config_name:
            config_data = self.configs_map.get(config_name)
            return list(config_data.tools.values()) if config_data else []
        
        # Return all tools from all configs
        all_tools = []
        for config_data in self.configs_map.values():
            all_tools.extend(config_data.tools.values())
        return all_tools
    
    def get_tools_by_category(self, category: str, config_name: str = None) -> List[UITool]:
        """Get tools by category, optionally scoped by config"""
        tools = self.get_all_tools(config_name)
        return [tool for tool in tools if tool.category == category]
    
    def clear_tools(self, config_name: str = None) -> None:
        """
        Clear tools, optionally for a specific config_name
        
        Args:
            config_name: Clear only this config_name (clears all if None)
        """
        if config_name:
            self.configs_map.pop(config_name, None)
        else:
            self.configs_map.clear()
        self.last_updated = datetime.now()


# Global registry instance
_ui_tools_registry: Optional[UIToolsRegistry] = None


def get_ui_tools_registry() -> UIToolsRegistry:
    """Get or create the global UI tools registry"""
    global _ui_tools_registry
    if _ui_tools_registry is None:
        _ui_tools_registry = UIToolsRegistry()
    return _ui_tools_registry
