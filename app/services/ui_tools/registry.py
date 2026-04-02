"""
UI Tools Registry and definitions for managing available UI tools and their metadata for frontend rendering.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime


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
    config_name: Optional[str] = None
    
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
            "config_name": self.config_name,
        }


@dataclass
class UIToolCall:
    """Represents a UI tool invocation"""
    tool_name: str
    input: Dict[str, Any]
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
class AvailableUITools:
    """Container for available UI tools"""
    tools: List[UITool]
    timestamp: datetime
    config: Optional[UIToolsConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools": [tool.to_dict() for tool in self.tools],
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.to_dict() if self.config else None,
        }


class UIToolsRegistry:
    """Registry for managing available UI tools scoped by configuration"""
    
    def __init__(self):
        # Structure: {config_name: {tool_name: UITool}}
        self.tools_by_config: Dict[str, Dict[str, UITool]] = {}
        self.config: UIToolsConfig = UIToolsConfig()
        self.last_updated = datetime.now()
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register default UI tools"""

    
    def register_tool(self, tool: UITool, config_name: str = None) -> None:
        """
        Register a UI tool from CRD, scoped by config
        
        Args:
            tool: The UITool to register
            config_name: The config this tool belongs to (default: 'default')
        """
        if config_name is None:
            config_name = "default"
        
        if config_name not in self.tools_by_config:
            self.tools_by_config[config_name] = {}
        
        tool.config_name = config_name
        self.tools_by_config[config_name][tool.name] = tool
        self.last_updated = datetime.now()
    
    def register_tools(self, tools: List[UITool], config_name: str = None) -> None:
        """Register multiple UI tools from CRDs, scoped by config"""
        for tool in tools:
            self.register_tool(tool, config_name)
    
    def register_config(self, config: UIToolsConfig) -> None:
        """Register the spec-level configuration"""
        self.config = config
        self.last_updated = datetime.now()
    
    def get_tool(self, name: str, config_name: str = None) -> Optional[UITool]:
        """
        Get a specific tool by name, optionally scoped by config
        
        Args:
            name: Tool name
            config_name: Config name (searches all if None)
            
        Returns:
            The UITool if found, None otherwise
        """
        if config_name:
            return self.tools_by_config.get(config_name, {}).get(name)
        
        # Search across all configs if no specific config given
        for tools in self.tools_by_config.values():
            if name in tools:
                return tools[name]
        
        return None
    
    def get_all_tools(self, config_name: str = None) -> List[UITool]:
        """
        Get all registered tools, optionally filtered by config
        
        Args:
            config_name: Config name (returns all if None)
            
        Returns:
            List of UITool objects
        """
        if config_name:
            return list(self.tools_by_config.get(config_name, {}).values())
        
        # Return all tools from all configs
        all_tools = []
        for tools in self.tools_by_config.values():
            all_tools.extend(tools.values())
        return all_tools
    
    def get_tools_by_category(self, category: str, config_name: str = None) -> List[UITool]:
        """Get tools by category, optionally scoped by config"""
        tools = self.get_all_tools(config_name)
        return [tool for tool in tools if tool.category == category]
    
    def get_available_tools(self, config_name: str = None) -> AvailableUITools:
        """Get all available tools with metadata, optionally scoped by config"""
        tools = self.get_all_tools(config_name)
        return AvailableUITools(
            tools=tools,
            timestamp=self.last_updated,
            config=self.config,
        )
    
    def validate_tool_input(self, tool_name: str, input_data: Dict[str, Any], config_name: str = None) -> tuple[bool, Optional[str]]:
        """Validate tool input against schema"""
        tool = self.get_tool(tool_name, config_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        
        # Check required fields
        for required_field in tool.schema.required:
            if required_field not in input_data:
                return False, f"Missing required field: {required_field}"
        
        return True, None
    
    def clear_tools(self, config_name: str = None) -> None:
        """
        Clear tools, optionally for a specific config
        
        Args:
            config_name: Clear only this config (clears all if None)
        """
        if config_name:
            self.tools_by_config.pop(config_name, None)
        else:
            self.tools_by_config.clear()
        self.last_updated = datetime.now()


# Global registry instance
_ui_tools_registry: Optional[UIToolsRegistry] = None


def get_ui_tools_registry() -> UIToolsRegistry:
    """Get or create the global UI tools registry"""
    global _ui_tools_registry
    if _ui_tools_registry is None:
        _ui_tools_registry = UIToolsRegistry()
    return _ui_tools_registry
