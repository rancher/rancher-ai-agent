"""
Data models and type definitions for UI tools.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum


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
    tools: List[UITool] = field(default_factory=list)


class UIToolCategory(str, Enum):
    """Builtin UI tool categories"""
    SELECTOR = "selector"
    
    def __str__(self) -> str:
        return self.value
