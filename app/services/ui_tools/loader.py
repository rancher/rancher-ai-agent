"""
UI Tools Loader
Loads UI tool definitions from Kubernetes ConfigMaps and registers them in the UI tools registry.
Expects ConfigMaps with label app=rancher-ai-ui-tools containing JSON-formatted UI tools data.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from .registry import UITool, UIToolSchema, UIToolsConfig, get_ui_tools_registry


def _get_ui_tools_object(data: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the outer config object from ConfigMap data.
    
    Args:
        data: ConfigMap data dictionary
    
    Returns:
        Parsed config object or None if missing/invalid
    """
    config_json = data.get("config")
    if not config_json:
        return None
    
    try:
        config_obj = json.loads(config_json)
        if not isinstance(config_obj, dict):
            logging.error(f"ConfigMap 'config' field is not a JSON object: {type(config_obj)}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse ConfigMap 'config' field as JSON: {e}")
        return None
    
    return config_obj


def _load_ui_tools_data_from_config_map(data: Dict[str, str]) -> tuple[List[UITool], UIToolsConfig]:
    """
    Load both UI tools and config from ConfigMap data in a single pass.
    
    Args:
        data: ConfigMap data dictionary
    
    Returns:
        Tuple of (tools list, config object)
    """
    tools: List[UITool] = []
    config = UIToolsConfig()
    
    if data is None:
        logging.error("ConfigMap data is None")
        return tools, config

    config_obj = _get_ui_tools_object(data)
    if config_obj is None:
        logging.warning("ConfigMap does not have valid 'config' field")
        return tools, config

    # Extract config
    config_data = config_obj.get("config", {})
    if config_data:
        try:
            config = _parse_ui_tool_config_definition(config_data)
            logging.debug(f"Loaded config: enabled={config.enabled}, revision={config.revision}, system_prompt={'set' if config.system_prompt else 'not set'}, max_tools={config.max_tools}")
        except Exception as e:
            logging.error(f"Failed to extract config from ConfigMap: {e}")
    else:
        logging.debug("ConfigMap 'config' object does not have 'config' key, using default config")

    # Extract tools
    tools_array = config_obj.get("tools", [])
    if tools_array:
        for tool_def in tools_array:
            try:
                tool = _parse_ui_tool_definition(tool_def)
                if tool:
                    tools.append(tool)
                    logging.info(f"Loaded UI tool: {tool.name}")
            except Exception as e:
                logging.error(f"Error parsing UI tool definition: {e}")
        logging.info(f"Loaded {len(tools)} UI tools from ConfigMap")
    else:
        logging.warning("ConfigMap 'config' object does not have 'tools' field")
    
    return tools, config


def _parse_ui_tool_config_definition(config_data: Dict[str, Any]) -> UIToolsConfig:
    """Parse UI tools config from config data"""
    
    enabled = config_data.get('enabled', True)
    revision = config_data.get('revision', 0)
    system_prompt = config_data.get('systemPrompt')
    max_tools = config_data.get('maxTools', 5)
    
    return UIToolsConfig(enabled=enabled, revision=revision, system_prompt=system_prompt, max_tools=max_tools)


def _parse_ui_tool_definition(tool_def: Dict[str, Any]) -> Optional[UITool]:
    """Parse a UI tool definition from the tools array"""
    
    tool_name = tool_def.get('name')
    description = tool_def.get('description', '')
    prompt = tool_def.get('prompt', '')
    category_str = tool_def.get('category', 'custom')
    schema_data = tool_def.get('schema', {})
    metadata_data = tool_def.get('metadata', {})
    revision = tool_def.get('revision', 0)
    enabled = tool_def.get('enabled', True)
    
    # Validate required fields
    if not tool_name or not description or not prompt:
        logging.warning(f"UI tool definition missing required fields (name, description, prompt): {tool_name}")
        return None
    
    # Use category string directly (validated by CRD schema)
    category = category_str if category_str else 'custom'
    
    # Parse schema
    schema = UIToolSchema(
        type=schema_data.get('type', 'object'),
        properties=schema_data.get('properties', {}),
        required=schema_data.get('required', []),
        description=schema_data.get('description'),
    )
    
    # Parse metadata as a simple dict
    tool_metadata = metadata_data if isinstance(metadata_data, dict) else {}
    
    return UITool(
        name=tool_name,
        description=description,
        prompt=prompt,
        category=category,
        schema=schema,
        metadata=tool_metadata,
        revision=revision,
        enabled=enabled,
    )


def reload_ui_tools_config(resource: Dict[str, Any]) -> None:
    """
    Reload UI tools and config from a UI tools ConfigMap into the registry.
    
    Extracts the data from the ConfigMap (JSON format), loads tools and config,
    and syncs them with the global registry. Clears any previous tools for this
    config before registering the new ones.
    
    Args:
        resource: The complete ConfigMap resource object
        
    Raises:
        Exception: If resource is malformed or missing required fields
    """
    resource_name = resource.get('metadata', {}).get('name', 'unknown')
    data = resource.get('data', {})
    
    if not data:
        logging.warning(f"UI tools ConfigMap '{resource_name}' has empty data, skipping reload")
        return
    
    tools, config = _load_ui_tools_data_from_config_map(data)
    
    # Sync with registry
    registry = get_ui_tools_registry()
    registry.clear_tools(config_name=resource_name)
    registry.register_tools_config(tools, config, config_name=resource_name)
    
    logging.info(f"Reloaded UI tools ConfigMap '{resource_name}' into registry ({len(tools)} tools)")


def clear_ui_tools_config(config_name: str) -> None:
    """
    Clear all UI tools for a specific UI tools ConfigMap from the registry.
    
    This is called when a ConfigMap with UI tools is deleted. Removes all tools
    associated with the given config name from the global registry.
    
    Args:
        config_name: The name of the ConfigMap to clear tools for
    """
    registry = get_ui_tools_registry()
    registry.clear_tools(config_name=config_name)
    logging.info(f"Cleared all UI tools for ConfigMap '{config_name}' from registry")
