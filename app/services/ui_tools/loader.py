"""
UI Tools Loader
Loads UI tool definitions from Kubernetes ConfigMaps.
Expects ConfigMaps containing JSON-formatted UI tools data.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from kubernetes import client, config as k8s_config
from .models import UITool, UIToolSchema, UIToolsConfig, UIToolsConfigData

NAMESPACE = "cattle-ai-agent-system"

def _init_k8s_client():
    """Initialize Kubernetes client."""
    try:
        # Try in-cluster config first
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        # Fall back to kubeconfig
        k8s_config.load_kube_config()
    
    return client.CoreV1Api()


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


def load_ui_tools_from_configmap(config_name: str) -> UIToolsConfigData | None:
    """
    Load UI tools from a specific ConfigMap and return the ui tools config.
    
    Args:
        config_name: Specific ConfigMap name to load
        
    Returns:
        UIToolsConfigData containing the list of tools and config settings, or None if loading fails
    """
    if config_name:
        # Load specific ConfigMap by name
        try:
            api = _init_k8s_client()
            cm = api.read_namespaced_config_map(config_name, NAMESPACE)
            
            cm_data = cm.data or {}
            if not cm_data:
                logging.warning(f"UI tools ConfigMap '{config_name}' has empty data, skipping")
                return None
            
            tools, config = _load_ui_tools_data_from_config_map(cm_data)
            logging.info(f"Loaded UI tools from ConfigMap '{config_name}' ({len(tools)} tools)")
            
            return UIToolsConfigData(config=config, tools=tools)
        except Exception as e:
            logging.error(f"Error loading ConfigMap '{config_name}': {e}")
    
    return None
