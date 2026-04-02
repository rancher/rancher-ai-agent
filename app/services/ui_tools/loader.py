"""
UI Tools CRD Loader
Loads UI tool definitions from Kubernetes CRDs and registers them in the UI tools registry.
"""

import logging
from typing import List, Optional, Dict, Any
from kubernetes import client, config
from .registry import UITool, UIToolSchema, UIToolsConfig, get_ui_tools_registry

NAMESPACE = "cattle-ai-agent-system"
GROUP = "ai.cattle.io"
VERSION = "v1alpha1"
PLURAL = "uitoolsconfigs"

def _init_k8s_client():
    """Initialize Kubernetes client."""
    try:
        # Try in-cluster config first
        config.load_incluster_config()
    except config.ConfigException:
        # Fall back to kubeconfig
        config.load_kube_config()
    
    return client.CustomObjectsApi()


def load_ui_tools_from_crds(spec: Optional[Dict[str, Any]]) -> List[UITool]:
    """
    Load UI tool definitions from Kubernetes CRD spec
    
    Args:
        spec: Spec dictionary from the UIToolsConfig resource.
    
    Returns:
        List of loaded UITool objects
    """
    if spec is None:
        logging.error("Spec dictionary is None")
        return []
    
    tools: List[UITool] = []
    tools_array = spec.get('tools', [])
    
    for tool_def in tools_array:
        try:
            tool = _parse_ui_tool_definition(tool_def)
            if tool:
                tools.append(tool)
                logging.info(f"Loaded UI tool: {tool.name}")
        except Exception as e:
            logging.error(f"Error parsing UI tool definition: {e}")
    
    logging.info(f"Loaded {len(tools)} UI tools from CRDs")
    return tools


def load_config_from_crds(spec: Optional[Dict[str, Any]]) -> UIToolsConfig:
    """
    Load spec-level configuration from Kubernetes CRD spec.config
    
    Args:
        spec: Spec dictionary from the UIToolsConfig resource.
    
    Returns:
        UIToolsConfig with spec-level enabled, revision, system_prompt, and max_tools fields
    """
    if spec is None:
        logging.warning("Spec dictionary is None, using default config")
        return UIToolsConfig()
    
    config_data = spec.get('config', {})
    enabled = config_data.get('enabled', True)
    revision = config_data.get('revision', 0)
    system_prompt = config_data.get('systemPrompt')
    max_tools = config_data.get('maxTools', 5)
    
    logging.debug(f"Loaded spec-level config: enabled={enabled}, revision={revision}, system_prompt={'set' if system_prompt else 'not set'}, max_tools={max_tools}")
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
    Reload UI tools and config from a UIToolsConfig CRD resource into the registry.
    
    Extracts the spec from the resource, loads tools and config, and syncs them
    with the global registry. Clears any previous tools for this config before
    registering the new ones.
    
    Args:
        resource: The complete UIToolsConfig CRD resource object
        
    Raises:
        Exception: If resource is malformed or missing required fields
    """
    resource_name = resource.get('metadata', {}).get('name', 'unknown')
    spec = resource.get('spec', {})
    
    if not spec:
        logging.warning(f"UIToolsConfig '{resource_name}' has empty spec, skipping reload")
        return
    
    # Extract and register tools
    tools = load_ui_tools_from_crds(spec)
    config_obj = load_config_from_crds(spec)
    
    # Sync with registry
    registry = get_ui_tools_registry()
    registry.clear_tools(config_name=resource_name)
    registry.register_tools(tools, config_name=resource_name)
    registry.register_config(config_obj)
    
    logging.info(f"Reloaded UIToolsConfig '{resource_name}' into registry ({len(tools)} tools)")


def clear_ui_tools_config(config_name: str) -> None:
    """
    Clear all UI tools for a specific UIToolsConfig from the registry.
    
    This is called when a UIToolsConfig CRD is deleted. Removes all tools
    associated with the given config name from the global registry.
    
    Args:
        config_name: The name of the UIToolsConfig to clear tools for
    """
    registry = get_ui_tools_registry()
    registry.clear_tools(config_name=config_name)
    logging.info(f"Cleared all UI tools for config '{config_name}' from registry")
