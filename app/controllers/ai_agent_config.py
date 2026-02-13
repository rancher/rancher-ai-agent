""" 
Kubernetes operator controller for managing AIAgentConfig custom resources.

This controller handles the lifecycle of AIAgentConfig CRDs, validating their
MCP server connections and updating their status accordingly.
"""

import kopf

from datetime import datetime, timezone
from ..services.agent.loader import AgentConfig
from ..services.agent.factory import create_mcp_client

def _set_status(patch, is_ready: bool, reason: str, message: str):
    """
    Update the status of an AIAgentConfig resource.
    
    Sets the Ready condition and overall phase based on the validation result.
    
    Args:
        patch: Kopf patch object to update the resource status
        is_ready: Whether the agent configuration is ready
        reason: Short reason code for the status (e.g., 'ConfigurationSucceeded')
        message: Detailed message explaining the status
    """
    patch.status['conditions'] = [{
        'type': 'Ready',
        'status': 'True' if is_ready else 'False',
        'reason': reason,
        'message': message,
        'lastTransitionTime': datetime.now(timezone.utc).isoformat()
    }]
    patch.status['phase'] = 'Ready' if is_ready else 'Failed'


async def _validate(agent_config: AgentConfig) -> None:
    """
    Validate an agent configuration by testing the MCP server connection.
    
    Attempts to connect to the MCP server and retrieve available tools
    to verify the configuration is valid and the server is reachable.
    
    Args:
        agent_config: The agent configuration to validate
        
    Raises:
        Exception: If the MCP server connection fails or tools cannot be retrieved
    """
    client = create_mcp_client(agent_config)

    # Test the connection by fetching available tools
    await client.get_tools()


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', field='spec')
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', field='spec')
@kopf.on.update('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', field='spec')
async def create_fn(spec, name, namespace, logger, patch, **kwargs):
    """
    Handle AIAgentConfig resource lifecycle events.
    
    This handler is triggered on create, update, and resume events for
    AIAgentConfig resources. It validates the MCP server connection and
    updates the resource status accordingly.
    
    Args:
        spec: The resource specification containing agent configuration
        name: Name of the AIAgentConfig resource
        namespace: Kubernetes namespace of the resource
        logger: Kopf logger instance
        patch: Kopf patch object to update resource status
        **kwargs: Additional kopf event data
        
    Raises:
        Exception: Re-raises validation exceptions after updating status
    """
    logger.debug(f"Creating AI Agent Config: {name} in namespace: {namespace}")

    # Parse the spec into an AgentConfig object
    agent_config = AgentConfig(
        name=name,
        displayName=spec.get('displayName', ''),
        description=spec.get('description', ''),
        system_prompt=spec.get('systemPrompt', ''),
        mcp_url=spec.get('mcpURL', ''),
        authentication=spec.get('authenticationType', ''),
        authentication_secret=spec.get('authenticationSecret', '')
    )
    try:
        # Validate the configuration by testing MCP server connection
        await _validate(agent_config)
        _set_status(patch, True, 'ConfigurationSucceeded', 'AI Agent configuration successful')

    except* Exception as eg:
        # Collect all exception messages from the exception group
        error_message = ""
        for e in eg.exceptions:
            error_message += f"{str(e)} "
        error_msg = f"Failed to load MCP tools: {error_message}"
        
        # Update status to reflect the failure
        _set_status(patch, False, 'ConfigurationFailed', error_msg)
        logger.warning(error_msg)

        raise kopf.PermanentError(f"Failed to load MCP tools: {error_message}") 
        
    