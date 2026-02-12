
import kopf
import httpx

from datetime import datetime, timezone
from ..services.agent.loader import AgentConfig
from ..services.agent.factory import  get_mcp_url_and_headers
from langchain_mcp_adapters.client import MultiServerMCPClient

def _set_status(patch, is_ready: bool, reason: str, message: str):
    """Update status with condition and phase."""
    patch.status['conditions'] = [{
        'type': 'Ready',
        'status': 'True' if is_ready else 'False',
        'reason': reason,
        'message': message,
        'lastTransitionTime': datetime.now(timezone.utc).isoformat()
    }]
    patch.status['phase'] = 'Ready' if is_ready else 'Failed'


async def _validate(agent_config: AgentConfig) -> None:
    mcp_url, header = get_mcp_url_and_headers(agent_config)
    client = MultiServerMCPClient({
        agent_config.name: {
            "url": mcp_url,
            "transport": "streamable_http",
            "headers": header,
        },
    })

    await client.get_tools()


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', retries=5)
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', retries=5)
@kopf.on.update('ai.cattle.io', 'v1alpha1', 'aiagentconfigs', retries=5)
async def create_fn(spec, name, namespace, logger, patch, **kwargs):
    logger.debug(f"Creating AI Agent Config: {name} in namespace: {namespace}")

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
        await _validate(agent_config)
        _set_status(patch, True, 'ConfigurationSucceeded', 'AI Agent configuration successful')

    except* Exception as eg:
        error_message = ""
        for e in eg.exceptions:
            error_message += f"{str(e)} "
        error_msg = f"Failed to load MCP tools: {error_message}"
        _set_status(patch, False, 'ConfigurationFailed', error_msg)
        raise 
        
    