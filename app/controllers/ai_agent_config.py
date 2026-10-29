""" 
Kubernetes operator controller for managing AIAgentConfig custom resources.

This controller handles the lifecycle of AIAgentConfig CRDs, validating their
MCP server connections and updating their status accordingly.
"""

import asyncio
import logging
import threading
import kopf
import httpx2

from kopf._cogs.configs.configuration import ScanningSettings, PostingSettings
from datetime import datetime, timezone
from ..services.agent.loader import AgentConfig, AuthenticationType, CABundleRef
from ..services.agent.factory import create_mcp_client, needs_oauth2_login
from ..services.agent._constants import NeedsOauth2
from ..services.oauth2.client import OAuthClientManager
from ..services.oauth2.credentials import get_oauth_secret_data
from ..services.oauth2.models import OAuthSecretError

# Transient exception types that indicate the MCP server may not be ready yet.
# These warrant a retry with backoff rather than a permanent failure.
_TRANSIENT_EXCEPTIONS = (
    ConnectionError,        # ConnectionRefusedError, ConnectionResetError, etc.
    TimeoutError,
    OSError,                # Low-level socket errors (e.g. "Network unreachable")
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.ReadTimeout,
    httpx2.WriteTimeout,
    httpx2.PoolTimeout,
)

# How far to follow ``__cause__`` when looking for a transient error. FastMCP wraps
# transport failures, e.g. RuntimeError("Client failed to connect") caused by httpx2.ConnectError.
_MAX_CAUSE_DEPTH = 5

_INITIAL_RETRY_DELAY = 1
_MAX_RETRY_DELAY = 30
_MAX_RETRIES = 200


class KopfManager:
    """
    Manages the Kopf operator lifecycle.
    
    This class handles starting and stopping the Kopf operator in a separate
    thread, similar to how MemoryManager handles database connections.
    """
    
    def __init__(self):
        self.stop_flag = None
        self.thread = None
        self.namespace = "cattle-ai-agent-system"
        
    def _run_operator(self, stop_flag):
        """
        Run the Kopf operator in a separate event loop.
        
        This method creates a new event loop and runs the Kopf operator
        until the stop flag is set.
        
        Args:
            stop_flag: Threading event to signal operator shutdown
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                kopf.operator(
                    standalone=True,
                    stop_flag=stop_flag,
                    namespaces=[self.namespace],
                    settings=kopf.OperatorSettings(
                        scanning=ScanningSettings(disabled=True),
                        posting=PostingSettings(enabled=False)
                    )
                )
            )
        except Exception as e:
            logging.error("Kopf operator crashed", exc_info=e)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    
    def start(self):
        """
        Start the Kopf operator in a background thread.
        
        This method initializes the stop flag and starts a daemon thread
        running the Kopf operator.
        """
        if self.thread is not None and self.thread.is_alive():
            logging.warning("Kopf operator is already running")
            return
            
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(
            target=self._run_operator,
            args=(self.stop_flag,),
            daemon=True,
        )
        self.thread.start()
        logging.info("Kopf operator started")
    
    def stop(self):
        """
        Stop the Kopf operator gracefully.
        
        This method sets the stop flag and waits for the operator thread
        to complete before returning.
        """
        if self.stop_flag is None or self.thread is None:
            logging.warning("Kopf operator is not running")
            return
            
        self.stop_flag.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                logging.warning("Kopf operator thread did not stop within timeout")
            else:
                logging.info("Kopf operator stopped")
        
        self.stop_flag = None
        self.thread = None


def create_kopf_manager() -> KopfManager:
    """
    Factory function to create a KopfManager instance.
    
    Returns:
        An instance of KopfManager.
    """
    manager = KopfManager()
    logging.info("KopfManager created")
    return manager

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
    client = await create_mcp_client(agent_config)

    # Test the connection by fetching available tools
    try:
        await client.list_tools()
    except Exception as e:
        if needs_oauth2_login(agent_config, client, _error_message(e)):
            raise NeedsOauth2(agent_config) from e
        raise


def _exception_leaves(exc: Exception) -> list[BaseException]:
    """Return the exceptions of a group, or the exception itself (MCP SDK 2 raises both)."""
    return list(exc.exceptions) if isinstance(exc, BaseExceptionGroup) else [exc]


def _error_message(exc: Exception) -> str:
    """Join the messages of an exception or exception group into a single string."""
    return " ".join(str(e) for e in _exception_leaves(exc))


def _is_transient(exc: BaseException) -> bool:
    """Return True if the exception, or one of its causes, is a transient connection error."""
    current: BaseException | None = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None:
            return False
        if isinstance(current, _TRANSIENT_EXCEPTIONS):
            return True
        current = current.__cause__
    return False


@kopf.on.resume('ai.cattle.io', 'v1alpha1', 'aiagentconfigs',  backoff=5.0, timeout=3600.0)
@kopf.on.create('ai.cattle.io', 'v1alpha1', 'aiagentconfigs')
@kopf.on.update('ai.cattle.io', 'v1alpha1', 'aiagentconfigs')
async def create_fn(spec, name, namespace, logger, patch, retry, **kwargs):
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
        authentication_secret=spec.get('authenticationSecret', ''),
        ca_bundle_ref=CABundleRef(**spec["caBundleRef"]) if spec.get("caBundleRef") else None,
    )
    try:
        # Validate the configuration by testing MCP server connection
        await _validate(agent_config)
        _set_status(patch, True, 'ConfigurationSucceeded', 'AI Agent configuration successful')

    except NeedsOauth2:
        _set_status(patch, True, 'ConfigurationSucceeded', 'Needs OAuth2 authentication')
        _register_oauth_client(agent_config, logger)
        return

    except Exception as e:
        error_message = _error_message(e)
        error_msg = f"Failed to load MCP tools: {error_message}"

        # Update status to reflect the failure
        _set_status(patch, False, 'ConfigurationFailed', error_msg)
        logger.warning(error_msg)

        # If any of the errors is transient, retry with exponential backoff
        if any(_is_transient(leaf) for leaf in _exception_leaves(e)):
            if retry >= _MAX_RETRIES:
                raise kopf.PermanentError(
                    f"Failed to load MCP tools for agent {name} after {retry} retries: {error_message}"
                )
            delay = min(_INITIAL_RETRY_DELAY * (2 ** retry), _MAX_RETRY_DELAY)
            raise kopf.TemporaryError(
                f"Failed to load MCP tools for agent {name} (retry {retry + 1}/{_MAX_RETRIES}): {error_message}",
                delay=delay,
            )

        raise kopf.PermanentError(f"Failed to load MCP tools for agent {name}: {error_message}")

def _register_oauth_client(agent_config: AgentConfig, logger) -> None:
    """
    Register or update the OAuth client for an agent with OAuth2 authentication.

    Reads the client credentials and metadata endpoint from the agent's
    authentication secret and registers the client in the singleton
    OAuthClientManager.

    Args:
        agent_config: The agent configuration to register.
        logger: Logger instance for diagnostic output.
    """
    if agent_config.authentication != AuthenticationType.OAUTH2:
        return

    if not agent_config.authentication_secret:
        logger.error(f"Agent '{agent_config.name}' uses OAuth2 but has no authentication secret")
        return

    try:
        credentials = get_oauth_secret_data(agent_config.authentication_secret)
    except OAuthSecretError as e:
        logger.warning(f"Cannot register OAuth client for '{agent_config.name}': {e}")
        raise kopf.TemporaryError("OAuth2 authentication requires an authentication secret, retrying...")

    if not credentials.client_id or not credentials.metadata_endpoint:
        logger.warning(
            f"Agent '{agent_config.name}' OAuth secret missing clientID or metadataEndpoint, "
            "skipping OAuth client registration"
        )
        return

    manager = OAuthClientManager.get_instance()
    manager.register_client(
        name=agent_config.name,
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        scope=credentials.scope,
        server_metadata_url=credentials.metadata_endpoint,
    )
    logger.info(f"Registered OAuth client for agent '{agent_config.name}'")
    