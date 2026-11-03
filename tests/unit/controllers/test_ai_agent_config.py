"""
Unit tests for the AIAgentConfig kopf controller.

Tests transient vs permanent error classification and retry behaviour.
"""
import pytest
import httpx2
import kopf
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.shared.exceptions import MCPError

from app.controllers.ai_agent_config import (
    create_fn,
    _validate,
    _INITIAL_RETRY_DELAY,
    _MAX_RETRY_DELAY,
    _MAX_RETRIES,
)
from app.services.agent._constants import NeedsOauth2
from app.services.agent.factory import _OAuth2BearerAuth
from app.services.agent.loader import AgentConfig, AuthenticationType


def _make_spec(**overrides):
    """Return a minimal valid spec dict."""
    base = {
        "displayName": "Test Agent",
        "description": "desc",
        "systemPrompt": "prompt",
        "mcpURL": "http://mcp:8080/sse",
        "authenticationType": "NONE",
        "authenticationSecret": "",
    }
    base.update(overrides)
    return base


def _build_kwargs(spec=None, retry=0):
    """Return the keyword arguments for invoking create_fn."""
    return dict(
        spec=spec or _make_spec(),
        name="test-agent",
        namespace="cattle-ai-agent-system",
        logger=MagicMock(),
        patch=MagicMock(status={}),
        retry=retry,
    )


# ============================================================================
# Successful validation
# ============================================================================

@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_create_fn_success(mock_validate):
    """Handler sets phase=Ready when validation succeeds."""
    kwargs = _build_kwargs()
    await create_fn(**kwargs)

    mock_validate.assert_awaited_once()
    patch_obj = kwargs["patch"]
    assert patch_obj.status["phase"] == "Ready"
    assert patch_obj.status["conditions"][0]["status"] == "True"


# ============================================================================
# Transient errors → kopf.TemporaryError with exponential backoff
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_class",
    [
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        OSError,
        httpx2.ConnectError,
        httpx2.ConnectTimeout,
        httpx2.ReadTimeout,
    ],
    ids=lambda c: c.__name__,
)
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_transient_error_raises_temporary(mock_validate, exc_class):
    """Transient connection errors should raise TemporaryError with a retry delay."""
    if issubclass(exc_class, (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.ReadTimeout)):
        exc = exc_class("connection failed", request=MagicMock())
    else:
        exc = exc_class("connection failed")
    mock_validate.side_effect = ExceptionGroup("eg", [exc])

    kwargs = _build_kwargs(retry=0)
    with pytest.raises(kopf.TemporaryError) as exc_info:
        await create_fn(**kwargs)

    assert exc_info.value.delay == _INITIAL_RETRY_DELAY
    patch_obj = kwargs["patch"]
    assert patch_obj.status["phase"] == "Failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry,expected_delay",
    [
        (0, _INITIAL_RETRY_DELAY),        # 1s
        (1, _INITIAL_RETRY_DELAY * 2),    # 2s
        (2, _INITIAL_RETRY_DELAY * 4),    # 4s
        (4, _INITIAL_RETRY_DELAY * 16),   # 16s
        (8, _MAX_RETRY_DELAY),            # capped at 30s
        (9, _MAX_RETRY_DELAY),            # capped at 30s
        (49, _MAX_RETRY_DELAY),           # still capped
    ],
)
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_exponential_backoff(mock_validate, retry, expected_delay):
    """Retry delay doubles each attempt, capped at _MAX_RETRY_DELAY."""
    mock_validate.side_effect = ExceptionGroup(
        "eg", [ConnectionRefusedError("refused")]
    )

    kwargs = _build_kwargs(retry=retry)
    with pytest.raises(kopf.TemporaryError) as exc_info:
        await create_fn(**kwargs)

    assert exc_info.value.delay == expected_delay


@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_max_retries_exceeded_becomes_permanent(mock_validate):
    """After _MAX_RETRIES transient failures, raise PermanentError."""
    mock_validate.side_effect = ExceptionGroup(
        "eg", [ConnectionRefusedError("refused")]
    )

    kwargs = _build_kwargs(retry=_MAX_RETRIES)
    with pytest.raises(kopf.PermanentError) as exc_info:
        await create_fn(**kwargs)

    assert "after" in str(exc_info.value).lower()


# ============================================================================
# Permanent errors → kopf.PermanentError
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_class,exc_args",
    [
        (ValueError, ("malformed URL",)),
        (KeyError, ("missing-key",)),
        (RuntimeError, ("unrecoverable",)),
    ],
    ids=lambda c: str(c),
)
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_permanent_error_raises_permanent(mock_validate, exc_class, exc_args):
    """Non-transient errors should raise PermanentError."""
    mock_validate.side_effect = ExceptionGroup("eg", [exc_class(*exc_args)])

    kwargs = _build_kwargs()
    with pytest.raises(kopf.PermanentError):
        await create_fn(**kwargs)

    patch_obj = kwargs["patch"]
    assert patch_obj.status["phase"] == "Failed"


# ============================================================================
# Mixed group: transient + permanent → still retries
# ============================================================================

@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_mixed_group_prefers_temporary(mock_validate):
    """When a group contains both transient and permanent errors, retry."""
    mock_validate.side_effect = ExceptionGroup(
        "eg",
        [ConnectionRefusedError("refused"), ValueError("bad config")],
    )

    kwargs = _build_kwargs()
    with pytest.raises(kopf.TemporaryError):
        await create_fn(**kwargs)


# ============================================================================
# Status fields are set correctly on failure
# ============================================================================

@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_status_fields_on_failure(mock_validate):
    """Verify status conditions contain correct reason and message on failure."""
    mock_validate.side_effect = ExceptionGroup(
        "eg", [ValueError("bad url")]
    )

    kwargs = _build_kwargs()
    with pytest.raises(kopf.PermanentError):
        await create_fn(**kwargs)

    patch_obj = kwargs["patch"]
    condition = patch_obj.status["conditions"][0]
    assert condition["type"] == "Ready"
    assert condition["status"] == "False"
    assert condition["reason"] == "ConfigurationFailed"
    assert "bad url" in condition["message"]


# ============================================================================
# MCP SDK 2 raises plain exceptions (not ExceptionGroups)
# ============================================================================

@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_plain_transient_error_raises_temporary(mock_validate):
    """A transient error raised on its own (not in a group) should still be retried."""
    mock_validate.side_effect = httpx2.ConnectError("connection failed", request=MagicMock())

    kwargs = _build_kwargs()
    with pytest.raises(kopf.TemporaryError):
        await create_fn(**kwargs)


@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_wrapped_transient_error_raises_temporary(mock_validate):
    """FastMCP wraps transport errors in RuntimeError; the httpx2 cause should mark it transient."""
    cause = httpx2.ConnectError("All connection attempts failed", request=MagicMock())
    exc = RuntimeError("Client failed to connect: All connection attempts failed")
    exc.__cause__ = cause
    mock_validate.side_effect = exc

    kwargs = _build_kwargs()
    with pytest.raises(kopf.TemporaryError):
        await create_fn(**kwargs)


@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_plain_mcp_error_raises_permanent(mock_validate):
    """A non-transient MCPError should mark the config as failed and raise PermanentError."""
    mock_validate.side_effect = MCPError(code=-32603, message="Server returned an error response")

    kwargs = _build_kwargs()
    with pytest.raises(kopf.PermanentError):
        await create_fn(**kwargs)

    condition = kwargs["patch"].status["conditions"][0]
    assert condition["reason"] == "ConfigurationFailed"
    assert "Server returned an error response" in condition["message"]


# ============================================================================
# OAuth2: a 401 during validation means the agent needs a user login
# ============================================================================

@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config._register_oauth_client")
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_needs_oauth2_marks_ready_and_registers_client(mock_validate, mock_register):
    """NeedsOauth2 from validation should succeed the config and register the OAuth client."""
    mock_validate.side_effect = NeedsOauth2(MagicMock())

    kwargs = _build_kwargs(spec=_make_spec(authenticationType="OAUTH2"))
    await create_fn(**kwargs)

    condition = kwargs["patch"].status["conditions"][0]
    assert condition["status"] == "True"
    assert condition["reason"] == "ConfigurationSucceeded"
    assert condition["message"] == "Needs OAuth2 authentication"
    mock_register.assert_called_once()


def _agent_config(authentication):
    return AgentConfig(
        name="test-agent",
        displayName="Test Agent",
        description="desc",
        system_prompt="prompt",
        mcp_url="https://mcp:8080/mcp",
        authentication=authentication,
        authentication_secret="",
    )


def _unauthorized_client():
    """An MCP client whose OAuth2 auth saw a 401, failing with the generic MCP SDK 2 error."""
    auth = _OAuth2BearerAuth(None)
    auth.unauthorized = True
    client = MagicMock()
    client.oauth2_auth = auth
    client.list_tools = AsyncMock(side_effect=MCPError(code=-32603, message="Server returned an error response"))
    return client


@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config.create_mcp_client", new_callable=AsyncMock)
async def test_validate_oauth2_unauthorized_raises_needs_oauth2(mock_create_client):
    """_validate should turn a 401 for an OAuth2 agent into NeedsOauth2."""
    mock_create_client.return_value = _unauthorized_client()

    with pytest.raises(NeedsOauth2):
        await _validate(_agent_config(AuthenticationType.OAUTH2))


@pytest.mark.asyncio
@patch("app.controllers.ai_agent_config.create_mcp_client", new_callable=AsyncMock)
async def test_validate_non_oauth2_reraises_original_error(mock_create_client):
    """_validate should re-raise the original error for agents that don't use OAuth2."""
    mock_create_client.return_value = _unauthorized_client()

    with pytest.raises(MCPError):
        await _validate(_agent_config(AuthenticationType.NONE))
