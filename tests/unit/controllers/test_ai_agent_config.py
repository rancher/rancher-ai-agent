"""
Unit tests for the AIAgentConfig kopf controller.

Tests transient vs permanent error classification and retry behaviour.
"""
import pytest
import httpx
import kopf
from unittest.mock import AsyncMock, MagicMock, patch

from app.controllers.ai_agent_config import (
    create_fn,
    _INITIAL_RETRY_DELAY,
    _MAX_RETRY_DELAY,
    _MAX_RETRIES,
)


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
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    ],
    ids=lambda c: c.__name__,
)
@patch("app.controllers.ai_agent_config._validate", new_callable=AsyncMock)
async def test_transient_error_raises_temporary(mock_validate, exc_class):
    """Transient connection errors should raise TemporaryError with a retry delay."""
    if issubclass(exc_class, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
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
        (8, _INITIAL_RETRY_DELAY * 256),  # 256s
        (9, _MAX_RETRY_DELAY),            # capped at 300s
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
