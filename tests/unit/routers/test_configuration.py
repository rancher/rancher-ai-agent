import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status, HTTPException
from app.routers import configuration as config_router
from app.routers.configuration import SettingsUpdate
from kubernetes.client.rest import ApiException
import json


@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = MagicMock()
    req.app.memory_manager.storage_type.value = "in-memory"
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    req.query_params = {}
    req.url.hostname = "localhost"
    return req


@pytest.mark.asyncio
async def test_get_models_openai_success(mock_request):
    """Test getting OpenAI models successfully."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        resp = await config_router.get_models(mock_request, llm_name="openai")
        assert resp.status_code == status.HTTP_200_OK
        content = json.loads(resp.body)
        assert "gpt-4o" in content
        assert "gpt-3.5-turbo" in content


@pytest.mark.asyncio
async def test_get_models_gemini_success(mock_request):
    """Test getting Gemini models successfully."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        resp = await config_router.get_models(mock_request, llm_name="gemini")
        assert resp.status_code == status.HTTP_200_OK
        content = json.loads(resp.body)
        assert "gemini-2.0-flash" in content


@pytest.mark.asyncio
async def test_get_models_bedrock_success(mock_request):
    """Test getting Bedrock models successfully."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        resp = await config_router.get_models(mock_request, llm_name="bedrock")
        assert resp.status_code == status.HTTP_200_OK
        content = json.loads(resp.body)
        assert "global.anthropic.claude-opus-4-5-20251101-v1:0" in content


@pytest.mark.asyncio
async def test_get_models_unsupported_provider(mock_request):
    """Test getting models for unsupported provider."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with pytest.raises(HTTPException) as exc:
            await config_router.get_models(mock_request, llm_name="invalid-provider")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_get_models_unauthorized(mock_request):
    """Test getting models without authentication."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await config_router.get_models(mock_request, llm_name="openai")
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_models_ollama_no_url(mock_request):
    """Test getting Ollama models without URL parameter."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with pytest.raises(HTTPException) as exc:
            await config_router.get_models(mock_request, llm_name="ollama")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_get_models_ollama_success(mock_request):
    """Test getting Ollama models successfully."""
    mock_request.query_params = {"url": "http://localhost:11434"}
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "model_1"},
            {"name": "model_2"}
        ]
    }
    
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_response)
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            resp = await config_router.get_models(mock_request, llm_name="ollama")
            assert resp.status_code == status.HTTP_200_OK
            content = json.loads(resp.body)
            assert "model_1" in content
            assert "model_2" in content


@pytest.mark.asyncio
async def test_get_models_ollama_connection_error(mock_request):
    """Test getting Ollama models with connection error."""
    mock_request.query_params = {"url": "http://localhost:11434"}
    
    import httpx
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await config_router.get_models(mock_request, llm_name="ollama")
            assert exc.value.status_code == status.HTTP_502_BAD_GATEWAY


@pytest.mark.asyncio
async def test_get_models_ollama_bad_status(mock_request):
    """Test getting Ollama models with bad HTTP status."""
    mock_request.query_params = {"url": "http://localhost:11434"}
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(return_value=mock_response)
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await config_router.get_models(mock_request, llm_name="ollama")
            assert exc.value.status_code == status.HTTP_502_BAD_GATEWAY


@pytest.mark.asyncio
async def test_get_settings_success(mock_request):
    """Test getting settings successfully."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        resp = await config_router.get_settings(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        content = json.loads(resp.body)
        assert "storageType" in content
        assert content["storageType"] == "in-memory"


@pytest.mark.asyncio
async def test_get_settings_unauthorized(mock_request):
    """Test getting settings without authentication."""
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await config_router.get_settings(mock_request)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_update_settings_unauthorized(mock_request):
    """Test updating settings without authentication."""
    settings = SettingsUpdate(OPENAI_API_KEY="test-key")
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as exc:
            await config_router.update_settings(settings, mock_request)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_update_settings_permission_denied(mock_request):
    """Test updating settings without permission."""
    settings = SettingsUpdate(OPENAI_API_KEY="test-key")
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("app.routers.configuration.check_k8s_permission", AsyncMock(return_value=False)):
            resp = await config_router.update_settings(settings, mock_request)
            assert resp.status_code == status.HTTP_403_FORBIDDEN
            content = json.loads(resp.body)
            assert "does not have permission" in content["detail"]


@pytest.mark.asyncio
async def test_update_settings_success(mock_request):
    """Test updating settings successfully."""
    settings = SettingsUpdate(
        OPENAI_API_KEY="test-key",
        OPENAI_URL="https://api.openai.com"
    )
    
    mock_secret = MagicMock()
    mock_secret.data = {
        "OPENAI_API_KEY": "old-key",
        "OPENAI_URL": "old-url",
        "MODEL": "gpt-4"
    }
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("app.routers.configuration.check_k8s_permission", AsyncMock(return_value=True)):
            with patch("app.routers.configuration.k8s_config.load_incluster_config"):
                with patch("app.routers.configuration.client.CoreV1Api") as mock_api:
                    mock_instance = MagicMock()
                    mock_api.return_value = mock_instance
                    mock_instance.read_namespaced_secret.return_value = mock_secret
                    
                    resp = await config_router.update_settings(settings, mock_request)
                    
                    assert resp.status_code == status.HTTP_200_OK
                    content = json.loads(resp.body)
                    assert "OPENAI_API_KEY" in content
                    assert "OPENAI_URL" in content
                    assert "MODEL" in content


@pytest.mark.asyncio
async def test_update_settings_k8s_error(mock_request):
    """Test updating settings with Kubernetes API error."""
    settings = SettingsUpdate(OPENAI_API_KEY="test-key")
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("app.routers.configuration.check_k8s_permission", AsyncMock(return_value=True)):
            with patch("app.routers.configuration.k8s_config.load_incluster_config"):
                with patch("app.routers.configuration.client.CoreV1Api") as mock_api:
                    mock_instance = MagicMock()
                    mock_api.return_value = mock_instance
                    mock_instance.read_namespaced_secret.side_effect = ApiException("Secret not found")
                    
                    resp = await config_router.update_settings(settings, mock_request)
                    
                    assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                    content = json.loads(resp.body)
                    assert "Failed to update llm-config secret" in content["detail"]


@pytest.mark.asyncio
async def test_update_settings_partial_fields(mock_request):
    """Test updating only some settings fields."""
    settings = SettingsUpdate(
        OPENAI_API_KEY="new-key",
        ACTIVE_LLM="openai"
    )
    
    mock_secret = MagicMock()
    mock_secret.data = {
        "OPENAI_API_KEY": "old-key",
        "ACTIVE_LLM": "gemini",
        "OLLAMA_URL": "http://localhost:11434"
    }
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("app.routers.configuration.check_k8s_permission", AsyncMock(return_value=True)):
            with patch("app.routers.configuration.k8s_config.load_incluster_config"):
                with patch("app.routers.configuration.client.CoreV1Api") as mock_api:
                    mock_instance = MagicMock()
                    mock_api.return_value = mock_instance
                    mock_instance.read_namespaced_secret.return_value = mock_secret
                    
                    resp = await config_router.update_settings(settings, mock_request)
                    
                    assert resp.status_code == status.HTTP_200_OK
                    content = json.loads(resp.body)
                    assert "OPENAI_API_KEY" in content
                    assert "ACTIVE_LLM" in content
                    assert "OLLAMA_URL" in content


@pytest.mark.asyncio
async def test_update_settings_nonexistent_field(mock_request):
    """Test updating a field that doesn't exist in the secret."""
    settings = SettingsUpdate(OPENAI_API_KEY="new-key")
    
    mock_secret = MagicMock()
    mock_secret.data = {
        "OLLAMA_URL": "http://localhost:11434"
    }
    
    with patch("app.routers.configuration.get_user_id_from_request", AsyncMock(return_value="test-user")):
        with patch("app.routers.configuration.check_k8s_permission", AsyncMock(return_value=True)):
            with patch("app.routers.configuration.k8s_config.load_incluster_config"):
                with patch("app.routers.configuration.client.CoreV1Api") as mock_api:
                    mock_instance = MagicMock()
                    mock_api.return_value = mock_instance
                    mock_instance.read_namespaced_secret.return_value = mock_secret
                    
                    resp = await config_router.update_settings(settings, mock_request)
                    
                    assert resp.status_code == status.HTTP_200_OK
                    content = json.loads(resp.body)
                    # Should return the secret data as-is
                    assert "OLLAMA_URL" in content
                    assert content["OLLAMA_URL"] == "http://localhost:11434"
