import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocketDisconnect

import app.services.llm as llm_module
from app.services.llm import (
    get_llm,
    get_active_llm,
    get_llm_model,
    close_openai_ca_clients,
)

@pytest.fixture(autouse=True)
def clear_openai_ca_clients():
    llm_module._openai_ca_clients.clear()
    yield
    llm_module._openai_ca_clients.clear()

@patch('app.services.llm.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model", "ACTIVE_LLM": "ollama", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_gemini(mock_chat_gemini):
    with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-pro", "ACTIVE_LLM": "gemini", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro")
        assert llm == mock_chat_gemini.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai(mock_openai):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4")
        assert llm == mock_openai.return_value

@patch('app.services.llm.httpx')
@patch('app.services.agent.loader.ssl.create_default_context')
@patch('app.services.llm.get_ca_cert_from_secret', return_value="PEM")
@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_with_custom_ca(mock_openai, mock_get_ca, mock_ssl_ctx, mock_httpx):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_URL": "https://openai.example.com", "OPENAI_CA_SECRET_NAME": "openai-ca", "OPENAI_CA_SECRET_KEY": "my-ca.crt"}, clear=True):
        llm = get_llm()
        mock_get_ca.assert_called_once_with("openai-ca", "my-ca.crt")
        mock_ssl_ctx.return_value.load_verify_locations.assert_called_once_with(cadata="PEM")
        mock_httpx.Client.assert_called_once_with(verify=mock_ssl_ctx.return_value)
        mock_httpx.AsyncClient.assert_called_once_with(verify=mock_ssl_ctx.return_value)
        mock_openai.assert_called_once_with(
            model="gpt-4",
            base_url="https://openai.example.com",
            http_client=mock_httpx.Client.return_value,
            http_async_client=mock_httpx.AsyncClient.return_value,
        )
        assert llm == mock_openai.return_value

@patch('app.services.llm.get_ca_cert_from_secret', side_effect=RuntimeError("not found"))
@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_custom_ca_load_failure(mock_openai, mock_get_ca):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_CA_SECRET_NAME": "openai-ca"}, clear=True):
        llm = get_llm()
        mock_get_ca.assert_called_once_with("openai-ca", "ca.crt")
        mock_openai.assert_called_once_with(model="gpt-4")
        assert llm == mock_openai.return_value

@patch('app.services.llm.httpx')
@patch('app.services.agent.loader.ssl.create_default_context')
@patch('app.services.llm.get_ca_cert_from_secret', return_value="PEM")
@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_custom_ca_empty_key_defaults(mock_openai, mock_get_ca, mock_ssl_ctx, mock_httpx):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_CA_SECRET_NAME": "openai-ca", "OPENAI_CA_SECRET_KEY": ""}, clear=True):
        get_llm()
        mock_get_ca.assert_called_once_with("openai-ca", "ca.crt")

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_custom_ca_empty_name_skipped(mock_openai):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_CA_SECRET_NAME": "", "OPENAI_CA_SECRET_KEY": "ca.crt"}, clear=True):
        get_llm()
        mock_openai.assert_called_once_with(model="gpt-4")

@patch('app.services.llm.httpx')
@patch('app.services.agent.loader.ssl.create_default_context')
@patch('app.services.llm.get_ca_cert_from_secret', return_value="PEM")
@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_custom_ca_clients_are_shared(mock_openai, mock_get_ca, mock_ssl_ctx, mock_httpx):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_CA_SECRET_NAME": "openai-ca"}, clear=True):
        get_llm()
        get_llm(model_override="gpt-4o")
        mock_get_ca.assert_called_once_with("openai-ca", "ca.crt")
        mock_httpx.Client.assert_called_once()
        mock_httpx.AsyncClient.assert_called_once()
        assert mock_openai.call_count == 2
        for call in mock_openai.call_args_list:
            assert call.kwargs["http_client"] is mock_httpx.Client.return_value
            assert call.kwargs["http_async_client"] is mock_httpx.AsyncClient.return_value

@patch('app.services.llm.get_ca_cert_from_secret', side_effect=RuntimeError("not found"))
@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_custom_ca_load_failure_is_retried(mock_openai, mock_get_ca):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_CA_SECRET_NAME": "openai-ca"}, clear=True):
        get_llm()
        get_llm()
        assert mock_get_ca.call_count == 2
        assert llm_module._openai_ca_clients == {}

@pytest.mark.asyncio
async def test_close_openai_ca_clients():
    http_client = MagicMock()
    http_async_client = MagicMock()
    http_async_client.aclose = AsyncMock()
    llm_module._openai_ca_clients[("openai-ca", "ca.crt")] = (http_client, http_async_client)

    await close_openai_ca_clients()

    http_client.close.assert_called_once()
    http_async_client.aclose.assert_awaited_once()
    assert llm_module._openai_ca_clients == {}

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_generic_openai(mock_openai):
    with patch.dict(os.environ, {"GENERIC_OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "generic-openai", "GENERIC_OPENAI_URL": "https://api.example.com", "GENERIC_OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4", base_url="https://api.example.com", api_key="fake-key")
        assert llm == mock_openai.return_value

def test_get_active_llm_not_configured():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="LLM not configured."):
            get_active_llm()

def test_get_active_llm_invalid():
    with patch.dict(os.environ, {"ACTIVE_LLM": "invalid-llm"}, clear=True):
        with pytest.raises(ValueError, match="LLM not configured."):
            get_active_llm()

def test_get_llm_model_not_configured():
    """Test get_llm_model when no model is configured for the active LLM"""
    with patch.dict(os.environ, {"ACTIVE_LLM": "ollama"}, clear=True):
        with pytest.raises(ValueError, match="LLM Model not configured"):
            get_llm_model("ollama")

@patch('app.services.llm.ChatOllama')
def test_get_llm_with_mock(mock_chat_ollama):
    """Test that mock URL is used when LLM_MOCK_ENABLED is true"""
    with patch.dict(os.environ, {
        "OLLAMA_MODEL": "test-model",
        "ACTIVE_LLM": "ollama",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_MOCK_ENABLED": "true",
        "LLM_MOCK_URL": "http://mock-server:8000"
    }, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://mock-server:8000")
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatOllama')
def test_get_llm_without_mock(mock_chat_ollama):
    """Test that original URL is used when LLM_MOCK_ENABLED is false"""
    with patch.dict(os.environ, {
        "OLLAMA_MODEL": "test-model",
        "ACTIVE_LLM": "ollama",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_MOCK_ENABLED": "false",
        "LLM_MOCK_URL": "http://mock-server:8000"
    }, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_gemini_with_mock(mock_chat_gemini):
    """Test that mock URL is used for Gemini when LLM_MOCK_ENABLED is true"""
    with patch.dict(os.environ, {
        "GEMINI_MODEL": "gemini-pro",
        "ACTIVE_LLM": "gemini",
        "GOOGLE_API_KEY": "fake-key",
        "LLM_MOCK_ENABLED": "true",
        "LLM_MOCK_URL": "http://mock-server:8000"
    }, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro", base_url="http://mock-server:8000", transport="rest")
        assert llm == mock_chat_gemini.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_with_mock(mock_openai):
    """Test that mock URL is used for OpenAI when LLM_MOCK_ENABLED is true"""
    with patch.dict(os.environ, {
        "OPENAI_MODEL": "gpt-4",
        "ACTIVE_LLM": "openai",
        "OPENAI_API_KEY": "fake-key",
        "LLM_MOCK_ENABLED": "true",
        "LLM_MOCK_URL": "http://mock-server:8000"
    }, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4", base_url="http://mock-server:8000")
        assert llm == mock_openai.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_with_custom_url(mock_openai):
    """Test that custom OPENAI_URL is used when provided"""
    with patch.dict(os.environ, {
        "OPENAI_MODEL": "gpt-4",
        "ACTIVE_LLM": "openai",
        "OPENAI_API_KEY": "fake-key",
        "OPENAI_URL": "http://custom-openai:8000"
    }, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4", base_url="http://custom-openai:8000")
        assert llm == mock_openai.return_value


@patch('app.services.llm.ChatOpenAI')
def test_get_llm_with_model_override(mock_openai):
    """Test that model_override overrides the env-configured model."""
    with patch.dict(os.environ, {
        "OPENAI_MODEL": "gpt-4",
        "ACTIVE_LLM": "openai",
        "OPENAI_API_KEY": "fake-key",
    }, clear=True):
        llm = get_llm(model_override="gpt-4o")
        mock_openai.assert_called_once_with(model="gpt-4o")
        assert llm == mock_openai.return_value


@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_with_model_override_gemini(mock_chat_gemini):
    """Test that model_override works with Gemini provider."""
    with patch.dict(os.environ, {
        "GEMINI_MODEL": "gemini-pro",
        "ACTIVE_LLM": "gemini",
        "GOOGLE_API_KEY": "fake-key",
    }, clear=True):
        llm = get_llm(model_override="gemini-2.0-flash")
        mock_chat_gemini.assert_called_once_with(model="gemini-2.0-flash")
        assert llm == mock_chat_gemini.return_value


@patch('app.services.llm.ChatOpenAI')
def test_get_llm_with_model_uses_get_llm(mock_openai):
    """Test that get_llm_with_model delegates to get_llm with the model override."""
    from app.services.llm import get_llm_with_model
    with patch.dict(os.environ, {
        "OPENAI_MODEL": "gpt-4",
        "ACTIVE_LLM": "openai",
        "OPENAI_API_KEY": "fake-key",
    }, clear=True):
        llm = get_llm_with_model("gpt-4o-mini")
        mock_openai.assert_called_once_with(model="gpt-4o-mini")
        assert llm == mock_openai.return_value

