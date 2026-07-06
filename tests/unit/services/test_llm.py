import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import WebSocketDisconnect

from app.services.llm import (
    get_llm,
    get_active_llm,
    get_llm_model,
)

@patch('app.services.llm.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model", "ACTIVE_LLM": "ollama", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434", disable_streaming=None)
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_gemini(mock_chat_gemini):
    with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-pro", "ACTIVE_LLM": "gemini", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro", disable_streaming=None)
        assert llm == mock_chat_gemini.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai(mock_openai):
    with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4", "ACTIVE_LLM": "openai", "OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4", disable_streaming=None)
        assert llm == mock_openai.return_value

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
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://mock-server:8000", disable_streaming=None)
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
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434", disable_streaming=None)
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
        mock_chat_gemini.assert_called_once_with(model="gemini-pro", base_url="http://mock-server:8000", transport="rest", disable_streaming=None)
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
        mock_openai.assert_called_once_with(model="gpt-4", base_url="http://mock-server:8000", disable_streaming=None)
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
        mock_openai.assert_called_once_with(model="gpt-4", base_url="http://custom-openai:8000", disable_streaming=None)
        assert llm == mock_openai.return_value

@patch('app.services.llm.ChatOllama')
def test_get_llm_with_streaming_disabled(mock_chat_ollama):
    """Test that disable_streaming='tool_calling' is passed when DISABLE_STREAMING=true"""
    with patch.dict(os.environ, {
        "OLLAMA_MODEL": "test-model",
        "ACTIVE_LLM": "ollama",
        "OLLAMA_URL": "http://localhost:11434",
        "DISABLE_STREAMING": "true"
    }, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434", disable_streaming="tool_calling")
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_gemini_with_streaming_disabled(mock_chat_gemini):
    """Test that disable_streaming='tool_calling' is passed for Gemini when DISABLE_STREAMING=true"""
    with patch.dict(os.environ, {
        "GEMINI_MODEL": "gemini-pro",
        "ACTIVE_LLM": "gemini",
        "GOOGLE_API_KEY": "fake-key",
        "DISABLE_STREAMING": "true"
    }, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro", disable_streaming="tool_calling")
        assert llm == mock_chat_gemini.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai_with_streaming_disabled(mock_openai):
    """Test that disable_streaming='tool_calling' is passed for OpenAI when DISABLE_STREAMING=true"""
    with patch.dict(os.environ, {
        "OPENAI_MODEL": "gpt-4",
        "ACTIVE_LLM": "openai",
        "OPENAI_API_KEY": "fake-key",
        "DISABLE_STREAMING": "true"
    }, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4", disable_streaming="tool_calling")
        assert llm == mock_openai.return_value

@patch('app.services.llm.ChatBedrockConverse')
def test_get_llm_bedrock_with_streaming_disabled(mock_bedrock):
    """Test that disable_streaming='tool_calling' is passed for Bedrock when DISABLE_STREAMING=true"""
    with patch.dict(os.environ, {
        "BEDROCK_MODEL": "anthropic.claude-3-sonnet-20240229-v1:0",
        "ACTIVE_LLM": "bedrock",
        "AWS_REGION": "us-east-1",
        "DISABLE_STREAMING": "true"
    }, clear=True):
        llm = get_llm()
        mock_bedrock.assert_called_once_with(model="anthropic.claude-3-sonnet-20240229-v1:0", disable_streaming="tool_calling")
        assert llm == mock_bedrock.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_generic_openai(mock_openai):
    """Test that generic-openai provider uses custom URL and API key"""
    with patch.dict(os.environ, {
        "GENERIC_OPENAI_MODEL": "custom-model",
        "ACTIVE_LLM": "generic-openai",
        "GENERIC_OPENAI_URL": "http://custom-endpoint:8000",
        "GENERIC_OPENAI_API_KEY": "custom-key"
    }, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(
            model="custom-model",
            base_url="http://custom-endpoint:8000",
            api_key="custom-key",
            disable_streaming=None
        )
        assert llm == mock_openai.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_generic_openai_with_streaming_disabled(mock_openai):
    """Test that disable_streaming='tool_calling' is passed for generic-openai when DISABLE_STREAMING=true"""
    with patch.dict(os.environ, {
        "GENERIC_OPENAI_MODEL": "custom-model",
        "ACTIVE_LLM": "generic-openai",
        "GENERIC_OPENAI_URL": "http://custom-endpoint:8000",
        "GENERIC_OPENAI_API_KEY": "custom-key",
        "DISABLE_STREAMING": "true"
    }, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(
            model="custom-model",
            base_url="http://custom-endpoint:8000",
            api_key="custom-key",
            disable_streaming="tool_calling"
        )
        assert llm == mock_openai.return_value

