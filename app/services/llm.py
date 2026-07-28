import os
import logging

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_aws import ChatBedrockConverse

class LLMManager:
    """
    Singleton manager for language model instances.
    
    This class ensures that only one instance of the language model is created
    and reused throughout the application, avoiding redundant initializations
    and ensuring consistent model configuration.
    """
    _instance: BaseLanguageModel = None

    @classmethod
    def get_instance(cls) -> BaseLanguageModel:
        """
        Retrieves the singleton instance of the language model.
        
        If the instance doesn't exist yet, it initializes it by calling get_llm().
        Subsequent calls return the same instance.
        
        Returns:
            The singleton language model instance.
        """
        if cls._instance is None:
            cls._instance = get_llm()
            logging.info(f"Using model: {cls._instance}")
        return cls._instance

def get_llm(model_override: str | None = None) -> BaseChatModel:
    """
    Selects and returns a language model instance based on environment variables.
    - If the active LLM or the model is not configured, it raises a ValueError.
    - If LLM mocking is enabled, it configures the connections to the mock server.
    
    Args:
        model_override: If provided, use this model name instead of the one from environment variables.

    Returns:
        An instance of a language model.
        
    Raises:
        ValueError: If the active LLM or the model is not configured.
    """

    activeLlm = get_active_llm()
    model = model_override if model_override else get_llm_model(activeLlm)
    
    llm_mock_enabled = os.environ.get("LLM_MOCK_ENABLED", "false").lower() == "true"
    llm_mock_url = os.environ.get("LLM_MOCK_URL", "")
    if llm_mock_enabled:
        logging.info(f"Connecting to LLM Mock server at {llm_mock_url}")

    if activeLlm == "ollama":
        if llm_mock_enabled:
            return ChatOllama(model=model, base_url=llm_mock_url)

        ollama_url = os.environ.get("OLLAMA_URL")
        return ChatOllama(model=model, base_url=ollama_url)
    if activeLlm == "gemini":
        if llm_mock_enabled:
            return ChatGoogleGenerativeAI(
                model=model,
                base_url=llm_mock_url,
                transport="rest"
            )
        if model == "gemini-2.5-flash":
             # Disable thinking budget for gemini-2.5-flash to avoid empty responses due to all tokens being used for thinking budget
             return ChatGoogleGenerativeAI(model=model, thinking_budget=0)
        
        return ChatGoogleGenerativeAI(model=model)
    if activeLlm == "openai":
        if llm_mock_enabled:
            return ChatOpenAI(model=model, base_url=llm_mock_url)
        
        openai_url = os.environ.get("OPENAI_URL")
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        return ChatOpenAI(model=model)
    if activeLlm == "bedrock":
        if llm_mock_enabled:
            os.environ["AWS_ENDPOINT_URL"] = llm_mock_url
        bedrock_url = os.environ.get("BEDROCK_URL")
        if bedrock_url:
            return ChatBedrockConverse(model=model, endpoint_url=bedrock_url)
        return ChatBedrockConverse(model=model)
    if activeLlm == "generic-openai":        
        generic_openai_url = os.environ.get("GENERIC_OPENAI_URL")
        generic_openai_api_key = os.environ.get("GENERIC_OPENAI_API_KEY")

        return ChatOpenAI(model=model, base_url=generic_openai_url, api_key=generic_openai_api_key)

    raise ValueError(f"Unsupported LLM provider: {activeLlm}")


def get_llm_with_model(model: str) -> BaseChatModel:
    """Create an LLM instance using the active provider but with a specific model override."""
    return get_llm(model_override=model)


def get_active_llm() -> str:
    """
    Retrieves the active LLM identifier from environment variables.
    
    Returns:
        The active LLM as a string, or None if not set.
    """
    llm = os.environ.get("ACTIVE_LLM", "")
    
    if llm not in ["ollama", "gemini", "openai", "bedrock", "generic-openai"]:
        raise ValueError("LLM not configured.")

    return llm

def get_llm_model(llm: str) -> str:
    """
    Retrieves the model name from environment variables.
    
    Args:
        llm: The LLM identifier, one of 'ollama', 'gemini', 'openai', 'bedrock', 'generic-openai'.

    Returns:
        The model name as a string.
    """

    model = None

    if llm:
        model = os.environ.get(f"{llm.upper().replace('-', '_')}_MODEL")

    if not model:
        raise ValueError("LLM Model not configured.")

    return model

