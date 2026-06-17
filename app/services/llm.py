import os
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_aws import ChatBedrockConverse

VALID_PROVIDERS = ("ollama", "gemini", "openai", "bedrock")

ROLE_ENV_VARS = {
    "supervisor": ("SUPERVISOR_LLM", "SUPERVISOR_MODEL"),
    "uitools": ("UITOOLS_LLM", "UITOOLS_MODEL"),
    "summary": ("SUMMARY_LLM", "SUMMARY_MODEL"),
}


class LLMManager:
    _instances: dict[str, BaseLanguageModel] = {}

    @classmethod
    def get_instance(cls, key: str = "default") -> BaseLanguageModel:
        if key not in cls._instances:
            cls._instances[key] = get_llm()
        return cls._instances[key]

    @classmethod
    def get_instance_for_role(cls, role: str) -> BaseLanguageModel:
        if role in cls._instances:
            return cls._instances[role]
        llm = get_llm_for_role(role)
        cls._instances[role] = llm
        return llm

    @classmethod
    def get_instance_for_agent(cls, agent_name: str, llm_provider: Optional[str], llm_model: Optional[str]) -> BaseLanguageModel:
        key = f"agent:{agent_name}"
        if key in cls._instances:
            return cls._instances[key]
        llm = get_llm_for_agent(llm_provider, llm_model)
        cls._instances[key] = llm
        return llm

    @classmethod
    def reset(cls):
        cls._instances = {}


def get_llm(llm_provider: Optional[str] = None, model: Optional[str] = None) -> BaseLanguageModel:
    if llm_provider is None:
        llm_provider = get_active_llm()
    if model is None:
        model = get_llm_model(llm_provider)

    llm_mock_enabled = os.environ.get("LLM_MOCK_ENABLED", "false").lower() == "true"
    llm_mock_url = os.environ.get("LLM_MOCK_URL", "")
    if llm_mock_enabled:
        logging.info(f"Connecting to LLM Mock server at {llm_mock_url}")

    if llm_provider == "ollama":
        if llm_mock_enabled:
            return ChatOllama(model=model, base_url=llm_mock_url)
        ollama_url = os.environ.get("OLLAMA_URL")
        return ChatOllama(model=model, base_url=ollama_url)
    if llm_provider == "gemini":
        if llm_mock_enabled:
            return ChatGoogleGenerativeAI(
                model=model,
                base_url=llm_mock_url,
                transport="rest"
            )
        if model == "gemini-2.5-flash":
             return ChatGoogleGenerativeAI(model=model, thinking_budget=0)
        return ChatGoogleGenerativeAI(model=model)
    if llm_provider == "openai":
        if llm_mock_enabled:
            return ChatOpenAI(model=model, base_url=llm_mock_url)
        openai_url = os.environ.get("OPENAI_URL")
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        return ChatOpenAI(model=model)
    if llm_provider == "bedrock":
        if llm_mock_enabled:
            os.environ["AWS_ENDPOINT_URL"] = llm_mock_url
        return ChatBedrockConverse(model=model)

    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def get_llm_for_role(role: str) -> BaseLanguageModel:
    env_llm, env_model = ROLE_ENV_VARS.get(role, (None, None))
    if not env_llm:
        raise ValueError(f"Unknown role: {role}")

    provider = os.environ.get(env_llm, "").strip() or None
    model = os.environ.get(env_model, "").strip() or None

    if provider and provider not in VALID_PROVIDERS:
        raise ValueError(f"Invalid LLM provider '{provider}' for role '{role}'.")

    if provider and model:
        return get_llm(llm_provider=provider, model=model)
    return get_llm()


def get_llm_for_agent(llm_provider: Optional[str], llm_model: Optional[str]) -> BaseLanguageModel:
    provider = (llm_provider or "").strip() or None
    model = (llm_model or "").strip() or None

    if provider and provider not in VALID_PROVIDERS:
        raise ValueError(f"Invalid LLM provider '{provider}' for agent.")

    if provider and model:
        return get_llm(llm_provider=provider, model=model)
    return get_llm()


def get_active_llm() -> str:
    llm = os.environ.get("ACTIVE_LLM", "")
    if llm not in VALID_PROVIDERS:
        raise ValueError("LLM not configured.")
    return llm


def get_llm_model(llm: str) -> str:
    model = None
    if llm:
        model = os.environ.get(f"{llm.upper()}_MODEL")
    if not model:
        raise ValueError("LLM Model not configured.")
    return model
