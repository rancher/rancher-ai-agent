from .services.llm import LLMManager
from langchain_core.language_models.llms import BaseLanguageModel

def get_llm() -> BaseLanguageModel:
    """
    FastAPI dependency that provides the default LLM instance.

    Returns:
        The default language model instance.
    """
    return LLMManager.get_instance()


def get_uitools_llm() -> BaseLanguageModel:
    """
    FastAPI dependency that provides the UI tools LLM instance.

    Returns:
        The UI tools language model instance (falls back to default if not configured).
    """
    return LLMManager.get_instance_for_role("uitools")