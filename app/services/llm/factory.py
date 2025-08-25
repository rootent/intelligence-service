from typing import Dict, Any, Optional
from app.services.llm.base import BaseLLM, LLMProvider
from app.services.llm.openai_llm import OpenAILLM
from app.services.llm.azure_openai_llm import AzureOpenAILLM
from app.services.llm.gemini_llm import GeminiLLM
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    _instances: Dict[str, BaseLLM] = {}

    @classmethod
    def create_llm(
        cls,
        provider: str,
        model: str = None,
        **kwargs
    ) -> BaseLLM:
        provider = provider.lower()
        cache_key = f"{provider}_{model}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        if provider == LLMProvider.OPENAI.value:
            llm = cls._create_openai_llm(model, **kwargs)
        elif provider == LLMProvider.AZURE_OPENAI.value:
            llm = cls._create_azure_openai_llm(model, **kwargs)
        elif provider == LLMProvider.GEMINI.value:
            llm = cls._create_gemini_llm(model, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        cls._instances[cache_key] = llm
        logger.info(f"Created {provider} LLM instance with model: {model}")
        return llm

    @classmethod
    def _create_openai_llm(cls, model: str = None, **kwargs) -> OpenAILLM:
        return OpenAILLM(
            model=model or "gpt-3.5-turbo",
            api_key=kwargs.get("api_key", settings.OPENAI_API_KEY),
            **kwargs
        )

    @classmethod
    def _create_azure_openai_llm(cls, model: str = None, **kwargs) -> AzureOpenAILLM:
        return AzureOpenAILLM(
            model=model or "gpt-35-turbo",
            azure_endpoint=kwargs.get("azure_endpoint", settings.AZURE_OPENAI_ENDPOINT),
            api_key=kwargs.get("api_key", settings.AZURE_OPENAI_API_KEY),
            api_version=kwargs.get("api_version", settings.AZURE_OPENAI_API_VERSION),
            deployment_name=kwargs.get("deployment_name", settings.AZURE_OPENAI_DEPLOYMENT_NAME),
            **kwargs
        )

    @classmethod
    def _create_gemini_llm(cls, model: str = None, **kwargs) -> GeminiLLM:
        return GeminiLLM(
            model=model or "gemini-pro",
            api_key=kwargs.get("api_key", settings.GEMINI_API_KEY),
            **kwargs
        )

    @classmethod
    def get_default_llm(cls) -> BaseLLM:
        return cls.create_llm(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=settings.DEFAULT_LLM_MODEL
        )

    @classmethod
    def clear_cache(cls):
        cls._instances.clear()
        logger.info("LLM factory cache cleared")

    @classmethod
    async def validate_provider(cls, provider: str, **kwargs) -> bool:
        try:
            llm = cls.create_llm(provider, **kwargs)
            return await llm.validate_connection()
        except Exception as e:
            logger.error(f"Provider validation failed for {provider}: {e}")
            return False

    @classmethod
    def list_available_providers(cls) -> list:
        return [provider.value for provider in LLMProvider]