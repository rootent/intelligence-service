from typing import List
from app.services.llm.factory import LLMFactory
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

async def generate_embedding(text: str, provider: str = None) -> List[float]:
    """
    Generate embeddings for text using the configured LLM provider.
    """
    try:
        if provider is None:
            provider = settings.DEFAULT_LLM_PROVIDER
        
        llm = LLMFactory.create_llm(provider=provider)
        embedding = await llm.generate_embedding(text)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding with {provider}: {e}")
        # Fallback to dummy embedding if actual service fails
        logger.warning(f"Falling back to dummy embedding of dimension {settings.VECTOR_DIMENSION}")
        return [0.1] * settings.VECTOR_DIMENSION