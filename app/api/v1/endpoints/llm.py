from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel, Field
from app.services.llm.factory import LLMFactory
from app.services.llm.base import ChatMessage, LLMProvider
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: str = Field(..., description="LLM provider (openai, azure_openai, gemini)")
    model: Optional[str] = Field(None, description="Model name (optional, uses default)")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8000)
    stream: bool = Field(False, description="Enable streaming response")

class EmbeddingRequest(BaseModel):
    text: str
    provider: str = Field(..., description="LLM provider (openai, azure_openai, gemini)")
    model: Optional[str] = None

class ProviderValidationRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None

@router.post("/chat")
async def chat_with_llm(request: ChatRequest):
    try:
        llm = LLMFactory.create_llm(
            provider=request.provider,
            model=request.model
        )
        
        if request.stream:
            async def generate():
                async for chunk in llm.generate_stream(
                    messages=request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = await llm.generate_response(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return response
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    try:
        llm = LLMFactory.create_llm(
            provider=request.provider,
            model=request.model
        )
        
        embedding = await llm.generate_embedding(request.text)
        
        return {
            "text": request.text,
            "embedding": embedding,
            "dimension": len(embedding),
            "provider": request.provider,
            "model": request.model or "default"
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/providers")
async def list_providers():
    return {
        "providers": LLMFactory.list_available_providers(),
        "default_provider": LLMFactory.get_default_llm().get_provider_name()
    }

@router.post("/validate")
async def validate_provider(request: ProviderValidationRequest):
    try:
        kwargs = {}
        if request.api_key:
            kwargs["api_key"] = request.api_key
        if request.azure_endpoint:
            kwargs["azure_endpoint"] = request.azure_endpoint
        if request.deployment_name:
            kwargs["deployment_name"] = request.deployment_name
            
        is_valid = await LLMFactory.validate_provider(
            provider=request.provider,
            **kwargs
        )
        
        return {
            "provider": request.provider,
            "is_valid": is_valid,
            "message": "Connection successful" if is_valid else "Connection failed"
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "provider": request.provider,
            "is_valid": False,
            "message": str(e)
        }

@router.delete("/cache")
async def clear_llm_cache():
    LLMFactory.clear_cache()
    return {"message": "LLM cache cleared successfully"}

@router.get("/default")
async def get_default_llm_info():
    try:
        default_llm = LLMFactory.get_default_llm()
        return {
            "provider": default_llm.get_provider_name(),
            "model": default_llm.model,
            "is_connected": await default_llm.validate_connection()
        }
    except Exception as e:
        logger.error(f"Default LLM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))