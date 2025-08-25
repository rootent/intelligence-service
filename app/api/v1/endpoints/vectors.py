from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.db.qdrant_client import search_similar_vectors
from app.services.embedding_service import generate_embedding

router = APIRouter()

class VectorSearchRequest(BaseModel):
    vector: List[float]
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class TextVectorSearchRequest(BaseModel):
    text: str
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

@router.post("/search")
async def search_vectors(request: VectorSearchRequest):
    try:
        results = await search_similar_vectors(
            vector=request.vector,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        return {
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/text")
async def search_vectors_by_text(request: TextVectorSearchRequest):
    try:
        embedding = await generate_embedding(request.text)
        results = await search_similar_vectors(
            vector=embedding,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        return {
            "query_text": request.text,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed")
async def generate_text_embedding(text: str):
    try:
        embedding = await generate_embedding(text)
        return {
            "text": text,
            "embedding": embedding,
            "dimension": len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))