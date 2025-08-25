from fastapi import APIRouter
from app.api.v1.endpoints import documents, vectors, health, llm, voice

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
