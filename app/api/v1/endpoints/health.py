from fastapi import APIRouter
from app.db.mongodb import get_database
from app.db.qdrant_client import get_qdrant_client

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "healthy", "service": "intelligence-service"}

@router.get("/db")
async def database_health():
    try:
        db = await get_database()
        mongo_status = "connected" if db else "disconnected"
        
        qdrant_client = await get_qdrant_client()
        qdrant_status = "connected" if qdrant_client else "disconnected"
        
        return {
            "mongodb": mongo_status,
            "qdrant": qdrant_status,
            "overall": "healthy" if mongo_status == "connected" and qdrant_status == "connected" else "unhealthy"
        }
    except Exception as e:
        return {
            "mongodb": "error",
            "qdrant": "error", 
            "overall": "unhealthy",
            "error": str(e)
        }