from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.router import api_router
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.qdrant_client import connect_to_qdrant, close_qdrant_connection

@asynccontextmanager
async def lifespan(_: FastAPI):
    await connect_to_mongo()
    await connect_to_qdrant()
    yield
    await close_mongo_connection()
    close_qdrant_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI application with MongoDB and Qdrant integration",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "FastAPI with MongoDB and Qdrant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)