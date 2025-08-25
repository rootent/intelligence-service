from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Match, MatchValue
from typing import List, Optional, Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QdrantDB:
    client: Optional[QdrantClient] = None

qdrant_db = QdrantDB()

def get_qdrant_client() -> QdrantClient:
    return qdrant_db.client

async def connect_to_qdrant():
    qdrant_db.client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
    )

    try:
        collections = qdrant_db.client.get_collections()
        logger.info(f"Connected to Qdrant. Available collections: {len(collections.collections)}")
        
        # Create document collection if it doesn't exist
        if not any(col.name == settings.QDRANT_COLLECTION_NAME for col in collections.collections):
            create_voiceprint_hybrid_collection()
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        raise

def close_qdrant_connection():
    logger.info("Closing Qdrant connection...")
    if qdrant_db.client:
        qdrant_db.client.close()
    logger.info("Qdrant connection closed")

def create_voiceprint_hybrid_collection():
    """Create a hybrid collection with both dense and sparse vectors for semantic and keyword search"""
    logger.info(f"Creating hybrid Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
    
    # Create collection with hybrid vectors
    qdrant_db.client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=settings.VECTOR_DIMENSION,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": {}
        }
    )
    
    # Create payload indexes for better filtering
    qdrant_db.client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="userId",
        field_schema="keyword"
    )
    
    qdrant_db.client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="name",
        field_schema="keyword"
    )
    
    qdrant_db.client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="tags",
        field_schema="keyword"
    )
    
    qdrant_db.client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="created_at",
        field_schema="datetime"
    )
    
    logger.info(f"Hybrid collection {settings.QDRANT_COLLECTION_NAME} created successfully")

def insert_vector(
    document_id: str, 
    vector: List[float], 
    payload: Dict[str, Any]
) -> bool:
    try:
        point = PointStruct(
            id=document_id,
            vector=vector,
            payload=payload
        )
        
        result = qdrant_db.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[point]
        )
        
        return result.status == "completed"
    except Exception as e:
        logger.error(f"Error inserting vector: {e}")
        return False

def search_similar_vectors(
    vector: List[float],
    limit: int = 10,
    score_threshold: float = 0.7,
    filter_conditions: Optional[Filter] = None
) -> List[Dict[str, Any]]:
    try:
        results = qdrant_db.client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_conditions
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        return []

def delete_vector(document_id: str) -> bool:
    try:
        result = qdrant_db.client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=[document_id]
        )
        return result.status == "completed"
    except Exception as e:
        logger.error(f"Error deleting vector: {e}")
        return False
