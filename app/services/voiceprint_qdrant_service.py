from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Match, MatchValue, SearchRequest, SparseVector
from app.core.config import settings
from app.db.qdrant_client import get_qdrant_client
from app.services.embedding_service import generate_embedding
import logging
import re
from collections import Counter


logger = logging.getLogger(__name__)

class VoicePrintQdrantHybridService:
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dense_vector_name = "dense"
        self.sparse_vector_name = "sparse"
        
    def get_client(self) -> QdrantClient:
        return get_qdrant_client()

    def search_voice_by_voiceprint(
        self,
        voiceprint: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for voices using dense vector similarity (voiceprint matching)"""
        try:
            # Build filter conditions
            filter_conditions = []
            
            if user_id:
                filter_conditions.append(
                    FieldCondition(key="userId", match=Match(value=MatchValue(value=user_id)))
                )
            
            if name:
                filter_conditions.append(
                    FieldCondition(key="name", match=Match(value=MatchValue(value=name)))
                )
            
            if tags:
                filter_conditions.append(
                    FieldCondition(key="tags", match=Match(any=tags))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Search using dense vector (voiceprint)
            results = self.get_client().search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=("dense", voiceprint),  # Use named vector
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter            )
            
            print(results)

            return [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching voice by voiceprint: {e}")
            return []

    
    def _generate_sparse_vector(self, text: str) -> SparseVector:
        """Generate sparse vector from text using simple TF-IDF like approach"""        
        # Simple tokenization and sparse vector generation
        tokens = re.findall(r'\b\w+\b', text.lower())
        token_counts = Counter(tokens)
        
        # Convert to sparse vector format
        indices = []
        values = []
        for token, count in token_counts.items():
            # Use hash of token as index to ensure consistency
            token_hash = abs(hash(token)) % 100000  # Limit to reasonable range, ensure positive
            indices.append(token_hash)
            values.append(float(count))
        
        return SparseVector(indices=indices, values=values)
    
    def upsert_voice(
        self,
        document_id: str,
        voiceprint: List[float],
        userId: str = "",
        name: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store voiceprint with both dense and sparse vectors for hybrid search"""
        try:
            client = self.get_client()
            
            # Generate sparse vector
            search_text = f"{userId}"
            sparse_vector = self._generate_sparse_vector(search_text)
            
            # Prepare payload
            payload = {
                "userId": userId,
                "name": name,
                "tags": tags or [],
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create point with hybrid vectors
            point = PointStruct(
                id=document_id,
                vector={
                    self.dense_vector_name: voiceprint,
                    self.sparse_vector_name: sparse_vector
                },
                payload=payload
            )
            


            result = client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            print(f"Result: {result}")

            success = result.status == "completed"
            if success:
                logger.info(f"Document {document_id} upserted successfully")
            else:
                logger.error(f"Failed to upsert document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error upserting document {document_id}: {e}")
            return False

    async def upsert_document(
        self,
        document_id: str,
        content: str,
        title: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store document with both dense and sparse vectors for hybrid search"""
        try:
            client = self.get_client()
            
            # Generate dense embedding
            dense_vector = await generate_embedding(content)
            
            # Generate sparse vector
            search_text = f"{title} {content}"
            sparse_vector = self._generate_sparse_vector(search_text)
            
            # Prepare payload
            payload = {
                "title": title,
                "content": content,
                "tags": tags or [],
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Create point with hybrid vectors
            point = PointStruct(
                id=document_id,
                vector={
                    self.dense_vector_name: dense_vector,
                    self.sparse_vector_name: sparse_vector
                },
                payload=payload
            )
            
            result = client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            success = result.status == "completed"
            if success:
                logger.info(f"Document {document_id} upserted successfully")
            else:
                logger.error(f"Failed to upsert document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error upserting document {document_id}: {e}")
            return False
    
# Global service instance
voiceprint_qdrant_hybrid_service = VoicePrintQdrantHybridService()