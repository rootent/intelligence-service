from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentUpdate
from app.db.mongodb import get_database
from app.db.qdrant_client import insert_vector, search_similar_vectors, delete_vector
from app.services.embedding_service import generate_embedding

class DocumentService:
    def __init__(self):
        self.collection_name = "documents"

    async def get_database(self) -> AsyncIOMotorDatabase:
        return await get_database()

    async def create_document(self, document_data: DocumentCreate) -> Document:
        db = await self.get_database()
        collection = db[self.collection_name]
        
        document_dict = document_data.model_dump()
        document_dict["created_at"] = datetime.now(timezone.utc)
        
        result = await collection.insert_one(document_dict)
        document_dict["_id"] = result.inserted_id
        
        document = Document(**document_dict)
        
        try:
            embedding = await generate_embedding(document_data.content)
            vector_inserted = await insert_vector(
                document_id=str(result.inserted_id),
                vector=embedding,
                payload={
                    "title": document_data.title,
                    "tags": document_data.tags,
                    "metadata": document_data.metadata
                }
            )
            
            if vector_inserted:
                await collection.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"vector_id": str(result.inserted_id)}}
                )
                document.vector_id = str(result.inserted_id)
                
        except Exception as e:
            print(f"Failed to create vector embedding: {e}")
        
        return document

    async def get_document(self, document_id: str) -> Optional[Document]:
        db = await self.get_database()
        collection = db[self.collection_name]
        
        document_data = await collection.find_one({"_id": ObjectId(document_id)})
        if document_data:
            return Document(**document_data)
        return None

    async def get_documents(
        self, 
        skip: int = 0, 
        limit: int = 10,
        tags: Optional[List[str]] = None
    ) -> List[Document]:
        db = await self.get_database()
        collection = db[self.collection_name]
        
        filter_query = {}
        if tags:
            filter_query["tags"] = {"$in": tags}
        
        cursor = collection.find(filter_query).skip(skip).limit(limit)
        documents = []
        async for document_data in cursor:
            documents.append(Document(**document_data))
        
        return documents

    async def update_document(
        self, 
        document_id: str, 
        document_update: DocumentUpdate
    ) -> Optional[Document]:
        db = await self.get_database()
        collection = db[self.collection_name]
        
        update_data = {k: v for k, v in document_update.model_dump().items() if v is not None}
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        result = await collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_data}
        )
        
        if result.modified_count:
            return await self.get_document(document_id)
        return None

    async def delete_document(self, document_id: str) -> bool:
        db = await self.get_database()
        collection = db[self.collection_name]
        
        document = await self.get_document(document_id)
        if not document:
            return False
            
        result = await collection.delete_one({"_id": ObjectId(document_id)})
        
        if result.deleted_count:
            if document.vector_id:
                await delete_vector(document.vector_id)
            return True
        return False

    async def search_similar_documents(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            embedding = await generate_embedding(query)
            
            filter_conditions = None
            if tags:
                filter_conditions = {"must": [{"key": "tags", "match": {"any": tags}}]}
            
            similar_vectors = await search_similar_vectors(
                vector=embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions
            )
            
            documents = []
            for vector_result in similar_vectors:
                document = await self.get_document(vector_result["id"])
                if document:
                    documents.append({
                        "document": document,
                        "score": vector_result["score"]
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

document_service = DocumentService()