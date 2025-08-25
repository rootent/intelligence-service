from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.schemas.document import (
    DocumentCreate, 
    DocumentUpdate, 
    DocumentResponse,
    VectorSearchRequest,
    VectorSearchResponse
)
from app.services.document_service import document_service

router = APIRouter()

@router.post("/", response_model=DocumentResponse)
async def create_document(document: DocumentCreate):
    try:
        created_document = await document_service.create_document(document)
        return DocumentResponse(
            id=str(created_document.id),
            title=created_document.title,
            content=created_document.content,
            vector_id=created_document.vector_id,
            tags=created_document.tags,
            metadata=created_document.metadata,
            created_at=created_document.created_at,
            updated_at=created_document.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    document = await document_service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=str(document.id),
        title=document.title,
        content=document.content,
        vector_id=document.vector_id,
        tags=document.tags,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at
    )

@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    tag_list = tags.split(",") if tags else None
    documents = await document_service.get_documents(skip=skip, limit=limit, tags=tag_list)
    
    return [
        DocumentResponse(
            id=str(doc.id),
            title=doc.title,
            content=doc.content,
            vector_id=doc.vector_id,
            tags=doc.tags,
            metadata=doc.metadata,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
        for doc in documents
    ]

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(document_id: str, document_update: DocumentUpdate):
    updated_document = await document_service.update_document(document_id, document_update)
    if not updated_document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=str(updated_document.id),
        title=updated_document.title,
        content=updated_document.content,
        vector_id=updated_document.vector_id,
        tags=updated_document.tags,
        metadata=updated_document.metadata,
        created_at=updated_document.created_at,
        updated_at=updated_document.updated_at
    )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    deleted = await document_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}

@router.post("/search", response_model=VectorSearchResponse)
async def search_documents(search_request: VectorSearchRequest):
    try:
        results = await document_service.search_similar_documents(
            query=search_request.query,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            tags=search_request.tags
        )
        
        documents = []
        for result in results:
            doc = result["document"]
            documents.append(DocumentResponse(
                id=str(doc.id),
                title=doc.title,
                content=doc.content,
                vector_id=doc.vector_id,
                tags=doc.tags,
                metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            ))
        
        return VectorSearchResponse(
            documents=documents,
            total_found=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))