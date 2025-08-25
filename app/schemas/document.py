from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    vector_id: Optional[str] = None
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None

class VectorSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None

class VectorSearchResponse(BaseModel):
    documents: List[DocumentResponse]
    total_found: int