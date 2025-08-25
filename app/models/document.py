from typing import Optional, List, Dict, Any
from pydantic import Field, ConfigDict
from .base import BaseDocument

class Document(BaseDocument):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Sample Document",
                "content": "This is a sample document content",
                "tags": ["sample", "document"],
                "metadata": {"author": "user", "category": "general"}
            }
        }
    )
    
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    vector_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)