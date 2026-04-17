from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Optional, List
from datetime import datetime

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True
    )

# Document Models
class DocumentBase(CamelModel):
    filename: str
    original_name: str
    file_size: int
    mime_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentMetadata(CamelModel):
    page_count: Optional[int] = None
    extracted_text: Optional[str] = None
    vector_store_id: Optional[str] = None

class DocumentResponse(DocumentBase):
    id: str
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    status: str  # uploading, processing, completed, failed
    version: int
    metadata: Optional[DocumentMetadata] = None

# Chat Models
class ChatMessageBase(CamelModel):
    content: str

class ChatMessageCreate(ChatMessageBase):
    document_ids: Optional[List[str]] = None

class DocumentContext(CamelModel):
    document_id: Optional[str] = None
    document_name: str
    num_documents: Optional[int] = 1
    relevant_chunks: Optional[List[str]] = None

class ChatMessageResponse(CamelModel):
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    document_context: Optional[DocumentContext] = None

class ChatSessionCreate(CamelModel):
    title: Optional[str] = None
    document_ids: Optional[List[str]] = None

class ChatSessionResponse(CamelModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []
    document_ids: List[str] = []

# Pagination
class PaginationMeta(CamelModel):
    page: int
    page_size: int
    total: int
    total_pages: int

class PaginatedDocuments(CamelModel):
    items: List[DocumentResponse]
    meta: PaginationMeta

# Generic API Response
class ApiResponse(CamelModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[dict] = None
