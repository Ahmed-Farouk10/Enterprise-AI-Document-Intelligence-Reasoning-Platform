from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Document Models
class DocumentBase(BaseModel):
    filename: str
    original_name: str
    file_size: int
    mime_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentMetadata(BaseModel):
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

    class Config:
        from_attributes = True

# Chat Models
class ChatMessageBase(BaseModel):
    content: str

class ChatMessageCreate(ChatMessageBase):
    document_ids: Optional[List[str]] = None

class DocumentContext(BaseModel):
    document_id: str
    document_name: str
    relevant_chunks: Optional[List[str]] = None

class ChatMessageResponse(BaseModel):
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    document_context: Optional[DocumentContext] = None

    class Config:
        from_attributes = True

class ChatSessionCreate(BaseModel):
    title: Optional[str] = None
    document_ids: Optional[List[str]] = None

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = []
    document_ids: List[str] = []

    class Config:
        from_attributes = True

# Pagination
class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int

class PaginatedDocuments(BaseModel):
    items: List[DocumentResponse]
    meta: PaginationMeta

# Generic API Response
class ApiResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[dict] = None
