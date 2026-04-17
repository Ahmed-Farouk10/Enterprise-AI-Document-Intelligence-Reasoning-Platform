from pydantic import BaseModel, ConfigDict, Field, AliasPath, AliasChoices
from pydantic.alias_generators import to_camel
from typing import Optional, List, Dict, Any
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

class DocumentRead(DocumentBase):
    id: str
    status: str
    # Map database 'created_at' (on DB model) to 'uploadedAt' (on Frontend JSON)
    uploaded_at: datetime = Field(validation_alias=AliasChoices("created_at", "uploaded_at"), alias="uploadedAt")
    # Map database 'updated_at' to 'processedAt'
    processed_at: Optional[datetime] = Field(None, validation_alias=AliasChoices("updated_at", "processed_at"), alias="processedAt")
    version: int = 1
    # CRITICAL: Map database 'extra_data' to 'metadata'. 
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        validation_alias=AliasChoices("extra_data", "metadata"), 
        alias="metadata"
    )

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


class DocumentResponse(DocumentRead):
    pass

# Chat Models
class ChatMessageCreate(CamelModel):
    content: str
    role: str = "user"

class ChatMessageResponse(CamelModel):
    id: str
    role: str
    content: str
    # Map database 'timestamp' to 'timestamp' (already named correctly but standardizing)
    timestamp: datetime = Field(validation_alias="timestamp", alias="timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, validation_alias="extra_data", alias="metadata")

class ChatSessionCreate(CamelModel):
    title: Optional[str] = None
    document_ids: Optional[List[str]] = None

class ChatSessionResponse(CamelModel):
    id: str
    title: str
    created_at: datetime = Field(validation_alias="created_at", alias="createdAt")
    updated_at: datetime = Field(validation_alias="updated_at", alias="updatedAt")
    messages: List[ChatMessageResponse] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)

# Pagination
class PaginationMeta(CamelModel):
    page: int
    page_size: int = Field(alias="pageSize")
    total: int
    total_pages: int = Field(alias="totalPages")

class PaginatedDocuments(CamelModel):
    items: List[DocumentResponse]
    meta: PaginationMeta

# Generic API Response
class ApiResponse(CamelModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[Any] = None
