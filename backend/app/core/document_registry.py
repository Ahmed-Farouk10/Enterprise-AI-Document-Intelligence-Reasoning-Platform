from datetime import datetime
from typing import Optional, Dict, Any, List
import hashlib
import json
from sqlalchemy import Column, String, DateTime, Text, Integer, create_engine, select
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.session_manager import Base, SessionManager 
# Re-using Base/Engine from session_manager or creating new? 
# The user's prompt suggests sharing it or having separate. 
# For simplicity and avoiding circular imports, I'll use the one from session_manager if possible, 
# or just define the model here and assume a shared engine setup in a real app.
# But provided code imports Base from sqlalchemy.ext.declarative which is deprecated.
# I will use the modern approach but keep it self-contained as per the snippet.

from app.core.logging_config import get_logger

logger = get_logger(__name__)

class DocumentVersion(Base):
    __tablename__ = "document_versions"
    
    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256
    extracted_entities = Column(Text)  # JSON
    vector_ids = Column(Text)  # JSON list of vector IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    change_summary = Column(Text)  # AI-generated summary of changes

class DocumentRegistry:
    """Central registry tracking all documents, their versions, and processing status."""
    
    def __init__(self, db_engine):
        self.Session = sessionmaker(bind=db_engine)
        # Ensure table exists
        Base.metadata.create_all(db_engine)
        
    async def register_document(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """Register new document or detect changes to existing."""
        content_hash = await self._compute_hash(file_path)
        
        # Check for existing document with same hash (exact duplicate)
        existing = await self._find_by_hash(content_hash)
        if existing:
            return {
                "document_id": existing["document_id"],
                "status": "duplicate",
                "version": existing["version_number"],
                "message": "Document already exists, no processing needed"
            }
        
        # Check for document with same name but different content (update)
        # For MVP, we presume finding by ID or Path. 
        # The user snippet implies _find_by_path
        similar = await self._find_by_path(file_path)
        if similar:
             # In a real impl, we'd need the doc ID from the 'similar' record
            return await self._create_version(similar["document_id"], file_path, content_hash)
        
        # New document
        import uuid
        new_doc_id = str(uuid.uuid4())
        return await self._create_new_document(new_doc_id, file_path, content_hash, user_id)

    async def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def _find_by_hash(self, content_hash: str) -> Optional[Dict]:
        session = self.Session()
        try:
            record = session.query(DocumentVersion).filter_by(content_hash=content_hash).first()
            if record:
                return {
                    "document_id": record.document_id,
                    "version_number": record.version_number
                }
            return None
        finally:
            session.close()

    async def _find_by_path(self, file_path: str) -> Optional[Dict]:
        # Placeholder: In a real system we'd check a 'Documents' table mapping path -> ID.
        # For now, returning None to treat everything as new or duplicates.
        return None

    async def _create_new_document(self, doc_id: str, file_path: str, content_hash: str, user_id: str):
        session = self.Session()
        import uuid
        try:
            version = DocumentVersion(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                version_number=1,
                content_hash=content_hash,
                extracted_entities="[]",
                vector_ids="[]",
                change_summary="Initial upload"
            )
            session.add(version)
            session.commit()
            return {
                "document_id": doc_id,
                "status": "created",
                "version": 1
            }
        finally:
            session.close()

    async def _create_version(self, document_id: str, file_path: str, content_hash: str):
        """Create new version of existing document."""
        # This would involve delta logic
        return {
            "document_id": document_id,
            "status": "updated",
            "version": 2, # simplified
            "changes": "Updated content"
        }

# Singleton (needs engine from session_manager)
from app.core.session_manager import session_manager
document_registry = DocumentRegistry(session_manager.engine)
