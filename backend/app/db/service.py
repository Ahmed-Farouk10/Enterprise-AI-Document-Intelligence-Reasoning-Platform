"""
Database service layer - provides high-level database operations using SQLAlchemy
Replaces the old JSON-based database.py
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import uuid

from app.db.models import Document, ChatSession, Message
from app.db.database import get_db


class DatabaseService:
    """Service layer for database operations"""
    
    # Document operations
    @staticmethod
    def create_document(
        db: Session,
        filename: str,
        original_name: str,
        file_size: int,
        mime_type: str,
        version: int = 1
    ) -> Document:
        """Create a new document record"""
        doc = Document(
            id=str(uuid.uuid4()),
            filename=filename,
            original_name=original_name,
            file_size=file_size,
            mime_type=mime_type,
            version=version,
            status="completed",
            extra_data={}
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    
    @staticmethod
    def get_documents(
        db: Session,
        skip: int = 0,
        limit: int = 20
    ) -> tuple[List[Document], int]:
        """Get paginated list of documents"""
        total = db.query(Document).count()
        docs = db.query(Document)\
            .order_by(Document.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        return docs, total
    
    @staticmethod
    def get_document(db: Session, doc_id: str) -> Optional[Document]:
        """Get a single document by ID"""
        return db.query(Document).filter(Document.id == doc_id).first()
    
    @staticmethod
    def delete_document(db: Session, doc_id: str) -> bool:
        """Delete a document"""
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            return True
        return False
    
    # Chat session operations
    @staticmethod
    def create_chat_session(
        db: Session,
        title: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> ChatSession:
        """Create a new chat session"""
        session_count = db.query(ChatSession).count()
        session = ChatSession(
            id=str(uuid.uuid4()),
            title=title or f"Chat {session_count + 1}",
            document_ids=document_ids or [],
            extra_data={}
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_chat_session(db: Session, session_id: str) -> Optional[ChatSession]:
        """Get a chat session with its messages"""
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    @staticmethod
    def get_all_sessions(db: Session) -> List[ChatSession]:
        """Get all chat sessions ordered by most recent"""
        return db.query(ChatSession)\
            .order_by(ChatSession.updated_at.desc())\
            .all()
    
    @staticmethod
    def delete_chat_session(db: Session, session_id: str) -> bool:
        """Delete a chat session and its messages"""
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)  # Cascade will delete messages
            db.commit()
            return True
        return False
    
    # Message operations
    @staticmethod
    def create_message(
        db: Session,
        session_id: str,
        role: str,
        content: str,
        document_context: Optional[Dict] = None
    ) -> Message:
        """Create a new message in a chat session"""
        message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            extra_data=document_context or {}
        )
        db.add(message)
        
        # Update session timestamp
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    def get_session_messages(db: Session, session_id: str) -> List[Message]:
        """Get all messages for a session"""
        return db.query(Message)\
            .filter(Message.session_id == session_id)\
            .order_by(Message.timestamp.asc())\
            .all()
