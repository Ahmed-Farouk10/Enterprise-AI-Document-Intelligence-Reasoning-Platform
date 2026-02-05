from typing import Dict, List, Optional
from datetime import datetime
import uuid

# In-memory storage (replace with database in production)
documents_db: Dict[str, dict] = {}
chat_sessions_db: Dict[str, dict] = {}
chat_messages_db: Dict[str, List[dict]] = {}

class Database:
    """Simple in-memory database for development"""
    
    # Document operations
    @staticmethod
    def create_document(filename: str, original_name: str, file_size: int, mime_type: str) -> dict:
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "filename": filename,
            "original_name": original_name,
            "file_size": file_size,
            "mime_type": mime_type,
            "uploaded_at": datetime.utcnow(),
            "processed_at": None,
            "status": "completed",  # Simplified for now
            "metadata": {
                "page_count": None,
                "extracted_text": None,
                "vector_store_id": None
            }
        }
        documents_db[doc_id] = document
        return document
    
    @staticmethod
    def get_documents(skip: int = 0, limit: int = 20) -> tuple[List[dict], int]:
        all_docs = list(documents_db.values())
        total = len(all_docs)
        # Sort by upload time descending
        all_docs.sort(key=lambda x: x["uploaded_at"], reverse=True)
        paginated = all_docs[skip:skip + limit]
        return paginated, total
    
    @staticmethod
    def get_document(doc_id: str) -> Optional[dict]:
        return documents_db.get(doc_id)
    
    @staticmethod
    def delete_document(doc_id: str) -> bool:
        if doc_id in documents_db:
            del documents_db[doc_id]
            return True
        return False
    
    # Chat session operations
    @staticmethod
    def create_chat_session(title: Optional[str] = None, document_ids: Optional[List[str]] = None) -> dict:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "title": title or f"Chat {len(chat_sessions_db) + 1}",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "document_ids": document_ids or []
        }
        chat_sessions_db[session_id] = session
        chat_messages_db[session_id] = []
        return session
    
    @staticmethod
    def get_chat_session(session_id: str) -> Optional[dict]:
        session = chat_sessions_db.get(session_id)
        if session:
            session["messages"] = chat_messages_db.get(session_id, [])
        return session
    
    @staticmethod
    def get_all_sessions() -> List[dict]:
        return list(chat_sessions_db.values())
    
    @staticmethod
    def delete_chat_session(session_id: str) -> bool:
        if session_id in chat_sessions_db:
            del chat_sessions_db[session_id]
            if session_id in chat_messages_db:
                del chat_messages_db[session_id]
            return True
        return False
    
    @staticmethod
    def create_message(session_id: str, role: str, content: str, document_context: Optional[dict] = None) -> dict:
        message_id = str(uuid.uuid4())
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "document_context": document_context
        }
        
        if session_id not in chat_messages_db:
            chat_messages_db[session_id] = []
        
        chat_messages_db[session_id].append(message)
        
        # Update session timestamp
        if session_id in chat_sessions_db:
            chat_sessions_db[session_id]["updated_at"] = datetime.utcnow()
        
        return message
