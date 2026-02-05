import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime

# File to store database
DB_FILE = "db_data.json"

class Database:
    """JSON-based persistent database"""
    
    # In-memory cache
    _documents: Dict[str, dict] = {}
    _chat_sessions: Dict[str, dict] = {}
    _chat_messages: Dict[str, List[dict]] = {}
    
    @classmethod
    def load_data(cls):
        """Load data from JSON file on startup"""
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "r") as f:
                    data = json.load(f)
                    cls._documents = data.get("documents", {})
                    cls._chat_sessions = data.get("chat_sessions", {})
                    cls._chat_messages = data.get("chat_messages", {})
                print(f"Loaded {len(cls._documents)} documents from disk.")
            except Exception as e:
                print(f"Error loading database: {e}")

    @classmethod
    def save_data(cls):
        """Save current state to JSON file"""
        try:
            data = {
                "documents": cls._documents,
                "chat_sessions": cls._chat_sessions,
                "chat_messages": cls._chat_messages
            }
            with open(DB_FILE, "w") as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")

    # Document operations
    @classmethod
    def create_document(cls, filename: str, original_name: str, file_size: int, mime_type: str) -> dict:
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "filename": filename,
            "original_name": original_name,
            "file_size": file_size,
            "mime_type": mime_type,
            "uploaded_at": datetime.utcnow().isoformat(),
            "processed_at": None,
            "status": "completed", 
            "metadata": {
                "page_count": None,
                "extracted_text": None,
                "vector_store_id": None
            }
        }
        cls._documents[doc_id] = document
        cls.save_data()
        return document
    
    @classmethod
    def get_documents(cls, skip: int = 0, limit: int = 20) -> tuple[List[dict], int]:
        all_docs = list(cls._documents.values())
        total = len(all_docs)
        # Sort by upload time descending
        all_docs.sort(key=lambda x: x["uploaded_at"], reverse=True)
        paginated = all_docs[skip:skip + limit]
        return paginated, total
    
    @classmethod
    def get_document(cls, doc_id: str) -> Optional[dict]:
        return cls._documents.get(doc_id)
    
    @classmethod
    def delete_document(cls, doc_id: str) -> bool:
        if doc_id in cls._documents:
            del cls._documents[doc_id]
            cls.save_data()
            return True
        return False
    
    # Chat session operations
    @classmethod
    def create_chat_session(cls, title: Optional[str] = None, document_ids: Optional[List[str]] = None) -> dict:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "title": title or f"Chat {len(cls._chat_sessions) + 1}",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "document_ids": document_ids or []
        }
        cls._chat_sessions[session_id] = session
        cls._chat_messages[session_id] = []
        cls.save_data()
        return session
    
    @classmethod
    def get_chat_session(cls, session_id: str) -> Optional[dict]:
        session = cls._chat_sessions.get(session_id)
        if session:
            session["messages"] = cls._chat_messages.get(session_id, [])
        return session
    
    @classmethod
    def get_all_sessions(cls) -> List[dict]:
        return list(cls._chat_sessions.values())
    
    @classmethod
    def delete_chat_session(cls, session_id: str) -> bool:
        if session_id in cls._chat_sessions:
            del cls._chat_sessions[session_id]
            if session_id in cls._chat_messages:
                del cls._chat_messages[session_id]
            cls.save_data()
            return True
        return False
    
    @classmethod
    def create_message(cls, session_id: str, role: str, content: str, document_context: Optional[dict] = None) -> dict:
        message_id = str(uuid.uuid4())
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "document_context": document_context
        }
        
        if session_id not in cls._chat_messages:
            cls._chat_messages[session_id] = []
        
        cls._chat_messages[session_id].append(message)
        
        # Update session timestamp
        if session_id in cls._chat_sessions:
            cls._chat_sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
        
        cls.save_data()
        return message

# Initialize by loading data
Database.load_data()
