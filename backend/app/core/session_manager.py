from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, AsyncGenerator
import json
import uuid
from dataclasses import dataclass, asdict
import logging

from app.core.redis_adapter import redis_adapter

logger = logging.getLogger(__name__)

@dataclass
class SessionContext:
    """Represents a chat session's complete state."""
    session_id: str
    user_id: str
    created_at: str
    last_activity: str
    messages: List[Dict[str, Any]]
    active_document_ids: List[str]
    user_preferences: Dict[str, Any]
    metadata: Dict[str, Any]
    
    @classmethod
    def create(cls, user_id: str, session_id: Optional[str] = None) -> "SessionContext":
        now = datetime.utcnow().isoformat()
        return cls(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            created_at=now,
            last_activity=now,
            messages=[],
            active_document_ids=[],
            user_preferences={},
            metadata={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        return cls(**data)


class SessionManager:
    """
    Manages user sessions with Redis persistence.
    Ensures continuity across page refreshes and server restarts.
    """
    
    def __init__(self):
        self.redis = redis_adapter
        self.session_ttl = timedelta(hours=24)  # 24 hour default TTL
        self.max_messages = 50  # Keep last 50 messages in active session
    
    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        return f"user_sessions:{user_id}"
    
    async def create_session(
        self,
        user_id: str,
        initial_documents: Optional[List[str]] = None,
        preferences: Optional[Dict] = None
    ) -> SessionContext:
        """Create new session for user."""
        session = SessionContext.create(user_id)
        
        if initial_documents:
            session.active_document_ids = initial_documents
        
        if preferences:
            session.user_preferences = preferences
        
        # Store in Redis
        await self._persist_session(session)
        
        # Add to user's session list
        await self.redis.sadd(self._user_sessions_key(user_id), session.session_id)
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve session by ID."""
        try:
            data = await self.redis.get(self._session_key(session_id))
            if data is None:
                logger.warning(f"Session {session_id} not found or expired")
                return None
            
            session = SessionContext.from_dict(data)
            
            # Update last activity
            session.last_activity = datetime.utcnow().isoformat()
            await self._persist_session(session)
            
            return session
            
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    async def get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        create_if_missing: bool = True
    ) -> Optional[SessionContext]:
        """
        Get existing session or create new one.
        This is the main entry point for session continuity.
        """
        if session_id:
            existing = await self.get_session(session_id)
            if existing:
                # Verify user ownership
                if existing.user_id != user_id:
                    logger.warning(f"Session {session_id} user mismatch")
                    return None
                return existing
        
        if create_if_missing:
            return await self.create_session(user_id)
        
        return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_activity = datetime.utcnow().isoformat()
        return await self._persist_session(session)
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        session.messages.append(message)
        
        if len(session.messages) > self.max_messages:
            session.messages = session.messages[-self.max_messages:]
        
        return await self._persist_session(session)
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20,
        include_system: bool = False
    ) -> List[Dict[str, Any]]:
        session = await self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if not include_system:
            messages = [m for m in messages if m["role"] != "system"]
        
        return messages[-limit:]

    async def update_session_context(self, session_id: str, message: Dict, document_ids: List[str] = None):
        """Backwards compatibility wrapper for previous implementation."""
        if document_ids:
            await self.update_session(session_id, {"active_document_ids": document_ids})
        
        await self.add_message(session_id, message.get("role", "user"), message.get("content", ""))

    async def clear_session(self, session_id: str) -> bool:
        session = await self.get_session(session_id)
        if not session: return False
        session.messages = []
        session.active_document_ids = []
        session.metadata["cleared_at"] = datetime.utcnow().isoformat()
        return await self._persist_session(session)
    
    async def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        session_ids = await self.redis.smembers(self._user_sessions_key(user_id))
        sessions = []
        for sid in session_ids:
            session = await self.get_session(sid)
            if session:
                sessions.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                    "message_count": len(session.messages)
                })
        return sorted(sessions, key=lambda x: x["last_activity"], reverse=True)

    async def _persist_session(self, session: SessionContext) -> bool:
        key = self._session_key(session.session_id)
        return await self.redis.set(key, session.to_dict(), ttl=self.session_ttl)


# Global instance
session_manager = SessionManager()
