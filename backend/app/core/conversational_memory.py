from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from dataclasses import dataclass
import logging

from app.core.redis_adapter import redis_adapter
# Assuming embedding_service exists, otherwise we mock it
try:
    from app.services.embedding_service import embedding_service
except ImportError:
    embedding_service = None

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: List[float]
    memory_type: str
    timestamp: str
    importance: float
    source_session: str
    metadata: Dict[str, Any]

class ConversationalMemory:
    def __init__(self):
        self.redis = redis_adapter
        self.short_term_ttl = timedelta(hours=1)
        self.working_memory_limit = 10
        
    def _memory_key(self, user_id: str, memory_type: str) -> str:
        return f"memory:{user_id}:{memory_type}"
        
    async def store_short_term(self, session_id: str, user_id: str, message: Dict[str, Any]) -> bool:
        key = self._memory_key(user_id, "short_term")
        entry = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": message
        }
        success = await self.redis.lpush(key, entry, max_length=100)
        if success:
            await self.redis.expire(key, self.short_term_ttl)
        return success
        
    async def get_short_term(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        key = self._memory_key(user_id, "short_term")
        return await self.redis.lrange(key, 0, limit - 1)

    async def update_working_memory(self, session_id: str, user_id: str, context_items: List[Dict[str, Any]]) -> bool:
        key = self._memory_key(user_id, f"working:{session_id}")
        working_memory = {
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat(),
            "context_items": context_items[-self.working_memory_limit:]
        }
        return await self.redis.set(key, working_memory, ttl=timedelta(hours=2))
        
    async def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        if not embedding_service: return []
        if query_embedding is None:
            # Mock if service unavailable
            query_embedding = [0.1] * 384
            # query_embedding = await embedding_service.embed(query)
            
        # Simplified: Retrieval logic to be expanded
        return []

conversational_memory = ConversationalMemory()
