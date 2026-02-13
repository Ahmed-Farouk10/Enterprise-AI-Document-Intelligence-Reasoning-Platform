from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from dataclasses import dataclass
import logging

from app.core.redis_adapter import redis_adapter
from app.services.embeddings import SentenceTransformerEmbeddingEngine

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
        try:
            self.embedding_engine = SentenceTransformerEmbeddingEngine()
        except Exception as e:
            logger.error(f"Failed to load embedding engine for memory: {e}")
            self.embedding_engine = None
        
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
        if not self.embedding_engine:
            return []
            
        try:
            # 1. Embed query
            if query_embedding is None:
                query_embedding = await self.embedding_engine.embed_text(query)
            
            # 2. Fetch recent short-term memories
            # In a real system, these would be in a vector DB (LanceDB/Qdrant) managed by Cognee.
            # Here for the "memory" module, we re-rank recent dictionary items.
            recent_memories = await self.get_short_term(user_id, limit=50)
            
            relevant = []
            query_vec = np.array(query_embedding)
            
            for mem in recent_memories:
                if not isinstance(mem, dict) or 'message' not in mem: continue
                
                msg_content = mem['message'].get('content', '')
                if not msg_content: continue
                
                # JIT Embedding (Note: This is slow for many items, but fine for <50 re-ranking)
                # Ideally, embeddings should be stored on write.
                mem_embedding = await self.embedding_engine.embed_text(msg_content)
                mem_vec = np.array(mem_embedding)
                
                # Cosine similarity
                norm = np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
                if norm == 0: continue
                
                score = np.dot(query_vec, mem_vec) / norm
                
                if score > 0.6: # Relevance threshold
                    mem['relevance_score'] = float(score)
                    relevant.append(mem)
            
            # Sort by relevance
            relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant[:5]
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

conversational_memory = ConversationalMemory()
