from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
import hashlib
import json
import logging
from functools import wraps

from app.core.redis_adapter import redis_adapter
from app.services.embeddings import SentenceTransformerEmbeddingEngine

logger = logging.getLogger(__name__)

class QueryResultCache:
    def __init__(self):
        self.redis = redis_adapter
        self.default_ttl = timedelta(hours=1)
        # Initialize embedding engine
        try:
           self.embedding_engine = SentenceTransformerEmbeddingEngine()
        except Exception as e:
            logger.error(f"Failed to initialize embedding engine for cache: {e}")
            self.embedding_engine = None
        
    def _cache_key(self, query_type: str, query_params: Dict[str, Any]) -> str:
        normalized = json.dumps(query_params, sort_keys=True, default=str)
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:24]
        return f"query:{query_type}:{query_hash}"
        
    async def get(self, query_type: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(query_type, query_params)
        cached = await self.redis.get(key)
        if cached:
            return cached.get("result")
        return None
        
    async def set(self, query_type: str, query_params: Dict[str, Any], result: Any, ttl: Optional[timedelta] = None) -> bool:
        key = self._cache_key(query_type, query_params)
        entry = {
            "result": result,
            "cached_at": datetime.utcnow().isoformat(),
            "query_params": query_params
        }
        return await self.redis.set(key, entry, ttl=ttl or self.default_ttl)

    async def get_semantic(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.embedding_engine:
            return None
            
        try:
            # Generate real embedding
            embedding = await self.embedding_engine.embed_text(query)
            # Match dimensionality (384 for all-MiniLM-L6-v2)
            return await self.redis.get_semantic_cache(embedding)
        except Exception as e:
            logger.error(f"Semantic cache retrieval failed: {e}")
            return None

    async def set_semantic(self, query: str, response: Any):
        if not self.embedding_engine:
            return
            
        try:
            embedding = await self.embedding_engine.embed_text(query)
            await self.redis.set_semantic_cache(query, embedding, response)
        except Exception as e:
            logger.error(f"Semantic cache set failed: {e}")

query_cache = QueryResultCache()
