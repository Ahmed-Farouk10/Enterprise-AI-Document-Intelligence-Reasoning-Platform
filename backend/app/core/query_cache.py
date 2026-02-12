from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
import hashlib
import json
import logging
from functools import wraps

from app.core.redis_adapter import redis_adapter

logger = logging.getLogger(__name__)

class QueryResultCache:
    def __init__(self):
        self.redis = redis_adapter
        self.default_ttl = timedelta(hours=1)
        
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
        # Mock embedding, real impl needs service
        embedded = [0.1] * 384 
        return await self.redis.get_semantic_cache(embedded)

    async def set_semantic(self, query: str, response: Any):
        embedded = [0.1] * 384
        await self.redis.set_semantic_cache(query, embedded, response)

query_cache = QueryResultCache()
