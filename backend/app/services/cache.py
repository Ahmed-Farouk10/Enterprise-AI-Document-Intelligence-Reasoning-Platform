"""
Redis caching service for performance optimization
"""

import redis
import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
import os
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Redis-based caching service
    Provides TTL-based caching for RAG queries and session data
    """
    
    def __init__(self):
        # Get Redis URL from environment
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Parse password from URL if present
        # Format: redis://:password@host:port/db
        try:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                encoding="utf-8"
            )
            # Test connection
            self.client.ping()
            self.enabled = True
            logger.info("redis_cache_initialized", url=redis_url.split('@')[-1])  # Hide password in logs
            
        except Exception as e:
            logger.warning("redis_cache_unavailable", error=str(e))
            self.enabled = False
            self.client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        # Create a deterministic string from args and kwargs
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        # Hash for fixed-length key
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug("cache_hit", key=key)
                return json.loads(value)
            else:
                logger.debug("cache_miss", key=key)
                return None
                
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self.enabled:
            return False
        
        try:
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug("cache_set", key=key, ttl=ttl)
            return True
            
        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled:
            return False
        
        try:
            self.client.delete(key)
            logger.debug("cache_delete", key=key)
            return True
            
        except Exception as e:
            logger.error("cache_delete_error", key=key, error=str(e))
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern
        
        Args:
            pattern: Glob-style pattern (e.g., "rag_query:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                logger.info("cache_pattern_delete", pattern=pattern, count=count)
                return count
            return 0
            
        except Exception as e:
            logger.error("cache_pattern_delete_error", pattern=pattern, error=str(e))
            return 0
    
    def cache_rag_query(self, query: str, doc_ids: list = None, ttl: int = 3600):
        """
        Decorator for caching RAG query results
        
        Usage:
            @cache_service.cache_rag_query(query, doc_ids, ttl=1800)
            def expensive_rag_operation():
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(
                    "rag_query",
                    query=query,
                    doc_ids=sorted(doc_ids) if doc_ids else []
                )
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.info("rag_cache_hit", query=query[:50])
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                if result:
                    self.set(cache_key, result, ttl=ttl)
                    logger.info("rag_cache_miss_stored", query=query[:50])
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_rag_cache(self):
        """Invalidate all RAG query caches"""
        return self.delete_pattern("rag_query:*")
    
    def cache_session(self, session_id: str, data: dict, ttl: int = 1800):
        """
        Cache session data
        
        Args:
            session_id: Session ID
            data: Session data
            ttl: Time to live (default: 30 minutes)
        """
        cache_key = f"session:{session_id}"
        return self.set(cache_key, data, ttl=ttl)
    
    def get_cached_session(self, session_id: str) -> Optional[dict]:
        """Get cached session data"""
        cache_key = f"session:{session_id}"
        return self.get(cache_key)
    
    def invalidate_session(self, session_id: str):
        """Invalidate cached session"""
        cache_key = f"session:{session_id}"
        return self.delete(cache_key)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.client.info()
            return {
                "enabled": True,
                "used_memory": info.get("used_memory_human"),
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error("cache_stats_error", error=str(e))
            return {"enabled": True, "error": str(e)}


# Global cache service instance
cache_service = CacheService()
