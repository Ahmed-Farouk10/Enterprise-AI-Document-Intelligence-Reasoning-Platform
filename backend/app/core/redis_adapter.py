import redis.asyncio as redis
import json
import pickle
import hashlib
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta
import numpy as np
from contextlib import asynccontextmanager
import logging

# Configure logger
logger = logging.getLogger(__name__)

class RedisCacheAdapter:
    """
    Unified Redis adapter for sessions, semantic caching, and query caching.
    Supports both standard key-value and vector similarity operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True
    ):
        pool_kwargs = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "max_connections": max_connections,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
            "retry_on_timeout": retry_on_timeout,
            "decode_responses": False
        }
        
        # Only add SSL if enabled, to avoid 'unexpected keyword argument' error
        if ssl:
            pool_kwargs["ssl"] = True
            
        self.connection_pool = redis.ConnectionPool(**pool_kwargs)
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection."""
        self._client = redis.Redis(connection_pool=self.connection_pool)
        try:
            await self._client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Don't raise here to allow app startup even if Redis is down (graceful degradation)
            
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")
    
    @asynccontextmanager
    async def pipeline(self):
        """Context manager for Redis pipeline operations."""
        if not self._client:
            await self.connect()
        async with self._client.pipeline() as pipe:
            yield pipe
    
    # =========================================================================
    # STANDARD KEY-VALUE OPERATIONS
    # =========================================================================
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic deserialization."""
        if not self._client: return None
        try:
            data = await self._client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        nx: bool = False
    ) -> bool:
        """Set value in cache with automatic serialization."""
        if not self._client: return False
        try:
            serialized = pickle.dumps(value)
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            
            kwargs = {}
            if ttl:
                kwargs['ex'] = ttl
            if nx:
                kwargs['nx'] = True
            
            return await self._client.set(key, serialized, **kwargs)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        if not self._client: return False
        try:
            return await self._client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        if not self._client: return False
        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
            
    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        if not self._client: return False
        try:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            return await self._client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
            
    # =========================================================================
    # HASH OPERATIONS
    # =========================================================================
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        if not self._client: return False
        try:
            serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            return await self._client.hset(key, field, serialized)
        except Exception as e:
            logger.error(f"Hash set error: {e}")
            return False
            
    async def hget(self, key: str, field: str) -> Optional[Any]:
        if not self._client: return None
        try:
            data = await self._client.hget(key, field)
            if data is None: return None
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data.decode('utf-8')
        except Exception as e:
            logger.error(f"Hash get error: {e}")
            return None
            
    async def hgetall(self, key: str) -> Dict[str, Any]:
        if not self._client: return {}
        try:
            data = await self._client.hgetall(key)
            result = {}
            for field, value in data.items():
                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                try:
                    result[field_str] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[field_str] = value.decode('utf-8') if isinstance(value, bytes) else value
            return result
        except Exception as e:
            logger.error(f"Hash getall error: {e}")
            return {}
            
    async def hdel(self, key: str, field: str) -> bool:
        if not self._client: return False
        try:
            return await self._client.hdel(key, field) > 0
        except Exception as e:
            logger.error(f"Hash delete error: {e}")
            return False

    # =========================================================================
    # LIST & SET OPERATIONS
    # =========================================================================

    async def lpush(self, key: str, value: Any, max_length: Optional[int] = None) -> bool:
        if not self._client: return False
        try:
            serialized = pickle.dumps(value)
            async with self._client.pipeline() as pipe:
                pipe.lpush(key, serialized)
                if max_length:
                    pipe.ltrim(key, 0, max_length - 1)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"List push error: {e}")
            return False
            
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        if not self._client: return []
        try:
            data = await self._client.lrange(key, start, end)
            return [pickle.loads(item) for item in data]
        except Exception as e:
            logger.error(f"List range error: {e}")
            return []
            
    async def sadd(self, key: str, member: str) -> bool:
        if not self._client: return False
        try:
            return await self._client.sadd(key, member) > 0
        except Exception as e:
            logger.error(f"Set add error: {e}")
            return False
            
    async def smembers(self, key: str) -> set:
        if not self._client: return set()
        try:
            members = await self._client.smembers(key)
            return {m.decode('utf-8') if isinstance(m, bytes) else m for m in members}
        except Exception as e:
            logger.error(f"Set members error: {e}")
            return set()
            
    async def srem(self, key: str, member: str) -> bool:
        if not self._client: return False
        try:
            return await self._client.srem(key, member) > 0
        except Exception as e:
            logger.error(f"Set remove error: {e}")
            return False

    # =========================================================================
    # SEMANTIC CACHING
    # =========================================================================

    def _generate_cache_key(self, query: str, prefix: str = "semantic") -> str:
        query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
        return f"{prefix}:{query_hash}"
        
    async def get_semantic_cache(
        self,
        query_embedding: List[float],
        threshold: float = 0.92,
        namespace: str = "default"
    ) -> Optional[Dict[str, Any]]:
        if not self._client: return None
        cache_key = f"semantic_cache:{namespace}"
        try:
             # Brute-force for now (optimize with RediSearch later)
            cached_entries = await self.hgetall(cache_key)
            if not cached_entries: return None
            
            best_match = None
            best_similarity = 0
            query_vec = np.array(query_embedding)
            
            for key, entry in cached_entries.items():
                # entry is already deserialized by hgetall
                if not isinstance(entry, dict) or 'embedding' not in entry:
                    continue
                    
                cached_vec = np.array(entry['embedding'])
                
                # Safety check for dimensions
                if query_vec.shape != cached_vec.shape:
                    logger.warning(f"Dimension mismatch in semantic cache: query={query_vec.shape}, cached={cached_vec.shape}")
                    continue

                similarity = np.dot(query_vec, cached_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)
                )
                
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match:
                logger.debug(f"Semantic cache hit: {best_similarity}")
                return {
                    'response': best_match['response'],
                    'similarity': best_similarity,
                    'original_query': best_match['query']
                }
            return None
        except Exception as e:
            logger.error(f"Semantic cache error: {e}")
            return None

    async def set_semantic_cache(
        self,
        query: str,
        query_embedding: List[float],
        response: Any,
        namespace: str = "default",
        ttl: Optional[int] = None
    ) -> bool:
        if not self._client: return False
        try:
            cache_key = f"semantic_cache:{namespace}"
            entry_key = self._generate_cache_key(query)
            
            entry = {
                'query': query,
                'embedding': query_embedding,
                'response': response,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Use hset directly with dict value (hset method handles json dumping)
            await self.hset(cache_key, entry_key, entry)
            if ttl:
                await self.expire(cache_key, ttl)
            return True
        except Exception as e:
            logger.error(f"Semantic set error: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        if not self._client: return {'status': 'disconnected'}
        try:
            info = await self._client.info()
            return {'status': 'healthy', 'used_memory': info.get('used_memory_human')}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

# Global instance
redis_adapter = RedisCacheAdapter()
