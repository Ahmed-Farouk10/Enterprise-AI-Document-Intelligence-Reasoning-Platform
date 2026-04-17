"""
CAG (Cache-Augmented Generation) Engine
Pre-computes and caches document context windows for instant retrieval
Eliminates vector search latency for repeated queries on same documents
"""
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from app.config import settings
from app.core.redis_adapter import redis_adapter
from app.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)


class CAGEngine:
    """
    Cache-Augmented Generation Engine
    
    Unlike traditional RAG which searches on every query,
    CAG pre-computes context windows for documents and caches them.
    This provides:
    - Instant retrieval (no vector search latency)
    - Perfect for repeated queries on same documents
    - Reduced LLM costs (no embedding computation per query)
    """
    
    def __init__(self):
        self.cache_ttl = settings.redis.CACHE_TTL
        self.max_context_size = 50000  # Max characters per cached context
        self.chunk_limit = 20  # Max chunks to cache per document
    
    def _get_cache_key(self, document_ids: List[str]) -> str:
        """Generate deterministic cache key for document set"""
        # Sort to ensure same key regardless of order
        sorted_ids = sorted(document_ids)
        key_str = "|".join(sorted_ids)
        return f"cag:context:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    async def get_cached_context(self, document_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve pre-computed context from CAG cache
        """
        if not document_ids:
            return None
        
        cache_key = self._get_cache_key(document_ids)
        
        try:
            cached = await redis_adapter.get(cache_key)  # redis_adapter.get is async
            
            if cached:
                # redis_adapter.get uses pickle, so cached is already deserialized
                if isinstance(cached, dict):
                    context_data = cached
                else:
                    context_data = json.loads(cached)
                logger.info(
                    f"cag_context_retrieved: docs={len(document_ids)}, "
                    f"length={len(context_data.get('full_context', ''))}"
                )
                return context_data
            
            logger.info(f"cag_cache_miss: docs={len(document_ids)}")
            return None
            
        except Exception as e:
            logger.error(f"cag_cache_read_failed: {e}")
            return None
    
    async def cache_context(
        self,
        document_ids: List[str],
        full_context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Pre-compute and cache document context
        """
        if not document_ids or not full_context:
            return False
        
        cache_key = self._get_cache_key(document_ids)
        
        # Truncate if too large
        if len(full_context) > self.max_context_size:
            logger.warning(
                f"cag_context_truncated: original={len(full_context)}, max={self.max_context_size}"
            )
            full_context = full_context[:self.max_context_size]
        
        context_data = {
            "full_context": full_context,
            "document_ids": document_ids,
            "metadata": metadata or {},
            "cached_at": datetime.utcnow().isoformat(),
            "ttl": self.cache_ttl
        }
        
        try:
            # redis_adapter.set uses pickle serialization internally
            await redis_adapter.set(
                cache_key,
                context_data,
                ttl=self.cache_ttl
            )
            
            logger.info(
                f"cag_context_cached: docs={len(document_ids)}, "
                f"length={len(full_context)}, ttl={self.cache_ttl}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"cag_cache_write_failed: {e}")
            return False
    
    async def precompute_context(
        self,
        document_ids: List[str],
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-compute context for documents (background task)
        
        This is called during document upload or when user first accesses documents.
        It fetches all relevant chunks and caches them for instant retrieval.
        
        Args:
            document_ids: List of document IDs to pre-compute
            force: Force recompute even if cached
            
        Returns:
            Cached context dict or None if failed
        """
        # Check if already cached (unless force)
        if not force:
            cached = await self.get_cached_context(document_ids)  # now async
            if cached:
                return cached
        
        try:
            # Fetch comprehensive context for all documents
            all_chunks = []
            
            for doc_id in document_ids:
                # Get more chunks for pre-computation
                chunks = await self._get_document_chunks(doc_id, limit=20)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                logger.warning(f"cag_precompute_no_chunks: docs={document_ids}")
                return None
            
            # Sort by relevance/distance
            all_chunks.sort(key=lambda x: x.get("distance", 0))
            
            # Format context
            context_texts = [chunk["text"] for chunk in all_chunks[:self.chunk_limit]]
            full_context = "\n---\n".join(context_texts)
            
            # Extract metadata
            metadata = {
                "document_type": self._detect_document_type(full_context),
                "chunk_count": len(all_chunks),
                "precomputed": True
            }
            
            # Cache (now async)
            success = await self.cache_context(document_ids, full_context, metadata)
            
            if success:
                logger.info(
                    f"cag_precompute_success: docs={len(document_ids)}, "
                    f"chunks={len(all_chunks)}, length={len(full_context)}"
                )
                
                return {
                    "full_context": full_context,
                    "metadata": metadata,
                    "document_ids": document_ids
                }
            
            return None
            
        except Exception as e:
            logger.error(f"cag_precompute_failed: {e}")
            return None
    
    async def _get_document_chunks(self, document_id: str, limit: int = 20) -> List[Dict]:
        """Get chunks for a specific document from vector store"""
        try:
            # Generic query to get representative chunks
            results = await vector_store_service.search(
                query="document content overview summary",
                limit=limit,
                document_id=document_id
            )
            return results
        except Exception as e:
            logger.error(f"cag_chunk_retrieval_failed: {e}")
            return []
    
    def _detect_document_type(self, context: str) -> str:
        """Simple document type detection from context"""
        context_lower = context.lower()[:5000]
        
        scores = {
            "resume": sum(1 for kw in ["experience", "education", "skills", "resume", "career"] if kw in context_lower),
            "contract": sum(1 for kw in ["agreement", "contract", "parties", "clause", "obligation"] if kw in context_lower),
            "invoice": sum(1 for kw in ["invoice", "total", "tax", "amount", "bill"] if kw in context_lower),
            "report": sum(1 for kw in ["report", "analysis", "findings", "conclusion", "executive"] if kw in context_lower)
        }
        
        best_match = max(scores, key=scores.get)
        return best_match if scores[best_match] >= 2 else "unknown"
    
    async def invalidate_context(self, document_ids: List[str]) -> bool:
        """Invalidate CAG cache for specific documents"""
        cache_key = self._get_cache_key(document_ids)
        
        try:
            await redis_adapter.delete(cache_key)
            logger.info(f"cag_cache_invalidated: docs={document_ids}")
            return True
        except Exception as e:
            logger.error(f"cag_invalidation_failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get CAG cache statistics"""
        try:
            pattern = "cag:context:*"
            keys = redis_adapter.keys(pattern)
            
            total_size = 0
            oldest_cache = None
            newest_cache = None
            
            for key in keys:
                cached = redis_adapter.get(key)
                if cached:
                    total_size += len(cached)
                    data = json.loads(cached)
                    cached_at = data.get("cached_at")
                    
                    if cached_at:
                        if not oldest_cache or cached_at < oldest_cache:
                            oldest_cache = cached_at
                        if not newest_cache or cached_at > newest_cache:
                            newest_cache = cached_at
            
            return {
                "cached_documents": len(keys),
                "total_cache_size_bytes": total_size,
                "oldest_cache": oldest_cache,
                "newest_cache": newest_cache,
                "hit_rate": "N/A"  # Would require tracking
            }
        except Exception as e:
            logger.error("cag_stats_failed", error=str(e))
            return {"error": str(e)}
    
    async def warm_cache(self, document_ids_list: List[List[str]]) -> int:
        """
        Pre-warm cache for multiple document sets
        Useful for popular document combinations
        
        Args:
            document_ids_list: List of document ID sets to cache
            
        Returns:
            Number of successfully cached sets
        """
        success_count = 0
        
        for doc_ids in document_ids_list:
            result = await self.precompute_context(doc_ids)
            if result:
                success_count += 1
        
        logger.info("cag_cache_warmed", total=len(document_ids_list), success=success_count)
        return success_count


# Global CAG engine instance
cag_engine = CAGEngine()
