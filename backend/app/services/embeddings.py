"""
Lightweight Embedding Engine — no HuggingFace, no SentenceTransformers.

Uses a deterministic hash-based embedding strategy for vector search.
This is a lightweight, zero-dependency approach that works entirely offline
with no model downloads, no network calls, and no GPU requirements.

The embeddings are consistent (same text always produces same vector),
which is essential for search/retrieval to work correctly.
"""
import asyncio
import hashlib
import logging
import math
import re
from typing import List, Any, Union

import numpy as np

logger = logging.getLogger(__name__)

DIMENSION = 384  # Same dimension as all-MiniLM-L6-v2 for LanceDB compatibility


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _text_to_vector(text: str) -> List[float]:
    """
    Generate a deterministic 384-dimensional embedding from text.
    
    Strategy: Hash overlapping character n-grams and word tokens into
    different vector dimensions, then L2-normalize. This produces
    vectors where semantically similar texts (shared words/phrases)
    have higher cosine similarity.
    """
    vec = np.zeros(DIMENSION, dtype=np.float32)
    
    if not text or not text.strip():
        return vec.tolist()
    
    tokens = _tokenize(text)
    
    # Word-level features (primary signal)
    for token in tokens:
        h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        # Spread each token across multiple dimensions for richer signal
        for offset in range(6):
            idx = (h + offset * 7) % DIMENSION
            sign = 1.0 if ((h >> offset) & 1) == 0 else -1.0
            vec[idx] += sign * (1.0 / (1 + offset * 0.3))
    
    # Bigram features (captures word order / phrases)
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}_{tokens[i+1]}"
        h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
        for offset in range(3):
            idx = (h + offset * 13) % DIMENSION
            sign = 1.0 if ((h >> offset) & 1) == 0 else -1.0
            vec[idx] += sign * 0.5
    
    # Character trigram features (catches partial word matches)
    clean = text.lower()
    for i in range(len(clean) - 2):
        trigram = clean[i:i+3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
        idx = h % DIMENSION
        vec[idx] += 0.15
    
    # L2-normalize so cosine similarity works correctly
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec.tolist()


class LocalEmbeddingEngine:
    """
    Zero-dependency embedding engine.
    
    Produces consistent 384-dim vectors from text using deterministic hashing.
    No model files, no network access, no GPU needed.
    Fully compatible with LanceDB vector search.
    """

    def __init__(self):
        self._dimension = DIMENSION
        logger.info(
            f"✅ LocalEmbeddingEngine initialized (dim={self._dimension}, no HuggingFace)"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def warmup(self):
        """No-op — engine is always ready."""
        pass

    def _encode_sync(self, text: Any) -> Any:
        """Blocking encode — safe for thread pool."""
        if isinstance(text, list):
            return [_text_to_vector(t) for t in text]
        return _text_to_vector(text)

    async def embed_text(self, text: Any) -> Any:
        """
        Embed text(s) asynchronously.
        Supports single string or List[str].
        """
        try:
            if not text:
                return [0.0] * self._dimension
            return await asyncio.to_thread(self._encode_sync, text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            if isinstance(text, list):
                return [[0.0] * self._dimension for _ in text]
            return [0.0] * self._dimension

    def embed_text_sync(self, text: Any) -> Any:
        """Sync shim for older code paths."""
        return self._encode_sync(text)

    def get_vector_size(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
embedding_engine = LocalEmbeddingEngine()
