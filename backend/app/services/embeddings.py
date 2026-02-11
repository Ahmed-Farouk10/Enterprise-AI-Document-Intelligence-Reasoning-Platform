import logging
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingEngine:
    """
    Custom wrapper for SentenceTransformers to be used as Cognee embedding engine.
    Implements the protocol expected by Cognee (embed_text).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            logger.info(f"Initializing local embedding model: {model_name}")
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Loaded SentenceTransformer: {model_name} (dim={self.dimension})")
        except Exception as e:
            logger.error(f"❌ Failed to load SentenceTransformer: {e}")
            raise e

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string asynchronously.
        Non-blocking implementation using thread pool.
        Protocol required by Cognee 0.5.2+.
        """
        import asyncio
        try:
            if not text:
                return [0.0] * self.dimension
                
            # Run blocking encode in thread pool to keep event loop responsive
            embedding = await asyncio.to_thread(
                self.model.encode, 
                text, 
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed for text preview '{text[:20]}...': {e}")
            return [0.0] * self.dimension

    # Shim for older Cognee versions or sync contexts if needed
    def embed_text_sync(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def get_vector_size(self) -> int:
        """Return embedding dimension"""
        return self.dimension
