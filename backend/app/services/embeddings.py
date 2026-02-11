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

    async def embed_text(self, text: Any) -> Any:
        """
        Embed text(s) asynchronously.
        Supports single string or List[str].
        Protocol required by Cognee 0.5.2+.
        """
        import asyncio
        try:
            if not text:
                return self._pad_vector([0.0] * self.dimension)
                
            if isinstance(text, list):
                # Batch processing
                embeddings = await asyncio.to_thread(
                    self.model.encode, 
                    text, 
                    convert_to_numpy=True
                )
                return [self._pad_vector(v.tolist()) for v in embeddings]
            
            # Single processing
            embedding = await asyncio.to_thread(
                self.model.encode, 
                text, 
                convert_to_numpy=True
            )
            return self._pad_vector(embedding.tolist())
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            if isinstance(text, list):
                return [self._pad_vector([0.0] * self.dimension) for _ in text]
            return self._pad_vector([0.0] * self.dimension)

    def _pad_vector(self, vector: List[float]) -> List[float]:
        """Pad vector to 3072 dimensions for Cognee compatibility."""
        if len(vector) < 3072:
            return vector + [0.0] * (3072 - len(vector))
        return vector[:3072]

    # Shim for older Cognee versions or sync contexts if needed
    def embed_text_sync(self, text: Any) -> Any:
        if isinstance(text, list):
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return [self._pad_vector(v.tolist()) for v in embeddings]
        return self._pad_vector(self.model.encode(text, convert_to_numpy=True).tolist())

    def get_vector_size(self) -> int:
        """Return embedding dimension (always 3072 for system-wide alignment)"""
        return 3072
