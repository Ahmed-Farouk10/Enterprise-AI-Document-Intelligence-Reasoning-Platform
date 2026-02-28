import logging
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingEngine:
    """
    Custom wrapper for SentenceTransformers to be used as Rag embedding engine.
    Implements the protocol expected by Rag (embed_text).
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
        Protocol required by Rag 0.5.2+.
        """
        try:
            if not text:
                return [0.0] * self.dimension
                
            if isinstance(text, list):
                # Batch processing
                embeddings = self.model.encode(text, convert_to_numpy=True)
                return [v.tolist() for v in embeddings]
            
            # Single processing
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            if isinstance(text, list):
                return [[0.0] * self.dimension for _ in text]
            return [0.0] * self.dimension

    # Shim for older Rag versions or sync contexts if needed
    def embed_text_sync(self, text: Any) -> Any:
        if isinstance(text, list):
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return [v.tolist() for v in embeddings]
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def get_vector_size(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2)"""
        return 384
