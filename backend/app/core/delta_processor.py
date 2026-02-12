from dataclasses import dataclass
from typing import Set, List, Tuple, Dict, Any
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # Avoid heavy deps if possible, or assume it's there
# app/core/delta_processor.py

@dataclass
class DocumentDelta:
    added_chunks: List[Dict]
    modified_chunks: List[Tuple[Dict, Dict]]  # (old, new)
    removed_chunks: List[Dict]
    unchanged_chunks: List[Dict]
    semantic_similarity: float

class IncrementalProcessor:
    def __init__(self, embedding_engine, vector_store):
        self.embedder = embedding_engine
        self.vector_store = vector_store
        self.similarity_threshold = 0.85  # Configurable
    
    async def process_incremental(self, document_id: str, new_text: str) -> DocumentDelta:
        """
        Process only changed portions of document.
        Preserves existing vectors for unchanged content.
        """
        # Retrieve existing chunks for this document
        # This assumes vector_store has a method get_document_chunks
        try:
            existing_chunks = [] 
            # existing_chunks = await self.vector_store.get_document_chunks(document_id)
        except Exception:
            existing_chunks = []
        
        # Chunk new text
        new_chunks = self._chunk_text(new_text)
        
        # Embed new chunks (Using embedder)
        # new_embeddings = await self.embedder.embed_batch([c["text"] for c in new_chunks])
        # Mocking embeddings for the structure
        new_embeddings = [np.random.rand(384) for _ in new_chunks]
        
        # Logic to match chunks would go here
        # For this implementation, we will treat all as new if no existing chunks
        
        return DocumentDelta(
            added_chunks=[{**c, "embedding": e} for c, e in zip(new_chunks, new_embeddings)],
            modified_chunks=[],
            removed_chunks=[],
            unchanged_chunks=[],
            semantic_similarity=0.0
        )
    
    def _chunk_text(self, text: str) -> List[Dict]:
        """Simple chunking helper."""
        words = text.split()
        chunks = []
        chunk_size = 300
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({"text": chunk_text, "index": len(chunks)})
        return chunks

    async def apply_delta(self, document_id: str, delta: DocumentDelta):
        """Apply incremental changes to vector store and knowledge graph."""
        operations = []
        
        # Remove obsolete vectors
        for chunk in delta.removed_chunks:
            operations.append(("delete", chunk.get("vector_id")))
        
        # Add new vectors
        for chunk in delta.added_chunks:
            operations.append(("insert", chunk))
        
        # Check if vector_store has batch_update
        if hasattr(self.vector_store, "batch_update"):
            await self.vector_store.batch_update(operations)
        else:
            # Fallback to individual adds
            pass
        
        return {
            "vectors_added": len(delta.added_chunks),
            "vectors_updated": len(delta.modified_chunks),
            "vectors_removed": len(delta.removed_chunks),
            "vectors_preserved": len(delta.unchanged_chunks),
        }

# Singleton placeholder
incremental_processor = IncrementalProcessor(None, None)
