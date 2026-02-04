from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info("Initializing Vector Store (FAISS + Sentence-Transformers)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
        self.dimension = 384
        self.index = None
        self.chunks = []
        
    def clear(self):
        """Reset the vector store"""
        self.index = None
        self.chunks = []
        
    def add_document(self, text: str, chunk_size: int = 300):
        """
        Split document into chunks and add to vector store
        """
        # Simple chunking by character count with overlap
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                # Overlap: keep last 50 words
                current_chunk = current_chunk[-50:]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        # Add remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        self.chunks = chunks
        
        # Encode chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
    def retrieve(self, query: str, k: int = 3):
        """
        Retrieve top-k most relevant chunks
        """
        if self.index is None or len(self.chunks) == 0:
            return []
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        k = min(k, len(self.chunks))  # Don't ask for more than we have
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return chunks
        return [self.chunks[i] for i in indices[0]]

# Singleton
vector_store = VectorStore()
