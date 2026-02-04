from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info("Initializing Enhanced Vector Store (Multi-Doc + Citations)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.metadata = []  # Store doc_id, chunk_index, original_text
        self.documents = {}  # doc_id -> document info
        
    def clear(self):
        """Reset the vector store"""
        self.index = None
        self.chunks = []
        self.metadata = []
        self.documents = {}
        
    def add_document(self, text: str, doc_id: str = None, doc_name: str = None, chunk_size: int = 300):
        """
        Add document with metadata for multi-doc RAG and citations
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
        
        if doc_name is None:
            doc_name = f"Document {doc_id}"
        
        # Store document info
        self.documents[doc_id] = {
            "name": doc_name,
            "text": text,
            "chunk_count": 0
        }
        
        # Chunking with overlap
        chunks = []
        chunk_metadata = []
        words = text.split()
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": chunk_idx,
                    "text": chunk_text
                })
                chunk_idx += 1
                
                # Overlap: keep last 50 words
                current_chunk = current_chunk[-50:]
                current_length = sum(len(w) + 1 for w in current_chunk)
        
        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            chunk_metadata.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_index": chunk_idx,
                "text": chunk_text
            })
        
        self.documents[doc_id]["chunk_count"] = len(chunks)
        
        # Append to existing store
        self.chunks.extend(chunks)
        self.metadata.extend(chunk_metadata)
        
        # Encode all chunks (rebuild index)
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        
        # Rebuild FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Added {len(chunks)} chunks from '{doc_name}' (ID: {doc_id}). Total chunks: {len(self.chunks)}")
        return doc_id
        
    def retrieve_with_citations(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k chunks WITH citation metadata
        Returns: [{"text": "...", "doc_id": "...", "doc_name": "...", "score": 0.95}, ...]
        """
        if self.index is None or len(self.chunks) == 0:
            return []
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Build results with citations
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "text": self.chunks[idx],
                "doc_id": self.metadata[idx]["doc_id"],
                "doc_name": self.metadata[idx]["doc_name"],
                "chunk_index": self.metadata[idx]["chunk_index"],
                "similarity_score": float(1 / (1 + dist))  # Convert distance to similarity
            })
        
        return results
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Legacy method - returns just text for backward compatibility"""
        results = self.retrieve_with_citations(query, k)
        return [r["text"] for r in results]
    
    def get_document_list(self) -> List[Dict]:
        """Get list of all stored documents"""
        return [
            {"doc_id": doc_id, **info}
            for doc_id, info in self.documents.items()
        ]

# Singleton
vector_store = VectorStore()
