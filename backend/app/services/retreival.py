from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple
import uuid
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class VectorStore:
import pickle
import os

    def __init__(self):
        logger.info("Initializing Advanced Vector Store (Hybrid Search + Re-Ranking)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # 80MB
        self.dimension = 384
        self.index = None
        self.chunks = []
        self.metadata = []
        self.documents = {}
        self.bm25 = None  # BM25 index
        self.tokenized_chunks = []  # For BM25
        logger.info("Models loaded: Bi-encoder + BM25 + Cross-encoder")
        
        # Load from disk if exists
        self.load_index()
        
    def save_index(self):
        """Save vector store to disk"""
        try:
            data = {
                "chunks": self.chunks,
                "metadata": self.metadata,
                "documents": self.documents,
                "embeddings": self.index.reconstruct_n(0, self.index.ntotal) if self.index and self.index.ntotal > 0 else None
            }
            with open("vector_store.pkl", "wb") as f:
                pickle.dump(data, f)
            logger.info("Vector store saved to disk")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_index(self):
        """Load vector store from disk"""
        if os.path.exists("vector_store.pkl"):
            try:
                with open("vector_store.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self.metadata = data.get("metadata", [])
                    self.documents = data.get("documents", {})
                    
                    # Rebuild indexes
                    embeddings = data.get("embeddings")
                    if embeddings is not None and len(embeddings) > 0:
                        self.index = faiss.IndexFlatL2(self.dimension)
                        self.index.add(embeddings)
                    
                    # Rebuild BM25
                    if self.chunks:
                        self.tokenized_chunks = [c.lower().split() for c in self.chunks]
                        self.bm25 = BM25Okapi(self.tokenized_chunks)
                        
                logger.info(f"Loaded {len(self.chunks)} chunks from persistence")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")

    def clear(self):
        """Reset the vector store"""
        self.index = None
        self.chunks = []
        self.metadata = []
        self.documents = {}
        self.bm25 = None
        self.tokenized_chunks = []
        if os.path.exists("vector_store.pkl"):
            os.remove("vector_store.pkl")
        
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
        
        # Encode all chunks (rebuild FAISS index)
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Build BM25 index
        self.tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        self.save_index() # Persist changes
        
        logger.info(f"Added {len(chunks)} chunks from '{doc_name}' (ID: {doc_id}). Total chunks: {len(self.chunks)}")
        return doc_id
    
    def _bm25_search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """BM25 keyword search"""
        if self.bm25 is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _semantic_search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Dense semantic search"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Convert distance to similarity score
        return [(int(indices[0][i]), float(1 / (1 + distances[0][i]))) for i in range(len(indices[0]))]
    
    def hybrid_search(self, query: str, k: int = 20, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search: BM25 + Semantic
        alpha=0.5 means equal weight, alpha=1.0 means full semantic, alpha=0.0 means full BM25
        """
        # Get candidates from both methods
        bm25_results = self._bm25_search(query, k=k)
        semantic_results = self._semantic_search(query, k=k)
        
        # Normalize scores to [0, 1]
        def normalize(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            range_score = max_score - min_score if max_score > min_score else 1.0
            return {idx: (score - min_score) / range_score for idx, score in results}
        
        bm25_norm = normalize(bm25_results)
        semantic_norm = normalize(semantic_results)
        
        # Combine scores
        all_indices = set(bm25_norm.keys()) | set(semantic_norm.keys())
        combined = {}
        for idx in all_indices:
            bm25_score = bm25_norm.get(idx, 0.0)
            semantic_score = semantic_norm.get(idx, 0.0)
            combined[idx] = alpha * semantic_score + (1 - alpha) * bm25_score
        
        # Sort by combined score
        sorted_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build results
        results = []
        for idx, score in sorted_indices:
            results.append({
                "text": self.chunks[idx],
                "doc_id": self.metadata[idx]["doc_id"],
                "doc_name": self.metadata[idx]["doc_name"],
                "chunk_index": self.metadata[idx]["chunk_index"],
                "hybrid_score": float(score)
            })
        
        return results
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, cand["text"]] for cand in candidates]
        
        # Score all pairs
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])
        
        # Sort by rerank score and take top-k
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        
        logger.info(f"Re-ranked {len(candidates)} candidates to top-{top_k}")
        return reranked
    
    def retrieve_with_citations(self, query: str, k: int = 3, use_hybrid: bool = True, use_reranking: bool = True) -> List[Dict]:
        """
        Two-stage retrieval:
        1. Hybrid search (BM25 + semantic) - get top-20 candidates
        2. Re-rank with cross-encoder - narrow to top-k
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Stage 1: Broad retrieval
        if use_hybrid:
            # Sync with disk to ensure we have latest documents (crucial for multi-worker setups)
            self.load_index()
            candidates = self.hybrid_search(query, k=20)
        else:
            # Fallback to pure semantic
            semantic_results = self._semantic_search(query, k=20)
            candidates = []
            for idx, score in semantic_results:
                candidates.append({
                    "text": self.chunks[idx],
                    "doc_id": self.metadata[idx]["doc_id"],
                    "doc_name": self.metadata[idx]["doc_name"],
                    "chunk_index": self.metadata[idx]["chunk_index"],
                    "similarity_score": score
                })
        
        # Stage 2: Re-ranking
        if use_reranking and len(candidates) > k:
            results = self.rerank(query, candidates, top_k=k)
            # Add similarity_score for backward compatibility
            for r in results:
                if "similarity_score" not in r:
                    r["similarity_score"] = r.get("rerank_score", r.get("hybrid_score", 0.5))
        else:
            results = candidates[:k]
            for r in results:
                if "similarity_score" not in r:
                    r["similarity_score"] = r.get("hybrid_score", 0.5)
        
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
