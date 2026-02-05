"""
Qdrant Vector Database Client
Replaces the pickle-based VectorStore with proper vector database
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
from typing import List, Dict, Optional
import os
import uuid
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Qdrant-based vector store for document embeddings
    Replaces the old pickle-based VectorStore
    """
    
    def __init__(
        self,
        collection_name: str = None,
        host: str = None,
        port: int = None,
        api_key: str = None
    ):
        # Get configuration from environment
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "documents")
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key if api_key else None
        )
        
        # Embedding dimension (sentence-transformers default)
        self.embedding_dim = 384
        
        # Initialize collection
        self._init_collection()
        
        logger.info("qdrant_initialized", collection=self.collection_name, host=host, port=port)
    
    def _init_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("qdrant_collection_created", collection=self.collection_name)
            else:
                logger.info("qdrant_collection_exists", collection=self.collection_name)
                
        except Exception as e:
            logger.error("qdrant_init_failed", error=str(e), exc_info=True)
            raise
    
    def add_document(self, text: str, doc_id: str, doc_name: str, metadata: Dict = None):
        """
        Add a document to the vector store
        
        Args:
            text: Document text content
            doc_id: Document ID
            doc_name: Document name
            metadata: Additional metadata
        """
        try:
            # Import embedding model here to avoid circular imports
            from sentence_transformers import SentenceTransformer
            
            # Initialize embedding model (cached)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Split text into chunks (simple chunking by paragraphs)
            chunks = []
            paragraphs = text.split("\n\n")
            
            for para in paragraphs:
                if len(para.strip()) > 100:  # Min chunk size
                    chunks.append(para.strip())
            
            # If no chunks, treat whole text as one chunk
            if not chunks:
                chunks = [text.strip()]
            
            # Generate embeddings for chunks
            embeddings = model.encode(chunks)
            
            # Prepare points for insertion
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                
                payload = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": i,
                    "text": chunk,
                    "metadata": metadata or {}
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
            
            # Upsert points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info("document_added_to_qdrant", doc_id=doc_id, chunks=len(chunks))
            
        except Exception as e:
            logger.error("qdrant_add_document_failed", doc_id=doc_id, error=str(e), exc_info=True)
            raise
    
    def search(
        self,
        query: str,
        k: int = 5,
        doc_id_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            k: Number of results
            doc_id_filter: Optional filter by document ID
            
        Returns:
            List of search results with text and metadata
        """
        try:
            # Import embedding model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Generate query embedding
            query_embedding = model.encode(query)
            
            # Build filter if doc_id provided
            search_filter = None
            if doc_id_filter:
                search_filter = Filter(
                    must=[FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id_filter)
                    )]
                )
            
            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=k
            )
            
            # Format results
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "text": hit.payload["text"],
                    "doc_id": hit.payload["doc_id"],
                    "doc_name": hit.payload["doc_name"],
                    "chunk_index": hit.payload["chunk_index"],
                    "similarity_score": hit.score,
                    "metadata": hit.payload.get("metadata", {})
                })
            
            logger.info("qdrant_search_completed", query_length=len(query), results=len(formatted_results))
            
            return formatted_results
            
        except Exception as e:
            logger.error("qdrant_search_failed", error=str(e), exc_info=True)
            return []
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )]
                )
            )
            logger.info("document_deleted_from_qdrant", doc_id=doc_id)
            
        except Exception as e:
            logger.error("qdrant_delete_failed", doc_id=doc_id, error=str(e), exc_info=True)
            raise
    
    def clear(self):
        """Clear all vectors from the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._init_collection()  # Recreate empty collection
            logger.info("qdrant_collection_cleared", collection=self.collection_name)
            
        except Exception as e:
            logger.error("qdrant_clear_failed", error=str(e), exc_info=True)
            raise
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error("qdrant_stats_failed", error=str(e))
            return {}


# Global instance (similar to old vector_store singleton)
qdrant_store = QdrantVectorStore()
