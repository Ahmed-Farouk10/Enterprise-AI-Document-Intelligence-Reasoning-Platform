import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import lancedb
import pyarrow as pa
from app.services.embeddings import SentenceTransformerEmbeddingEngine

logger = logging.getLogger(__name__)

# LanceDB setup location
LANCEDB_DIR = "/app/.cache/lancedb_data"
os.makedirs(LANCEDB_DIR, mode=0o777, exist_ok=True)

class VectorStoreService:
    def __init__(self):
        self.db = lancedb.connect(LANCEDB_DIR)
        self.table_name = "document_chunks"
        self.embedding_engine = SentenceTransformerEmbeddingEngine()
        
        # Initialize table if it doesn't exist
        if self.table_name not in self.db.table_names():
            # Define schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_engine.dimension)),
                pa.field("metadata", pa.string()) # JSON string for flexibility
            ])
            self.db.create_table(self.table_name, schema=schema)
            logger.info(f"Created new LanceDB table: {self.table_name}")
        else:
             self.table = self.db.open_table(self.table_name)
             logger.info(f"Opened existing LanceDB table: {self.table_name}")

    def get_table(self):
        return self.db.open_table(self.table_name)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple sliding window chunker."""
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Adjust to nearest space if not at end
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + chunk_size / 2:
                    end = last_space
            chunks.append(text[start:end].strip())
            
            # Prevent infinite loop if text has no spaces and end == start
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
            
        return chunks

    async def ingest_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        Chunk document text, embed it, and store in LanceDB.
        Returns the number of chunks ingested.
        """
        import json
        if not text.strip():
            return 0
            
        chunks = self._chunk_text(text)
        
        # Parallel embedding (though local model might be blocking, CPU-bound)
        # We rely on the async wrapper in SentenceTransformerEmbeddingEngine
        vectors = await self.embedding_engine.embed_text(chunks)
        
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        data_to_insert = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
             data_to_insert.append({
                 "id": f"{document_id}_{i}",
                 "document_id": document_id,
                 "text": chunk,
                 "vector": vector,
                 "metadata": meta_str
             })
             
        table = self.get_table()
        try:
            table.add(data_to_insert)
        except Exception as e:
            logger.error(f"❌ LanceDB insertion error: {e}")
            if data_to_insert:
                logger.error(f"Data sample: {data_to_insert[0]}")
            raise e
        
        logger.info(f"✅ Indexed {len(chunks)} chunks for document {document_id}")
        return len(chunks)

    async def search(self, query: str, limit: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search LanceDB for the most relevant chunks.
        """
        import json
        query_vector = await self.embedding_engine.embed_text(query)
        
        table = self.get_table()
        search = table.search(query_vector).limit(limit)
        
        if document_id:
            # Filter by document if specified
            search = search.where(f"document_id = '{document_id}'")
            
        results = search.to_list()
        
        processed_results = []
        for r in results:
             try:
                 meta = json.loads(r.get("metadata", "{}"))
             except:
                 meta = {}
             
             processed_results.append({
                 "id": r["id"],
                 "document_id": r["document_id"],
                 "text": r["text"],
                 "metadata": meta,
                 "distance": r.get("_distance", 0.0)
             })
             
        return processed_results

vector_store_service = VectorStoreService()
