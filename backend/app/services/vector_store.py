import os
import uuid
import logging
import json
from typing import List, Dict, Any, Optional
import pyarrow as pa
from app.config import settings
from app.services.embeddings import embedding_engine

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.store_type = settings.vector_store.VECTOR_STORE_TYPE
        self.embedding_engine = embedding_engine
        self.table_name = "document_chunks"
        
        if self.store_type == "supabase":
            try:
                self._init_supabase()
            except Exception as e:
                logger.error(f"CRITICAL: Supabase initialization failed ({e}). Falling back to local LanceDB.")
                self.store_type = "lancedb"
                self._init_lancedb()
        else:
            self._init_lancedb()

    def _init_lancedb(self):
        import lancedb
        lancedb_dir = settings.vector_store.LANCEDB_URI
        if lancedb_dir.startswith("file://"):
            lancedb_dir = lancedb_dir.replace("file://", "")
        os.makedirs(lancedb_dir, exist_ok=True)
        
        self.db = lancedb.connect(lancedb_dir)
        if self.table_name not in self.db.table_names():
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_engine.dimension)),
                pa.field("metadata", pa.string())
            ])
            try:
                self.table = self.db.create_table(self.table_name, schema=schema)
                logger.info(f"Created LanceDB table: {self.table_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    self.table = self.db.open_table(self.table_name)
                    logger.info(f"LanceDB table {self.table_name} was created by another process.")
                else:
                    raise e
        else:
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Opened LanceDB table: {self.table_name}")

    def _init_supabase(self):
        from supabase import create_client, Client
        url = settings.database.SUPABASE_URL
        key = settings.database.SUPABASE_SERVICE_ROLE_KEY
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for supabase vector store")
        self.supabase: Client = create_client(url, key)
        logger.info("Initialized Supabase Vector Store (pgvector)")

    def get_table(self):
        if self.store_type == "lancedb":
            return self.db.open_table(self.table_name)
        return None

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + chunk_size / 2:
                    end = last_space
            chunks.append(text[start:end].strip())
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    async def ingest_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> int:
        if not text.strip():
            return 0
            
        chunks = self._chunk_text(text)
        vectors = await self.embedding_engine.embed_text(chunks)
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        if self.store_type == "supabase":
            data = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                data.append({
                    "id": f"{document_id}_{i}",
                    "document_id": document_id,
                    "content": chunk,
                    "embedding": vector,
                    "metadata": metadata or {}
                })
            # Insert into 'document_chunks' table in Supabase
            try:
                self.supabase.table(self.table_name).insert(data).execute()
            except Exception as e:
                logger.error(f"Supabase insertion error: {e}")
                raise e
        else:
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
            table.add(data_to_insert)
        
        logger.info(f"✅ Indexed {len(chunks)} chunks for document {document_id}")
        return len(chunks)

    async def search(self, query: str, limit: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_vector = await self.embedding_engine.embed_text(query)
        processed_results = []

        if self.store_type == "supabase":
            # Using Supabase RPC for vector search (match_documents)
            params = {
                "query_embedding": query_vector,
                "match_threshold": 0.5,
                "match_count": limit,
            }
            if document_id:
                params["filter_document_id"] = document_id
            
            try:
                # Call the 'match_documents' SQL function
                response = self.supabase.rpc("match_documents", params).execute()
                for r in response.data:
                    processed_results.append({
                        "id": r.get("id"),
                        "document_id": r.get("document_id"),
                        "text": r.get("content"),
                        "metadata": r.get("metadata", {}),
                        "distance": 1.0 - r.get("similarity", 0.0)
                    })
            except Exception as e:
                logger.error(f"Supabase search error: {e}")
        else:
            table = self.get_table()
            search = table.search(query_vector).limit(limit)
            if document_id:
                search = search.where(f"document_id = '{document_id}'")
            results = search.to_list()
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
