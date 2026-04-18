import os
from app.config import settings

# --- CRITICAL: SYNC RAG ENV VARS WITH CENTRAL SETTINGS ---
# This ensures that Rag's internal logic respects our LLM_PROVIDER choice (OpenRouter)
# and Database choice (Supabase/PostgreSQL)

# 1. Coordinate Storage Root
from app.storage_setup import STORAGE_ROOT as RAG_ROOT
_rag_root = RAG_ROOT

# 2. Synchronize environment variables for internal RAG modules
os.environ["RAG_ROOT_DIR"] = _rag_root
os.environ["RAG_DATABASE_URL"] = settings.database.DATABASE_URL
os.environ["RAG_LLM_PROVIDER"] = settings.llm.LLM_PROVIDER
os.environ["RAG_LLM_MODEL"] = settings.llm.active_model
os.environ["LLM_PROVIDER"] = settings.llm.LLM_PROVIDER
os.environ["LLM_MODEL"] = settings.llm.active_model
os.environ["LLM_API_KEY"] = settings.llm.active_api_key or ""

# Also set standard keys for fallbacks
if settings.llm.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.llm.GROQ_API_KEY
if settings.llm.OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = settings.llm.OPENROUTER_API_KEY

# 3. Vector Store
os.environ["RAG_VECTOR_DB_TYPE"] = settings.vector_store.VECTOR_STORE_TYPE
os.environ["RAG_VECTOR_DB_URL"] = settings.vector_store.LANCEDB_URI # Or supabase URI if applicable

# 4. Standardize Embedding Provider
os.environ["EMBEDDING_PROVIDER"] = settings.vector_store.EMBEDDING_MODEL
os.environ["RAG_EMBEDDING_PROVIDER"] = "fastembed" # Standard for our setup

class RagSettings:
    """Mock class to satisfy imports while using central settings"""
    def __init__(self):
        self.LLM_PROVIDER = settings.llm.LLM_PROVIDER
        self.LLM_MODEL = settings.llm.active_model
        self.LLM_API_KEY = settings.llm.active_api_key
        self.GROQ_API_KEY = settings.llm.GROQ_API_KEY
        self.RAG_VECTOR_DB_TYPE = settings.vector_store.VECTOR_STORE_TYPE
        self.RAG_VECTOR_DB_URL = settings.database.DATABASE_URL # Pass Postgres URL if set
        
        # Hardcoded for our verified setup
        self.EMBEDDING_PROVIDER = "fastembed"
        self.EMBEDDING_MODEL = settings.vector_store.EMBEDDING_MODEL
        self.EXTRACTION_MODEL = self.LLM_MODEL
        self.GRAPH_DATABASE_URL = os.getenv("GRAPH_DATABASE_URL") or settings.database.DATABASE_URL


rag_settings = RagSettings()
# For backwards compatibility with the rest of the code
import sys
module = sys.modules[__name__]
module.settings = rag_settings
