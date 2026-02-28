from pydantic_settings import BaseSettings
from typing import Optional
import os

# --- CRITICAL: SET LLM_API_KEY BEFORE RAG IMPORTS ---
# Rag checks for this during module import, so it MUST be set first
if not os.getenv("LLM_API_KEY"):
    # Use HF_TOKEN if available.
    if os.getenv("HF_TOKEN"):
        os.environ["LLM_API_KEY"] = os.getenv("HF_TOKEN")
        print(f"🔑 LLM_API_KEY set to: {os.environ['LLM_API_KEY'][:10]}...")
    else:
        print("⚠️ No LLM_API_KEY or HF_TOKEN found. Rag may fail LLM connection tests.")

# --- FAILSAFE: Force Clean Env Vars ---
# Removed aggressive overrides to allow custom OpenRouter/Gemini configurations
# --------------------------------------

# --- CRITICAL: INHERIT FROM CENTRAL SETUP ---
# We reuse the logic from app/rag_setup.py to ensure consistency
try:
    from app.rag_setup import RAG_ROOT, verify_rag_setup
except ImportError:
    # If import fails (e.g. running script directly), define basic fallback
    # But ideally, this should never happen in the app context
    print("⚠️ WARNING: Could not import rag_setup. Using fallback defaults.")
    RAG_ROOT = "/app/.cache/rag_data"

_rag_root = RAG_ROOT
print(f"📦 Rag Config inherits Root: {_rag_root}")

# Ensure directory exists (redundant but safe)
os.makedirs(_rag_root, exist_ok=True)

# Set env vars for Rag (Reinforcing what setup already did)
os.environ["RAG_ROOT_DIR"] = _rag_root
os.environ["RAG_DATABASE_URL"] = f"sqlite:///{_rag_root}/databases/rag_db.db"

# CRITICAL FIX: Skip LLM connection test on HF Spaces
# Rag's setup_and_check_environment() calls test_llm_connection() which hangs for 30+ seconds
# when LLM_API_KEY is invalid. This causes pipeline timeout.
if os.getenv("HF_TOKEN"):
    # Force HuggingFace Provider for Rag
    os.environ["RAG_LLM_PROVIDER"] = "huggingface"
    os.environ["RAG_LLM_MODEL"] = "Qwen/Qwen2.5-72B-Instruct" 
    # Skip LLM connection test on HF Spaces
    os.environ["RAG_SKIP_LLM_TEST"] = "true"
    logger_msg = "⚙️ Rag: Skipping LLM connection test (HF Spaces / no valid token)"
    print(logger_msg)
# ---------------------------------------------------------

class RagSettings(BaseSettings):
    # Graph Database (Neo4j)
    RAG_GRAPH_DB_TYPE: str = "neo4j"
    RAG_GRAPH_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "changeme123")

    # Default User for Rag operations
    DEFAULT_USER_ID: str = os.getenv("RAG_DEFAULT_USER_ID", "5e5ab0cc-892c-4c79-a8f7-2938f649efcd")
    RAG_API_KEY: Optional[str] = os.getenv("RAG_API_KEY")

    # Vector Store (Local LanceDB for spaces)
    RAG_VECTOR_DB_TYPE: str = "lancedb"
    RAG_VECTOR_DB_URL: str = os.getenv("LANCEDB_URI", "/app/rag_data/lancedb")
    # RAG_VECTOR_DB_KEY: Not needed for LanceDB

    # LLM & Embedding (Defaulting to OpenRouter free models for Spaces Compatibility)
    LLM_PROVIDER: str = os.getenv("RAG_LLM_PROVIDER", "openrouter")
    LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "google/gemini-2.0-flash-exp:free")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Standard Embedding Config
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("HF_TOKEN")

    # Rag Processing Options
    EXTRACTION_MODEL: str = os.getenv("RAG_LLM_MODEL", "google/gemini-2.0-flash-exp:free")
    GRAPH_DATABASE_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = RagSettings()

# Configure Rag environment variables
# These are picked up by Rag's internal configuration
os.environ["RAG_LLM_PROVIDER"] = settings.LLM_PROVIDER
os.environ["RAG_LLM_MODEL"] = settings.LLM_MODEL
os.environ["RAG_GRAPH_DB_TYPE"] = settings.RAG_GRAPH_DB_TYPE
os.environ["RAG_GRAPH_URL"] = settings.RAG_GRAPH_URL
os.environ["RAG_VECTOR_DB_TYPE"] = settings.RAG_VECTOR_DB_TYPE
os.environ["RAG_VECTOR_DB_URL"] = settings.RAG_VECTOR_DB_URL
os.environ["RAG_API_KEY"] = settings.RAG_API_KEY or ""

# CRITICAL: Set standard variables for Rag 0.5.x
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
os.environ["RAG_EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER # Alias for safety

# Update API keys if explicitly provided in settings
if settings.LLM_API_KEY and settings.LLM_API_KEY != "local":
    os.environ["LLM_API_KEY"] = settings.LLM_API_KEY
    
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
