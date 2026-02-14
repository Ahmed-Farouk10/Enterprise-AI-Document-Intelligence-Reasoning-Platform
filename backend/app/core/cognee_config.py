from pydantic_settings import BaseSettings
from typing import Optional
import os

# --- CRITICAL: SET LLM_API_KEY BEFORE COGNEE IMPORTS ---
# Cognee checks for this during module import, so it MUST be set first
if not os.getenv("LLM_API_KEY"):
    # Use HF_TOKEN if available.
    if os.getenv("HF_TOKEN"):
        os.environ["LLM_API_KEY"] = os.getenv("HF_TOKEN")
        print(f"🔑 LLM_API_KEY set to: {os.environ['LLM_API_KEY'][:10]}...")
    else:
        print("⚠️ No LLM_API_KEY or HF_TOKEN found. Cognee may fail LLM connection tests.")

# --- CRITICAL: INHERIT FROM CENTRAL SETUP ---
# We reuse the logic from app/cognee_setup.py to ensure consistency
try:
    from app.cognee_setup import COGNEE_ROOT, verify_cognee_setup
except ImportError:
    # If import fails (e.g. running script directly), define basic fallback
    # But ideally, this should never happen in the app context
    print("⚠️ WARNING: Could not import cognee_setup. Using fallback defaults.")
    COGNEE_ROOT = "/app/.cache/cognee_data"

_cognee_root = COGNEE_ROOT
print(f"📦 Cognee Config inherits Root: {_cognee_root}")

# Ensure directory exists (redundant but safe)
os.makedirs(_cognee_root, exist_ok=True)

# Set env vars for Cognee (Reinforcing what setup already did)
os.environ["COGNEE_ROOT_DIR"] = _cognee_root
os.environ["COGNEE_DATABASE_URL"] = f"sqlite:///{_cognee_root}/databases/cognee_db.db"

# CRITICAL FIX: Skip LLM connection test on HF Spaces
# Cognee's setup_and_check_environment() calls test_llm_connection() which hangs for 30+ seconds
# when LLM_API_KEY is invalid. This causes pipeline timeout.
if os.getenv("HF_TOKEN"):
    # Force HuggingFace Provider for Cognee
    os.environ["COGNEE_LLM_PROVIDER"] = "huggingface"
    os.environ["COGNEE_LLM_MODEL"] = "Qwen/Qwen2.5-72B-Instruct" 
    
    # AGGRESSIVELY REMOVE GOOGLE KEYS to prevent LiteLLM auto-discovery
    if "GOOGLE_API_KEY" in os.environ:
        print("⚠️ Unsetting GOOGLE_API_KEY to prevent conflict")
        del os.environ["GOOGLE_API_KEY"]
    if "GEMINI_API_KEY" in os.environ:
         print("⚠️ Unsetting GEMINI_API_KEY to prevent conflict")
         del os.environ["GEMINI_API_KEY"]

    # Skip LLM connection test on HF Spaces
    os.environ["COGNEE_SKIP_LLM_TEST"] = "true"
    logger_msg = "⚙️ Cognee: Skipping LLM connection test (HF Spaces / no valid token)"
    print(logger_msg)
# ---------------------------------------------------------

class CogneeSettings(BaseSettings):
    # Graph Database (Neo4j)
    COGNEE_GRAPH_DB_TYPE: str = "neo4j"
    COGNEE_GRAPH_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "changeme123")

    # Default User for Cognee operations
    DEFAULT_USER_ID: str = os.getenv("COGNEE_DEFAULT_USER_ID", "5e5ab0cc-892c-4c79-a8f7-2938f649efcd")
    COGNEE_API_KEY: Optional[str] = os.getenv("COGNEE_API_KEY")

    # Vector Store (Local LanceDB for spaces)
    COGNEE_VECTOR_DB_TYPE: str = "lancedb"
    COGNEE_VECTOR_DB_URL: str = os.getenv("LANCEDB_URI", "/app/cognee_data/lancedb")
    # COGNEE_VECTOR_DB_KEY: Not needed for LanceDB

    # LLM & Embedding (Using Gemini + FastEmbed/SentenceTransformers)
    # Changed default to "gemini" as per user request and availability of API key
    LLM_PROVIDER: str = os.getenv("COGNEE_LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "gemini/gemini-2.0-flash")
    LLM_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
    
    # Standard Embedding Config (Matches Cognee 0.5.x expectations)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("HF_TOKEN")

    # Cognee Processing Options
    EXTRACTION_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "gemini/gemini-2.0-flash")
    GRAPH_DATABASE_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = CogneeSettings()

# Configure Cognee environment variables
# These are picked up by Cognee's internal configuration
os.environ["COGNEE_LLM_PROVIDER"] = settings.LLM_PROVIDER
os.environ["COGNEE_LLM_MODEL"] = settings.LLM_MODEL
os.environ["COGNEE_GRAPH_DB_TYPE"] = settings.COGNEE_GRAPH_DB_TYPE
os.environ["COGNEE_GRAPH_URL"] = settings.COGNEE_GRAPH_URL
os.environ["COGNEE_VECTOR_DB_TYPE"] = settings.COGNEE_VECTOR_DB_TYPE
os.environ["COGNEE_VECTOR_DB_URL"] = settings.COGNEE_VECTOR_DB_URL
os.environ["COGNEE_API_KEY"] = settings.COGNEE_API_KEY or ""

# CRITICAL: Set standard variables for Cognee 0.5.x
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
os.environ["COGNEE_EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER # Alias for safety

# Update LLM_API_KEY if explicitly provided in settings
if settings.LLM_API_KEY and settings.LLM_API_KEY != "local":
    os.environ["LLM_API_KEY"] = settings.LLM_API_KEY


