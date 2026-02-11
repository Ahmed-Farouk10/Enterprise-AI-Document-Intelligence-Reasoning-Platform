from pydantic_settings import BaseSettings
from typing import Optional
import os

# --- CRITICAL: SET LLM_API_KEY BEFORE COGNEE IMPORTS ---
# Cognee checks for this during module import, so it MUST be set first
if not os.getenv("LLM_API_KEY"):
    # Use HF_TOKEN if available, otherwise use 'local' placeholder
    os.environ["LLM_API_KEY"] = os.getenv("HF_TOKEN", "local")
    print(f"🔑 LLM_API_KEY set to: {os.environ['LLM_API_KEY'][:10]}..." if len(os.environ['LLM_API_KEY']) > 10 else "local")

# --- COGNEE PATH CONFIGURATION (MUST BE BEFORE IMPORT) ---
# Detect writable directory for HuggingFace Spaces / Docker
if os.getenv("HF_HOME"):
    _cognee_root = os.path.join(os.getenv("HF_HOME"), "cognee_data")
else:
    _cognee_root = os.path.join(os.getcwd(), ".cognee_system")

# Ensure directory exists
os.makedirs(_cognee_root, exist_ok=True)

# Set env vars for Cognee to pick up
os.environ["COGNEE_ROOT_DIR"] = _cognee_root

# CRITICAL FIX: Skip LLM connection test on HF Spaces
# Cognee's setup_and_check_environment() calls test_llm_connection() which hangs for 30+ seconds
# when LLM_API_KEY is invalid. This causes pipeline timeout.
if os.getenv("HF_HOME") or not os.getenv("HF_TOKEN"):
    os.environ["COGNEE_SKIP_LLM_TEST"] = "true"
    logger_msg = "⚙️ Cognee: Skipping LLM connection test (HF Spaces / no valid token)"
    print(logger_msg)
# ---------------------------------------------------------

class CogneeSettings(BaseSettings):
    # Graph Database (Neo4j)
    COGNEE_GRAPH_DB_TYPE: str = "neo4j"
    COGNEE_GRAPH_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "changeme123")

    # Default User for Cognee operations
    DEFAULT_USER_ID: str = os.getenv("COGNEE_DEFAULT_USER_ID", "5e5ab0cc-892c-4c79-a8f7-2938f649efcd")

    # Vector Store (Local LanceDB for spaces)
    COGNEE_VECTOR_DB_TYPE: str = "lancedb"
    COGNEE_VECTOR_DB_URL: str = os.getenv("LANCEDB_URI", "/app/cognee_data/lancedb")
    # COGNEE_VECTOR_DB_KEY: Not needed for LanceDB

    # LLM & Embedding (Using Qwen + FastEmbed/SentenceTransformers)
    LLM_PROVIDER: str = os.getenv("COGNEE_LLM_PROVIDER", "huggingface")
    LLM_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LLM_API_KEY: Optional[str] = os.getenv("HF_TOKEN")
    
    # Standard Embedding Config (Matches Cognee 0.5.x expectations)
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("HF_TOKEN")

    # Cognee Processing Options
    EXTRACTION_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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

# CRITICAL: Set standard variables for Cognee 0.5.x
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
os.environ["COGNEE_EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER # Alias for safety

# Update LLM_API_KEY if explicitly provided in settings
if settings.LLM_API_KEY and settings.LLM_API_KEY != "local":
    os.environ["LLM_API_KEY"] = settings.LLM_API_KEY

# Set Cognee root directory for data storage
COGNEE_ROOT_DIR = os.getenv("COGNEE_ROOT_DIR", "/app/cognee_data")
os.environ["COGNEE_ROOT_DIR"] = COGNEE_ROOT_DIR
