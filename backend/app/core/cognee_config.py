from pydantic_settings import BaseSettings
from typing import Optional
import os

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
# ---------------------------------------------------------

class CogneeSettings(BaseSettings):
    # Graph Database (Neo4j)
    COGNEE_GRAPH_DB_TYPE: str = "neo4j"
    COGNEE_GRAPH_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "changeme123")

    # Vector Store (Qdrant - managed internally by Cognee or external)
    COGNEE_VECTOR_DB_TYPE: str = "qdrant"
    COGNEE_VECTOR_DB_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COGNEE_VECTOR_DB_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # LLM & Embedding (Using Qwen via Hugging Face)
    # Cognee uses these to extract entities and relations
    LLM_PROVIDER: str = os.getenv("COGNEE_LLM_PROVIDER", "huggingface")
    LLM_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    LLM_API_KEY: Optional[str] = os.getenv("HF_TOKEN")  # Optional for HF
    EMBEDDING_API_KEY: Optional[str] = os.getenv("HF_TOKEN")

    # Cognee Processing Options
    EXTRACTION_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    GRAPH_DATABASE_URL: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = CogneeSettings()

# Configure Cognee environment variables
# These are picked up by Cognee's internal configuration
os.environ["GRAPH_DATABASE_URL"] = settings.NEO4J_URI
os.environ["GRAPH_DATABASE_USERNAME"] = settings.NEO4J_USER
os.environ["GRAPH_DATABASE_PASSWORD"] = settings.NEO4J_PASSWORD
os.environ["VECTOR_DB_URL"] = settings.COGNEE_VECTOR_DB_URL
os.environ["LLM_PROVIDER"] = settings.LLM_PROVIDER
os.environ["LLM_MODEL"] = settings.LLM_MODEL

# Set Cognee root directory for data storage
COGNEE_ROOT_DIR = os.getenv("COGNEE_ROOT_DIR", "/app/cognee_data")
os.environ["COGNEE_ROOT_DIR"] = COGNEE_ROOT_DIR
