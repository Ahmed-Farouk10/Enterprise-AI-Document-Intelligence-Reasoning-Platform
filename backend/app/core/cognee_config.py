from pydantic_settings import BaseSettings
from typing import Optional
import os

class CogneeSettings(BaseSettings):
    """Configuration for Cognee Knowledge Graph Engine"""
    
    # Graph Database (Neo4j)
    COGNEE_GRAPH_DB_TYPE: str = "neo4j"
    COGNEE_GRAPH_URL: str = os.getenv("COGNEE_GRAPH_URL", "bolt://localhost:7687")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Vector Store (Qdrant - managed internally by Cognee or external)
    COGNEE_VECTOR_DB_TYPE: str = "qdrant"
    COGNEE_VECTOR_DB_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COGNEE_VECTOR_DB_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # LLM & Embedding (Using OpenAI/LiteLLM standard)
    # Cognee uses these to extract entities and relations
    LLM_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") 
    EMBEDDING_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Processing options
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = CogneeSettings()
