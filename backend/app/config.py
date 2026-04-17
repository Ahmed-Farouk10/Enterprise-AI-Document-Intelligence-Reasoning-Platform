"""
Professional Configuration Management
Centralized settings with environment-based profiles
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal
import os
from pathlib import Path


# Allow extra env vars in all configs
class BaseSettingsExtra(BaseSettings):
    """Base settings class that allows extra fields"""
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


class LLMConfig(BaseSettingsExtra):
    """LLM Provider Configuration"""
    LLM_PROVIDER: Literal["openai", "groq", "gemini", "ollama", "openrouter"] = Field(default="groq")
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    OPENROUTER_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-exp")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b")
    OPENROUTER_MODEL: str = Field(default="google/gemini-2.0-flash-exp:free")
    TEMPERATURE: float = Field(default=0.2, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=4096, ge=1)
    TIMEOUT: int = Field(default=60, ge=1)

    @property
    def active_model(self) -> str:
        model_map = {
            "openai": self.OPENAI_MODEL,
            "groq": self.GROQ_MODEL,
            "gemini": self.GEMINI_MODEL,
            "ollama": self.OLLAMA_MODEL,
            "openrouter": self.OPENROUTER_MODEL,
        }
        return model_map.get(self.LLM_PROVIDER, self.GROQ_MODEL)

    @property
    def active_api_key(self) -> Optional[str]:
        key_map = {
            "openai": self.OPENAI_API_KEY,
            "groq": self.GROQ_API_KEY,
            "gemini": self.GEMINI_API_KEY,
            "ollama": None,
            "openrouter": self.OPENROUTER_API_KEY,
        }
        return key_map.get(self.LLM_PROVIDER)


class DatabaseConfig(BaseSettingsExtra):
    """Database Configuration"""
    DATABASE_URL: str = Field(default="sqlite:///./data/app.db")
    STORAGE_ROOT: str = Field(default="./data")
    UPLOAD_DIR: str = Field(default="./data/uploads")
    # Supabase Specifics
    SUPABASE_URL: Optional[str] = Field(default=None)
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(default=None)
    USE_SUPABASE_STORAGE: bool = Field(default=False)


class VectorStoreConfig(BaseSettingsExtra):
    """Vector Store Configuration"""
    VECTOR_STORE_TYPE: Literal["lancedb", "supabase"] = Field(default="lancedb")
    LANCEDB_URI: str = Field(default="./data/vectordb")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=5000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0)


class RedisConfig(BaseSettingsExtra):
    """Redis Configuration for Caching & Sessions"""
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = Field(default=50)
    REDIS_SOCKET_TIMEOUT: int = Field(default=5)
    SESSION_TTL: int = Field(default=86400)
    SESSION_MAX_MESSAGES: int = Field(default=50)
    CACHE_TTL: int = Field(default=3600)
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.92)


class LangGraphConfig(BaseSettingsExtra):
    """LangGraph Workflow Configuration"""
    NODE_TIMEOUT: int = Field(default=30, ge=1)
    WORKFLOW_TIMEOUT: int = Field(default=120, ge=1)
    MAX_CONTEXT_MESSAGES: int = Field(default=10)
    MEMORY_TTL: int = Field(default=7200)
    VERIFICATION_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    REQUIRE_CITATIONS: bool = Field(default=True)


class SearchConfig(BaseSettingsExtra):
    """External Search Configuration"""
    TAVILY_API_KEY: Optional[str] = Field(default=None)
    DUCKDUCKGO_ENABLED: bool = Field(default=True)
    MAX_SEARCH_RESULTS: int = Field(default=5)


class LoggingConfig(BaseSettingsExtra):
    """Logging Configuration"""
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    JSON_LOGS: bool = Field(default=True)
    LOG_FORMAT: str = Field(default="%(message)s")


class RateLimitConfig(BaseSettingsExtra):
    """Rate Limiting Configuration"""
    DEFAULT_RATE: str = Field(default="200/minute")
    DAILY_LIMIT: str = Field(default="5000/day")
    UPLOAD_RATE: str = Field(default="100/minute")
    SESSION_CREATE_RATE: str = Field(default="20/minute")
    CHAT_MESSAGE_RATE: str = Field(default="30/minute")
    RAG_QUERY_RATE: str = Field(default="50/minute")


class Settings(BaseSettingsExtra):
    """Master Settings - Aggregates All Configurations"""
    APP_NAME: str = Field(default="DocuCentric")
    APP_VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(default="development")
    DEBUG: bool = Field(default=False)

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    def initialize_storage(self):
        """Create all required directories"""
        dirs = [
            self.database.STORAGE_ROOT,
            self.database.UPLOAD_DIR,
            self.vector_store.LANCEDB_URI,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
