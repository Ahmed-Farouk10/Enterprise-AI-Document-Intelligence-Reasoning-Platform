import os
from typing import Literal, Optional, List
from pydantic import Field, field_validator, ValidationInfo

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import socket
from urllib.parse import urlparse, urlunparse



# Allow extra env vars in all configs
class BaseSettingsExtra(BaseSettings):
    """Base settings class that allows extra fields and flattened env vars"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        env_nested_delimiter="__",
        protected_namespaces=('settings_', 'model_')
    )




class LLMConfig(BaseSettingsExtra):
    """LLM Provider Configuration"""
    LLM_PROVIDER: Literal["openai", "groq", "gemini", "ollama", "openrouter"] = Field(
        default="groq"
    )


    @field_validator("*", mode="before")
    @classmethod
    def validate_from_env(cls, v, info: ValidationInfo):
        """Allow loading from flat env vars (e.g. OPENROUTER_API_KEY)"""
        # Mapping for special cases if needed
        env_val = os.getenv(info.field_name)
        if info.field_name == "LLM_PROVIDER":
            env_val = env_val or os.getenv("AI_PROVIDER")
        
        return env_val if env_val else v


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
    DATABASE_URL: str = Field(
        default="sqlite:////app/.cache/app_data/databases/app.db"
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_db_url(cls, v):

        # 1. Gather all possible sources
        sources = [
            os.getenv("postgresql"),
            os.getenv("DATABASE_URL"),
            os.getenv("DB_URL"),
            v # The default
        ]
        
        # 2. Find the first non-empty, non-None value
        val = next((s for s in sources if s and str(s).strip()), v)
        
        # 3. Sanitize if it's a string
        if isinstance(val, str):
            # Remove ALL whitespace (including internal spaces, tabs, newlines)
            # URL components should never have unescaped spaces
            val = "".join(val.split())
            
            # SQLAlchemy 1.4+ requirement: postgres:// -> postgresql://
            # We also handle cases where the protocol might be missing or triple-slashed
            if val.startswith("postgres://"):
                val = val.replace("postgres://", "postgresql://", 1)
            elif val.startswith("postgres:"):
                val = val.replace("postgres:", "postgresql:", 1)
            elif val.startswith("//"):
                val = "postgresql:" + val

            # 4. Professional Patch: Force IPv4 for Supabase on Hugging Face
            # HF infrastructure often fails with IPv6 "Network is unreachable"
            try:
                parsed = urlparse(val)
                if parsed.hostname and (".supabase.co" in parsed.hostname or ".supabase.com" in parsed.hostname):
                    # Resolve hostname to IPv4 specifically
                    ipv4 = socket.gethostbyname(parsed.hostname)
                    if ipv4:
                        # Rebuild netloc with IPv4 IP
                        new_netloc = parsed.netloc.replace(parsed.hostname, ipv4)
                        val = urlunparse(parsed._replace(netloc=new_netloc))
                        print(f"[CONFIG] IPv4 Resolved for Supabase: {ipv4}")
            except Exception as e:
                print(f"[CONFIG] IPv4 Resolution check skipped: {e}")


        
        # Critical: Print metadata for debugging
        # We'll use a safer split and print the first 15 chars
        safe_preview = val[:15] + "..." if val and len(val) > 15 else val
        print(f"[CONFIG] DB Source Selected. Preview: {safe_preview}, Length: {len(val) if val else 0}")

        
        return val



    STORAGE_ROOT: str = Field(default="/app/.cache/app_data")
    UPLOAD_DIR: str = Field(default="/app/.cache/app_data/uploads")
    
    SUPABASE_URL: Optional[str] = Field(default=None)
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(default=None)
    USE_SUPABASE_STORAGE: bool = Field(default=False)


class VectorStoreConfig(BaseSettingsExtra):
    """Vector Store Configuration"""
    VECTOR_STORE_TYPE: Literal["lancedb", "supabase"] = Field(
        default="lancedb"
    )

    @field_validator("VECTOR_STORE_TYPE", mode="before")
    @classmethod
    def validate_vector_type(cls, v):

        return os.getenv("VECTOR_STORE_TYPE") or os.getenv("DB_TYPE") or v

    LANCEDB_URI: str = Field(default="./data/vectordb")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)



class RedisConfig(BaseSettingsExtra):
    """Redis Configuration for Caching & Sessions"""
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0"
    )

    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def validate_redis_url(cls, v):

        return os.getenv("REDIS_URL") or os.getenv("CACHE_URL") or v

    REDIS_MAX_CONNECTIONS: int = Field(default=50)
    REDIS_SOCKET_TIMEOUT: int = Field(default=5)
    SESSION_TTL: int = Field(default=86400)
    CACHE_TTL: int = Field(default=3600)
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.9)




class LoggingConfig(BaseSettingsExtra):
    """Logging Configuration"""
    LOG_LEVEL: str = Field(default="INFO")
    JSON_LOGS: bool = Field(default=False)


class RateLimitConfig(BaseSettingsExtra):
    """Rate Limiting Configuration"""
    UPLOAD_RATE: str = Field(default="10/minute")
    CHAT_RATE: str = Field(default="60/minute")


class SearchConfig(BaseSettingsExtra):
    """Search & External Tool Configuration"""
    TAVILY_API_KEY: Optional[str] = Field(default=None)





class Settings(BaseSettingsExtra):
    """Global Application Settings"""
    PROJECT_NAME: str = Field(default="DocuCentric")
    APP_NAME: str = Field(default="DocuCentric Intelligence")
    VERSION: str = Field(default="0.3.0")
    APP_VERSION: str = Field(default="0.3.0")
    API_V1_STR: str = Field(default="/api")
    ENVIRONMENT: str = Field(default="production")

    
    # Nested configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)



    def initialize_storage(self):
        """Creates necessary storage directories"""
        for path_str in [self.database.STORAGE_ROOT, self.database.UPLOAD_DIR]:
            path = Path(path_str)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"[CONFIG] Created directory: {path_str}")




# Global instance
settings = Settings()
