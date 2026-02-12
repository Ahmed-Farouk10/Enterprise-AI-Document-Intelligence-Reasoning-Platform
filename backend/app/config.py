from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Enterprise AI Document Intelligence"
    DEBUG: bool = True
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Session Configuration
    SESSION_TTL_HOURS: int = 24
    SESSION_MAX_MESSAGES: int = 50
    
    # Cache Configuration
    CACHE_DEFAULT_TTL_MINUTES: int = 60
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_NAMESPACE: str = "document_intelligence"
    
    # Memory Configuration
    MEMORY_SHORT_TERM_TTL_MINUTES: int = 60
    MEMORY_WORKING_LIMIT: int = 10
    MEMORY_CONSOLIDATION_INTERVAL_HOURS: int = 24
    
    class Config:
        env_file = ".env"
        extra = "ignore" # Allow extra fields in env file

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
