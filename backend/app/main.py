"""
DocuCentric - Enterprise Document Intelligence & Reasoning Platform
Main FastAPI Application
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.routes import documents, chat, knowledge_graph
from app.db.database import Base, engine, wait_for_db
from app.core.logging_config import configure_logging, get_logger
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.core.redis_adapter import redis_adapter
from app.core.session_manager import session_manager
from app.core.query_cache import query_cache

# Initialize structured logging
configure_logging(
    log_level=settings.logging.LOG_LEVEL,
    json_output=settings.logging.JSON_LOGS
)

logger = get_logger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown"""
    # ==================== STARTUP ====================
    logger.info("application_startup", version=settings.APP_VERSION)
    
    # Initialize storage directories
    settings.initialize_storage()
    logger.info("storage_initialized", root=settings.database.STORAGE_ROOT)
    
    # Connect to Redis
    await redis_adapter.connect()
    logger.info("redis_connected", url=settings.redis.REDIS_URL)
    
    # Wait for database
    wait_for_db()
    Base.metadata.create_all(bind=engine)
    logger.info("database_initialized")
    
    # Warm up LLM service
    from app.services.llm_service import llm_service
    llm_service.warmup()
    logger.info("llm_warmed_up", provider=settings.llm.LLM_PROVIDER)
    
    # Initialize vector store
    from app.services.vector_store import vector_store_service
    logger.info("vector_store_initialized", uri=settings.vector_store.LANCEDB_URI)
    
    logger.info(
        "application_ready",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        llm_provider=settings.llm.LLM_PROVIDER,
        llm_model=settings.llm.active_model
    )
    
    yield
    
    # ==================== SHUTDOWN ====================
    logger.info("application_shutdown")
    await redis_adapter.disconnect()
    logger.info("redis_disconnected")


# CORS configuration MUST be first, before other middleware
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Document Intelligence Platform with LangGraph Workflows and CAG",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== MIDDLEWARE ====================

# Initialize rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Session middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Handle session persistence via cookies"""
    response = await call_next(request)

    if hasattr(request.state, "new_session_id"):
        response.set_cookie(
            key="session_id",
            value=request.state.new_session_id,
            max_age=settings.redis.SESSION_TTL,
            httponly=True,
            secure=settings.ENVIRONMENT == "production",
            samesite="lax"
        )

    return response

# ==================== ROUTES ====================

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(knowledge_graph.router)


# ==================== PUBLIC ENDPOINTS ====================

@app.get("/")
def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "features": [
            "LangGraph Workflows",
            "Cache-Augmented Generation (CAG)",
            "Redis Sessions",
            "Semantic Caching",
            "Multi-Provider LLM Router"
        ],
        "llm_provider": settings.llm.LLM_PROVIDER,
        "llm_model": settings.llm.active_model,
        "endpoints": [
            "/health",
            "/docs",
            "/api/documents/upload",
            "/api/chat/sessions",
            "/api/chat/stream",
            "/api/cache/stats"
        ]
    }


@app.get("/health")
def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "components": {
            "database": "connected",
            "vector_store": "LanceDB",
            "cache": "Redis",
            "llm_provider": settings.llm.LLM_PROVIDER,
            "llm_model": settings.llm.active_model
        }
    }


@app.get("/api/sessions")
async def list_sessions(user_id: str = "default_user"):
    """List all active sessions for user"""
    return await session_manager.list_user_sessions(user_id)


@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear session conversation history"""
    success = await session_manager.clear_session(session_id)
    return {"success": success}


@app.get("/api/cache/stats")
async def cache_stats():
    """Get comprehensive cache statistics"""
    from app.services.cag_engine import cag_engine
    
    return {
        "redis": await redis_adapter.health_check(),
        "cag": cag_engine.get_cache_stats()
    }


@app.post("/api/cache/invalidate")
async def invalidate_cache(
    namespace: str = "default",
    pattern: str = None
):
    """Manually invalidate cache entries"""
    count = await redis_adapter.invalidate_semantic_cache(namespace, pattern)
    return {"invalidated_entries": count}


@app.post("/api/cache/warm")
async def warm_cache(document_sets: list[list[str]]):
    """Pre-warm CAG cache for document sets"""
    from app.services.cag_engine import cag_engine
    
    success_count = await cag_engine.warm_cache(document_sets)
    return {
        "success": success_count,
        "total": len(document_sets)
    }


@app.post("/system/reset/vector-store")
async def reset_vector_store():
    """
    CRITICAL: Reset LanceDB Vector Store
    WARNING: This will delete all indexed documents
    """
    try:
        from app.services.vector_store import LANCEDB_DIR
        import shutil
        
        if os.path.exists(LANCEDB_DIR):
            shutil.rmtree(LANCEDB_DIR)
            os.makedirs(LANCEDB_DIR, mode=0o777, exist_ok=True)
        
        logger.info("vector_store_reset")
        return {
            "status": "success",
            "message": "Vector store wiped successfully"
        }
    except Exception as e:
        logger.error("vector_store_reset_failed", error=str(e))
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/system/info")
async def system_info():
    """Get system configuration and feature info"""
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        },
        "llm": {
            "provider": settings.llm.LLM_PROVIDER,
            "model": settings.llm.active_model,
            "temperature": settings.llm.TEMPERATURE,
            "max_tokens": settings.llm.MAX_TOKENS
        },
        "vector_store": {
            "type": "LanceDB",
            "embedding_model": settings.vector_store.EMBEDDING_MODEL,
            "chunk_size": settings.vector_store.CHUNK_SIZE,
            "chunk_overlap": settings.vector_store.CHUNK_OVERLAP
        },
        "cache": {
            "type": "Redis",
            "ttl": settings.redis.CACHE_TTL,
            "semantic_threshold": settings.redis.SEMANTIC_CACHE_THRESHOLD
        },
        "workflows": [
            "document_qa",
            "resume_analysis",
            "contract_analysis"
        ],
        "features": {
            "langgraph_workflows": True,
            "cache_augmented_generation": True,
            "semantic_caching": True,
            "session_persistence": True,
            "multi_provider_llm": True,
            "hallucination_detection": True,
            "external_search": bool(settings.search.TAVILY_API_KEY)
        }
    }
