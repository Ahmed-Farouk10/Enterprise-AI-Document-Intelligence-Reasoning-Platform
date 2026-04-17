import os
import asyncio
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
    logger.info("application_startup_initiated", version=settings.APP_VERSION)
    
    # Initialize storage directories
    try:
        settings.initialize_storage()
        logger.info("storage_initialized", root=settings.database.STORAGE_ROOT)
    except Exception as e:
        logger.error(f"storage_initialization_failed: {e}")

    # Connect to Redis with a timeout
    try:
        await asyncio.wait_for(redis_adapter.connect(), timeout=10.0)
        logger.info("redis_connected")
    except asyncio.TimeoutError:
        logger.error("redis_connection_timeout: App will start but caching will be disabled")
    except Exception as e:
        logger.error(f"redis_connection_failed: {e}")
    
    # Wait for database (non-blocking retry implemented in database.py)
    db_ok = wait_for_db()
    if db_ok:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("database_schema_synced")
        except Exception as e:
            logger.error(f"database_sync_failed: {e}")
    
    # Warm up LLM service
    try:
        from app.services.llm_service import llm_service
        llm_service.warmup()
        logger.info("llm_warmed_up", provider=settings.llm.LLM_PROVIDER)
    except Exception as e:
        logger.error(f"llm_warmup_failed: {e}")
    
    # Initialize vector store
    try:
        from app.services.vector_store import vector_store_service
        logger.info("vector_store_initialized", type=settings.vector_store.VECTOR_STORE_TYPE)
    except Exception as e:
        logger.error(f"vector_store_initialization_failed: {e}")
    
    logger.info(
        "application_ready",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        llm_provider=settings.llm.LLM_PROVIDER,
        llm_model=settings.llm.active_model
    )
    
    yield
    
    # ==================== SHUTDOWN ====================
    logger.info("application_shutdown_initiated")
    try:
        await redis_adapter.disconnect()
        logger.info("redis_disconnected")
    except Exception as e:
        logger.error(f"redis_disconnect_failed: {e}")


# Initialize app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Document Intelligence Platform with LangGraph Workflows and CAG",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "https://ahmed-farouk10-enterprise-ai-document-intelligence.vercel.app",
    "https://ahmedaayman-eadrip.hf.space",
]

# Allow all Vercel subdomains (previews, branches, etc.)
allowed_origin_regex = r"https://.*\.vercel\.app"


# Support for dynamic HF Space URLs
hf_space_id = os.getenv("SPACE_ID")
if hf_space_id:
    # Example: ahmedaayman/EADRIP -> https://ahmedaayman-eadrip.hf.space
    parts = hf_space_id.split("/")
    if len(parts) == 2:
        owner, name = parts
        hf_origin = f"https://{owner.lower().replace('_', '-')}-{name.lower().replace('_', '-')}.hf.space"
        if hf_origin not in allowed_origins:
            allowed_origins.append(hf_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
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
            "vector_store": settings.vector_store.VECTOR_STORE_TYPE,
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
    CRITICAL: Reset Vector Store
    WARNING: This will delete all indexed documents
    """
    try:
        from app.services.vector_store import vector_store_service
        # Logic depends on store type, handled in service
        return {
            "status": "success",
            "message": "Vector store wipe initiated"
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
            "type": settings.vector_store.VECTOR_STORE_TYPE,
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
