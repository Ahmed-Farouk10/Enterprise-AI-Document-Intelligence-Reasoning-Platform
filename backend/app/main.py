# FastAPI Main Application
import os
import pathlib
import sys

# ===== CRITICAL: Configure Cognee Path BEFORE Any Other Imports =====
# Force Cognee to use writable directory (fix for HF Spaces permissions)
# Default is site-packages which is read-only
cognee_root = "/app/.cache/cognee_data"
os.environ["COGNEE_ROOT"] = cognee_root
os.environ["COGNEE_ROOT_DIR"] = cognee_root 
os.environ["SYSTEM_ROOT_DIRECTORY"] = cognee_root
os.environ["COGNEE_DATA_STORAGE"] = f"{cognee_root}/data_storage"
os.environ["COGNEE_ANON_ID_PATH"] = "/tmp/.cognee_anon_id"  # Fix telemetry warning

# Create directories explicitly
pathlib.Path(f"{cognee_root}/data_storage").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{cognee_root}/databases").mkdir(parents=True, exist_ok=True)

# NOW import cognee setup
from app.cognee_setup import COGNEE_ROOT, verify_cognee_setup
# ====================================================================

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, chat, jobs, graph
from app.db.database import Base, engine, wait_for_db
from app.core.logging_config import configure_logging, get_logger
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import logging
import asyncio

# --- MEMORY ENRICHMENT IMPORTS ---
from app.core.redis_adapter import redis_adapter
from app.core.session_manager import session_manager
from app.core.query_cache import query_cache

# Initialize structured logging
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("JSON_LOGS", "true").lower() == "true"
configure_logging(log_level=log_level, json_output=json_logs)

logger = get_logger(__name__)

# Lifespan Context Manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("application_startup", status="connecting_to_redis")
    await redis_adapter.connect()
    
    # Existing Startup Logic
    logger.info("application_startup", status="waiting_for_database")
    wait_for_db()
    
    logger.info("application_startup", status="creating_database_tables")
    Base.metadata.create_all(bind=engine)
    logger.info("application_startup", status="database_initialized")

    # Warmup LLM Service
    from app.services.llm_service import llm_service
    asyncio.create_task(asyncio.to_thread(llm_service.warmup))

    # Initialize Cognee Engine
    async def init_cognee():
        try:
            from app.services.cognee_engine import cognee_engine
            await cognee_engine.initialize()
            logger.info("application_startup", status="cognee_initialized")
        except Exception as e:
            logger.error("application_startup", status="cognee_initialization_failed", error=str(e))
    
    asyncio.create_task(init_cognee())
    
    yield
    
    # Shutdown
    await redis_adapter.disconnect()
    logger.info("application_shutdown", status="redis_disconnected")

app = FastAPI(
    title="Enterprise Document Intelligence API - Enhanced Self-RAG",
    version="0.4.1",
    description="Multi-document AI with Self-RAG: Citations, Query Rewriting, Hallucination Detection + Async Processing",
    lifespan=lifespan
)

# Initialize Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# --- SESSION MIDDLEWARE ---
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Middleware to handle session persistence."""
    response = await call_next(request)
    
    # Set session cookie if new session created
    if hasattr(request.state, "new_session_id"):
        response.set_cookie(
            key="session_id",
            value=request.state.new_session_id,
            max_age=86400,  # 24 hours
            httponly=True,
            secure=True,
            samesite="lax"
        )
    
    return response

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(jobs.router)
app.include_router(graph.router)

# --- NEW SESSION & CACHE ENDPOINTS ---
@app.get("/api/sessions")
async def list_sessions(user_id: str):
    """List all active sessions for user."""
    return await session_manager.list_user_sessions(user_id)

@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear session conversation history."""
    success = await session_manager.clear_session(session_id)
    return {"success": success}

@app.get("/api/cache/stats")
async def cache_stats():
    """Get Redis cache statistics."""
    return await redis_adapter.health_check()

@app.post("/api/cache/invalidate")
async def invalidate_cache(
    namespace: str = "default",
    pattern: str = None
):
    """Manually invalidate cache entries."""
    count = await redis_adapter.invalidate_semantic_cache(namespace, pattern)
    return {"invalidated_entries": count}

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Enterprise Document Intelligence",
        "version": "0.4.1",
        "features": ["Redis Sessions", "Semantic Caching", "Graph RAG"],
        "endpoints": [
            "/health",
            "/api/documents/upload",
            "/api/chat/stream",
            "/api/sessions",
            "/api/cache/stats"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "0.3.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }
