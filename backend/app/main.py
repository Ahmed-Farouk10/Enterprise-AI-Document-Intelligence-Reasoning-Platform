# FastAPI Main Application
import os
import pathlib
import sys
import logging

# ===== CRITICAL: Configure Cognee Path BEFORE Any Other Imports =====
# This ensures environment variables and monkey patches are applied 
# BEFORE Cognee or any other module imports it.
try:
    from app.cognee_setup import COGNEE_ROOT, verify_cognee_setup
except ImportError:
    # Fallback if running from root without package context
    sys.path.append(os.path.join(os.getcwd(), 'app'))
    from app.cognee_setup import COGNEE_ROOT, verify_cognee_setup

# Create directories explicitly (Redundant but safe)
pathlib.Path(f"{COGNEE_ROOT}/data_storage").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{COGNEE_ROOT}/databases").mkdir(parents=True, exist_ok=True)
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
    
    # [ADDED] Force Cognee Table Creation (Synchronous Wrapper)
    # This is critical for HF Spaces where async init might be flaky
    try:
        from cognee.infrastructure.databases.relational import create_db_and_tables
        logger.info("⚙️ STARTUP: Forcing Cognee table verification...")
        await create_db_and_tables()
        logger.info("✅ STARTUP: Cognee tables initialized.")
    except Exception as e:
        logger.error(f"⚠️ STARTUP WARNING: Cognee init failed: {e}")

    logger.info("application_startup", status="database_initialized")

    # Warmup LLM Service
    from app.services.llm_service import llm_service
    asyncio.create_task(asyncio.to_thread(llm_service.warmup))

    # Initialize Cognee Engine
    # Initialize Cognee Engine
    async def init_cognee():
        try:
            # Standard maintenance: Prune stale metadata on startup (User Requested Fix)
            try:
                import cognee
                # This clears the local metadata registry safely to resolve UNIQUE constraint errors
                if hasattr(cognee, "prune_system"):
                    await cognee.prune_system()
                    logger.info("application_startup", status="cognee_system_pruned")
                else:
                    # Fallback for older/different versions causing issues
                    cognee.prune()
                    logger.info("application_startup", status="cognee_pruned")
            except Exception as e:
                 logger.warning(f"Prune failed (non-critical): {e}")

            from app.services.cognee_engine import cognee_engine
            await cognee_engine.initialize()
            logger.info("application_startup", status="cognee_initialized")
        except Exception as e:
            logger.error("application_startup", status="cognee_initialization_failed", error=str(e))
    
    asyncio.create_task(init_cognee())

    # Start Memify Background Service (Self-Improvement)
    from app.services.cognee_background import memify_service
    await memify_service.start()
    logger.info("application_startup", status="memify_service_started")
    
    yield
    
    # Shutdown
    await memify_service.stop()
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
def health_check():
    """Diagnostic endpoint to check Cognee status."""
    return {
        "status": "online",
        "service": "Enterprise Document Intelligence",
        "cognee_root": os.environ.get("COGNEE_ROOT"),
        "anon_id_path": os.environ.get("COGNEE_ANONYMOUS_ID_PATH"),
        "writable": os.access(os.environ.get("COGNEE_ROOT", "/"), os.W_OK),
        "db_exists": os.path.exists(os.path.join(os.environ.get("COGNEE_ROOT", ""), "databases", "cognee_db.db")),
        "permissions_patch": "applied"
    }

@app.post("/system/reset")
async def system_reset():
    """CRITICAL: Prune all data from Cognee. Use with caution."""
    try:
        import cognee
        cognee.prune()
        return {"status": "success", "message": "System pruned. All data removed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/system/cognify/{dataset_name}")
async def manual_cognify(dataset_name: str):
    """Manually trigger Cognify for a dataset."""
    try:
        import cognee
        # Trigger cognify
        await cognee.cognify(datasets=[dataset_name])
        return {"status": "success", "message": f"Cognify triggered for {dataset_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
