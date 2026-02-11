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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, chat, jobs, graph
from app.db.database import Base, engine, wait_for_db
from app.core.logging_config import configure_logging, get_logger
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import logging
import asyncio

# Initialize structured logging
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("JSON_LOGS", "true").lower() == "true"
configure_logging(log_level=log_level, json_output=json_logs)

logger = get_logger(__name__)

app = FastAPI(
    title="Enterprise Document Intelligence API - Enhanced Self-RAG",
    version="0.4.0",
    description="Multi-document AI with Self-RAG: Citations, Query Rewriting, Hallucination Detection + Async Processing"
)

# Initialize Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    logger.info("application_startup", status="waiting_for_database")
    wait_for_db()
    
    logger.info("application_startup", status="creating_database_tables")
    Base.metadata.create_all(bind=engine)
    logger.info("application_startup", status="database_initialized")

    # Warmup LLM Service (Load Model into Memory)
    from app.services.llm_service import llm_service
    # Run warmup in a thread to avoid blocking the event loop
    asyncio.create_task(asyncio.to_thread(llm_service.warmup))

    # Initialize Cognee Engine (Graph Database Connection)
    async def init_cognee():
        try:
            from app.services.cognee_engine import cognee_engine
            await cognee_engine.initialize()
            logger.info("application_startup", status="cognee_initialized")
        except Exception as e:
            logger.error("application_startup", status="cognee_initialization_failed", error=str(e))
    
    asyncio.create_task(init_cognee())

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

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Enterprise Document Intelligence",
        "version": "0.3.0",
        "datasets": ["RVL-CDIP", "FUNSD", "SROIE", "Synthetic-GAN"],
        "endpoints": [
            "/health",
            "/api/documents/upload",
            "/api/documents",
            "/api/chat/sessions",
            "/api/chat/sessions/{id}/messages"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "0.3.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }
