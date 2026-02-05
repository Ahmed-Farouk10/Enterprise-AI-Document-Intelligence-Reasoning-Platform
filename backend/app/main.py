# FastAPI Main Application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, chat, jobs
from app.db.database import Base, engine, wait_for_db
from app.core.logging_config import configure_logging, get_logger
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import logging

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
    logger.info("application_startup", event="waiting_for_database")
    wait_for_db()
    
    logger.info("application_startup", event="creating_database_tables")
    Base.metadata.create_all(bind=engine)
    logger.info("application_startup", event="database_initialized")

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
