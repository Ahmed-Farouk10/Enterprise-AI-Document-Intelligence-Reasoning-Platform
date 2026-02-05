# FastAPI Main Application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, chat

app = FastAPI(
    title="Enterprise Document Intelligence API - Enhanced Self-RAG",
    version="0.3.0",
    description="Multi-document AI with Self-RAG: Citations, Query Rewriting, Hallucination Detection"
)

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
