# Edit backend/app/main.py - change version number
# Or just touch it to modify timestamp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Enterprise Document Intelligence API - Enhanced Self-RAG",
    version="0.2.0",
    description="Multi-document AI with Self-RAG: Citations, Query Rewriting, Hallucination Detection"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Enterprise Document Intelligence",
        "version": "0.1.0",
        "datasets": ["RVL-CDIP", "FUNSD", "SROIE", "Synthetic-GAN"],
        "endpoints": ["/health", "/upload", "/classify", "/extract", "/chat"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "components": {
            "ocr": "tesseract",
            "pdf": "pdfplumber",
            "word": "python-docx",
            "excel": "pandas/openpyxl"
        }
    }

# Import routes
from app.routes import upload, classify, extract, chat

app.include_router(upload.router, prefix="/upload", tags=["Ingestion"])
app.include_router(classify.router, prefix="/classify", tags=["Classification"])
app.include_router(extract.router, prefix="/extract", tags=["Extraction"])
app.include_router(chat.router, prefix="/chat", tags=["QA"])