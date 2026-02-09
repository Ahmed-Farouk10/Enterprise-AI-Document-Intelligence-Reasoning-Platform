from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
from pathlib import Path

from app.schemas import DocumentResponse, PaginatedDocuments
from app.db.database import get_db
from app.db.models import Document
from app.db.service import DatabaseService
from app.services.retreival import vector_store
from app.core.rate_limiter import limiter

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload", response_model=DocumentResponse)
@limiter.limit("10/hour")
async def upload_document(request: Request, response: Response, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a document and dispatch async processing job"""
    from app.core.logging_config import get_logger
    from app.workers.tasks import process_document_task
    import uuid
    
    logger = get_logger(__name__)
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not supported. Only PDF, TXT, and DOCX are allowed.")
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    file_stem = Path(file.filename).stem
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file_stem}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        
        # Check for existing versions
        existing_docs = db.query(Document).filter(
            Document.original_name == file.filename
        ).order_by(Document.version.desc()).first()
        
        new_version = 1
        if existing_docs:
            new_version = existing_docs.version + 1
            
        # Create document record in database with "pending" status
        document = DatabaseService.create_document(
            db=db,
            filename=unique_filename,
            original_name=file.filename,
            file_size=file_size,
            mime_type=file.content_type,
            version=new_version
        )
        
        # Set initial status to pending
        document.status = "pending"
        db.commit()
        db.refresh(document)
        
        # Dispatch processing task (Async via Celery Worker)
        # Using .delay() offloads to Redis queue for background worker
        task = process_document_task.delay(
            doc_id=document.id,
            file_path=unique_filename,
            mime_type=file.content_type,
            filename=file.filename
        )
        
        logger.info("document_processed_inline", doc_id=document.id, filename=file.filename)
        
        # Return immediately with pending status
        return {
            "id": document.id,
            "filename": document.filename,
            "original_name": document.original_name,
            "file_size": document.file_size,
            "mime_type": document.mime_type,
            "uploaded_at": document.created_at.isoformat(),
            "processed_at": None,
            "status": "pending",
            "version": document.version,
            "metadata": {
                "task_id": task.id,
                "message": "Document uploaded successfully. Processing in background."
            }
        }
    
    except Exception as e:
        logger.error("document_upload_failed", error=str(e), exc_info=True)
        # Clean up file if database operation fails
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.get("", response_model=PaginatedDocuments)
async def get_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all documents with pagination"""
    skip = (page - 1) * page_size
    documents, total = DatabaseService.get_documents(db, skip=skip, limit=page_size)
    
    # Convert ORM objects to dicts
    items = [{
        "id": doc.id,
        "filename": doc.filename,
        "original_name": doc.original_name,
        "file_size": doc.file_size,
        "mime_type": doc.mime_type,
        "uploaded_at": doc.created_at.isoformat(),
        "processed_at": doc.updated_at.isoformat() if doc.status == "completed" else None,
        "status": doc.status,
        "version": doc.version,
        "metadata": doc.extra_data or {}
    } for doc in documents]
    
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "items": items,
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages
        }
    }

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get a specific document"""
    document = DatabaseService.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "original_name": document.original_name,
        "file_size": document.file_size,
        "mime_type": document.mime_type,
        "uploaded_at": document.created_at.isoformat(),
        "processed_at": document.updated_at.isoformat() if document.status == "completed" else None,
        "status": document.status,
        "version": document.version,
        "metadata": document.extra_data or {}
    }

@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document"""
    document = DatabaseService.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from disk
    file_path = UPLOAD_DIR / document.filename
    if file_path.exists():
        os.remove(file_path)
    
    # Delete from database
    DatabaseService.delete_document(db, document_id)
    
    return {"success": True, "message": "Document deleted successfully"}

@router.get("/{document_id}/status")
async def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Get document processing status"""
    document = DatabaseService.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "status": document.status,
        "progress": 100 if document.status == "completed" else 0
    }

@router.get("/{document_id}/download")
async def download_document(document_id: str, db: Session = Depends(get_db)):
    """Download or view a document"""
    from fastapi.responses import FileResponse
    
    document = DatabaseService.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = UPLOAD_DIR / document.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=document.original_name,
        media_type=document.mime_type
    )

