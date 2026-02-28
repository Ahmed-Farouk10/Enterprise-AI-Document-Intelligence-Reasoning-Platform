from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request, Response, BackgroundTasks

from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
from pathlib import Path

from app.schemas import DocumentResponse, PaginatedDocuments
from app.db.database import get_db
from app.db.models import Document
from app.db.service import DatabaseService
from app.core.rate_limiter import limiter
from app.core.logging_config import get_logger
import uuid

router = APIRouter(prefix="/api/documents", tags=["Documents"])
logger = get_logger(__name__)

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def process_document_background(document_id: str, file_path: Path, mime_type: str, original_filename: str):
    """Background task for document processing"""
    try:
        logger.info(f"🔄 Starting background processing for doc_id={document_id}")
        from app.db.database import SessionLocal
        
        # New DB session for background task
        with SessionLocal() as db:
            document = DatabaseService.get_document(db, document_id)
            if not document:
                logger.error(f"❌ Document {document_id} not found in background task")
                return

            try:
                # Extract text using OCR Service
                from app.services.ocr import ocr_service
                text = ocr_service.extract_text(file_path, mime_type)
                
                # Index in Vector Database
                if text and len(text.strip()) > 10:
                    try:
                        # 1. Store chunks in LanceDB
                        from app.services.vector_store import vector_store_service
                        logger.info(f"💾 Vectorizing document {document_id}")
                        
                        metadata = {
                            "filename": original_filename,
                            "upload_date": document.created_at.isoformat(),
                            "mime_type": mime_type
                        }
                        
                        chunks_inserted = await vector_store_service.ingest_document(
                            document_id=str(document.id),
                            text=text,
                            metadata=metadata
                        )
                        
                        # 2. Extract structured entities (Resumes, Invoices, etc.)
                        from app.services.cognee_pipelines import route_to_pipeline
                        logger.info(f" Extracted structured entities for {document_id}")
                        
                        pipeline_result = await route_to_pipeline(
                            text=text,
                            document_id=str(document.id),
                            document_type="auto_detect"
                        )
                        
                        # Extract stats safely
                        is_success = chunks_inserted > 0
                        error_msg = pipeline_result.error if not pipeline_result.success else None
                        
                        if is_success:
                           extra_data = {
                                "graph_stats": {
                                    "entity_count": chunks_inserted, # Use chunks indexed as "entity" equivalent
                                    "document_type": pipeline_result.document_type,
                                    "dataset_name": pipeline_result.dataset,
                                    "entities": pipeline_result.entities or {} # Contains resume counts etc
                                }
                           }
                           
                           if error_msg:
                               extra_data["extraction_warning"] = str(error_msg)
                               logger.warning(f"⚠️ Partial success for {document_id}: Extraction failed - {error_msg}")
                           
                           document.extra_data = extra_data
                    
                    except Exception as store_error:
                         logger.error(f"⚠️ Vector insertion failed: {store_error}", exc_info=True)
                         if not document.extra_data: document.extra_data = {}
                         document.extra_data["vector_error"] = str(store_error)
                         document.status = "failed"
                         db.commit()
                         return

                    document.status = "completed"
                    logger.info(f"✅ Document {document_id} processing complete")
                
                else:
                    document.status = "failed"
                    logger.warning(f"⚠️ No text extracted for {document_id}")
            
            except Exception as e:
                logger.error(f"❌ Processing failed for {document_id}: {e}")
                document.status = "failed"
            
            finally:
                db.commit()

    except Exception as e_outer:
        logger.error(f"🔥 Critical background task error: {e_outer}")


@router.post("/upload", response_model=DocumentResponse)
@limiter.limit("10/hour")
async def upload_document(
    request: Request, 
    response: Response, 
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Upload a document and dispatch background processing"""
    
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
            
        # Create document record
        document = DatabaseService.create_document(
            db=db,
            filename=unique_filename,
            original_name=file.filename,
            file_size=file_size,
            mime_type=file.content_type,
            version=new_version
        )
        
        # Set initial status
        document.status = "pending"
        db.commit()
        db.refresh(document)
        
        # Dispatch background task
        background_tasks.add_task(
            process_document_background, 
            str(document.id), 
            file_path, 
            file.content_type, 
            file.filename
        )
        
        logger.info(f"🚀 Dispatched background processing for {document.id}")
        
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
                "message": "Upload successful. Processing in background."
            }
        }
    
    except Exception as e:
        logger.error("document_upload_failed", error=str(e), exc_info=True)
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

