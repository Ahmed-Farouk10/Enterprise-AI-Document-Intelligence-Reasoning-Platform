from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List
import os
import shutil
from pathlib import Path

from app.schemas import DocumentResponse, PaginatedDocuments, PaginationMeta
from app.database import Database

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and process it for RAG"""
    import logging
    import uuid
    logger = logging.getLogger(__name__)
    
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
        
        # Create document record with processing status
        document = Database.create_document(
            filename=unique_filename,
            original_name=file.filename,
            file_size=file_size,
            mime_type=file.content_type
        )
        
        # Process document asynchronously (extract text and add to vector store)
        try:
            from app.services.retreival import vector_store
            
            # Extract text based on file type
            if file.content_type == "text/plain":
                # Plain text
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file.content_type == "application/pdf":
                # PDF extraction
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
                        document["metadata"]["page_count"] = len(pdf.pages)
                except Exception as e:
                    logger.error(f"PDF extraction failed: {e}")
                    text = ""
            elif "wordprocessingml" in file.content_type:
                # DOCX extraction
                try:
                    from docx import Document as DocxDocument
                    doc = DocxDocument(file_path)
                    text = "\n\n".join([para.text for para in doc.paragraphs])
                except Exception as e:
                    logger.error(f"DOCX extraction failed: {e}")
                    text = ""
            else:
                text = ""
            
            # Add to vector store if text extracted
            if text and len(text.strip()) > 50:
                vector_store.add_document(
                    text=text,
                    doc_id=document["id"],
                    doc_name=file.filename
                )
                document["metadata"]["extracted_text"] = text[:500] + "..." if len(text) > 500 else text
                document["metadata"]["vector_store_id"] = document["id"]
                document["status"] = "completed"
                logger.info(f"Document {file.filename} processed and added to vector store")
            else:
                document["status"] = "failed"
                document["metadata"]["error"] = "Could not extract text from document"
                logger.warning(f"No text extracted from {file.filename}")
                
        except Exception as e:
            logger.error(f"Document processing error: {e}", exc_info=True)
            document["status"] = "failed"
            document["metadata"]["error"] = str(e)
        
        return document
    
    except Exception as e:
        # Clean up file if database operation fails
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.get("", response_model=PaginatedDocuments)
async def get_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get all documents with pagination"""
    skip = (page - 1) * page_size
    documents, total = Database.get_documents(skip=skip, limit=page_size)
    
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "items": documents,
        "meta": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages
        }
    }

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document"""
    document = Database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    document = Database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from disk
    file_path = UPLOAD_DIR / document["filename"]
    if file_path.exists():
        os.remove(file_path)
    
    # Delete from database
    Database.delete_document(document_id)
    
    return {"success": True, "message": "Document deleted successfully"}

@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    document = Database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "status": document["status"],
        "progress": 100 if document["status"] == "completed" else 0
    }

@router.get("/{document_id}/download")
async def download_document(document_id: str):
    """Download or view a document"""
    from fastapi.responses import FileResponse
    
    document = Database.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = UPLOAD_DIR / document["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=document["original_name"],
        media_type=document["mime_type"]
    )
