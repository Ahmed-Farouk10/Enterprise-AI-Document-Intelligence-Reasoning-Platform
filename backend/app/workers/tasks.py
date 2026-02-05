"""
Celery tasks for async document processing
"""

from celery import Task
from app.workers.celery_app import celery_app
from app.core.logging_config import get_logger
from app.db.database import SessionLocal
from app.db.service import DatabaseService
from sqlalchemy.orm import Session
import os
from pathlib import Path

logger = get_logger(__name__)

UPLOAD_DIR = Path("uploads")


class DatabaseTask(Task):
    """
    Base task that provides a database session
    Automatically handles session lifecycle
    """
    _session = None

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = SessionLocal()
        return self._session

    def after_return(self, *args, **kwargs):
        if self._session is not None:
            self._session.close()
            self._session = None


@celery_app.task(bind=True, base=DatabaseTask, name="app.workers.tasks.process_document_task")
def process_document_task(self, doc_id: str, file_path: str, mime_type: str, filename: str):
    """
    Process a document asynchronously:
    1. Extract text from file
    2. Add to vector store
    3. Update document status
    
    Args:
        doc_id: Database document ID
        file_path: Path to the uploaded file
        mime_type: MIME type of the file
        filename: Original filename
    """
    logger.info("document_processing_started", doc_id=doc_id, filename=filename, task_id=self.request.id)
    
    try:
        # Get database document
        document = DatabaseService.get_document(self.session, doc_id)
        if not document:
            logger.error("document_not_found", doc_id=doc_id)
            return {"status": "failed", "error": "Document not found"}
        
        # Update status to processing
        document.status = "processing"
        self.session.commit()
        
        # Extract text based on file type
        text = ""
        full_path = UPLOAD_DIR / file_path
        
        if mime_type == "text/plain":
            # Plain text
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
                
        elif mime_type == "application/pdf":
            # PDF extraction
            try:
                import pdfplumber
                with pdfplumber.open(full_path) as pdf:
                    text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
                    logger.info("pdf_extracted", doc_id=doc_id, pages=len(pdf.pages))
            except Exception as e:
                logger.error("pdf_extraction_failed", doc_id=doc_id, error=str(e))
                
        elif "wordprocessingml" in mime_type:
            # DOCX extraction
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(full_path)
                text = "\n\n".join([para.text for para in doc.paragraphs])
                logger.info("docx_extracted", doc_id=doc_id)
            except Exception as e:
                logger.error("docx_extraction_failed", doc_id=doc_id, error=str(e))
        
        # Add to vector store if text extracted
        if text and len(text.strip()) > 50:
            # Import Qdrant vector store
            from app.services.qdrant_store import qdrant_store
            
            qdrant_store.add_document(
                text=text,
                doc_id=doc_id,
                doc_name=filename
            )
            
            document.status = "completed"
            logger.info("document_processed_successfully", doc_id=doc_id, text_length=len(text))
        else:
            document.status = "failed"
            logger.warning("no_text_extracted", doc_id=doc_id)
        
        # Update document in database
        self.session.commit()
        
        return {
            "status": document.status,
            "doc_id": doc_id,
            "text_length": len(text) if text else 0
        }
        
    except Exception as e:
        logger.error("document_processing_error", doc_id=doc_id, error=str(e), exc_info=True)
        
        # Update document status to failed
        try:
            document = DatabaseService.get_document(self.session, doc_id)
            if document:
                document.status = "failed"
                self.session.commit()
        except Exception as db_error:
            logger.error("failed_to_update_document_status", doc_id=doc_id, error=str(db_error))
        
        # Re-raise to mark Celery task as failed
        raise
