from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request, Response, BackgroundTasks

# ... imports ...

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
                
                # Index in vector store
                if text and len(text.strip()) > 10:
                    from app.services.retreival import vector_store
                    vector_store.add_document(text=text, doc_id=document.id, doc_name=original_filename)
                    vector_store.save_index()
                    
                    # Graph Processing (Cognee)
                    try:
                        from app.services.cognee_engine import cognee_engine
                        logger.info(f" Processing graph for {document_id}")
                        
                        document_graph = await cognee_engine.ingest_document_professional(
                            document_text=text,
                            document_id=str(document.id),
                            document_type="auto_detect",
                            metadata={
                                "filename": original_filename,
                                "upload_date": document.created_at.isoformat(),
                                "mime_type": mime_type
                            }
                        )
                        
                        # Extract stats safely
                        is_success = False
                        if hasattr(document_graph, 'success') and document_graph.success: is_success = True
                        elif isinstance(document_graph, dict) and document_graph.get('success', False): is_success = True
                        
                        if is_success:
                           def get_val(obj, key, default):
                               if isinstance(obj, dict): return obj.get(key, default)
                               return getattr(obj, key, default)

                           document.extra_data = {
                                "graph_stats": {
                                    "entity_count": get_val(document_graph, "entity_count", 0),
                                    "document_type": get_val(document_graph, "document_type", "unknown"),
                                    "dataset_name": get_val(document_graph, "dataset_name", ""),
                                    "entities": get_val(document_graph, "entities", {})
                                }
                           }
                    
                    except Exception as cognee_error:
                         logger.error(f"⚠️ Cognee failed: {cognee_error}")
                         if not document.extra_data: document.extra_data = {}
                         document.extra_data["cognee_error"] = str(cognee_error)

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

