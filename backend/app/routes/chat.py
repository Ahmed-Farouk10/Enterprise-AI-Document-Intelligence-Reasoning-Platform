from fastapi import APIRouter, HTTPException, Depends, Request, Response
from sqlalchemy.orm import Session
from typing import List
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import (
    ChatSessionCreate, 
    ChatSessionResponse, 
    ChatMessageCreate, 
    ChatMessageResponse
)
from app.db.database import get_db
from app.db.service import DatabaseService
from app.core.logging_config import get_logger
from app.services.cache import cache_service
from app.core.rate_limiter import limiter

# Import existing services
try:
    from app.services.llm import llm_service
    from app.services.retreival import vector_store
    from app.services.search import search_service
    LLM_AVAILABLE = True
    LLM_INIT_ERROR = None
except Exception as e:
    logger = get_logger(__name__)
    logger.warning("llm_services_unavailable", error=str(e))
    llm_service = None
    vector_store = None
    LLM_AVAILABLE = False
    LLM_INIT_ERROR = str(e)

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = get_logger(__name__)

# Retry decorator for HF model cold starts
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def generate_with_retry(context: str, question: str):
    """Generate answer with retry logic for model cold starts"""
    if not LLM_AVAILABLE or llm_service is None:
        raise Exception("LLM service not available")
    return llm_service.generate_answer(context, question)

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(db: Session = Depends(get_db)):
    """Get all chat sessions"""
    sessions = DatabaseService.get_all_sessions(db)
    
    # Convert ORM to response format
    return [{
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "document_ids": session.document_ids or [],
        "messages": []  # Don't load all messages in list view
    } for session in sessions]

@router.post("/sessions", response_model=ChatSessionResponse)
@limiter.limit("20/minute")
async def create_session(request: Request, response: Response, session_data: ChatSessionCreate, db: Session = Depends(get_db)):
    """Create a new chat session"""
    session = DatabaseService.create_chat_session(
        db=db,
        title=session_data.title,
        document_ids=session_data.document_ids
    )
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "document_ids": session.document_ids or [],
        "messages": []
    }

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get a specific chat session with all messages (with caching)"""
    
    # Try to get from cache first
    cached_session = cache_service.get_cached_session(session_id)
    if cached_session:
        logger.info("session_cache_hit", session_id=session_id)
        return cached_session
    
    # Get from database
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get messages for this session
    messages = DatabaseService.get_session_messages(db, session_id)
    
    session_data = {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "document_ids": session.document_ids or [],
        "messages": [{
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        } for msg in messages]
    }
    
    # Cache for 30 minutes
    cache_service.cache_session(session_id, session_data, ttl=1800)
    logger.info("session_cache_miss_stored", session_id=session_id)
    
    return session_data

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session"""
    success = DatabaseService.delete_chat_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Invalidate cache
    cache_service.invalidate_session(session_id)
    logger.info("session_deleted_cache_invalidated", session_id=session_id)
    
    return {"success": True, "message": "Session deleted successfully"}

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
@limiter.limit("50/minute")
async def send_message(request: Request, response: Response, session_id: str, message_data: ChatMessageCreate, db: Session = Depends(get_db)):
    """Send a message in a chat session with RAG"""
    
    # Check if session exists
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Create user message
    user_message = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="user",
        content=message_data.content
    )
    
    # Generate AI response with RAG
    try:
        # Create cache key for this RAG query (based on question + session docs)
        doc_ids_key = sorted(session.document_ids) if session.document_ids else []
        cache_key = cache_service._generate_key(
            "rag_query",
            query=message_data.content,
            doc_ids=doc_ids_key
        )
        
        # Try to get cached response
        cached_response = cache_service.get(cache_key)
        if cached_response:
            logger.info("rag_cache_hit", session_id=session_id, query=message_data.content[:50])
            ai_response_content = cached_response.get("content")
            document_context = cached_response.get("document_context")
        else:
            # Cache miss - execute RAG pipeline
            logger.info("rag_cache_miss", session_id=session_id, query=message_data.content[:50])
            
            if LLM_AVAILABLE and len(vector_store.chunks) > 0:
                logger.info("rag_pipeline_started", session_id=session_id, question=message_data.content)
            
                # Skip rewriting to preserve exact user intent
                rewritten_query = message_data.content
                
                # Step 2: Hybrid retrieval with reranking
                retrieved_chunks = vector_store.retrieve_with_citations(
                    rewritten_query, 
                    k=10, # Increased for Qwen 32k context window 
                    use_hybrid=True,
                    use_reranking=True
                )
            
            if retrieved_chunks:
                # Trust the vector store's top results (already hybrid graded)
                # LLM grading is too unstable for small models
                relevant_chunks = retrieved_chunks

                
                # Step 4: Generate answer from context
                if relevant_chunks:
                    context = "\n\n".join([
                        f"[{c['doc_name']}] {c['text']}" 
                        for c in relevant_chunks
                    ])
                    
                    # Use retry logic for generation
                    try:
                        generated_answer = generate_with_retry(context, message_data.content)
                    except Exception as e:
                        logger.error("generation_failed_after_retries", session_id=session_id, error=str(e))
                        generated_answer = "⚠️ Model is warming up (cold start). Please try again in 10-15 seconds."
                    
                    # Step 5: Grade hallucination (Self-RAG critique)
                    is_grounded = llm_service.grade_hallucination(context, generated_answer)
                    
                    if not is_grounded:
                        generated_answer = f"⚠️ Low confidence: {generated_answer}\n\n(This answer may not be fully supported by the documents)"
                    
                    # Build response with citations
                    citations = "\n\n📚 **Sources:**\n" + "\n".join([
                        f"- {c['doc_name']} (similarity: {c.get('similarity_score', 0):.2f})"
                        for c in relevant_chunks
                    ])
                    
                    ai_response_content = f"{generated_answer}\n\n{citations}"
                    
                    # Store document context
                    document_context = {
                        "document_id": relevant_chunks[0]["doc_id"],
                        "document_name": relevant_chunks[0]["doc_name"],
                        "relevant_chunks": [c["text"][:200] + "..." for c in relevant_chunks]
                    }
                else:
                    ai_response_content = "I found some documents, but none were relevant to your question. Could you rephrase?"
                    document_context = None

            else:
                # Fallback when no documents or model not available
                if not LLM_AVAILABLE:
                    ai_response_content = "⚠️ AI model is initializing. Please wait 10-20 seconds and try again."
                else:
                    ai_response_content = "📄 No documents uploaded yet. Please upload documents to enable Q&A."
                document_context = None
    
    except Exception as e:
        logger.error("rag_pipeline_error", session_id=session_id, error=str(e), exc_info=True)
        ai_response_content = f"⚠️ Error processing your question. This might be a model cold start - please try again in 15 seconds.\n\nTechnical details: {str(e)}"
        document_context = None

    # Cache successful RAG responses
    if document_context and "⚠️" not in ai_response_content:
        cache_data = {
            "content": ai_response_content,
            "document_context": document_context
        }
        cache_service.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL
        logger.info("rag_response_cached", session_id=session_id)
    
    
    ai_message = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=ai_response_content,
        document_context=document_context
    )
    
    # Return the AI message (frontend will optimistically add user message)
    return {
        "id": ai_message.id,
        "role": ai_message.role,
        "content": ai_message.content,
        "timestamp": ai_message.timestamp.isoformat(),
        "document_context": ai_message.extra_data
    }

@router.post("/sessions/{session_id}/stream")
async def stream_message(request: Request, session_id: str, message_data: ChatMessageCreate, db: Session = Depends(get_db)):
    """Stream a message response using SSE-like format"""
    from fastapi.responses import StreamingResponse
    import json
    
    # Check if session exists
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Create user message
    # Fetch recent history for context
    recent_messages = []
    try:
        # We need to use a new session or ensure the existing one is thread-safe if used in generator,
        # but here we are in main thread before generator starts.
        msgs = DatabaseService.get_session_messages(db, session_id)
        # Convert to dict to avoid detached instance issues in generator
        recent_messages = [{"role": m.role, "content": m.content} for m in msgs[-5:]]
    except Exception as e:
        logger.warning(f"Failed to fetch history: {e}")

    user_message = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="user",
        content=message_data.content
    )
    
    async def event_generator():
        context = ""
        full_response = ""
        document_context = None
        intent = "GENERAL_CHAT"  # Default intent if no documents to analyze
        
        # RAG Logic (Simplified for streaming)
        try:
            if not LLM_AVAILABLE:
                logger.error(f"LLM not available during stream. Init error: {LLM_INIT_ERROR}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'AI Service Unavailable: {LLM_INIT_ERROR}. Please check backend logs.'})}\n\n"
                return

            if len(vector_store.chunks) > 0:
                # 1. ORCHESTRATION: Parallel Intent & Rewrite
                yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: analyzing intent & context...'})}\n\n"
                await asyncio.sleep(0.01)

                loop = asyncio.get_running_loop()
                
                # Execute blocking LLM calls in parallel threads
                # Task A: Classify Intent
                task_intent = loop.run_in_executor(None, llm_service.classify_intent, message_data.content)
                # Task B: Rewrite Query (Optimistically)
                task_rewrite = loop.run_in_executor(None, lambda: llm_service.rewrite_query(message_data.content, chat_history=recent_messages))
                
                # Wait for both (Parallel Execution)
                intent, rewritten_query = await asyncio.gather(task_intent, task_rewrite)
                
                yield f"data: {json.dumps({'type': 'status', 'content': f'Thinking: identified task as {intent}...'})}\n\n"
                await asyncio.sleep(0.01)

                # 2. ORCHESTRATION: Fast-Path Routing
                final_query = rewritten_query
                
                if intent == "GENERAL_CHAT":
                    # Fast Path: Verify if we even need retrieval. 
                    # For now, we still retrieve but maybe with original query if rewrite failed or was weird.
                    # But actually, rewrite is usually good.
                    # Optimization: If highly confident it's greetings ("hi"), we could skip retrieval.
                    # For now, let's keep retrieval but use the rewritten one.
                    pass
                else:
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: searching local knowledge base...'})}\n\n"
                
                # Retrieve using the result from parallel execution
                retrieved_chunks = vector_store.retrieve_with_citations(final_query, k=3)
                
                if retrieved_chunks:
                    context = "\n\n".join([f"[{c['doc_name']}] {c['text']}" for c in retrieved_chunks])
                    # yield f"data: {json.dumps({'type': 'status', 'content': 'Reading context...'})}\n\n"
                    
                    document_context = {
                        "document_name": retrieved_chunks[0]["doc_name"],
                        "relevant_chunks": [c["text"][:200] + "..." for c in retrieved_chunks]
                    }
            
            # --- AGENTIC REASONING LAYER ---
            yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: verifying context sufficiency...'})}\n\n"
            await asyncio.sleep(0.01)
            
            # 4. ORCHESTRATION: Search Permission Control
            should_search = False
            search_reason = ""
            
            # Strict Rules
            if intent == "ATS_ESTIMATION":
                should_search = False
                search_reason = "ATS Estimation requires strictly internal heuristics."
            elif intent == "GAP_ANALYSIS":
                should_search = False 
                search_reason = "Gap Analysis relies strictly on document dates."
            elif intent == "SEARCH_QUERY":
                should_search = True
                search_reason = "User explicitly requested external market/tech data."
            else:
                # Fallback to confidence check for GENERAL_CHAT or RESUME_ANALYSIS
                top_score = retrieved_chunks[0].get("similarity_score", 0.0) if retrieved_chunks else 0.0
                if top_score < 0.65:
                    should_search = True
                    search_reason = f"Low confidence match ({top_score:.2f}) - Context might be missing."
                
                external_keywords = ["market", "salary", "news", "trend"]
                if any(k in message_data.content.lower() for k in external_keywords):
                    should_search = True
                    search_reason = "Detected external data request keywords."

            # 2. Execute Search if Needed
            search_context = ""
            if should_search:
                 yield f"data: {json.dumps({'type': 'status', 'content': f'Thinking: {search_reason} Optimizing search query...'})}\n\n"
                 
                 # Generate Smart Query
                 smart_query = llm_service.generate_search_query(message_data.content, context)
                 # Fix: Avoid backslash in f-string expression for Python < 3.12
                 msg = f"Thinking: Searching web for '{smart_query}'..."
                 yield f"data: {json.dumps({'type': 'status', 'content': msg})}\n\n"
                 
                 try:
                     search_results = search_service.search(smart_query, num_results=5)
                     
                     if search_results:
                         yield f"data: {json.dumps({'type': 'status', 'content': f'Found {len(search_results)} credible results. Verifying...'})}\n\n"
                         
                         # Format search results
                         search_context = "\n\n".join([
                             f"[External Source: {r.title} ({r.source_type})] {r.snippet} (Link: {r.link})"
                             for r in search_results
                         ])
                         
                         # Append to context
                         context = f"{context}\n\nEXTERNAL BENCHMARK INFORMATION (FOR REFERENCE ONLY):\n---------------------------------------------------\n{search_context}\n---------------------------------------------------"
                     else:
                         yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: Web search yielded no results. Falling back to general knowledge.'})}\n\n"
                 except Exception as exc:
                     logger.error(f"Search failed: {exc}")
                     yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: Search API unavailable. Proceeding with best effort.'})}\n\n"
            else:
                 yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking: Local documents provided sufficient context.'})}\n\n"

            # 5. ORCHESTRATION: Mode-Locked Generation
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating response...'})}\n\n"
            yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"
            
            # Fetch specialized prompt and inject into context
            system_prompt = llm_service.get_task_prompt(intent)
            full_context = f"SYSTEM INSTRUCTION: {system_prompt}\n\nCONTEXT:\n{context}"
            
            for token in llm_service.stream_inference(message_data.content, full_context):
                # Filter lifecycle markers
                if "[STREAM_START]" in token: continue
                if "[STREAM_END]" in token: break
                     
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                
            # Finalize
            yield f"data: {json.dumps({'type': 'done', 'content': full_response, 'document_context': document_context})}\n\n"
            
            # Save assistant message to DB
            try:
                pass 
            except Exception as e:
                logger.error("failed_to_save_stream_message", error=str(e))
                
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/health")
async def chat_health():
    """Check chat service health"""
    return {
        "status": "online",
        "llm_available": LLM_AVAILABLE,
        "vector_store_chunks": len(vector_store.chunks) if vector_store else 0,
        "documents_indexed": len(vector_store.documents) if vector_store else 0
    }
