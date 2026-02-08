# app/api/chat.py
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.services.llm_service import LLMService, AnalysisConfig, llm_service
from app.db.database import get_db
from app.db.service import DatabaseService
from app.services.cache import cache_service
from app.core.rate_limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


# ==================== SCHEMAS ====================

class ChatSessionCreate(BaseModel):
    title: str = Field(default="New Analysis Session")
    document_ids: List[str] = Field(default_factory=list)


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    document_ids: List[str]
    messages: List[Dict]


class ChatMessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    document_context: Optional[Dict] = None
    metadata: Optional[Dict] = None


# ==================== DEPENDENCIES ====================

def get_llm_service() -> LLMService:
    """Dependency injection for LLM service"""
    return llm_service


# ==================== SESSION MANAGEMENT ====================

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(db: Session = Depends(get_db)):
    """Retrieve all chat sessions (lightweight list view)"""
    sessions = DatabaseService.get_all_sessions(db)
    
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
            "document_ids": s.document_ids or [],
            "messages": []  # Don't load messages in list view
        }
        for s in sessions
    ]


@router.post("/sessions", response_model=ChatSessionResponse)
@limiter.limit("20/minute")
async def create_session(
    request: Request,
    response: Response,
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Create new analysis session"""
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
async def get_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get specific session with messages (cached)"""
    
    # Cache check
    cached = cache_service.get_cached_session(session_id)
    if cached:
        logger.info("session_cache_hit", session_id=session_id)
        return cached
    
    # DB fetch
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = DatabaseService.get_session_messages(db, session_id)
    
    response_data = {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "document_ids": session.document_ids or [],
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat()
            }
            for m in messages
        ]
    }
    
    # Cache for 30 min
    cache_service.cache_session(session_id, response_data, ttl=1800)
    return response_data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete session and invalidate cache"""
    success = DatabaseService.delete_chat_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cache_service.invalidate_session(session_id)
    logger.info("session_deleted", session_id=session_id)
    
    return {"success": True, "message": "Session deleted"}


# ==================== CORE CHAT ENDPOINTS ====================

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
@limiter.limit("30/minute")
async def send_message(
    request: Request,
    response: Response,
    session_id: str,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Main chat endpoint - document analysis with RAG.
    Non-streaming for simple queries, handles complex analysis.
    """
    
    # 1. Validate session
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 2. Save user message
    user_msg = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="user",
        content=message_data.content
    )
    
    # 3. Retrieve document context (RAG)
    # TODO: Replace with actual vector store retrieval
    document_text = _get_session_documents(session)
    
    # 4. SAFETY GATE: No document = No generation
    if not document_text or len(document_text.strip()) < 100:
        return _create_error_response(
            session_id=session_id,
            content="⚠️ No document uploaded or document too short. Please upload a document to analyze.",
            db=db
        )
    
    # 5. Classify and configure analysis
    intent = llm.classify_intent(message_data.content)
    depth = llm.classify_depth(message_data.content)
    scope = llm.detect_scope(message_data.content)
    
    config = AnalysisConfig(
        intent=intent,
        depth=depth,
        scope=scope,
        allow_external_search=False,  # Default safe mode
        require_citations=True
    )
    
    # 6. Extract scoped context (Anti-hallucination filter)
    scoped_context = llm.extract_scope_context(document_text, scope)
    
    # 7. Build system prompt (Single Source of Truth)
    system_prompt = llm.build_system_prompt(config)
    
    # 8. Route by intent
    try:
        if intent == LLMService.INTENT_SCORING:
            # Deterministic scoring - no streaming
            answer = _handle_scoring_intent(llm, system_prompt, scoped_context, message_data.content)
        elif intent == LLMService.INTENT_GAP_ANALYSIS:
            # Structured gap analysis
            answer = _handle_gap_analysis(llm, system_prompt, scoped_context, message_data.content)
        else:
            # Standard generation
            answer = llm.generate(
                system_prompt=system_prompt,
                document_context=scoped_context,
                question=message_data.content,
                temperature=0.2
            )
            
        # 9. Save and return
        ai_msg = DatabaseService.create_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=answer,
            document_context={"scope": scope, "intent": intent}
        )
        
        return {
            "id": ai_msg.id,
            "role": "assistant",
            "content": answer,
            "timestamp": ai_msg.timestamp.isoformat(),
            "document_context": {"retrieved_scope": scope, "intent": intent, "depth": depth},
            "metadata": {"tokens_used": len(answer.split())}  # Approximate
        }
        
    except Exception as e:
        logger.error("generation_error", session_id=session_id, error=str(e))
        return _create_error_response(
            session_id=session_id,
            content=f"⚠️ Analysis failed: {str(e)}. Please try again.",
            db=db
        )


@router.post("/sessions/{session_id}/stream")
async def stream_message(
    request: Request,
    session_id: str,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Streaming endpoint for real-time analysis feedback.
    Used for: Long-form improvements, detailed evaluations.
    """
    
    # Validate session
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get document
    document_text = _get_session_documents(session)
    
    if not document_text:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': 'No document uploaded'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    # Save user message (non-blocking)
    asyncio.create_task(
        asyncio.to_thread(
            DatabaseService.create_message,
            db, session_id, "user", message_data.content
        )
    )
    
    # Configure analysis
    intent = llm.classify_intent(message_data.content)
    depth = llm.classify_depth(message_data.content)
    scope = llm.detect_scope(message_data.content)
    
    config = AnalysisConfig(
        intent=intent,
        depth=depth,
        scope=scope,
        require_citations=True
    )
    
    scoped_context = llm.extract_scope_context(document_text, scope)
    system_prompt = llm.build_system_prompt(config)
    
    async def event_generator():
        full_response = ""
        document_context = {"scope": scope, "intent": intent}
        
        try:
            # Status update
            yield _sse_event("status", f"Analyzing: {intent} | Scope: {', '.join(scope)}")
            
            # Stream generation
            async for token in _async_stream_wrapper(
                llm.generate_stream(system_prompt, scoped_context, message_data.content)
            ):
                if "[STREAM_START]" in token:
                    continue
                if "[STREAM_END]" in token:
                    break
                    
                full_response += token
                yield _sse_event("token", token)
            
            # Completion
            yield _sse_event("done", full_response, document_context)
            
            # Background save
            asyncio.create_task(
                asyncio.to_thread(
                    DatabaseService.create_message,
                    db, session_id, "assistant", full_response, document_context
                )
            )
            
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield _sse_event("error", str(e))
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== UTILITY FUNCTIONS ====================

def _get_session_documents(session) -> str:
    """
    Retrieve and concatenate all documents for a session.
    TODO: Integrate with actual vector store / document service
    """
    # Placeholder: In production, fetch from vector store using session.document_ids
    # For now, return empty to trigger safety gate if not implemented
    return ""  # Implement actual document retrieval


def _handle_scoring_intent(llm: LLMService, system_prompt: str, context: str, question: str) -> str:
    """
    Special handler for scoring intents - enforces deterministic calculation.
    """
    # First, check if document has measurable criteria
    pre_check = llm.generate(
        system_prompt="Does this document contain measurable/quantifiable criteria? Answer only YES or NO.",
        document_context=context,
        question="Can this be scored?",
        max_new_tokens=10,
        temperature=0.0
    )
    
    if "no" in pre_check.lower():
        return "❌ **Scoring Not Possible**\n\nThe document does not contain sufficient quantitative or measurable criteria to generate a meaningful score. Please upload a document with clear metrics, requirements, or evaluation criteria."
    
    # Proceed with structured scoring
    return llm.generate(
        system_prompt=system_prompt,
        document_context=context,
        question=question,
        temperature=0.1  # Low creativity for scoring
    )


def _handle_gap_analysis(llm: LLMService, system_prompt: str, context: str, question: str) -> str:
    """
    Special handler for gap analysis - temporal verification.
    """
    # Extract dates first (could use NLP library in production)
    date_extraction = llm.generate(
        system_prompt="Extract all dates and date ranges from this document. List them in format: [Date/Range] - [Context]",
        document_context=context,
        question="List all temporal information",
        max_new_tokens=512
    )
    
    # Then analyze gaps
    full_context = f"TEMPORAL DATA:\n{date_extraction}\n\nDOCUMENT:\n{context}"
    
    return llm.generate(
        system_prompt=system_prompt,
        document_context=full_context,
        question=question
    )


def _create_error_response(session_id: str, content: str, db: Session) -> Dict:
    """Helper for consistent error responses"""
    # Save error as assistant message for continuity
    try:
        msg = DatabaseService.create_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=content
        )
        return {
            "id": msg.id,
            "role": "assistant",
            "content": content,
            "timestamp": msg.timestamp.isoformat(),
            "document_context": None,
            "metadata": {"error": True}
        }
    except Exception:
        return {
            "id": "error",
            "role": "assistant",
            "content": content,
            "timestamp": "",
            "document_context": None,
            "metadata": {"error": True}
        }


def _sse_event(event_type: str, content: str, extra: Optional[Dict] = None) -> str:
    """Format Server-Sent Event"""
    import json
    data = {"type": event_type, "content": content}
    if extra:
        data.update(extra)
    return f"data: {json.dumps(data)}\n\n"


async def _async_stream_wrapper(sync_generator):
    """Convert sync generator to async for FastAPI streaming"""
    loop = asyncio.get_event_loop()
    iterator = iter(sync_generator)
    
    while True:
        try:
            token = await loop.run_in_executor(None, next, iterator)
            yield token
        except StopIteration:
            break
        except Exception as e:
            logger.error("Stream wrapper error", error=str(e))
            break


# ==================== HEALTH & DEBUG ====================

@router.get("/health")
async def chat_health():
    """Service health check"""
    return {
        "status": "online",
        "llm_status": llm_service.get_service_health(),
        "version": "2.0.0-enterprise"
    }


@router.post("/debug/analyze")
async def debug_analyze(
    document: str,
    question: str,
    llm: LLMService = Depends(get_llm_service)
):
    """
    Debug endpoint to see analysis configuration without saving.
    Useful for testing prompt engineering.
    """
    intent = llm.classify_intent(question)
    depth = llm.classify_depth(question)
    scope = llm.detect_scope(question)
    
    config = AnalysisConfig(intent=intent, depth=depth, scope=scope)
    system_prompt = llm.build_system_prompt(config)
    scoped_context = llm.extract_scope_context(document, scope)
    
    return {
        "config": {
            "intent": intent,
            "depth": depth,
            "scope": scope
        },
        "system_prompt": system_prompt,
        "context_length": len(scoped_context),
        "context_preview": scoped_context[:500] + "..." if len(scoped_context) > 500 else scoped_context
    }