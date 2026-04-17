"""
Optimized Chat API Routes
Multi-document support with token efficiency and personality
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.db.database import get_db, SessionLocal
from app.db.service import DatabaseService
from app.services.llm_service import llm_service
from app.services.vector_store import vector_store_service
from app.services.cag_engine import cag_engine
from app.core.rate_limiter import limiter
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


# ==================== SCHEMAS ====================

class ChatSessionCreate(BaseModel):
    title: str = Field(default="New Analysis Session")
    document_ids: List[str] = Field(default_factory=list)


class ChatMessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


# ==================== SESSION MANAGEMENT ====================

@router.get("/sessions")
async def get_sessions(db: Session = Depends(get_db)):
    """Retrieve all chat sessions"""
    sessions = DatabaseService.get_all_sessions(db)
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
            "document_ids": s.document_ids or [],
            "message_count": len(s.messages) if hasattr(s, 'messages') else 0
        }
        for s in sessions
    ]


@router.post("/sessions")
@limiter.limit("20/minute")
async def create_session(
    request: Request,
    response: Response,
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Create new analysis session with multi-document support"""
    session = DatabaseService.create_chat_session(
        db=db,
        title=session_data.title,
        document_ids=session_data.document_ids
    )

    # Pre-compute CAG context for multi-document sets
    if session_data.document_ids and len(session_data.document_ids) > 1:
        asyncio.create_task(
            cag_engine.precompute_context(session_data.document_ids)
        )

    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "document_ids": session.document_ids or [],
        "messages": []
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get session with messages"""
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = DatabaseService.get_session_messages(db, session_id)

    return {
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


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete session"""
    success = DatabaseService.delete_chat_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "message": "Session deleted"}


class ChatSessionUpdate(BaseModel):
    document_ids: List[str]


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    update_data: ChatSessionUpdate,
    db: Session = Depends(get_db)
):
    """Update session metadata (like document selections)"""
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.document_ids = update_data.document_ids
    db.commit()
    db.refresh(session)
    
    # Trigger CAG pre-computation for the new set
    if len(update_data.document_ids) > 1:
        asyncio.create_task(
            cag_engine.precompute_context(update_data.document_ids)
        )
        
    return {
        "success": True,
        "document_ids": session.document_ids
    }


# ==================== MESSAGE PROCESSING ====================

@router.post("/sessions/{session_id}/messages")
@limiter.limit("30/minute")
async def send_message(
    request: Request,
    response: Response,
    session_id: str,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db)
):
    """
    Send message with multi-document context and optimized token usage.
    Non-streaming response.
    """
    # Validate session
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_msg = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="user",
        content=message_data.content
    )

    # Get conversation history
    history = DatabaseService.get_session_messages(db, session_id)
    conversation_history = [
        {"role": m.role, "content": m.content}
        for m in history[:-1]  # Exclude current user message
    ]

    # Classify intent (token-efficient)
    intent, depth = llm_service.classify_intent(message_data.content)

    # Retrieve multi-document context
    context = await _retrieve_context(
        message_data.content,
        session.document_ids or [],
        depth=depth
    )

    # Generate response with personality and optimization
    response = llm_service.generate_with_context(
        question=message_data.content,
        context=context,
        conversation_history=conversation_history[-6:] if conversation_history else None,
        intent=intent,
        num_documents=len(session.document_ids or []),
        max_context_chars=3000  # Token optimization
    )

    # Check for errors
    if response.startswith("Error:"):
        ai_content = response
    else:
        ai_content = response

    # Save AI response
    ai_msg = DatabaseService.create_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=ai_content,
        document_context={
            "intent": intent,
            "depth": depth,
            "num_documents": len(session.document_ids or []),
            "context_length": len(context)
        }
    )

    return {
        "id": ai_msg.id,
        "role": "assistant",
        "content": ai_content,
        "timestamp": ai_msg.timestamp.isoformat(),
        "metadata": {
            "intent": intent,
            "depth": depth,
            "num_documents": len(session.document_ids or []),
            "tokens_estimated": len(ai_content.split())
        }
    }


@router.post("/sessions/{session_id}/stream")
@limiter.limit("30/minute")
async def stream_message(
    request: Request,
    response: Response,
    session_id: str,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db)
):
    """
    Streaming endpoint with multi-document support.
    Real-time token streaming with personality.
    """
    session = DatabaseService.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    DatabaseService.create_message(db, session_id, "user", message_data.content)

    # Get conversation history
    history = DatabaseService.get_session_messages(db, session_id)
    conversation_history = [
        {"role": m.role, "content": m.content}
        for m in history[:-1]
    ]

    async def event_generator():
        full_response = ""

        try:
            # Check documents exist
            if not session.document_ids:
                yield _sse_event("status", "⚠️ No documents here yet! Upload some and I'll analyze them for you. 😊")
                yield _sse_event("error", "Please upload documents before asking questions")
                yield _sse_event("done", "")
                return

            yield _sse_event("status", f"🔍 Analyzing {len(session.document_ids)} document(s)...")

            # Classify intent
            intent, depth = llm_service.classify_intent(message_data.content)
            yield _sse_event("status", f"💡 Got it! Analyzing with {depth} focus...")

            # Retrieve context
            context = await _retrieve_context(
                message_data.content,
                session.document_ids or [],
                depth=depth
            )

            if not context or len(context.strip()) < 50:
                yield _sse_event("status", "⚠️ Hmm, the documents seem a bit empty. Let me know if you'd like to re-upload them!")
                yield _sse_event("done", "")
                return

            yield _sse_event("status", "✨ Great finds! Composing my thoughts...")

            # Generate with streaming
            stream_gen = llm_service.generate_with_context(
                question=message_data.content,
                context=context,
                conversation_history=conversation_history[-6:] if conversation_history else None,
                intent=intent,
                num_documents=len(session.document_ids or []),
                max_context_chars=3000,
                stream=True
            )

            # Stream tokens
            for token in stream_gen:
                full_response += token
                yield _sse_event("token", token)

            if not full_response:
                full_response = "Hmm, my brain went blank for a moment! Try asking again and I'll give you a better answer. 😅"

            # Save response in background
            def _save():
                db_session = SessionLocal()
                try:
                    DatabaseService.create_message(
                        db_session,
                        session_id,
                        "assistant",
                        full_response,
                        {"intent": intent, "context_length": len(context)}
                    )
                finally:
                    db_session.close()

            await asyncio.to_thread(_save)

            yield _sse_event("done", full_response, {
                "intent": intent,
                "num_documents": len(session.document_ids or []),
                "tokens_used": len(full_response.split())
            })

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield _sse_event("error", f"Oops! Something went wrong: {str(e)}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== HELPER FUNCTIONS ====================

async def _retrieve_context(
    query: str,
    document_ids: List[str],
    depth: str = "shallow"
) -> str:
    """Retrieve context from multiple documents with token optimization"""
    if not document_ids:
        return ""

    # Check CAG cache first (instant retrieval)
    if len(document_ids) > 1:
        cag_result = await cag_engine.get_cached_context(document_ids)
        if cag_result:
            return cag_result["full_context"]

    # Retrieve from vector store
    limit = 15 if depth == "deep" else 8
    all_chunks = []

    for doc_id in document_ids:
        chunks = await vector_store_service.search(
            query=query,
            limit=limit,
            document_id=doc_id
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return ""

    # Sort by relevance
    all_chunks.sort(key=lambda x: x.get("distance", 0))

    # Format context with document separators
    context_texts = []
    for chunk in all_chunks[:12]:  # Limit chunks for token efficiency
        context_texts.append(chunk["text"])

    full_context = "\n---\n".join(context_texts)

    # Cache for multi-document sets
    if len(document_ids) > 1:
        await cag_engine.cache_context(
            document_ids=document_ids,
            full_context=full_context,
            metadata={"query": query, "chunk_count": len(all_chunks)}
        )

    return full_context


def _sse_event(event_type: str, data: Any, metadata: Optional[Dict] = None) -> str:
    """Format SSE event"""
    import json
    payload = {"type": event_type, "data": str(data)}
    if metadata:
        payload["metadata"] = metadata
    return f"data: {json.dumps(payload)}\n\n"
