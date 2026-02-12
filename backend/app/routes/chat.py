# app/api/chat.py
import asyncio
import logging
from app.core.logging_config import get_logger
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.services.llm_service import LLMService, AnalysisConfig, llm_service
from app.db.database import get_db, SessionLocal
from app.db.service import DatabaseService
from app.services.cache import cache_service
from app.core.rate_limiter import limiter
from app.services.retreival import vector_store
from app.services.knowledge_graph import kg_service
from app.services.cognee_engine import cognee_engine, AnalysisMode, GraphQueryResult
from app.services.verification_service import get_verification_service
import json
from app.core.session_manager import session_manager
from app.core.conversational_memory import conversational_memory
from app.core.query_cache import query_cache

logger = get_logger(__name__)
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


# --- Cognee Enhanced Schemas ---

class CogneeQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    analysis_mode: str = Field(default="auto", description="entities, temporal, comparative, anomalies, summary, or auto")
    include_graph_viz: bool = Field(default=False)
    session_id: Optional[str] = None
    document_id: Optional[str] = None


class CogneeResponse(BaseModel):
    answer: str
    confidence: float
    evidence_paths: List[List[Dict]]
    entities: List[Dict]
    analysis_mode: str
    graph_stats: Optional[Dict] = None
    processing_time_ms: int


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
    
    # 3. Classify and configure analysis (MOVED UP)
    intent = llm.classify_intent(message_data.content)
    depth = llm.classify_depth(message_data.content)
    scope = llm.detect_scope(message_data.content)
    
    # 4. Retrieve document context (Cognee Graph Reasoning)
    retrieval_data = await _get_retrieved_context(
        message_data.content, 
        depth=depth,
        document_ids=session.document_ids or []
    )
    document_text = retrieval_data["full_context"]
    
    # 5. SAFETY GATE: No document = No generation
    if not document_text or len(document_text.strip()) < 50:
        return _create_error_response(
            session_id=session_id,
            content="⚠️ No relevant document context found. Please ensure you have uploaded a document.",
            db=db
        )

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
    
    # --- SESSION MANAGER INTEGRATION ---
    from app.core.session_manager import session_manager
    
    # Get/Create Session Context
    # We use the session_id from the URL as the consistent ID
    sm_session = session_manager.get_or_create_session(user_id="default_user", session_id=session_id)
    history = sm_session.get("context", [])
    
    # Save user message to context manager
    session_manager.update_session_context(
        session_id=session_id,
        message={"role": "user", "content": message_data.content},
        document_ids=session.document_ids
    )

    config = AnalysisConfig(
        intent=intent,
        depth=depth,
        scope=scope,
        require_citations=True
    )
    
    scoped_context = llm.extract_scope_context(document_text, scope)
    system_prompt = llm.build_system_prompt(config)
    
    # Inject History into System Prompt
    if history:
        history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]])
        system_prompt += f"\n\nCONVERSATION HISTORY:\n{history_text}\n"

    async def event_generator():
        full_response = ""
        document_context = {"scope": scope, "intent": intent}
        
        try:
            # 0. SEMANTIC CACHE CHECK
            # We check cache immediately to save compute
            cached_result = await query_cache.get_semantic(message_data.content)
            if cached_result:
                yield _sse_event("status", "⚡ Semantic Cache Hit (Efficiency: 100%)")
                yield _sse_event("reasoning", f"Found similar query (Similarity: {cached_result.get('similarity', 0):.2f})")
                
                # Stream cached content as if it were being generated (fast replay)
                cached_content = cached_result['response']
                chunk_size = 10
                for i in range(0, len(cached_content), chunk_size):
                    yield _sse_event("token", cached_content[i:i+chunk_size])
                    await asyncio.sleep(0.01) # fast stream simulation
                
                yield _sse_event("done", cached_content, {"source": "semantic_cache"})
                return

            # 1. RETRIEVE MEMORIES & HISTORY
            # Get consistent history from Redis
            history_list = await session_manager.get_conversation_history(session_id, limit=6)
            
            # Get relevant long-term memories
            relevant_memories = await conversational_memory.retrieve_relevant_memories(
                user_id=sm_session.user_id,
                query=message_data.content
            )
            
            # Phase 1: Show ACTUAL retrieval method used
            retrieval_method = retrieval_data.get("retrieval_method", "unknown")
            
            if retrieval_method == "hybrid":
                yield _sse_event("status", f"🔍 Hybrid Mode: Vector Store + Graph | Depth: {depth}")
                yield _sse_event("reasoning", "🕸️ Combining vector search with knowledge graph...")
            elif retrieval_method == "cognee":
                yield _sse_event("status", f"🔍 Graph Mode: {intent} | Depth: {depth}")
                yield _sse_event("reasoning", "🕸️ Traversing knowledge graph for evidence...")
            else:  # vector_store or none
                yield _sse_event("status", f"🔍 Vector Search Mode: {intent} | Depth: {depth}")
                yield _sse_event("reasoning", "📚 Searching document embeddings for relevant context...")
            
            await asyncio.sleep(0.5) # UX for reasoning perception
            
            if retrieval_data.get("entities"):
                entities = [e["name"] for e in retrieval_data["entities"]]
                yield _sse_event("reasoning", f"✓ Found {len(entities)} relevant entities: {', '.join(entities[:3])}...")
            
            # Show memory usage if relevant
            if relevant_memories:
                 yield _sse_event("reasoning", f"🧠 Recalled {len(relevant_memories)} relevant facts from memory.")

            # Phase 2: Synthesis
            if retrieval_method == "cognee" or retrieval_method == "hybrid":
                yield _sse_event("status", "Synthesizing answer from graph connections...")
            else:
                yield _sse_event("status", "Synthesizing answer from retrieved documents...")
            
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
            
            # TIER 3: Post-generation verification (hallucination detection)
            yield _sse_event("status", "🔍 Verifying response against source context...")
            
            verification_service = get_verification_service()
            verification_report = verification_service.verify_response(
                response=full_response,
                context=document_text,  # Use full context, not just scoped
                include_evidence=False  # Don't send evidence to client (too verbose)
            )
            
            # Log verification results
            logger.info(
                f"📊 Verification: {verification_report['overall_score']}% | "
                f"Verified: {len(verification_report['verified_facts'])} | "
                f"Hallucinated: {len(verification_report['hallucinated_facts'])}"
            )
            
            # Add warning message if hallucinations detected
            final_response = full_response
            if verification_report['flagged']:
                warning = verification_service.format_warning_message(verification_report)
                if warning:
                    final_response = f"{full_response}\n\n{warning}"
                    # Send additional warning event
                    yield _sse_event("warning", warning)
            
            # Update document context with verification info
            document_context["verification"] = {
                "score": verification_report['overall_score'],
                "flagged": verification_report['flagged'],
                "hallucinated_count": len(verification_report['hallucinated_facts'])
            }
            
            # --- MEMORY & CACHE UPDATES ---
            # 1. Update Session Context (Redis)
            await session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_response
            )
            
            # 2. Update Semantic Cache (for future queries)
            await query_cache.set_semantic(
                query=message_data.content,
                response=final_response
            )
            
            # 3. Update Working Memory (Short-term context)
            current_context = history_list + [{"role": "user", "content": message_data.content}, {"role": "assistant", "content": final_response}]
            await conversational_memory.update_working_memory(
                session_id=session_id,
                user_id=sm_session.user_id,
                context_items=current_context
            )

            # Completion
            yield _sse_event("done", final_response, document_context)
            
            # Background save (SQL) - Legacy but kept for analytics
            def _save_background():
                db_session = SessionLocal()
                try:
                    DatabaseService.create_message(
                        db_session, session_id, "assistant", full_response, document_context
                    )
                finally:
                    db_session.close()

            asyncio.create_task(
                asyncio.to_thread(_save_background)
            )
            
        except Exception as e:
            # Structlog requires keyword arguments for context
            logger.error("stream_error", error=str(e), session_id=session_id)
            yield _sse_event("error", str(e))
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== UTILITY FUNCTIONS ====================

async def _get_retrieved_context(query: str, depth: str, document_ids: List[str] = []) -> Dict[str, Any]:
    """
    Retrieve context using the new HybridSearchService (Vector + Graph Fusion).
    """
    from app.services.hybrid_search_service import hybrid_search_service, SearchConfig
    
    # Configure search based on depth/intent
    config = SearchConfig()
    
    if depth == LLMService.DEPTH_EVALUATIVE:
        config.graph_weight = 0.5  # Higher graph weight for detailed evaluation
        config.hops = 2            # Deeper traversal
    elif depth == LLMService.DEPTH_IMPROVEMENT:
        config.alpha = 0.4         # More keyword focus to find specific errors
        config.graph_weight = 0.3
    
    try:
        # execute hybrid search
        result = await hybrid_search_service.search(
            query=query,
            document_ids=document_ids,
            config=config
        )
        
        return {
            "full_context": result["full_context"],
            "document_name": "Hybrid Knowledge Base",
            "confidence": result["confidence"],
            "entities": [{"name": e} for e in result["entities"]],
            "graph_evidence": [], # populated by graph expansion
            "retrieval_method": "hybrid_fusion"
        }
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        # Fallback to simple vector search if everything explodes
        return {
            "full_context": "Error retrieving context. Please try again.",
            "document_name": "Error",
            "confidence": 0.0,
            "entities": [],
            "graph_evidence": [],
            "retrieval_method": "error"
        }


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
        return " **Scoring Not Possible**\n\nThe document does not contain sufficient quantitative or measurable criteria to generate a meaningful score. Please upload a document with clear metrics, requirements, or evaluation criteria."
    
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


# --- Cognee Specialized Endpoints ---

@router.post("/query", response_model=CogneeResponse)
@limiter.limit("50/minute")
async def cognee_query(
    request: Request,
    query: CogneeQuery,
    db: Session = Depends(get_db)
):
    """
    Execute Cognee knowledge graph query.
    Primary analysis endpoint - replaces/augments standard RAG.
    """
    start_time = asyncio.get_event_loop().time()
    
    # Resolve document IDs
    doc_ids = []
    if query.session_id:
        session = DatabaseService.get_chat_session(db, query.session_id)
        if session:
            doc_ids = session.document_ids or []
    elif query.document_id:
        doc_ids = [query.document_id]
    
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No documents specified.")
    
    # Map analysis mode
    mode = _map_query_to_mode(query.question, query.analysis_mode)
    
    try:
        # Execute Cognee query
        result: GraphQueryResult = await cognee_engine.query(
            question=query.question,
            document_ids=doc_ids,
            mode=mode,
            include_subgraph=query.include_graph_viz
        )
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Save to history if session provided
        if query.session_id:
            DatabaseService.create_message(db, query.session_id, "user", query.question)
            DatabaseService.create_message(
                db, query.session_id, "assistant", result.answer,
                {"confidence": result.confidence_score, "mode": mode.value}
            )
        
        return CogneeResponse(
            answer=result.answer,
            confidence=result.confidence_score,
            evidence_paths=result.evidence_paths,
            entities=result.entities_involved,
            analysis_mode=mode.value,
            graph_stats=result.subgraph if query.include_graph_viz else None,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Cognee query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/gaps")
async def analyze_temporal_gaps(document_id: str):
    """Specialized temporal gap analysis using Cognee graph."""
    try:
        result = await cognee_engine.analyze_gaps(document_id)
        return {
            "document_id": document_id,
            "analysis": result.answer,
            "confidence": result.confidence_score,
            "evidence": result.evidence_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/compare")
async def compare_documents(source_id: str, target_id: str):
    """Compare two documents using graph alignment."""
    try:
        result = await cognee_engine.compare_to_standards(source_id, target_id)
        return {
            "source_id": source_id,
            "target_id": target_id,
            "alignment_score": result.confidence_score,
            "analysis": result.answer,
            "matched_entities": result.entities_involved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/trajectory")
async def get_career_trajectory(document_id: str):
    """Extract career progression from knowledge graph."""
    try:
        trajectory = await cognee_engine.extract_career_trajectory(document_id)
        return {"document_id": document_id, "trajectory": trajectory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/graph")
async def get_document_graph(document_id: str):
    """Export document knowledge graph for visualization."""
    try:
        # In a real scenario, this would return a subgraph
        return {"document_id": document_id, "nodes": [], "edges": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Utility Functions for Cognee ---

def _map_query_to_mode(question: str, requested_mode: str) -> AnalysisMode:
    """Intelligent mode selection based on question content."""
    if requested_mode != "auto":
        mode_map = {
            "entities": AnalysisMode.ENTITY_EXTRACTION,
            "temporal": AnalysisMode.TEMPORAL_REASONING,
            "comparative": AnalysisMode.COMPARATIVE_ANALYSIS,
            "anomalies": AnalysisMode.ANOMALY_DETECTION,
            "summary": AnalysisMode.SUMMARIZATION,
            "relationships": AnalysisMode.RELATIONSHIP_MAPPING
        }
        return mode_map.get(requested_mode, AnalysisMode.ENTITY_EXTRACTION)
    
    q = question.lower()
    if any(k in q for k in ["gap", "break", "between", "when", "timeline"]):
        return AnalysisMode.TEMPORAL_REASONING
    if any(k in q for k in ["compare", "fit", "match", "against"]):
        return AnalysisMode.COMPARATIVE_ANALYSIS
    if any(k in q for k in ["missing", "incomplete", "wrong", "error", "anomaly"]):
        return AnalysisMode.ANOMALY_DETECTION
    return AnalysisMode.SUMMARIZATION


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