from fastapi import APIRouter, HTTPException
from typing import List
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import (
    ChatSessionCreate, 
    ChatSessionResponse, 
    ChatMessageCreate, 
    ChatMessageResponse
)
from app.database import Database

# Import existing services
try:
    from app.services.llm import llm_service
    from app.services.retreival import vector_store
    LLM_AVAILABLE = True
except Exception as e:
    logging.warning(f"LLM services not available: {e}")
    llm_service = None
    vector_store = None
    LLM_AVAILABLE = False

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

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
async def get_sessions():
    """Get all chat sessions"""
    sessions = Database.get_all_sessions()
    
    # Add messages to each session
    for session in sessions:
        session["messages"] = []  # Don't load all messages in list view
    
    return sessions

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(session_data: ChatSessionCreate):
    """Create a new chat session"""
    session = Database.create_chat_session(
        title=session_data.title,
        document_ids=session_data.document_ids
    )
    session["messages"] = []
    return session

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str):
    """Get a specific chat session with all messages"""
    session = Database.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    success = Database.delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"success": True, "message": "Session deleted successfully"}

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(session_id: str, message_data: ChatMessageCreate):
    """Send a message in a chat session with RAG"""
    
    # Check if session exists
    session = Database.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Create user message
    user_message = Database.create_message(
        session_id=session_id,
        role="user",
        content=message_data.content
    )
    
    # Generate AI response with RAG
    try:
        if LLM_AVAILABLE and len(vector_store.chunks) > 0:
            logger.info(f"Using RAG pipeline for question: {message_data.content}")
            
            # Step 1: Query rewriting (optional)
            rewritten_query = llm_service.rewrite_query(message_data.content)
            
            # Step 2: Hybrid retrieval with reranking
            retrieved_chunks = vector_store.retrieve_with_citations(
                rewritten_query, 
                k=3,
                use_hybrid=True,
                use_reranking=True
            )
            
            if retrieved_chunks:
                # Step 3: Grade relevance (Self-RAG)
                relevant_chunks = []
                for chunk in retrieved_chunks:
                    is_relevant = llm_service.grade_relevance(chunk["text"], message_data.content)
                    if is_relevant:
                        relevant_chunks.append(chunk)
                    else:
                        logger.info(f"Filtered out irrelevant chunk from {chunk['doc_name']}")
                
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
                        logger.error(f"Generation failed after retries: {e}")
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
                ai_response_content = "No documents found. Please upload documents first."
                document_context = None
        else:
            # Fallback when no documents or model not available
            if not LLM_AVAILABLE:
                ai_response_content = "⚠️ AI model is initializing. Please wait 10-20 seconds and try again."
            else:
                ai_response_content = "📄 No documents uploaded yet. Please upload documents to enable Q&A."
            document_context = None
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
        ai_response_content = f"⚠️ Error processing your question. This might be a model cold start - please try again in 15 seconds.\n\nTechnical details: {str(e)}"
        document_context = None
    
    ai_message = Database.create_message(
        session_id=session_id,
        role="assistant",
        content=ai_response_content,
        document_context=document_context
    )
    
    # Return the AI message (frontend will optimistically add user message)
    return ai_message

@router.get("/health")
async def chat_health():
    """Check chat service health"""
    return {
        "status": "online",
        "llm_available": LLM_AVAILABLE,
        "vector_store_chunks": len(vector_store.chunks) if vector_store else 0,
        "documents_indexed": len(vector_store.documents) if vector_store else 0
    }
