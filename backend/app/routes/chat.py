from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.retreival import vector_store
from app.services.llm import llm_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    context_text: str = None
    question: str
    doc_id: str = None

@router.post("/")
async def chat_with_document(request: ChatRequest):
    """
    Self-RAG Endpoint:
    1. RETRIEVE: Get top chunks from vector store
    2. SELF-REFLECT: Grade chunk relevance to question
    3. GENERATE: Answer using valid chunks only
    4. CRITIQUE: Verify answer is factually supported
    5. RESPOND: Return answer with confidence score
    """
    try:
        # Index new document if provided
        if request.context_text:
            vector_store.clear()
            vector_store.add_document(request.context_text)

        if not request.question:
            raise HTTPException(400, "Question is required.")

        # Step 1: RETRIEVE - Get top chunks
        raw_chunks = vector_store.retrieve(request.question, k=3)
        
        if not raw_chunks:
            return {
                "status": "success",
                "answer": "I don't have enough context to answer that question.",
                "confidence": "low",
                "workflow": "no_retrieval",
                "model": "flan-t5-self-rag-lite"
            }

        # Step 2: SELF-REFLECT - Relevance Grading
        logger.info(f"Grading {len(raw_chunks)} retrieved chunks for relevance...")
        relevant_chunks = []
        for chunk in raw_chunks:
            if llm_service.grade_relevance(chunk, request.question):
                relevant_chunks.append(chunk)
                logger.info(f"✓ Chunk deemed relevant")
            else:
                logger.info(f"✗ Chunk filtered out (irrelevant)")
        
        # Fallback if everything was filtered out
        if not relevant_chunks:
            return {
                "status": "success",
                "answer": "I couldn't find relevant information in the document for that specific question.",
                "confidence": "low",
                "workflow": "all_chunks_filtered",
                "retrieved_chunks": len(raw_chunks),
                "relevant_chunks": 0,
                "model": "flan-t5-self-rag-lite"
            }

        # Step 3: GENERATE - Create answer from relevant chunks
        context_block = " ".join(relevant_chunks)
        logger.info("Generating answer from relevant chunks...")
        answer = llm_service.generate_answer(context_block, request.question)

        # Step 4: CRITIQUE - Hallucination Check
        logger.info("Checking answer for factual support...")
        is_supported = llm_service.grade_hallucination(context_block, answer)
        
        # Step 5: RESPOND - Format final answer with confidence
        final_answer = answer
        confidence = "high"
        
        if not is_supported:
            final_answer = f"(Uncertain) {answer}\n\n[Note: This answer might not be fully supported by the document text.]"
            confidence = "medium"
            logger.warning("Answer flagged as potentially unsupported")
        else:
            logger.info("✓ Answer verified as factually supported")

        return {
            "status": "success",
            "question": request.question,
            "answer": final_answer,
            "confidence": confidence,
            "retrieved_context": relevant_chunks,
            "workflow": "self_rag_complete",
            "retrieved_chunks": len(raw_chunks),
            "relevant_chunks": len(relevant_chunks),
            "is_supported": is_supported,
            "model": "flan-t5-self-rag-lite"
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "google/flan-t5-base",
        "retrieval": "FAISS + all-MiniLM-L6-v2",
        "workflow": "self-rag (reflect + critique)"
    }