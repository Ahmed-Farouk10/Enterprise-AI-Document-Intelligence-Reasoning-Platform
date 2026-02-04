from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.retreival import vector_store
from app.services.llm import llm_service
import logging
import json
from typing import Optional, List

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    context_text: str = None
    question: str
    doc_id: str = None
    doc_name: str = None
    use_query_rewriting: bool = True

class MultiDocRequest(BaseModel):
    documents: List[dict]  # [{"text": "...", "name": "..."}]
    question: str

@router.post("/")
async def chat_with_document(request: ChatRequest):
    """
    Enhanced Self-RAG Endpoint with:
    - Multi-document support
    - Citation tracking
    - Query rewriting
    - Confidence scoring
    """
    try:
        # Index new document if provided
        if request.context_text:
            doc_id = vector_store.add_document(
                request.context_text,
                doc_id=request.doc_id,
                doc_name=request.doc_name
            )
            logger.info(f"Indexed document: {doc_id}")

        if not request.question:
            raise HTTPException(400, "Question is required.")

        # ENHANCEMENT 1: Query Rewriting
        original_question = request.question
        search_query = request.question
        
        if request.use_query_rewriting:
            search_query = llm_service.rewrite_query(request.question)
        
        # Step 1: RETRIEVE with Citations
        raw_results = vector_store.retrieve_with_citations(search_query, k=5)
        
        if not raw_results:
            return {
                "status": "success",
                "answer": "I don't have enough context to answer that question.",
                "confidence": "low",
                "workflow": "no_retrieval",
                "model": "flan-t5-self-rag-enhanced"
            }

        # Step 2: SELF-REFLECT - Relevance Grading with Citation Tracking
        logger.info(f"Grading {len(raw_results)} retrieved chunks for relevance...")
        relevant_results = []
        filtered_sources = []
        
        for result in raw_results:
            if llm_service.grade_relevance(result["text"], request.question):
                relevant_results.append(result)
                logger.info(f"✓ Chunk from '{result['doc_name']}' deemed relevant (score: {result['similarity_score']:.3f})")
            else:
                filtered_sources.append(result["doc_name"])
                logger.info(f"✗ Chunk filtered out (irrelevant)")
        
        # Fallback if everything was filtered out
        if not relevant_results:
            return {
                "status": "success",
                "answer": "I couldn't find relevant information in the document(s) for that specific question.",
                "confidence": "low",
                "workflow": "all_chunks_filtered",
                "retrieved_chunks": len(raw_results),
                "relevant_chunks": 0,
                "documents_searched": len(set(r["doc_name"] for r in raw_results)),
                "model": "flan-t5-self-rag-enhanced"
            }

        # Step 3: GENERATE - Create answer from relevant chunks
        context_block = " ".join([r["text"] for r in relevant_results])
        logger.info("Generating answer from relevant chunks...")
        answer = llm_service.generate_answer(context_block, request.question)

        # Step 4: CRITIQUE - Hallucination Check
        logger.info("Checking answer for factual support...")
        is_supported = llm_service.grade_hallucination(context_block, answer)
        
        # Step 5: RESPOND with Citations
        final_answer = answer
        confidence = "high"
        
        if not is_supported:
            final_answer = f"(Uncertain) {answer}\n\n[Note: This answer might not be fully supported by the document text.]"
            confidence = "medium"
            logger.warning("Answer flagged as potentially unsupported")
        else:
            logger.info("✓ Answer verified as factually supported")

        # ENHANCEMENT 2: Citation Information
        citations = [
            {
                "doc_name": r["doc_name"],
                "doc_id": r["doc_id"],
                "chunk_text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "similarity_score": round(r["similarity_score"], 3)
            }
            for r in relevant_results
        ]

        return {
            "status": "success",
            "question": original_question,
            "rewritten_query": search_query if search_query != original_question else None,
            "answer": final_answer,
            "confidence": confidence,
            "citations": citations,
            "workflow": "self_rag_complete",
            "retrieved_chunks": len(raw_results),
            "relevant_chunks": len(relevant_results),
            "documents_searched": len(set(r["doc_name"] for r in raw_results)),
            "is_supported": is_supported,
            "model": "flan-t5-self-rag-enhanced"
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multi-doc")
async def chat_multi_document(request: MultiDocRequest):
    """
    Upload and query multiple documents at once
    """
    try:
        # Clear and index all documents
        vector_store.clear()
        doc_ids = []
        
        for idx, doc in enumerate(request.documents):
            doc_id = vector_store.add_document(
                text=doc.get("text", ""),
                doc_name=doc.get("name", f"Document {idx+1}")
            )
            doc_ids.append(doc_id)
        
        logger.info(f"Indexed {len(doc_ids)} documents")
        
        # Use standard chat flow
        chat_req = ChatRequest(question=request.question)
        return await chat_with_document(chat_req)
        
    except Exception as e:
        logger.error(f"Multi-doc chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def list_documents():
    """Get list of all indexed documents"""
    return {
        "status": "success",
        "documents": vector_store.get_document_list()
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "google/flan-t5-base",
        "retrieval": "FAISS + all-MiniLM-L6-v2",
        "workflow": "self-rag-enhanced (query rewriting + citations)",
        "features": [
            "Multi-document RAG",
            "Citation tracking",
            "Query rewriting",
            "Relevance grading",
            "Hallucination detection"
        ]
    }