from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.layout import layout_parser
from app.services.receipt_parser import receipt_parser
from app.services.resume_parser import resume_parser
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class ExtractRequest(BaseModel):
    text: str
    doc_type: str = "unknown"


@router.post("/layout")
async def extract_layout(request: ExtractRequest):
    """D2: Extract layout structure (heuristic-based)."""
    try:
        result = layout_parser.parse(request.text)
        return {
            "status": "success",
            "layout": result,
            "document_structure": {
                "has_headers": result["entity_counts"]["headers"] > 0,
                "has_questions": result["entity_counts"]["questions"] > 0,
                "has_answers": result["entity_counts"]["answers"] > 0,
                "is_form_like": result["is_form_like"]
            }
        }
    except Exception as e:
        logger.error(f"Layout extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def extract_entities(request: ExtractRequest):
    """
    D3: Smart Router - Detects Receipt vs Resume and uses specialized parsers
    """
    try:
        text_lower = request.text.lower()
        
        # 1. Detect Type
        is_resume = (request.doc_type == "resume") or \
                   ("education" in text_lower and "experience" in text_lower)
        
        # 2. Extract using specialized parser
        if is_resume:
            data = resume_parser.extract(request.text)
            model = "resume-heuristic-v1"
        else:
            data = receipt_parser.extract(request.text)
            model = "receipt-heuristic-v1"
            
        return {
            "status": "success",
            "entities": data,
            "model": model,
            "doc_type_detected": "resume" if is_resume else "receipt"
        }

    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "d2_model": "heuristic-layout-parser",
        "d3_model": "receipt-heuristic-v1 + resume-heuristic-v1"
    }