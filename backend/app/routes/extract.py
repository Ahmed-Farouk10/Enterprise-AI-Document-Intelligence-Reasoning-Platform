from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.layout import layout_parser
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ExtractRequest(BaseModel):
    text: str

@router.post("/layout")
async def extract_layout(request: ExtractRequest):
    """
    D2: Extract layout structure (heuristic-based).
    Identifies headers, questions, answers in forms.
    """
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
    D3: Entity extraction placeholder (SROIE - Phase 4)
    """
    return {
        "status": "placeholder",
        "message": "D3 entity extraction (SROIE) coming in Phase 4",
        "text_length": len(request.text),
        "dataset": "SROIE-pending"
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "d2_model": "heuristic-layout-parser",
        "d3_model": "SROIE-pending",
        "note": "LayoutLMv3 skipped (1.6GB too large for free tier)"
    }