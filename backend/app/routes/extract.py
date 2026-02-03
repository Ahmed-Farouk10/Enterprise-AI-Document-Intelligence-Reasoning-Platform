from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.layout import layout_parser
from app.services.entity_extractor import entity_extractor
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/layout")
async def extract_layout(request: TextRequest):
    """
    D2: Extract layout structure (heuristic-based).
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
async def extract_entities(request: TextRequest):
    """
    D3: Entity extraction from receipts (SROIE pattern-based).
    Extracts: total_amount, tax_amount, date, company_name, items
    """
    try:
        result = entity_extractor.extract(request.text)
        
        return {
            "status": "success",
            "extraction": result,
            "summary": {
                "total_found": result['entities']['total_amount']['value'] is not None,
                "date_found": result['entities']['date']['value'] is not None,
                "company_found": result['entities']['company_name']['value'] is not None,
                "items_found": len(result['entities']['items'])
            }
        }
    
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "d2_model": "heuristic-layout-parser",
        "d3_model": "deterministic-regex-extractor",
        "capabilities": ["total_amount", "tax_amount", "date", "company_name", "line_items"]
    }