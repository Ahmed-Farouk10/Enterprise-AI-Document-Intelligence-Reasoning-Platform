from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.layout import layout_parser
import logging
import re

logger = logging.getLogger(__name__)
router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/layout")
async def extract_layout(request: TextRequest):
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
async def extract_entities(request: TextRequest):
    """
    D3: Simple entity extraction using regex (no external dependencies).
    """
    text = request.text
    
    # Simple regex extraction
    money_pattern = r'\$\s*\d[\d,]*\.?\d{0,2}'
    date_pattern = r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b'
    
    money_matches = re.findall(money_pattern, text)
    date_matches = re.findall(date_pattern, text)
    
    # Find largest amount as potential total
    total = None
    if money_matches:
        # Clean and find max
        amounts = []
        for m in money_matches:
            clean = re.sub(r'[^\d.]', '', m)
            try:
                amounts.append((m, float(clean)))
            except:
                pass
        if amounts:
            total = max(amounts, key=lambda x: x[1])
    
    return {
        "status": "success",
        "extraction": {
            "total_amount": {
                "value": total[0] if total else None,
                "numeric": total[1] if total else None
            },
            "dates": date_matches,
            "all_money_found": money_matches
        },
        "model": "simple-regex",
        "dataset": "SROIE-placeholder"
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "d2_model": "heuristic-layout-parser",
        "d3_model": "simple-regex-extractor"
    }