from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.layout import layout_parser
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ExtractRequest(BaseModel):
    image_bytes: str  # Base64 encoded image (simplified)
    text: str = None  # Optional OCR text

@router.post("/layout")
async def extract_layout(image_bytes: bytes):
    """
    D2: Extract layout structure using FUNSD-trained LayoutLMv3.
    Identifies headers, questions, answers in forms.
    """
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Parse layout
        result = layout_parser.parse(image)
        
        return {
            "status": "success",
            "layout": result,
            "document_structure": {
                "has_headers": result["entity_counts"]["headers"] > 0,
                "has_questions": result["entity_counts"]["questions"] > 0,
                "has_answers": result["entity_counts"]["answers"] > 0,
                "is_form_like": result["entity_counts"]["questions"] > 0 and result["entity_counts"]["answers"] > 0
            }
        }
    
    except Exception as e:
        logger.error(f"Layout extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def extract_entities(text: str):
    """
    Placeholder for D3 NER (SROIE) - Phase 4
    """
    return {
        "status": "placeholder",
        "message": "D3 entity extraction (SROIE) coming in Phase 4",
        "text_length": len(text),
        "dataset": "SROIE-pending"
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy" if layout_parser.model else "loading",
        "d2_model": "LayoutLMv3-FUNSD",
        "d3_model": "SROIE-pending"
    }