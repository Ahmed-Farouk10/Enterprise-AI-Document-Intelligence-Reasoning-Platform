from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

CANDIDATE_LABELS = [
    "invoice", "resume", "financial_statement", "contract", 
    "bank_statement", "technical_specification", "legal_letter", "id_card_or_passport"
]

print("Loading zero-shot classifier...")
classifier = None
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("Zero-shot classifier loaded successfully")
except Exception as e:
    logger.error(f"Error loading zero-shot classifier: {e}")

class ClassificationRequest(BaseModel):
    text: str

@router.post("/")
async def classify_document(request: ClassificationRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is currently loading or failed to load."
        )
    
    try:
        truncated_text = request.text[:1000]
        result = classifier(truncated_text, CANDIDATE_LABELS)
        
        return {
            "status": "success",
            "predicted_class": result["labels"][0],
            "confidence": round(result["scores"][0], 4),
            "is_confident": result["scores"][0] > 0.5,
            "all_scores": {
                label: round(score, 4) 
                for label, score in zip(result["labels"], result["scores"])
            },
            "model": "facebook/bart-large-mnli",
            "dataset": "RVL-CDIP-domain-adapted"
        }
    
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.get("/health")
async def health_check():
    return {
        "status": "healthy" if classifier else "unavailable",
        "model_loaded": classifier is not None,
        "model_name": "facebook/bart-large-mnli",
        "candidate_labels": CANDIDATE_LABELS
    }
