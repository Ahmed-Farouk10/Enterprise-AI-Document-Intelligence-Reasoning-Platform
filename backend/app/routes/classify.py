from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def classify_document(text: str):
    return {"predicted_class": "pending", "dataset": "RVL-CDIP"}