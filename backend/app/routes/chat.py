from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def chat(query: str, doc_id: str = None):
    return {"answer": "System initializing...", "confidence": 0.0}