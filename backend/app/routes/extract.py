from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def extract_entities(text: str):
    return {"entities": {}, "dataset": "SROIE"}