from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    return {"status": "received", "filename": file.filename}