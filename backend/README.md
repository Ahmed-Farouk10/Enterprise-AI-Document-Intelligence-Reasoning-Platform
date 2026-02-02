---

title: Enterprise Document Intelligence
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Enterprise Document Intelligence API

Multimodal AI for document understanding using 4 datasets: RVL-CDIP, FUNSD, SROIE, GAN.

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /upload` - Document ingestion (PDF, DOCX, XLSX, images)
- `POST /classify` - Document classification (D1)
- `POST /extract` - Entity extraction (D3)
- `POST /chat` - RAG-based Q&A

## Architecture

- **Backend**: FastAPI + Docker on Hugging Face Spaces
- **Training**: Kaggle notebooks → HF Hub models
- **Frontend**: Vercel (separate)

## Datasets

| Dataset | Purpose |
|---------|---------|
| RVL-CDIP | Document classification |
| FUNSD | Layout understanding |
| SROIE | Receipt entity extraction |
| Synthetic GAN | Data augmentation |