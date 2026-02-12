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
- **State Management**: Redis (Conversational Memory & Caching)

## Stateful Architecture (New)

The system uses a persistent memory layer to support stateless deployments (e.g., HF Spaces):

1.  **Session Manager**: backed by Redis, persists user context across server restarts.
2.  **Semantic Cache**: Caches RAG queries to reduce LLM costs and latency.
3.  **Conversational Memory**: Maintains short-term and working memory for context-aware chat.

### Requirements
- **Redis**: A running Redis instance (v5.0+)
- **Environment Variables**:
    - `REDIS_HOST`: Hostname (default: localhost)
    - `REDIS_PORT`: Port (default: 6379)
    - `REDIS_PASSWORD`: Optional password

## Datasets

| Dataset | Purpose |
|---------|---------|
| RVL-CDIP | Document classification |
| FUNSD | Layout understanding |
| SROIE | Receipt entity extraction |
| Synthetic GAN | Data augmentation |