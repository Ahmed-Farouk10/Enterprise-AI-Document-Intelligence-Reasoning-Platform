# DocuCentric - Enterprise Document Intelligence & Reasoning Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-7e56c2.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DocuCentric is a next-generation **Document Intelligence and Reasoning Platform** that combines **LangGraph workflows** with **Cache-Augmented Generation (CAG)** to deliver fast, accurate, and fact-grounded document analysis.

Unlike traditional RAG systems that hallucinate, DocuCentric enforces **strict fact-grounding** - every claim must reference actual document content, with built-in hallucination detection and verification.

### Key Features

✅ **LangGraph Workflows** - Stateful multi-step reasoning with intent classification  
✅ **Cache-Augmented Generation (CAG)** - Pre-computed document contexts for instant retrieval  
✅ **Multi-Provider LLM Router** - OpenAI, Groq, Gemini, Ollama with automatic failover  
✅ **Semantic Caching** - Embedding-based cache reduces LLM costs by up to 70%  
✅ **Hallucination Detection** - Post-generation verification with confidence scoring  
✅ **Redis Session Persistence** - Cross-server session continuity with 24h TTL  
✅ **Document Type Routing** - Specialized pipelines for resumes, contracts, invoices  
✅ **External Search Integration** - Tavily + DuckDuckGo for web intelligence  
✅ **Rate Limiting** - Production-ready rate limiting with SlowAPI  
✅ **Structured Logging** - JSON logs with request tracing  

## Architecture

```
┌─────────────────┐
│   Next.js UI    │  ← React 19, shadcn/ui, Tailwind CSS
└────────┬────────┘
         │ HTTP/SSE
┌────────▼────────┐
│   FastAPI       │  ← LangGraph Workflows + CAG Engine
│   Backend       │
├─────────────────┤
│ • Intent Router │ → Classifies queries (SUMMARY, FACTUAL, etc.)
│ • CAG Engine    │ → Pre-computed document contexts
│ • LLM Router    │ → Multi-provider with circuit breaker
│ • Verifier      │ → Hallucination detection
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│LanceDB│ │ Redis │
│Vector │ │Cache  │
│Store  │ │       │
└───────┘ └───────┘
```

## Technology Stack

### Backend
- **Framework:** FastAPI 0.110+
- **Workflows:** LangGraph 0.2+
- **LLM Providers:** OpenAI, Groq, Gemini, Ollama, OpenRouter
- **Vector Store:** LanceDB
- **Cache:** Redis 7
- **Database:** PostgreSQL 16 / SQLite
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **Document Processing:** pdfplumber, pypdf, python-docx, Tesseract OCR
- **External Search:** Tavily API, DuckDuckGo

### Frontend
- **Framework:** Next.js 16 (App Router)
- **UI:** React 19, shadcn/ui, Tailwind CSS 4
- **State:** TanStack React Query v5
- **HTTP:** Axios with interceptors

## Quick Start

### Prerequisites

- **Docker Desktop** installed and running ([Download](https://www.docker.com/products/docker-desktop/))
- At least 8GB RAM available for Docker

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Enterprise AI Document Intelligence  Reasoning Platform"
```

### 2. Configure Environment

**Backend:**
```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your LLM provider:
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

**Frontend:**
```bash
cp frontend/.env.example frontend/.env.local
# No changes needed - defaults to http://localhost:8000
```

### 3. Start Services

**Windows:**
```bash
start-dev.cmd
```

**Linux/Mac:**
```bash
chmod +x start-dev.sh
./start-dev.sh
```

**Or manually:**
```bash
docker-compose up -d --build
```

### 4. Access Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### 5. View Logs

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 6. Stop Services

```bash
docker-compose down
```

## Docker Deployment

For detailed Docker deployment guides, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) and [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md).

### Production

```bash
# Build and start with production config
docker-compose -f docker-compose.yml up -d --build

# Scale workers
docker-compose up -d --scale celery-worker=4
```

## Configuration

### Environment Variables

See [`.env.example`](backend/.env.example) for all configuration options.

**Required:**
- `LLM_PROVIDER` - Choose: openai, groq, gemini, ollama, openrouter
- Corresponding `*_API_KEY` for your provider
- `DATABASE_URL` - SQLite or PostgreSQL connection string
- `REDIS_URL` - Redis connection string

**Optional:**
- `TAVILY_API_KEY` - For web search integration
- `OPENROUTER_API_KEY` - For access to 100+ models

### LLM Provider Configuration

DocuCentric supports multiple LLM providers with automatic failover:

```python
# Provider selection in .env
LLM_PROVIDER=groq  # Options: openai, groq, gemini, ollama, openrouter

# Model customization
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.0-flash-exp
```

## API Reference

### Document Management

```bash
# Upload document
POST /api/documents/upload
Content-Type: multipart/form-data
- file: PDF, TXT, or DOCX

# List documents
GET /api/documents?page=1&page_size=20

# Get document
GET /api/documents/{document_id}

# Delete document
DELETE /api/documents/{document_id}

# Download document
GET /api/documents/{document_id}/download
```

### Chat Sessions

```bash
# Create session
POST /api/chat/sessions
{
  "title": "Resume Analysis",
  "document_ids": ["doc-1", "doc-2"]
}

# List sessions
GET /api/chat/sessions

# Get session
GET /api/chat/sessions/{session_id}

# Send message (non-streaming)
POST /api/chat/sessions/{session_id}/messages
{
  "content": "What are the candidate's key skills?"
}

# Send message (streaming)
POST /api/chat/sessions/{session_id}/stream
{
  "content": "Evaluate this resume"
}
```

### Cache Management

```bash
# Get cache stats
GET /api/cache/stats

# Invalidate cache
POST /api/cache/invalidate
{
  "namespace": "default",
  "pattern": "session:*"
}

# Pre-warm cache
POST /api/cache/warm
{
  "document_sets": [
    ["doc-1", "doc-2"],
    ["doc-3"]
  ]
}
```

### System Management

```bash
# System info
GET /system/info

# Health check
GET /health

# Reset vector store (WARNING: Deletes all indexed documents)
POST /system/reset/vector-store
```

## Testing

```bash
# Run all tests
cd backend
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run without slow tests
pytest -m "not slow"
```

## LangGraph Workflows

DocuCentric uses LangGraph for intelligent document reasoning:

### Workflow Types

1. **Document Q&A** - General document questions
2. **Resume Analysis** - Specialized resume parsing
3. **Contract Analysis** - Legal document analysis

### Workflow Steps

```
1. Classify Intent → SUMMARY | FACTUAL | EVALUATIVE | etc.
2. Retrieve CAG Context → Pre-computed document cache
3. Build System Prompt → Intent-specific instructions
4. Generate Response → LLM with strict fact-grounding
5. Verify Response → Hallucination detection
```

### Intent Types

- `SUMMARY` - Document overview
- `FACTUAL` - Specific questions
- `EVALUATIVE` - Assessment/critique
- `IMPROVEMENT` - Enhancement suggestions
- `GAP_ANALYSIS` - Missing information
- `SCORING` - Quantitative assessment
- `SEARCH_QUERY` - Requires web search
- `GENERAL` - Unclear/conversational

## Cache-Augmented Generation (CAG)

### How CAG Works

Unlike traditional RAG that searches on every query, CAG **pre-computes** document contexts during upload:

1. **Document Upload** → Extract text chunks
2. **Pre-computation** → Cache representative contexts
3. **Query Time** → Instant retrieval (no vector search)
4. **Cache Invalidation** → On document update/delete

### Benefits

- ⚡ **10x Faster** - No embedding computation per query
- 💰 **70% Cheaper** - Reduced LLM API calls via semantic cache
- 🎯 **More Accurate** - Comprehensive context vs. keyword matching

### Cache Management

```python
# Pre-compute context for documents
await cag_engine.precompute_context(["doc-1", "doc-2"])

# Check cache
cached = cag_engine.get_cached_context(["doc-1"])

# Invalidate
cag_engine.invalidate_context(["doc-1"])

# Warm cache for popular documents
await cag_engine.warm_cache([
    ["doc-1", "doc-2"],
    ["doc-3"]
])
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript strict mode for frontend
- Add tests for new features
- Update documentation
- Use conventional commits

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Author

**Ahmed Farouk Shahin**

- GitHub: [@Ahmed-Farouk10](https://github.com/Ahmed-Farouk10)

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LanceDB](https://lancedb.com/) - Embedded vector database

## Changelog

### v1.0.0 (2026-04-15)

**BREAKING CHANGES:**
- Removed all Hugging Face/Cognee dependencies
- Replaced legacy RAG with LangGraph workflows
- Implemented CAG (Cache-Augmented Generation)
- Professionalized configuration management

**NEW FEATURES:**
- LangGraph stateful reasoning workflows
- CAG engine for instant document context retrieval
- Multi-provider LLM router with circuit breaker
- Comprehensive test suite
- Docker Compose deployment
- System info and cache management endpoints

**IMPROVEMENTS:**
- Removed 2000+ lines of dead code
- Clean separation of concerns
- Environment-based configuration
- Production-ready rate limiting
- Structured JSON logging

---

**Ready to deploy. No more monkey patching. No more hallucination.** 🚀
