# Architecture Documentation

## System Overview

DocuCentric is an enterprise-grade document intelligence platform that combines **LangGraph workflows** with **Cache-Augmented Generation (CAG)** to deliver fast, accurate, and fact-grounded document analysis.

## Design Decisions

### Why LangGraph + CAG Instead of GraphRAG?

| Approach | Pros | Cons | Our Decision |
|----------|------|------|--------------|
| **Pure GraphRAG** (Neo4j) | Rich entity relationships | High infra overhead, slow ingestion | ❌ Not suitable |
| **CAG Only** | Fast, simple | No reasoning capabilities | ❌ Too limited |
| **LangGraph + CAG** | Intelligent workflows + speed | Moderate complexity | ✅ **Optimal** |

### Architecture Principles

1. **Fact-Grounding First** - No hallucination tolerated
2. **Performance by Design** - CAG eliminates retrieval latency
3. **Multi-Provider Flexibility** - No vendor lock-in
4. **Separation of Concerns** - Clean modular architecture
5. **Production Ready** - Rate limiting, caching, monitoring

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                         │
│                    Next.js 16 Frontend                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────┐
│                       API LAYER                              │
│  FastAPI Routes (documents.py, chat.py)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   SERVICE LAYER                              │
│                                                              │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ LangGraph        │  │ CAG Engine   │  │ LLM Service   │ │
│  │ Workflows        │  │              │  │ (Router)      │ │
│  └────────┬─────────┘  └──────┬───────┘  └───────┬───────┘ │
│           │                   │                   │         │
│  ┌────────▼─────────┐  ┌──────▼───────┐  ┌───────▼───────┐ │
│  │ Intent           │  │ Redis Cache  │  │ Providers:    │ │
│  │ Classifier       │  │              │  │ OpenAI,Groq,  │ │
│  │                  │  │              │  │ Gemini,Ollama │ │
│  └──────────────────┘  └──────────────┘  └───────────────┘ │
│                                                              │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Vector Store     │  │ Verification │  │ Search        │ │
│  │ (LanceDB)        │  │ Service      │  │ (Tavily)      │ │
│  └──────────────────┘  └──────────────┘  └───────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   DATA LAYER                                 │
│                                                              │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ PostgreSQL │  │ Redis    │  │ LanceDB  │  │ File     │ │
│  │ /SQLite    │  │          │  │          │  │ System   │ │
│  └────────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Workflow Architecture

### LangGraph State Machine

```
                    ┌──────────────┐
                    │  User Query  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Classify    │
                    │  Intent      │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  Retrieve CAG Context   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Build System Prompt    │
              │  (Intent-Specific)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Generate Response      │
              │  (LLM with Context)     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Verify Response        │
              │  (Hallucination Check)  │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │  Response    │
                    └──────────────┘
```

### Intent Classification

The system classifies queries into 8 intents:

1. **SUMMARY** - Document overview request
2. **FACTUAL** - Specific factual questions
3. **EVALUATIVE** - Assessment/critique
4. **IMPROVEMENT** - Enhancement suggestions
5. **GAP_ANALYSIS** - Missing information
6. **SCORING** - Quantitative assessment
7. **SEARCH_QUERY** - Requires external search
8. **GENERAL** - Unclear/conversational

Each intent triggers specialized system prompts and processing logic.

## CAG (Cache-Augmented Generation)

### How It Works

**Traditional RAG:**
```
Query → Embed Query → Vector Search → Format Context → Generate
(Latency: ~500ms-2s per query)
```

**CAG Approach:**
```
Upload → Pre-compute Context → Cache
Query → Retrieve from Cache → Generate
(Latency: ~10-50ms per query)
```

### Cache Strategy

1. **Document Upload** triggers pre-computation
2. **Context Windows** are cached per document set
3. **Semantic Cache** stores similar queries
4. **TTL Management** ensures freshness (1 hour default)

### Cache Keys

Cache keys are deterministic based on document IDs:
```python
key = sha256(sorted(document_ids))
# Example: cag:context:a1b2c3d4e5f6...
```

## LLM Router Architecture

### Circuit Breaker Pattern

```
Provider 1 (Primary)
    ↓ [5 failures]
Circuit Breaker Trips (60s timeout)
    ↓
Provider 2 (Fallback 1)
    ↓ [5 failures]
Provider 3 (Fallback 2)
    ↓
Provider 4 (Fallback 3)
```

### Supported Providers

| Provider | Models | Cost | Speed |
|----------|--------|------|-------|
| Groq | Llama-3.3-70B, Mixtral | Free tier | ⚡⚡⚡⚡⚡ |
| OpenAI | GPT-4o, GPT-4o-mini | $$ | ⚡⚡⚡⚡ |
| Gemini | Gemini-2.0-flash | Free tier | ⚡⚡⚡⚡⚡ |
| Ollama | Local models | Free | ⚡⚡ |
| OpenRouter | 100+ models | Varies | Varies |

## Data Flow

### Document Upload Flow

```
1. Client uploads file (PDF/TXT/DOCX)
2. FastAPI validates file type
3. Creates document record (pending status)
4. Returns immediately to client
5. Background task:
   a. Extract text (OCR if needed)
   b. Chunk document (1000 chars, 200 overlap)
   c. Generate embeddings
   d. Store in LanceDB
   e. Pre-compute CAG context
   f. Update status to "completed"
```

### Chat Query Flow

```
1. Client sends message
2. Validate session
3. Check semantic cache (hit? return cached)
4. Execute LangGraph workflow:
   a. Classify intent
   b. Retrieve CAG context
   c. Build system prompt
   d. Generate response
   e. Verify response
5. Save to database
6. Return to client
7. Update semantic cache
```

## Security Considerations

### Rate Limiting

| Endpoint | Rate Limit |
|----------|-----------|
| Default | 200/min, 5000/day |
| Upload | 10/hour |
| Session Create | 20/min |
| Chat Messages | 30/min |
| RAG Query | 50/min |

### Data Protection

- API keys stored in environment variables (never in code)
- Redis sessions with TTL (auto-expire)
- File uploads validated for type and size
- CORS configured for specific origins in production

## Monitoring & Observability

### Structured Logging

All logs use `structlog` with JSON format:

```json
{
  "timestamp": "2026-04-15T10:30:00Z",
  "level": "info",
  "event": "workflow_completed",
  "workflow": "document_qa",
  "processing_time_ms": 1234,
  "verification_score": 95.5
}
```

### Health Checks

- `/health` - Component status (database, cache, vector store)
- `/system/info` - Full system configuration
- `/api/cache/stats` - Cache hit rates and size

## Performance Optimization

### Caching Layers

1. **CAG Cache** - Pre-computed document contexts (1h TTL)
2. **Semantic Cache** - Similar query responses (1h TLL, 92% threshold)
3. **Session Cache** - Full session state (30m TTL)

### Vector Store Optimization

- Chunk size: 1000 characters
- Overlap: 200 characters
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Retrieval limit: 5-25 chunks based on query depth

## Deployment Strategies

### Development

```bash
docker-compose up -d
```

### Production

```bash
# With PostgreSQL and Redis cluster
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up -d --scale celery-worker=8
```

### Hugging Face Spaces (Legacy)

**Not supported in v1.0.0** - Removed HF dependencies for professional deployment.

## Migration from v0.4.x

### Breaking Changes

1. **Removed:**
   - All Hugging Face dependencies (`huggingface-hub`, `fastembed`)
   - Cognee/Rag legacy code (2000+ lines)
   - Neo4j graph database
   - Monkey patching workarounds

2. **Changed:**
   - Configuration: Environment variables → Pydantic settings
   - RAG engine → LangGraph workflows
   - Basic cache → CAG + Semantic cache

3. **Added:**
   - LangGraph 0.2+
   - CAG engine
   - Multi-provider LLM router
   - Comprehensive test suite
   - Docker Compose deployment

### Migration Steps

1. **Backup data:**
   ```bash
   cp -r backend/data backup/
   cp backend/docucentric.db backup/
   ```

2. **Update dependencies:**
   ```bash
   pip install -r requirements.txt  # New requirements
   ```

3. **Update .env:**
   - Remove `HF_TOKEN`, `COGNEE_*`, `NEO4J_*` variables
   - Add `LLM_PROVIDER` and corresponding API key
   - See `.env.example` for new format

4. **Migrate database:**
   ```bash
   # Old SQLite schema → New schema
   alembic upgrade head
   ```

5. **Test:**
   ```bash
   pytest
   ```

## Future Roadmap

### Q2 2026
- [ ] Human-in-the-loop workflow review
- [ ] Multi-language document support
- [ ] Advanced graph visualization
- [ ] Workflow builder UI

### Q3 2026
- [ ] OAuth2 authentication
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)

### Q4 2026
- [ ] Custom model fine-tuning
- [ ] Workflow marketplace
- [ ] API rate limit tiers
- [ ] Enterprise SSO integration
