# DocuCentric: Gemini CLI Mandates

This document provides foundational mandates and technical standards for the **DocuCentric** project. These instructions take absolute precedence over general defaults.

## Project Essence
DocuCentric is an Enterprise Document Intelligence platform that replaces traditional RAG with **LangGraph workflows** and **Cache-Augmented Generation (CAG)**.

### Core Philosophy
1.  **Zero Hallucination Tolerance:** Strict fact-grounding is the highest priority. Every LLM claim must be verified against document context.
2.  **CAG Over RAG:** Prioritize pre-computed document contexts (CAG) for speed and consistency. Vector search (LanceDB) is a fallback for deep retrieval.
3.  **Stateful Reasoning:** Complex queries must go through LangGraph state machines, not single-shot LLM calls.
4.  **Multi-Provider Resilience:** The LLM Router must handle provider failures gracefully using the circuit breaker pattern.

## Technical Stack & Constraints

### Backend (FastAPI)
- **Framework:** FastAPI 0.110+ utilizing `asynccontextmanager` for lifespan management.
- **Orchestration:** LangGraph 0.2+ for all reasoning workflows.
- **Storage:** 
  - **Relational:** PostgreSQL (production) / SQLite (local) via SQLAlchemy/Alembic.
  - **Vector:** LanceDB (stored in `backend/.lancedb_data`).
  - **Cache:** Redis 7 for CAG context, semantic cache, and session persistence.
- **Logging:** Mandatory structured JSON logging via `structlog`. Use `app.core.logging_config.get_logger`.
- **Validation:** Pydantic v2 for all schemas and settings.

### Frontend (Next.js)
- **Framework:** Next.js 16 (App Router) with React 19.
- **Styling:** Tailwind CSS 4 and shadcn/ui. **Prefer Vanilla CSS/Tailwind 4 primitives.**
- **State Management:** TanStack React Query v5.
- **Type Safety:** Strict TypeScript mode is mandatory.

## Engineering Standards

### 1. Workflow Implementation (LangGraph)
All reasoning logic must reside in `backend/app/services/langgraph_workflows.py`. 
- Every workflow must include an **Intent Classification** step (SUMMARY, FACTUAL, EVALUATIVE, etc.).
- Every generation must pass through the **Verification Service** (`backend/app/services/verification_service.py`).
- **Personality Mandate:** Responses must be "Extrovert" (enthusiastic, witty, conversational, and encouraging) while remaining strictly fact-grounded. Use `HumanPersonality.SYSTEM_BASE` for tone.

### 2. CAG Engine & Token Optimization
- **Performance:** Prioritize pre-computed contexts cached in Redis.
- **Context Compression:** Always compress document context (e.g., 5000 -> 3000 chars) before sending to LLM.
- **History Truncation:** Limit conversation history to the last 3 turns (6 messages) to save tokens.
- **Smart Retrieval:** Use shallow retrieval (8 chunks) for simple queries and deep retrieval (15 chunks) for complex analysis.
- **Cache Keys:** Use deterministic keys: `cag:context:{sha256(sorted_doc_ids)}`.

### 3. Frontend & API Standards
- **Streaming:** Use Server-Sent Events (SSE) for all chat responses (`/api/chat/sessions/{id}/stream`).
- **State Management:** Use TanStack React Query v5 for caching and persistence.
- **UI:** Strictly adhere to Tailwind CSS 4 and shadcn/ui components.
- **Resilience:** All LLM calls must use the `llm_service.py` router with circuit breaker logic.

### 4. Error Handling & Logging
- API endpoints must use **SlowAPI rate limiting**.
- **Logging:** Mandatory JSON logging via `structlog`. Use `get_logger(__name__)`.
- **Validation:** Pydantic v2 for all data models.

### 4. Code Style
- **Python:** Follow PEP 8. Use explicit type hints. Prefer async/await for I/O bound tasks.
- **Frontend:** Use functional components and hooks. Maintain clean separation between UI (shadcn) and logic (hooks).

## Development Workflow

### Environment Configuration
- Backend: `backend/.env` (based on `.env.example`).
- Frontend: `frontend/.env.local`.
- **CRITICAL:** Never hardcode API keys. Access via `app.config.settings`.

### Testing Mandates
- Every bug fix requires a reproduction test in `backend/tests/` or `tests/`.
- Run tests using `pytest` from the `backend` directory.
- Maintain >80% coverage for core services (`cag_engine`, `llm_service`, `verification_service`).

### Deployment
- Development is Docker-first: `docker-compose up -d --build`.
- Ports: `8000` (Backend), `3000` (Frontend).

## Prohibited Actions
- **No Monkey Patching:** Do not use legacy Hugging Face or Cognee patterns removed in v1.0.0.
- **No Direct Vector Search for Q&A:** Always check CAG cache first.
- **No Unstructured Logs:** Never use `print()` or standard `logging` without JSON formatting in backend code.
