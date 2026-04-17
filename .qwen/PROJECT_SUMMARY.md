# Project Summary: DocuCentric Platform

## Architecture
- **Backend:** Python 3.11, FastAPI, Celery (Redis Broker), LanceDB (Vector Store), PostgreSQL (Auth/Metadata).
- **Frontend:** Next.js 14, Tailwind CSS, Shadcn UI.
- **AI Stack:** Groq (LLM), Custom Hash-based Local Embeddings (no HuggingFace).

## Current State
- **Stabilization Phase:** Fixed async coroutine bugs and removed heavy dependencies.
- **Offline Readiness:** The system is now 100% offline-capable for embeddings.
- **Session Management:** Functional chat deletion implemented.

## Tooling Warning for AI
- Only use `glob`, `grep_search`, `edit`, `agent`, and `web_fetch`.
- `todo` tool IS NOT INSTALLED. Write plans as plain text.
