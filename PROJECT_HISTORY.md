# DocuCentric: Project Development History & Technical Iterations

This document tracks the technical evolution of DocuCentric from a proof-of-concept RAG application to a production-hardened Document Intelligence platform.

---

## 🚀 Phase 1: The Core Pivot (Architectural Shift)
**Goal:** Replace unreliable, "black-box" RAG patterns with stateful, verifiable workflows.

- **Legacy Removal**: Excised over 2,000 lines of monkey-patched code, including legacy Hugging Face and Cognee dependencies.
- **Workflow Engine**: Integrated **LangGraph 0.2** for stateful multi-step reasoning.
- **CAG Implementation**: Developed the **Cache-Augmented Generation (CAG)** engine to pre-compute document contexts, reducing latency by 10x compared to per-query vector searches.
- **Multi-Provider Router**: Built a resilient LLM router supporting Groq, OpenAI, and Gemini with circuit-breaker logic.

## 🏗️ Phase 2: Infrastructure & Production Hardening
**Goal:** Stabilize the platform for cloud deployment (Hugging Face / Vercel).

- **Database Migration**: Shifted to **Supabase (PostgreSQL/pgvector)** via Port 6543 (Connection Pooler) for production-grade persistence.
- **Network Stabilization**: Implemented a DNS/IPv4 shim locally on the backend to bypass Hugging Face's IPv6-only outbound restrictions, restoring connectivity to external database endpoints.
- **Asynchronous Pipeline**: Refactored the `LLMService` to use `AsyncOpenAI`, ensuring non-blocking streaming responses in the chat interface.

## 📡 Phase 3: Streaming & Performance Optimization
**Goal:** Achieve "instant-feel" responsiveness in the UI.

- **Anti-Proxy Buffering**: Added `X-Accel-Buffering: no` and `Cache-Control: no-cache` headers to SSE streams to prevent proxy-level delays.
- **Broadened Retrieval**: Lowered semantic thresholds (0.5 → 0.1) and added a "Nuclear Fallback" to raw database retrieval to ensure context is never lost, even for small queries.
- **Context Partitioning**: Implemented explicit document boundaries (`--- DOCUMENT START ---`) to help the LLM distinguish between multiple uploaded files (e.g., a Resume vs. a Handbook).

## 🛠️ Key Technical Edits

| Component | Improvement | Impact |
|-----------|-------------|--------|
| **Backend** | Pydantic V2 Migration | Robust environment variable validation and performance. |
| **LLM Service** | Direct Env-Var Fallback | Ensured API keys propagate correctly in Docker/HF environments. |
| **Vector Store** | match_documents RPC | Highly efficient semantic search directly inside PostgreSQL. |
| **Chat Routing** | Streaming SSE | Real-time token delivery with zero latency. |
| **Storage** | Dynamic Path Shims | Seamless cross-platform file handling (Windows/WSL/Docker). |

---

## 🎯 Current Status (Production Ready)
- **Primary LLM**: Groq (Llama 3.3 70B) for 200+ tokens/sec performance.
- **Storage**: Supabase Bucket + PostgreSQL (pgvector).
- **Frontend**: Next.js 16 (App Router) on Vercel.
- **Live Link**: [https://docucentric.vercel.app/](https://docucentric.vercel.app/)

---

*Compiled by Ahmed Farouk Shahin - Senior Developer Narrative*
