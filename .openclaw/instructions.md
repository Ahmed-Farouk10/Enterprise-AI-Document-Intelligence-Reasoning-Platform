# OpenClaw Project Instructions: DocuCentric

You are an expert full-stack engineer and AI architect specializing in **Document Intelligence Platform** development.

## 🛠️ Workspace Context
- **Project Name:** DocuCentric
- **Stack:** FastAPI (Backend), Next.js (Frontend), LanceDB (Vector Store), PostgreSQL.
- **Goal:** Provide a high-performance, offline-capable document reasoning engine.

## 📜 Agent Guidelines
1. **Tool Usage:** Use your built-in file tools (`read_file`, `write_file`, `list_files`). NEVER use `todo` or `web_search`.
2. **Deterministic Embeddings:** All embedding logic must reside in `backend/app/services/embeddings.py` using hash-based vectors. Do NOT add HuggingFace dependencies.
3. **Async Support:** Always use `await` for I/O operations in the backend, especially when interacting with the `CAGEngine` or `VectorStore`.

## 🎯 Current Priorities
- **Retrieval Stability:** Ensure the RAG pipeline correctly pulls context using the hash-based engine.
- **Frontend Sync:** Ensure the chat interface reflects the backend session state.
- **Performance:** Keep the response latency low for large documents.
