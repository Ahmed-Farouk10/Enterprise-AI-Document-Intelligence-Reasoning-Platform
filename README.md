# DocuCentric: Enterprise AI Document Intelligence Platform

Welcome to the core repository for **DocuCentric**. This project is a next-generation Document Reasoning and RAG (Retrieval-Augmented Generation) platform designed to handle complex unstructured data—contracts, resumes, technical manuals—with the precision and wit of a senior analyst.

[![Deployment Status](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/Ahmed-Farouk10/Enterprise-AI-Document-Intelligence-Reasoning-Platform)
[![Website](https://img.shields.io/badge/Live-Vercel-black)](https://docucentric.vercel.app/)

---

## 🚀 The Vision
Document intelligence isn't just about finding text; it's about **reasoning** over it. Most RAG systems suffer from "hallucination bloat" or a dry, robotic tone. DocuCentric was built to solve this by implementing a **Strict Fact-Grounding** persona with a witty, human-centric interface. If it’s not in your document, the AI won’t invent it—but it will tell you why with a bit of personality.

---

## 🏗️ Technical Architecture
The system follows a modular **Asynchronous Self-RAG** architecture, optimized for deployment on resource-constrained environments like Hugging Face Spaces.

### Core Stack:
- **FastAPI**: High-performance backend engine.
- **LanceDB**: An embedded vector database for lightning-fast semantic retrieval without the overhead of external clusters.
- **Groq API**: Utilizing Llama-3.3-70B and Qwen-2.5-32B for sub-second reasoning and extraction.
- **Sentence-Transformers**: Local embeddings using the `all-MiniLM-L6-v2` model.
- **Redis & Celery**: Background processing for OCR and multi-tier document vectorization.

---

## 🛠️ Key Components

### 1. The Reasoning Engine (`llm_service.py`)
A custom model router that intelligently switches between local fallbacks, the Hugging Face Inference API, and the Groq native SDK. It manages the **DocuCentric Persona**—a witty assistant that enforces strict citations.

### 2. Standardized RAG Pipeline (`vector_store.py`)
Moving away from complex graph overheads, we use a refined chunking and indexing strategy. Every document is split based on semantic boundaries, indexed into LanceDB, and retrieved using hybrid search techniques.

### 3. Credible Search Service (`search.py`)
When the document context isn't enough, the system triggers a "Search Intent." It uses **Tavily** to fetch high-credible web results, which are then injected into the LLM context as `[EXTERNAL RESULTS]`.

### 4. Semantic Cache (`query_cache.py`)
To keep costs low and responses instant, DocuCentric implements a Redis-backed semantic cache. If a similar question has been asked before, the system replays the verified answer instantly.

---

## 📅 Project Phases (The Journey)

### Phase 1: The Graph RAG Experiment
Initially, we integrated **Cognee** and Neo4j to build a Knowledge Graph. This provided deep entity links but proved too heavy and resource-intensive for standard cloud deployments and free-tier spaces.

### Phase 2: The "Clean Break" (Standard RAG)
We refactored the entire ingestion pipeline. We stripped out the Cognee/Graph dependencies in favor of a customized **LanceDB** implementation. This reduced the Docker image size by 70% and cut document processing time from minutes to seconds.

### Phase 3: Persona & Web Intelligence
Our current focus. We integrated **Groq** for instantaneous streaming responses. We also refined the AI's persona to ensure it strikes the perfect balance between human-like conversation and "deadly serious" factual accuracy.

---

## 💻 Usage & Local Setup

### 1. Environment Configuration
Create a `.env` file in the `backend/` directory:
```bash
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_optional
```

### 2. Run with Docker (Recommended)
```bash
docker-compose up --build
```

### 3. Manual Startup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

---

## 🔗 Live Implementation
You can interact with the latest version of DocuCentric here:
👉 **[DocuCentric Live Dashboard](https://docucentric.vercel.app/)**

---
*Created by Ahmed Farouk Shahin — Building the future of document reasoning.*
