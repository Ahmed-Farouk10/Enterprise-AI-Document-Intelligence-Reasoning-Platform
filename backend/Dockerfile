FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/.cache && \
    chown -R appuser:appuser /app

# Set environment variables for model caching
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install heavy ML dependencies first (improves caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.40.0 sentence-transformers==3.0.0

# Install remaining dependencies including Cognee
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download the LLM and Embedding models to bake them into the image
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    from sentence_transformers import SentenceTransformer; \
    llm_name = 'Qwen/Qwen2.5-1.5B-Instruct'; \
    AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, low_cpu_mem_usage=True); \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
ENV DEPLOY_TIMESTAMP=20240206_1200
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Expose port 7860
EXPOSE 7860

# Default command: Start Redis -> Start Celery Worker -> Start FastAPI
CMD redis-server --daemonize yes && \
    celery -A app.workers.celery_app worker --loglevel=info --concurrency=1 -D && \
    uvicorn "app.main:app" --host "0.0.0.0" --port "7860"
