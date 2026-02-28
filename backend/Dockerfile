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

# Create app user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/.cache && \
    chown -R appuser:appuser /app

# Set environment variables
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV PYTHONUNBUFFERED=1

# Add standard cache-busting ARG
ARG CACHEBUST=1
ENV FORCE_REBUILD_DEPS=$CACHEBUST

# Copy requirements
COPY requirements.txt .

# Install dependencies in a single layer with CPU-specific torch
# Using --prefer-binary to speed up build and avoid hung compilations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Ensure environment is healthy
RUN python -c "print('Environment ready!')"

# Install Playwright browsers and dependencies
# Using --with-deps to install system dependencies (requires root, which we are)
RUN playwright install --with-deps chromium

# Pre-download Models to speed up HF Spaces startup
# This bakes the Embedding Model into the Docker image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Fix permissions on downloaded cache so app user can access it
RUN chown -R appuser:appuser /app/.cache

# Copy application code
ENV DEPLOY_TIMESTAMP=20240208_2041
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Expose port 7860
EXPOSE 7860

# Default command: Start Everything
CMD redis-server --daemonize yes && \
    celery -A app.workers.celery_app worker --loglevel=info --concurrency=1 -D && \
    uvicorn "app.main:app" --host "0.0.0.0" --port "7860"
