# ========================================
# DocuCentric Backend Dockerfile
# Optimized for Hugging Face Spaces
# ========================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user (Hugging Face uses user with UID 1000)
RUN useradd -m -u 1000 user

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/uploads && \
    chown -R user:user /app

# Copy application code
COPY --chown=user:user . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port (Default for HF Spaces is 7860)
EXPOSE 7860

# Switch to user
USER user

# Startup command - HF Spaces uses the PORT environment variable
CMD uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 2
