FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user for security
RUN mkdir -p /app && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY --chown=appuser:appuser . .

# Create uploads directory
RUN mkdir -p uploads && chown -R appuser:appuser uploads

# Switch to non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Expose port 7860 (Required by HuggingFace Spaces)
EXPOSE 7860

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
