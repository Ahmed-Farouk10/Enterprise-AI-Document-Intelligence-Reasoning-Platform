FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && \
    useradd -m -u 1000 user && \
    chown -R user:user /app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    chown -R user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
