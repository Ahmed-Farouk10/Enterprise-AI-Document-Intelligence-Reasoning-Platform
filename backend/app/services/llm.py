from transformers import T5Tokenizer, T5ForConditionalGeneration, TextIteratorStreamer
import logging
import threading
import time
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Default to large model for decent reasoning (fits in 16GB RAM)
        self.model_name = os.getenv("LLM_MODEL", "google/flan-t5-large")
        self.tokenizer = None
        self.model = None

    # ... (rest of class) ...

    def generate_answer(self, context: str, question: str) -> str:
        """Standard generation from context with improved prompt"""
        input_text = f"Read the following documents and answer the question in detail. Use bullet points if applicable. If the answer is not in the documents, say 'I cannot find the answer'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self._run_inference(input_text, max_length=512)
