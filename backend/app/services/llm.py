from transformers import T5Tokenizer, T5ForConditionalGeneration, TextIteratorStreamer
import logging
import threading
import time

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        import os
        # Default to base model for better quality (still fits in free tier)
        self.model_name = os.getenv("LLM_MODEL", "google/flan-t5-base")
        self.tokenizer = None
        self.model = None

    # ... (existing loading logic unchanged)

    def generate_answer(self, context: str, question: str) -> str:
        """Standard generation from context with improved prompt"""
        input_text = f"Review the context below and answer the question truthfullly. If the answer is not in the context, say 'I cannot find the answer in the document'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self._run_inference(input_text, max_length=256)

# Singleton
llm_service = LLMService()
