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
        
    def load_model(self):
        """Lazy load the model"""
        if self.model is not None:
            return

        logger.info(f"Loading LLM ({self.model_name})...")
        
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, 
                low_cpu_mem_usage=True
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

    def _ensure_model_loaded(self):
        if self.model is None:
            self.load_model()

    def _run_inference(self, prompt: str, max_length: int = 64) -> str:
        """Helper to run model generation"""
        self._ensure_model_loaded()
        # Note: True timeout for blocking CPU tasks in Python requires multiprocessing
        # For now we rely on logical constraints and max_length
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids
        
        outputs = self.model.generate(
            input_ids, 
            max_length=max_length, 
            num_beams=2, 
            early_stopping=True,
            repetition_penalty=1.2,
            do_sample=False,
            max_time=60.0 # Enforce 60s timeout at generation level (HuggingFace feature)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def stream_inference(self, prompt: str, max_length: int = 256):
        """
        Generator that yields tokens as they are generated.
        """
        self._ensure_model_loaded()
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_length=max_length,
            do_sample=False, # Use greedy decoding for stability
            repetition_penalty=1.2, # Prevent loops
            max_time=60.0
        )

        # Run generation in a separate thread so we can yield from the streamer
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def rewrite_query(self, question: str) -> str:
        """
        Query rewriting: Expand complex questions into clearer search queries
        """
        prompt = f"""Task: Rewrite this question to be clearer and more specific for document search.
Question: {question}
Rewritten:"""
        
        rewritten = self._run_inference(prompt, max_length=64)
        
        # Only use rewrite if it's substantially different and not malformed
        if len(rewritten) > 5 and rewritten.lower() != question.lower():
            logger.info(f"Query rewritten: '{question}' -> '{rewritten}'")
            return rewritten
        return question

    def grade_relevance(self, context: str, question: str) -> bool:
        """
        Self-RAG Step 1: Retrieval Grading
        Check if the retrieved chunk is actually relevant to the question.
        """
        prompt = f"""Task: Determine if the context is relevant to the question. Answer 'yes' or 'no'.
Question: {question}
Context: {context}
Relevant:"""
        
        response = self._run_inference(prompt, max_length=5)
        return "yes" in response.lower()

    def grade_hallucination(self, context: str, answer: str) -> bool:
        """
        Self-RAG Step 2: Critique Token
        Check if the answer is supported by the context (Fact-checking).
        """
        prompt = f"""Task: Determine if the answer is supported by the context. Answer 'yes' or 'no'.
Context: {context}
Answer: {answer}
Supported:"""
        
        response = self._run_inference(prompt, max_length=5)
        return "yes" in response.lower()

    def generate_answer(self, context: str, question: str) -> str:
        """Standard generation from context with improved prompt"""
        input_text = f"Read the following documents and answer the question in detail. Use bullet points if applicable. If the answer is not in the documents, say 'I cannot find the answer'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self._run_inference(input_text, max_length=512)

# Singleton
llm_service = LLMService()
