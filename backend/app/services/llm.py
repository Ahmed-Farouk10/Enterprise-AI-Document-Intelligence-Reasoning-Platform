from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import logging
import threading
import time
import os
import torch

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Upgrade to Qwen2.5-1.5B-Instruct (SOTA for <3B params)
        self.model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Lazy load the model"""
        if self.model is not None:
            return

        logger.info(f"Loading LLM ({self.model_name})...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

    def _ensure_model_loaded(self):
        if self.model is None:
            logger.info(f"Model not loaded. Loading now... (PID: {os.getpid()})")
            self.load_model()
        else:
            logger.debug(f"Model already loaded (PID: {os.getpid()})")

    def warmup(self):
        """Warmup the model by loading it into memory"""
        logger.info("Warming up LLM Service...")
        self._ensure_model_loaded()
        logger.info("LLM Service ready.")

    def _run_inference(self, messages: list, max_new_tokens: int = 512, temperature: float = 0.4) -> str:
        """Helper to run model generation using chat templates"""
        self._ensure_model_loaded()
        
        # Proper chat template handling
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            max_length=4096 # Large context for Qwen
        )
        
        outputs = self.model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            max_time=90.0
        )
        
        # Decode only the new tokens
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    def stream_inference(self, message: str, context: str = ""):
        """
        Generator for streaming response
        """
        self._ensure_model_loaded()
        
        if not context or len(context.strip()) < 10:
            # Fallback to General Knowledge
            logger.info("Context empty. Using General Knowledge System Prompt.")
            messages = [
                {"role": "system", "content": "You are Qwen, a helpful assistant. The user's question could not be answered by local documents or web search. Answer the question to the best of your ability using your general knowledge. Start your answer by saying 'I couldn't find specific documents, but generally speaking...'"},
                {"role": "user", "content": f"Question:\n{message}"}
            ]
        else:
            # Standard RAG
            messages = [
                {"role": "system", "content": "You are Qwen, a helpful and precise document assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say 'I cannot find the answer'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{message}"}
            ]

        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            max_length=8192 # Support larger docs
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.1,
            max_time=120.0
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def rewrite_query(self, question: str) -> str:
        """
        Query rewriting: Expand complex questions into clearer search queries
        """
        messages = [
            {"role": "system", "content": "You are a search query optimizer. Rewrite the user's question to be clearer and more specific for document retrieval. Return ONLY the rewritten query."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        rewritten = self._run_inference(messages, max_new_tokens=128, temperature=0.3)
        
        # Clean up if model is chatty
        rewritten = rewritten.strip().strip('"').strip("'")
        
        if len(rewritten) > 5:
            logger.info(f"Query rewritten: '{question}' -> '{rewritten}'")
            return rewritten
        return question

    def grade_relevance(self, context: str, question: str) -> bool:
        """
        Self-RAG Step 1: Retrieval Grading
        """
        messages = [
            {"role": "system", "content": "You are a relevance grader. Determine if the context is relevant to the question. Answer only 'yes' or 'no'."},
            {"role": "user", "content": f"Question: {question}\n\nContext: {context}"}
        ]
        
        response = self._run_inference(messages, max_new_tokens=10, temperature=0.1)
        return "yes" in response.lower()

    def grade_hallucination(self, context: str, answer: str) -> bool:
        """
        Self-RAG Step 2: Critique Token
        """
        messages = [
            {"role": "system", "content": "You are a fact checker. Determine if the answer is fully supported by the context. Answer only 'yes' or 'no'."},
            {"role": "user", "content": f"Context: {context}\n\nAnswer: {answer}"}
        ]
        
        response = self._run_inference(messages, max_new_tokens=10, temperature=0.1)
        return "yes" in response.lower()

    def generate_answer(self, context: str, question: str) -> str:
        """Standard generation"""
        messages = [
            {"role": "system", "content": "You are a senior document analyst. "
             "Analyze the provided context holistically. "
             "When evaluating timelines (work history, education), pay close attention to overlapping dates. "
             "Do not assume gaps exist if education or other activities cover the period. "
             "Answer in detailed bullet points."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
        return self._run_inference(messages, max_new_tokens=1024)

# Singleton
llm_service = LLMService()
