from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        logger.info("Loading LLM (FLAN-T5-Base)...")
        self.model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        logger.info("LLM loaded successfully")

    def _run_inference(self, prompt: str, max_length: int = 64) -> str:
        """Helper to run model generation"""
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
            temperature=0.3
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        """Standard generation from context"""
        input_text = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
        return self._run_inference(input_text, max_length=128)

# Singleton
llm_service = LLMService()
