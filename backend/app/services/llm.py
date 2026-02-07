from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import logging
import threading
import time
import os
import torch
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Upgrade to Qwen2.5-1.5B-Instruct (SOTA for <3B params)
        self.model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer = None
        self.model = None
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load_model(self):
        """Lazy load the model with retries"""
        if self.model is not None:
            return

        logger.info(f"Loading LLM ({self.model_name})...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                dtype=torch.float32, 
                low_cpu_mem_usage=True
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

    def generate_search_query(self, user_query: str, context: str = "") -> str:
        """
        Generate a precise, keyword-optimized web search query.
        Output is GUARANTEED to be a single clean query string.
        """
        self._ensure_model_loaded()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an enterprise-grade search query generator.\\n"
                    "Your output feeds directly into an automated search engine.\\n\\n"
                    "RULES:\\n"
                    "- Output ONE single-line search query\\n"
                    "- NO explanations\\n"
                    "- NO prefixes or labels\\n"
                    "- NO punctuation except hyphens\\n"
                    "- Focus on standards benchmarks definitions\\n"
                    "- Use professional technical keywords only"
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Question: {user_query}\\n"
                    f"Domain Context: {context[:200]}"
                )
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,          # ← Deterministic for search
            temperature=0.0,
            repetition_penalty=1.05
        )

        raw_query = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        # Final hard sanitation layer (never trust the model blindly)
        query = (
            raw_query
            .replace("\\n", " ")
            .replace('"', "")
            .strip()
            .lower()
        )

        logger.info(f"Search Query Generated: {query}")
        return query

    def stream_inference(self, message: str, context: str = ""):
        """
        Enterprise-grade streaming response with lifecycle signaling
        and strict reasoning discipline.
        """
        self._ensure_model_loaded()

        if not context or len(context.strip()) < 10:
            system_prompt = (
                "You are an expert AI assistant.\\n"
                "No internal documents were found.\\n"
                "Answer confidently using general professional knowledge.\\n"
                "Clearly indicate when reasoning is based on general knowledge."
            )
            user_prompt = f"Question:\\n{message}"
        else:
            system_prompt = (
                "You are an Enterprise AI Document Intelligence Analyst.\\n\\n"
                "DOCUMENT RULES:\\n"
                "- Local documents are ground truth\\n"
                "- External knowledge is for benchmarks only\\n"
                "- Never mix the two implicitly\\n"
                "- Explicitly state when information is missing\\n"
                "- Maintain professional, confident tone"
            )
            user_prompt = f"Context:\\n{context}\\n\\nQuestion:\\n{message}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
            max_length=8192
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.25,         # ← lower for factual streaming
            repetition_penalty=1.1,
            max_time=120.0
        )

        def _generate():
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # Stream lifecycle signaling (frontend gold)
        yield "[STREAM_START]\\n"

        for token in streamer:
            yield token

        yield "\\n[STREAM_END]"

    def classify_intent(self, query: str) -> str:
        """
        Classifies the user query into a specific task intent.
        Returns: 'ATS_ESTIMATION', 'GAP_ANALYSIS', 'RESUME_ANALYSIS', 'GENERAL_CHAT', or 'SEARCH_QUERY'
        """
        prompt = f"""You are an intent classifier for a Resume Analysis AI.
        Classify the following user query into exactly ONE category:
        
        1. ATS_ESTIMATION: User asks for an ATS score, parse rate, or keyword match.
        2. GAP_ANALYSIS: User asks about time gaps, employment breaks, or timeline issues.
        3. RESUME_ANALYSIS: User asks for general feedback, improvements, or critique of the resume.
        4. SEARCH_QUERY: User asks for EXTERNAL market data, salaries, or specific tech trends (requires web search).
        5. GENERAL_CHAT: Greetings, clarification, or questions not about the resume.

        Query: "{query}"

        Return ONLY the category name (e.g. ATS_ESTIMATION). Do not add any punctuation.
        """
        
        messages = [
            {"role": "system", "content": "You are a precise intent classifier."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Quick inference for classification
            intent = self._run_inference(messages, max_new_tokens=10)
            intent = intent.strip().upper()
            
            # Fallback for hallway hallucinations
            valid_intents = {"ATS_ESTIMATION", "GAP_ANALYSIS", "RESUME_ANALYSIS", "SEARCH_QUERY", "GENERAL_CHAT"}
            if intent not in valid_intents:
                # Basic keyword fallback
                q = query.lower()
                if "ats" in q or "score" in q: return "ATS_ESTIMATION"
                if "gap" in q or "break" in q: return "GAP_ANALYSIS"
                if "salary" in q or "market" in q: return "SEARCH_QUERY"
                # If short query, assume general chat or follow up
                if len(query.split()) < 3: return "GENERAL_CHAT"
                return "RESUME_ANALYSIS"
                
            return intent
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "RESUME_ANALYSIS"

    def rewrite_query(self, original_query: str, chat_history: List[dict] = None) -> str:
        """
        Rewrite the query to be self-contained for vector retrieval, using chat history for context.
        """
        if not chat_history:
            chat_history = []
            
        # Format history for context (last 3 turns)
        history_context = ""
        if chat_history:
            # Filter out system messages and large context blocks if stored
            recent_history = [m for m in chat_history if m['role'] in ('user', 'assistant')][-3:]
            # Helper to truncate content
            def trunc(s): return s[:200] + "..." if len(s) > 200 else s
            history_text = "\n".join([f"{msg['role']}: {trunc(msg['content'])}" for msg in recent_history])
            history_context = f"Conversation History:\n{history_text}\n"

        prompt = f"""{history_context}
        Current Query: "{original_query}"
        
        Task: Rewrite the current query to be a standalone search query for a vector database.
        - Resolve pronouns (e.g., "it", "mine", "his") using the history.
        - If the user says "What about mine", and the history discusses "Ahmed's Resume", rewrite to "Ahmed's Resume".
        - Improve keywords for retrieval.
        - Do NOT answer the question. ONLY return the rewritten query string.
        
        Rewritten Query:"""
        
        messages = [
            {"role": "system", "content": "You are a query rewriting engine. output ONLY the rewritten query."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._run_inference(messages, max_new_tokens=64)
            return response.strip().replace('"', '')
        except Exception:
            return original_query

    def get_task_prompt(self, intent: str) -> str:
        """Returns the specialized system prompt for the given intent."""
        
        base_prompt = """You are an Enterprise AI Document Intelligence & Reasoning Platform.

Your role is to help users achieve their objective with a provided document
(resume, legal document, invoice, policy, technical file, or similar).

CORE PRINCIPLES:
1. The uploaded document is the SINGLE source of truth.
2. Never invent, assume, or repeat information.
3. Answer ONLY what the user asks — no extra analysis unless requested.
4. If information is missing, unclear, or ambiguous, explicitly say so.
5. Only search the web if the document itself is insufficient AND
   the question requires external validation or enrichment.

EVIDENCE GATING & SENIORITY CALIBRATION:
- Do not extrapolate seniority. (e.g., "Student Lead" ≠ "Engineering Manager").
- Distinguish clearly between "Internship/Student Project" and "Full-Time Industry Experience".
- If a role is "Member" or "Participant", do not frame it as "Strategic Leadership" unless explicitly proven by bullets.

DOCUMENT AWARENESS:
- Treat the document as persistent across follow-up questions.
- Do not re-summarize the document unless explicitly asked.
- Do not duplicate entries or infer missing dates, roles, or facts.

QUESTION HANDLING RULES:
- Factual question → factual answer only.
- Evaluative question → compare against the stated objective.
- Gap/missing question → identify absence without guessing.
- Improvement question → provide actionable, document-grounded feedback.

WEB SEARCH RULES:
- Trigger web search ONLY if:
  a) Required information is not present in the document, AND
  b) The question cannot be answered without external data.
- Use only credible, ranked sources.
- Clearly separate document facts from external findings.

OUTPUT RULES:
- Be concise, structured, and accurate.
- Do not repeat content unnecessarily.
- Never add assumptions, praise, or critique unless requested."""

        if intent == "ATS_ESTIMATION":
            return f"""{base_prompt}
            
            TASK: Estimate ATS Compatibility based on industry heuristics.
            RESPONSE FORMAT:
            - ATS Match Level: [High / Medium / Low] (NOT an arbitrary number)
            - Role-Specific Fit: [Assessment against target role keywords]
            - Parsing Quality: [High/Low] (Layout, tables, fonts)
            - Keyword Analysis: [Critical missing vs present terms]
            - Breakdown: [Brief bullet points]
            
            DO NOT output a fake "Score out of 100".
            Do NOT search the web. Use internal knowledge of ATS systems."""
            
        elif intent == "GAP_ANALYSIS":
            return f"""{base_prompt}
            
            TASK: Identify time gaps in work history. Do not evaluate skills or fit.
            RESPONSE FORMAT:
            - Chronological Timeline: [List start-end dates]
            - Gaps Identified: [List gaps > 3 months with specific dates]
            - Analysis: [Brief grounded comment on continuity]
            If no gaps exist, state: "No significant employment gaps found."
            Do NOT search the web."""
            
        elif intent == "RESUME_ANALYSIS":
            return f"""{base_prompt}
            
            TASK: Detailed Resume Critique for Impact and Clarity.
            RESPONSE FORMAT:
            - Executive Summary: [2 lines - Be Honest about Seniority Level]
            - Key Strengths: [3 bullets]
            - Critical Weaknesses: [3 bullets]
            - Actionable Improvements: [Specific edits to make]
            
            Calibration Rule: Clearly distinguish between Internships and Full-Time roles. 
            Do not overstate student leadership as corporate management."""
            
        elif intent == "SEARCH_QUERY":
            return f"""{base_prompt}
            
            TASK: Enriched External Information Retrieval.
            User has explicitly requested outside information.
            Synthesize search results with document context if relevant.
            Use header: EXTERNAL BENCHMARK INFORMATION (FOR REFERENCE ONLY)."""

        else:
            return base_prompt

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
