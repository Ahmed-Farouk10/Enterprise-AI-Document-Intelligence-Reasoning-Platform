# app/services/llm_service.py
import os
import re
import torch
import logging
import threading
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for document analysis tasks"""
    intent: str
    depth: str  # FACTUAL, EVALUATIVE, IMPROVEMENT
    scope: List[str]
    allow_external_search: bool = False
    require_citations: bool = True


class LLMService:
    """
    Enterprise Document Intelligence Engine.
    
    Core Principles:
    1. Document is ground truth - no hallucination permitted
    2. Intent-driven prompt selection - single source of truth
    3. Evidence-gated generation - no output without context
    4. Deterministic scoring - LLM explains, doesn't score arbitrarily
    """
    
    # Intent Constants
    INTENT_SUMMARY = "SUMMARY"
    INTENT_FACTUAL = "FACTUAL"
    INTENT_EVALUATIVE = "EVALUATIVE"
    INTENT_IMPROVEMENT = "IMPROVEMENT"
    INTENT_GAP_ANALYSIS = "GAP_ANALYSIS"
    INTENT_SCORING = "SCORING"
    INTENT_SEARCH = "SEARCH_QUERY"
    INTENT_GENERAL = "GENERAL_CHAT"
    
    # Depth Constants
    DEPTH_FACTUAL = "FACTUAL"
    DEPTH_EVALUATIVE = "EVALUATIVE"
    DEPTH_IMPROVEMENT = "IMPROVEMENT"
    
    def __init__(self):
        # TIER 3 UPGRADE: Qwen2.5-32B for production-grade accuracy
        # Previous: microsoft/Phi-3-mini-4k-instruct (hallucination rate ~40%)
        # Qwen2.5-32B: hallucination rate ~15-20%, better document analysis
        self.model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._model_lock = threading.Lock()
        
        # Pre-compiled prompt templates (Phase 13 optimization)
        self._prompts = self._compile_prompts()
        
    def _compile_prompts(self) -> Dict[str, str]:
        """Compile all system prompts for consistency"""
        return {
            "base_persona": """[CRITICAL INSTRUCTION] IF THE DOCUMENT CONTEXT DOES NOT CONTAIN THIS INFORMATION, YOU MUST SAY "The document does not mention [X]". NEVER INVENT INFORMATION. NEVER USE YOUR TRAINING DATA.

You are the intelligent assistant for the Enterprise AI Document Intelligence Platform. 
Your role is to help users analyze, extract, and reason over their documents (resumes, legal contracts, technical manuals, etc.). 
You have a good, funny, humanly personality—friendly, witty, and concise. But when it comes to facts about the document, you are DEADLY SERIOUS and strictly accurate. 
Keep your responses well-balanced: not too long, not too short.

ABSOLUTE RULES (VIOLATION PROHIBITED):
1. SOURCE TRUTH: Use ONLY information explicitly present in the provided document.
2. MISSING DATA: If information is absent, state exactly: "The document does not mention [X]."
3. NO INFERENCE: Never assume, infer, or extrapolate beyond explicit text.
4. NO EXTERNAL: Do not use training data or external knowledge unless explicitly flagged as [EXTERNAL RESULTS from Web Search].
5. CITATION: Every claim must reference specific document sections/evidence.
6. SCOPE BOUNDARY: Analyze ONLY within the specified scope sections.
7. HALLUCINATION CHECK: Before outputting ANY fact, verify it exists in the context.

DOCUMENT TYPE: Generic (Auto-detected: Resume, Legal, Technical, Medical, Academic)
""",
            
            "factual_mode": """
MODE: FACTUAL EXTRACTION
TASK: Extract specific information directly stated in the document.
OUTPUT RULES:
- Provide exact quotes or close paraphrases with citations.
- If not found: "The document does not mention this information."
- No analysis, no judgment, no recommendations.
""",
            
            "evaluative_mode": """
MODE: EVALUATIVE ANALYSIS
TASK: Assess document quality, completeness, or fit against standards.
OUTPUT STRUCTURE:
1. VERIFIED EVIDENCE: (Items explicitly found in document with citations)
2. MISSING STANDARD ELEMENTS: (Critical items standard for domain but absent)
3. CRITICAL GAPS: (Logical inconsistencies, timeline issues, structural problems)
4. VERDICT: (Evidence-based assessment only)

RULE: Do not invent standards. Use only domain-typical expectations.
""",
            
            "improvement_mode": """
MODE: IMPROVEMENT & OPTIMIZATION
TASK: Provide specific, actionable improvements to the document.
OUTPUT STRUCTURE:
1. ISSUES IDENTIFIED: (Specific problems with evidence citations)
2. CONCRETE IMPROVEMENTS: (Actionable changes, not generic advice)
3. EXAMPLE REWRITE: (Specific section improvement, if applicable)

RULE: All suggestions must be grounded in document content. No generic templates.
""",
            
            "gap_analysis_mode": """
MODE: TEMPORAL & INFORMATIONAL GAP ANALYSIS
TASK: Identify significant gaps in timelines or missing critical information.
GAP CRITERIA:
- Employment gaps: >3 months without overlapping education/projects
- Missing sections: Standard for document type but absent
- Logical gaps: Inconsistent dates, missing prerequisites

OUTPUT:
- List each gap with specific date ranges or section references
- Note: "No significant gaps detected" only if document is complete
""",
            
            "scoring_mode": """
MODE: STRUCTURED SCORING
TASK: Provide numerical evaluation based on explicit document criteria.

SCORING METHODOLOGY:
1. Identify measurable criteria present in document
2. Count/verify each criterion against evidence
3. Calculate: (Met Criteria / Total Criteria) × 100

OUTPUT FORMAT:
- Score: [X]% (calculation shown)
- Criteria Breakdown: (List with evidence)
- Missing: (Unscored due to absent data)
- Confidence: (High/Medium/Low based on document completeness)

WARNING: If document lacks measurable criteria, return "Scoring not possible - insufficient quantitative data."
""",
            
            "search_integration": """
EXTERNAL SEARCH INTEGRATION:
The following [EXTERNAL] information was retrieved to provide benchmarks.
EXTERNAL data is for context only - never treated as document fact.
Clearly label: "[EXTERNAL BENCHMARK]" vs "[DOCUMENT FACT]"
"""
        }

    # ==================== MODEL LIFECYCLE ====================j
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def load_model(self) -> None:
        """Thread-safe model loading with retry logic"""
        with self._model_lock:
            if self.model is not None:
                return
            
            # Check if running on HuggingFace Spaces (limited resources)
            is_hf_spaces = os.getenv("SPACE_ID") or os.getenv("HF_HOME", "").startswith("/app/.cache")
            
            # Tiny models that are safe to load in 16GB free-tier HF Spaces RAM
            tiny_models = ["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/phi-1_5"]
            
            if is_hf_spaces and self.model_name not in tiny_models:
                logger.warning(f"🌐 Running on HuggingFace Spaces - Skipping local model load for large model {self.model_name}")
                logger.warning("💡 Will use HuggingFace Inference API for completions")
                
                # MEMORY OPTIMIZATION: Skip tokenizer if low memory mode is enabled
                if os.getenv("RAG_LOW_MEMORY_MODE") == "true":
                     logger.info("⚡ Low Memory Mode: Skipping local tokenizer load (Will rely on API)")
                     return

                # Skip for OpenRouter or non-HF Repo ID formats
                provider = os.getenv("LLM_PROVIDER", "").lower()
                if provider in ["openrouter", "openai"] or "/" in self.model_name and not self.model_name.startswith(("Qwen", "microsoft", "meta-llama", "mistralai")):
                     logger.info(f"⚡ Skipping local tokenizer load for remote provider API model: {self.model_name}")
                     return

                # Only load tokenizer for text processing
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    logger.info("✅ Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Tokenizer loading failed: {e}")
                    # Even tokenizer can fail on HF spaces, that's OK
                return
                
            logger.info(f"Loading LLM: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                # Determine torch dtype safely
                use_bf16 = False
                if torch.cuda.is_available():
                    use_bf16 = torch.cuda.is_bf16_supported()
                elif hasattr(torch.cpu, 'is_bf16_supported'):
                    use_bf16 = torch.cpu.is_bf16_supported()
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                raise

    def _ensure_loaded(self) -> None:
        """Lazy loading wrapper"""
        if self.model is None:
            self.load_model()

    def warmup(self) -> None:
        """Eager loading for production deployment"""
        self._ensure_loaded()
        logger.info("LLM Service warmed up and ready")
    
    # ==================== CIRCUIT BREAKER ====================
    
    _failure_count: int = 0
    _last_failure_time: float = 0
    _circuit_open: bool = False
    CB_THRESHOLD: int = 5
    CB_RESET_TIMEOUT: int = 60  # seconds

    def _check_circuit(self):
        """Check if circuit is open and if it should reset."""
        import time
        if self._circuit_open:
            if time.time() - self._last_failure_time > self.CB_RESET_TIMEOUT:
                logger.info("🔄 Circuit Breaker HALF-OPEN: Testing service...")
                self._circuit_open = False
                self._failure_count = 0
                return
            raise Exception("Circuit Breaker OPEN: Service unavailable due to repeated failures.")

    def _record_failure(self):
        """Record a failure and potentially open circuit."""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        logger.warning(f"⚠️ LLM Service Failure ({self._failure_count}/{self.CB_THRESHOLD})")
        
        if self._failure_count >= self.CB_THRESHOLD:
            self._circuit_open = True
            logger.error("🔥 Circuit Breaker TRIPPED: Stopping requests for 60s")

    def _record_success(self):
        """Reset failure count on success."""
        if self._failure_count > 0:
            self._failure_count = 0
            self._circuit_open = False
            logger.info("✅ Circuit Breaker RESET: Service healthy")

    def _generate_via_gemini_direct(self, prompt: Any, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Fallback for Gemini models on HF Spaces using google-generativeai"""
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            
            # If no explicit Gemini key, check if LLM_API_KEY exists and is not a HF token
            if not api_key:
                fallback_key = os.environ.get("LLM_API_KEY", "")
                if fallback_key and not fallback_key.startswith("hf_") and fallback_key != "local":
                    api_key = fallback_key

            if not api_key or api_key == "local" or api_key.startswith("hf_"):
                 raise ValueError("No valid Google AI API key found. Please add 'GEMINI_API_KEY' to your Hugging Face Space Variables & Secrets.")
                 
            genai.configure(api_key=api_key)
            
            # Extract just the model name if it has the litellm "gemini/" prefix
            model_name = self.model_name.replace("gemini/", "") if self.model_name.startswith("gemini/") else self.model_name
            if not model_name or model_name == "gemini":
                model_name = "gemini-2.0-flash"
                
            model = genai.GenerativeModel(model_name)
            
            # Combine role messages safely for native gemini SDK
            if isinstance(prompt, list):
                text_prompt = ""
                for msg in prompt:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")
                    text_prompt += f"{role}: {content}\n\n"
            else:
                text_prompt = str(prompt)
                
            response = model.generate_content(
                text_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Native Gemini API failed: {e}")
            raise e

    def _generate_via_gemini_stream(self, prompt: Any, max_tokens: int = 1024, temperature: float = 0.3) -> Generator[str, None, None]:
        """Streaming fallback for Gemini models via native SDK"""
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            
            if not api_key:
                fallback_key = os.environ.get("LLM_API_KEY", "")
                if fallback_key and not fallback_key.startswith("hf_") and fallback_key != "local":
                    api_key = fallback_key

            if not api_key or api_key == "local" or api_key.startswith("hf_"):
                 raise ValueError("No valid Google AI API key found. Please add 'GEMINI_API_KEY' to your Hugging Face Space Variables & Secrets.")
                 
            genai.configure(api_key=api_key)
            
            model_name = self.model_name.replace("gemini/", "") if self.model_name.startswith("gemini/") else self.model_name
            if not model_name or model_name == "gemini":
                model_name = "gemini-2.0-flash"
                
            model = genai.GenerativeModel(model_name)
            
            if isinstance(prompt, list):
                text_prompt = ""
                for msg in prompt:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")
                    text_prompt += f"{role}: {content}\n\n"
            else:
                text_prompt = str(prompt)
                
            yield "[STREAM_START]\n"
            
            response = model.generate_content(
                text_prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
            yield "\n[STREAM_END]"
            
        except Exception as e:
            logger.error(f"❌ Native Gemini API Stream failed: {e}")
            yield f"\n⚠️ [ERROR] Gemini API failed: {str(e)}"
            yield "\n[STREAM_END]"

    def _generate_via_inference_api(
        self,
        prompt: Any,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
        raise_errors: bool = False
    ) -> str:
        """
        Generate using HuggingFace Inference API with Circuit Breaker.
        """
        self._check_circuit()
        
        try:
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            
            # Format inputs as messages for the 'conversational' task
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            
            # Use chat_completion for the conversational task
            response = client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            self._record_success()
            return response.choices[0].message.content
            
        except Exception as e:
            self._record_failure()
            error_msg = str(e)
            logger.error(f"HF Inference API failed: {error_msg}")
            
            if raise_errors:
                raise e
            
            if "403 Forbidden" in error_msg and "Inference Providers" in error_msg:
                advice = (
                    "\n\n🔑 TIP: Your Hugging Face fine-grained token needs 'Make calls to Inference Providers' permission enabled. "
                    "Alternatively, enable 'Make calls to the serverless Inference API'."
                )
                return f"I apologize, but I'm unable to authenticate with the Inference API. {advice}"
            
            # If circuit tripped, re-raise to stop immediate retries higher up
            if self._circuit_open:
                return "⚠️ System Alert: LLM Service is temporarily unavailable. Please try again in a minute."
                
            return "I apologize, but I'm currently unable to process this request due to system limitations. Please try again later."

    def _generate_via_inference_api_stream(
        self,
        prompt: Any,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> Generator[str, None, None]:
        """
        Streaming generation via HuggingFace Inference API.
        """
        self._check_circuit()
        
        try:
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            
            # Format inputs
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            
            yield "[STREAM_START]\n"
            
            # Use chat_completion with stream=True
            for chunk in client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            ):
                token = chunk.choices[0].delta.content
                if token:
                    yield token
            
            yield "\n[STREAM_END]"
            self._record_success()
            
            yield "\n[STREAM_END]"
            self._record_success()
            
        except Exception as e:
            self._record_failure()
            logger.error(f"HF Inference API Stream failed: {e}")
            yield f"\n⚠️ [ERROR] Inference API failed: {str(e)}"
            yield "\n[STREAM_END]"

    def _get_openai_client(self):
        """Helper to get an OpenAI compatible client based on environment"""
        import openai
        provider = os.getenv("LLM_PROVIDER", "").lower()
        
        # 1. Check Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key or provider == "groq":
             safe_key = groq_key or "local"
             logger.info(f"⚡ Groq Init | Provider: {provider} | Key Prefix: {str(safe_key)[:6]}...")
             return openai.OpenAI(
                 base_url="https://api.groq.com/openai/v1",
                 api_key=safe_key
             )
        
        # 2. Check OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY") 
        if not api_key:
             llm_key = os.getenv("LLM_API_KEY", "")
             # If provider is explicitly openrouter or key looks like one, inherit the key
             if llm_key.startswith("sk-or-") or provider == "openrouter":
                 api_key = llm_key
                 
        if api_key or provider == "openrouter":
             safe_key = api_key or "local"
             logger.info(f"⚡ OpenRouter Init | Provider: {provider} | Key Prefix: {str(safe_key)[:6]}... | Key Len: {len(str(safe_key))}")
             return openai.OpenAI(
                 base_url="https://openrouter.ai/api/v1",
                 api_key=safe_key  # Prevent 'dummy' since 'local' will be caught better by underlying errors if it breaks
             )
             
        # 3. Defaults to Generic OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        return openai.OpenAI(
            base_url=base_url if base_url else None,
            api_key=api_key or "local"
        )

    def _generate_via_openai_compatible(self, prompt: Any, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Route to OpenAI-compatible endpoints like OpenRouter."""
        try:
            client = self._get_openai_client()
            
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
                
            model_name = self.model_name.replace("openrouter/", "").replace("groq/", "")
            
            # Default Groq model if unset or generic
            if os.getenv("LLM_PROVIDER", "").lower() == "groq" and model_name in ["", "groq", "Qwen/Qwen2.5-7B-Instruct"]:
                model_name = "llama-3.3-70b-versatile"
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0.5,
                stop=["<|eot_id|>", "</s>", "<|im_end|>", "<eos>"]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ OpenAI Compatible API failed: {e}")
            raise e

    def _generate_via_openai_compatible_stream(self, prompt: Any, max_tokens: int = 1024, temperature: float = 0.3) -> Generator[str, None, None]:
        """Streaming route for OpenAI-compatible endpoints."""
        try:
            client = self._get_openai_client()
            
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
                
            model_name = self.model_name.replace("openrouter/", "").replace("groq/", "")
            
            # Default Groq model if unset or generic
            if os.getenv("LLM_PROVIDER", "").lower() == "groq" and model_name in ["", "groq", "Qwen/Qwen2.5-7B-Instruct"]:
                model_name = "llama-3.3-70b-versatile"
            
            yield "[STREAM_START]\n"
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0.5,
                stop=["<|eot_id|>", "</s>", "<|im_end|>", "<eos>"],
                stream=True
            )
            
            for chunk in response:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
                    
            yield "\n[STREAM_END]"
            
        except Exception as e:
            logger.error(f"❌ OpenAI Compatible API Stream failed: {e}")
            yield f"\n⚠️ [ERROR] OpenAI API failed: {str(e)}"
            yield "\n[STREAM_END]"

    def _generate_via_groq_native(self, prompt: Any, max_tokens: int = 4096, temperature: float = 0.6) -> str:
        """Native Groq generation using groq package."""
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
            
        model_name = self.model_name.replace("groq/", "")
        
        # Default Groq model if unset or generic
        if os.getenv("LLM_PROVIDER", "").lower() == "groq" and model_name in ["", "groq", "Qwen/Qwen2.5-7B-Instruct"]:
            model_name = "qwen-2.5-32b"
            
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "top_p": 0.95,
            }
            if "deepseek" in model_name.lower() or "qwen" in model_name.lower():
                kwargs["reasoning_effort"] = "default"
                
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Native Groq API failed: {e}")
            raise e

    def _generate_via_groq_native_stream(self, prompt: Any, max_tokens: int = 4096, temperature: float = 0.6) -> Generator[str, None, None]:
        """Native Groq streaming generation using groq package."""
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
            
        model_name = self.model_name.replace("groq/", "")
        
        # Default Groq model if unset or generic
        if os.getenv("LLM_PROVIDER", "").lower() == "groq" and model_name in ["", "groq", "Qwen/Qwen2.5-7B-Instruct"]:
            model_name = "qwen-2.5-32b"
            
        try:
            yield "[STREAM_START]\n"
            
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "top_p": 0.95,
                "stream": True,
                "stop": None
            }
            if "deepseek" in model_name.lower() or "qwen" in model_name.lower():
                kwargs["reasoning_effort"] = "default"
                
            response = client.chat.completions.create(**kwargs)
            
            for chunk in response:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
                    
            yield "\n[STREAM_END]"
        except Exception as e:
            logger.error(f"❌ Native Groq API Stream failed: {e}")
            yield f"\n⚠️ [ERROR] Groq API failed: {str(e)}"
            yield "\n[STREAM_END]"

    def _generate_via_groq(
        self,
        prompt: Any,
        model: str = "groq/llama3-70b-8192",
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> str:
        """
        Generate using Groq API via LiteLLM.
        """
        try:
            import litellm
            # litellm.drop_params = True # prevent sending unsupported params if any
            
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY not found in environment")
                
            # Format inputs
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            logger.info(f"⚡ Calling Groq API ({model}) via LiteLLM...")
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=os.getenv("GROQ_API_KEY")
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Groq API failed: {e}")
            raise e

    # ==================== INTENT & SCOPE CLASSIFICATION ====================
    
    def classify_intent(self, query: str) -> str:
        """
        Deterministic intent classification with keyword fallback.
        Returns stable intent categories for routing.
        """
        q = query.lower().strip()
        
        # Direct pattern matching (deterministic layer)
        if any(k in q for k in ["summarize", "summary", "tldr", "overview"]):
            return self.INTENT_SUMMARY
            
        if any(k in q for k in ["score", "percentage", "rate", "grade", "percent", "ats"]):
            return self.INTENT_SCORING
            
        if any(k in q for k in ["gap", "break", "missing time", "unemployed", "between jobs"]):
            return self.INTENT_GAP_ANALYSIS
            
        if any(k in q for k in ["improve", "rewrite", "enhance", "optimize", "better", "fix"]):
            return self.INTENT_IMPROVEMENT
            
        if any(k in q for k in ["evaluate", "assess", "fit", "suitable", "good for", "compare"]):
            return self.INTENT_EVALUATIVE
            
        if any(k in q for k in ["search", "web", "find online", "google", "look up", "salary", "market", "industry standard", "trend", "news", "external", "tavily"]):
            return self.INTENT_SEARCH
            
        if any(k in q for k in ["what is", "when did", "where", "who", "how many", "list"]):
            return self.INTENT_FACTUAL
            
        if len(q.split()) < 4:
            return self.INTENT_GENERAL
            
        return self.INTENT_EVALUATIVE  # Default for complex queries

    def classify_depth(self, query: str) -> str:
        """
        Classify required reasoning depth.
        IMPROVEMENT = generative rewriting (highest risk, needs guardrails)
        """
        q = query.lower()
        
        improvement_keywords = [
            "improve", "rewrite", "write", "draft", "create", "generate",
            "enhance", "optimize", "polish", "refine", "make better"
        ]
        
        if any(k in q for k in improvement_keywords):
            return self.DEPTH_IMPROVEMENT
            
        evaluative_keywords = [
            "evaluate", "assess", "analyze", "review", "critique",
            "fit", "suitable", "recommend", "suggestion"
        ]
        
        if any(k in q for k in evaluative_keywords):
            return self.DEPTH_EVALUATIVE
            
        return self.DEPTH_FACTUAL

    def detect_scope(self, query: str) -> List[str]:
        """
        Detect which document sections are relevant.
        Returns list of scope identifiers.
        """
        q = query.lower()
        scopes = []
        
        # Temporal/Experience scopes
        if any(k in q for k in ["work", "job", "experience", "employment", "career", "position"]):
            scopes.append("WORK_HISTORY")
            
        if any(k in q for k in ["education", "degree", "university", "college", "school", "graduated"]):
            scopes.append("EDUCATION")
            
        if any(k in q for k in ["skill", "technology", "tool", "language", "framework", "proficiency"]):
            scopes.append("SKILLS")
            
        if any(k in q for k in ["project", "portfolio", "achievement", "accomplishment"]):
            scopes.append("PROJECTS")
            
        if any(k in q for k in ["certification", "certificate", "license", "accreditation"]):
            scopes.append("CERTIFICATIONS")
            
        if any(k in q for k in ["leadership", "managed", "led", "team", "supervised"]):
            scopes.append("LEADERSHIP")
            
        # Document quality scopes
        if any(k in q for k in ["format", "layout", "structure", "template", "design"]):
            scopes.append("FORMATTING")
            
        if any(k in q for k in ["keyword", "ats", "parse", "machine readable"]):
            scopes.append("ATS_COMPATIBILITY")
            
        if not scopes:
            scopes.append("ENTIRE_DOCUMENT")
            
        return scopes

    def extract_scope_context(self, full_document: str, scopes: List[str]) -> str:
        """
        Extract relevant sections from document based on scope.
        This is the CRITICAL anti-hallucination filter.
        """
        if "ENTIRE_DOCUMENT" in scopes or not scopes:
            return full_document
            
        # Simple section extraction (can be enhanced with NLP)
        sections = []
        lines = full_document.split('\n')
        current_section = "GENERAL"
        section_buffer = []
        
        section_keywords = {
            "WORK_HISTORY": ["experience", "employment", "work history", "professional experience", "career"],
            "EDUCATION": ["education", "academic", "degree", "university", "qualification"],
            "SKILLS": ["skills", "technologies", "competencies", "proficiencies", "expertise"],
            "PROJECTS": ["projects", "portfolio", "accomplishments"],
            "CERTIFICATIONS": ["certifications", "certificates", "licenses"],
            "LEADERSHIP": ["leadership", "management", "supervision"]
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            detected_section = None
            for scope, keywords in section_keywords.items():
                if any(kw in line_lower for kw in keywords) and len(line) < 100:  # Likely header
                    detected_section = scope
                    break
                    
            if detected_section:
                # Save previous section if it was in scope
                if current_section in scopes and section_buffer:
                    sections.append(f"\n=== {current_section} ===\n" + '\n'.join(section_buffer))
                current_section = detected_section
                section_buffer = []
            else:
                section_buffer.append(line)
                
        # Don't forget last section
        if current_section in scopes and section_buffer:
            sections.append(f"\n=== {current_section} ===\n" + '\n'.join(section_buffer))
            
        return '\n'.join(sections) if sections else full_document

    # ==================== PROMPT BUILDING ====================
    
    def build_system_prompt(self, config: AnalysisConfig) -> str:
        """
        Build the complete system prompt from config.
        SINGLE SOURCE OF TRUTH - no other prompts allowed in generation.
        """
        parts = [self._prompts["base_persona"]]
        
        # Add mode-specific instructions
        if config.depth == self.DEPTH_FACTUAL:
            parts.append(self._prompts["factual_mode"])
        elif config.depth == self.DEPTH_EVALUATIVE:
            parts.append(self._prompts["evaluative_mode"])
        elif config.depth == self.DEPTH_IMPROVEMENT:
            parts.append(self._prompts["improvement_mode"])
            
        # Add intent-specific overrides
        if config.intent == self.INTENT_GAP_ANALYSIS:
            parts.append(self._prompts["gap_analysis_mode"])
        elif config.intent == self.INTENT_SCORING:
            parts.append(self._prompts["scoring_mode"])
            
        # Add search integration notice if external data present
        if config.allow_external_search:
            parts.append(self._prompts["search_integration"])
            
        # Inject scope constraint
        scope_str = ", ".join(config.scope)
        parts.append(f"\nANALYSIS SCOPE: {scope_str}\n")
        
        # Add citation requirement
        if config.require_citations:
            parts.append("CITATION FORMAT: Use [Section: X] or [Line: Y] for every claim.\n")
            
        return "\n".join(parts)

    # ==================== CORE GENERATION ====================
    
    def _prepare_messages(
        self, 
        system_prompt: str, 
        context: str, 
        question: str
    ) -> List[Dict[str, str]]:
        """Prepare message list for chat template"""
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        
        # DEBUG: Log what we're sending to LLM
        logger.info(f"🔍 LLM Context Preview ({len(context)} chars): {context[:500]}...")
        
        # Format with clear boundary markers to help model distinguish context from question
        user_content = f"""DOCUMENT CONTEXT (THIS IS THE ONLY SOURCE OF TRUTH):
═══════════════════════════════════════════════════════════════
{context}
═══════════════════════════════════════════════════════════════

USER QUESTION: {question}

REMINDER: Answer ONLY from the text between the ═══ markers above. If not found in that exact text, you MUST say "The document does not mention this information."
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def generate(
        self,
        prompt: Any,
        max_tokens: int = 1024,
        temperature: float = 0.3,  # Lower = more deterministic
        top_p: float = 0.9
    ) -> str:
        """
        Generate completion using local model OR HuggingFace Inference API.
        
        Args:
            prompt: Either a string or a list of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)

        self._ensure_loaded()
        
        # If model not loaded (HF Spaces), use Inference API or Gemini SDK
        if self.model is None:
            provider = os.getenv("LLM_PROVIDER", "").lower()
            key = os.getenv("LLM_API_KEY", "")
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
            groq_key = os.getenv("GROQ_API_KEY", "")
            
            if provider == "groq" or groq_key:
                logger.info(f"⚡ Using Native Groq API for {self.model_name}")
                return self._generate_via_groq_native(prompt, max_tokens, temperature)
                
            if provider in ["openrouter", "openai"] or key.startswith("sk-or-") or openrouter_key:
                logger.info(f"⚡ Using OpenAI-compatible API for {self.model_name}")
                return self._generate_via_openai_compatible(prompt, max_tokens, temperature)
                
            if "gemini" in self.model_name.lower() or provider == "gemini":
                logger.info("⚡ Using Native Gemini SDK for generation")
                return self._generate_via_gemini_direct(prompt, max_tokens, temperature)
                
            logger.info("Using HuggingFace Inference API with chat completions")
            return self._generate_via_inference_api(prompt, max_tokens, temperature, top_p)

        
        # Local generation
        logger.debug(f"Generating with temperature={temperature}, max_tokens={max_tokens}")
        
        if isinstance(prompt, list):
            # Apply template if messages passed
            text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = str(prompt)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response.strip()

    def generate_stream(
        self,
        system_prompt: str,
        document_context: str,
        question: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.2  # Low enough for factual accuracy, high enough to prevent vocab loops
    ) -> Generator[str, None, None]:
        """
        Streaming generation for UX responsiveness.
        Use for: Long-form analysis, improvement suggestions.
        """
        self._ensure_loaded()
        
        messages = self._prepare_messages(system_prompt, document_context, question)

        # Fallback to external APIs if model not loaded
        if self.model is None:
            provider = os.getenv("LLM_PROVIDER", "").lower()
            key = os.getenv("LLM_API_KEY", "")
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
            groq_key = os.getenv("GROQ_API_KEY", "")
            
            if provider == "groq" or groq_key:
                logger.info(f"⚡ Using Native Groq API for streaming: {self.model_name}")
                yield from self._generate_via_groq_native_stream(
                    prompt=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return
                
            if provider in ["openrouter", "openai"] or key.startswith("sk-or-") or openrouter_key:
                logger.info(f"⚡ Using OpenAI-compatible API for streaming: {self.model_name}")
                yield from self._generate_via_openai_compatible_stream(
                    prompt=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return
                
            if "gemini" in self.model_name.lower() or provider == "gemini":
                logger.info("⚡ Using Native Gemini SDK for streaming")
                yield from self._generate_via_gemini_stream(
                    prompt=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return
                
            logger.info("Using HuggingFace Inference API for streaming")
            yield from self._generate_via_inference_api_stream(
                prompt=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            return
        
        # Local generation (original logic)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=16384  # Large context for long documents
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.3,  # Increased to prevent training data regurgitation
            top_p=0.9,  # Nucleus sampling for more focused responses
            top_k=50,   # Limit vocabulary to reduce hallucination
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Run generation in background thread
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True
        )
        thread.start()
        
        # Stream with lifecycle markers
        yield "[STREAM_START]\n"
        
        for token in streamer:
            yield token
            
        yield "\n[STREAM_END]"
        
        thread.join(timeout=1.0)

    # ==================== SPECIALIZED METHODS ====================
    
    def generate_search_query(self, user_query: str, document_summary: str = "") -> str:
        """
        Generate optimized search query for external data.
        Deterministic, keyword-focused output.
        """
        self._ensure_loaded()
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a precise web search query. "
                    "Output ONLY the query string. No explanations. "
                    "Focus on: industry standards, benchmarks, definitions. "
                    "Max 10 words."
                )
            },
            {
                "role": "user",
                "content": f"User Question: {user_query}\nDocument Context: {document_summary[:200]}"
            }
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # Deterministic
            temperature=0.0
        )
        
        query = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # Hard sanitation
        query = query.replace('"', '').replace("'", "").strip().lower()
        query = re.sub(r'\s+', ' ', query)
        
        return query[:100]  # Hard limit

    def verify_evidence(self, claim: str, context: str) -> Dict[str, Any]:
        """
        Self-RAG: Verify if claim is supported by context.
        Returns verification result with confidence.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Verify if the CLAIM is fully supported by the CONTEXT. "
                    "Answer ONLY: 'VERIFIED', 'PARTIAL', or 'UNSUPPORTED'. "
                    "Then explain in one sentence."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"
            }
        ]
        
        try:
            response = self.generate(
                system_prompt=messages[0]["content"],
                document_context=context,
                question=f"Verify this claim: {claim}",
                max_new_tokens=50,
                temperature=0.0
            )
            
            result = "UNSUPPORTED"
            if "verified" in response.lower():
                result = "VERIFIED"
            elif "partial" in response.lower():
                result = "PARTIAL"
                
            return {
                "status": result,
                "explanation": response,
                "claim": claim
            }
        except Exception as e:
            logger.error(f"Evidence verification failed: {e}")
            return {"status": "ERROR", "explanation": str(e), "claim": claim}

    # ==================== UTILITY ====================
    
    def get_service_health(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        return {
            "status": "healthy" if self.model else "uninitialized",
            "model_name": self.model_name,
            "loaded": self.model is not None
        }


# Singleton instance
llm_service = LLMService()