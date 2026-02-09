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
        self.model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._model_lock = threading.Lock()
        
        # Pre-compiled prompt templates (Phase 13 optimization)
        self._prompts = self._compile_prompts()
        
    def _compile_prompts(self) -> Dict[str, str]:
        """Compile all system prompts for consistency"""
        return {
            "base_persona": """You are an Enterprise Document Intelligence Engine.

ABSOLUTE RULES (VIOLATION PROHIBITED):
1. SOURCE TRUTH: Use ONLY information explicitly present in the provided document.
2. MISSING DATA: If information is absent, state exactly: "The document does not mention [X]."
3. NO INFERENCE: Never assume, infer, or extrapolate beyond explicit text.
4. NO EXTERNAL: Do not use training data or external knowledge unless explicitly flagged as [EXTERNAL].
5. CITATION: Every claim must reference specific document sections/evidence.
6. SCOPE BOUNDARY: Analyze ONLY within the specified scope sections.

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
            
        if any(k in q for k in ["salary", "market", "industry standard", "trend", "news", "external"]):
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
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
            }
        ]

    def generate(
        self,
        system_prompt: str,
        document_context: str,
        question: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2
    ) -> str:
        """
        Synchronous generation with full safety controls.
        Use for: ATS scoring, gap analysis, short answers.
        """
        self._ensure_loaded()
        
        messages = self._prepare_messages(system_prompt, document_context, question)
        
        # Apply chat template
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
        temperature: float = 0.25
    ) -> Generator[str, None, None]:
        """
        Streaming generation for UX responsiveness.
        Use for: Long-form analysis, improvement suggestions.
        """
        self._ensure_loaded()
        
        messages = self._prepare_messages(system_prompt, document_context, question)
        
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
            repetition_penalty=1.1,
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