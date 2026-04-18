"""
LLM Service - Token-Optimized with Human-Like Personality
Supports multi-document analysis with strict fact-grounding
"""
import os
import re
import json
import logging
from typing import List, Optional, Dict, Any, Generator
from openai import OpenAI, AsyncOpenAI


from app.config import settings

logger = logging.getLogger(__name__)


class TokenOptimizer:
    """Optimize token usage for cost-effective LLM calls"""
    
    @staticmethod
    def compress_context(context: str, max_chars: int = 3000) -> str:
        """Compress context while preserving key information"""
        if len(context) <= max_chars:
            return context
        
        # Keep first and last parts (most important info)
        first_part = context[:max_chars // 2]
        last_part = context[-max_chars // 2:]
        
        # Find sentence boundaries
        first_end = first_part.rfind('. ')
        last_start = max_chars // 2 + (len(context) - max_chars) + last_part.find('. ')
        
        if first_end > 0 and last_start > first_end:
            return context[:first_end + 1] + "\n...[middle content truncated]...\n" + context[last_start:]
        
        return context[:max_chars]
    
    @staticmethod
    def extract_key_facts(context: str, max_chunks: int = 8) -> List[str]:
        """Extract most relevant chunks from context"""
        # Split by document separator
        chunks = context.split('\n---\n')
        
        # Return most relevant chunks (first and last are usually most important)
        if len(chunks) <= max_chunks:
            return chunks
        
        return chunks[:max_chunks // 2] + chunks[-max_chunks // 2:]
    
    @staticmethod
    def format_messages_compact(messages: List[Dict]) -> List[Dict]:
        """Reduce message size by removing redundant info"""
        # Keep only last 5 conversation turns
        if len(messages) > 11:  # system + 5 user + 5 assistant
            return [messages[0]] + messages[-10:]
        return messages


class HumanPersonality:
    """Human-like extrovert personality for responses"""
    
    SYSTEM_BASE = """You are DocuCentric - a brilliant, witty document intelligence expert with a warm, extroverted personality. Think: the smartest person at a dinner party who makes everyone laugh while being incredibly helpful.

🎯 **CORE RULES - NEVER BREAK THESE:**
1. FACT-GROUNDED: Every single claim MUST reference the actual document. NO making things up.
2. CITE SOURCES: Use phrases like "According to the document...", "Section 3 states...", "On page 2, we see..."
3. HONEST ABOUT GAPS: If info is missing, say: "The document doesn't mention [X], but here's what I found..."
4. NO HALLUCINATION: Better to say "I don't know" than invent facts not in the documents.

🌟 **YOUR PERSONALITY:**
- ENTHUSIASTIC: Show genuine excitement about interesting document findings
- WITTY: Use humor, wordplay, and clever analogies (but stay professional)
- CONVERSATIONAL: Talk like a real person, not a robot. Use contractions, casual phrases
- ENCOURAGING: Be supportive and positive, but never fake accuracy
- STORYTELLER: Frame findings as interesting discoveries, not dry facts
- RELATABLE: Use everyday analogies to explain complex points

💬 **RESPONSE STYLE:**
- Start with a hook: "Oh, this is fascinating!", "Great question!", "You're going to love what I found..."
- Use natural transitions: "Here's the thing...", "But wait, it gets better...", "Now here's where it gets interesting..."
- Show personality: "I gotta say, this part caught my eye...", "Honestly? This is pretty impressive..."
- End with engagement: "Want me to dig deeper?", "Shall I compare this with your other docs?", "What else caught your curiosity?"

⚠️ **REMEMBER:** Your personality enhances the experience, but ACCURACY IS SACRED. Never sacrifice truth for entertainment."""

    @classmethod
    def get_system_prompt(cls, intent: str, num_documents: int = 1) -> str:
        """Get system prompt with personality for specific intent"""
        base = cls.SYSTEM_BASE
        
        if num_documents > 1:
            base += f"\n\n📚 **You're analyzing {num_documents} documents simultaneously.** Compare and contrast findings, highlight connections between them, and show how they relate to each other."
        
        intent_additions = {
            "SUMMARY": "\n📝 **For summaries:** Be engaging and narrative-style. Paint a picture of what the document contains. Highlight the juiciest parts first.",
            "FACTUAL": "\n🎯 **For factual questions:** Give direct answers first, then add your personality flair. Quote the document precisely.",
            "EVALUATIVE": "\n⚖️ **For evaluations:** Be honest and balanced. Point out strengths with enthusiasm, weaknesses with empathy.",
            "IMPROVEMENT": "\n💡 **For improvements:** Be constructive and creative. Suggest practical, actionable enhancements with examples.",
            "GAP_ANALYSIS": "\n🔍 **For gap analysis:** Play detective. Show what's missing and why it matters. Build curiosity about what could be there.",
            "SCORING": "\n📊 **For scoring:** Be fair but lively. Explain each score with evidence and a touch of wit.",
            "SEARCH_QUERY": "\n🌐 **For web search:** Clearly distinguish document facts from external knowledge.",
            "GENERAL": "\n💬 **For general chat:** Be conversational and helpful. Keep it light but informative."
        }
        
        return base + intent_additions.get(intent, intent_additions["GENERAL"])


class LLMService:
    """
    Production LLM service with:
    - Token optimization for cost efficiency
    - Human-like extrovert personality
    - Multi-document analysis support
    - Strict fact-grounding
    """

    def __init__(self):
        # Use centralized settings
        self.provider = settings.llm.LLM_PROVIDER.lower()
        self.token_optimizer = TokenOptimizer()
        self.personality = HumanPersonality()
        
        self.model_name = settings.llm.active_model
        # CRITICAL FAIL-SAFE: Re-check environment variables directly if settings are empty
        raw_key = settings.llm.active_api_key
        if not raw_key or raw_key == "dummy-key":
             raw_key = (
                 os.getenv("OPENROUTER_API_KEY") or 
                 os.getenv("GROQ_API_KEY") or
                 os.getenv("OPENAI_API_KEY") or 
                 os.getenv("GEMINI_API_KEY") or 
                 "dummy-key"
             )
        
        # Strip any hidden whitespace/newlines from characters
        self.api_key = str(raw_key).strip().strip('"').strip("'")
        
        url_map = {
            "openai": None,
            "groq": "https://api.groq.com/openai/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "openrouter": "https://openrouter.ai/api/v1",
        }
        self.base_url = url_map.get(self.provider)
        
        # Log key availability (obscured)
        key_status = f"PRESENT (len={len(self.api_key)})" if self.api_key and self.api_key != "dummy-key" else "MISSING"
        logger.info(f"🚀 Initializing {self.provider} with model {self.model_name}. Key status: {key_status}")

        try:
            # Custom headers for OpenRouter (sometimes improves auth success)
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://huggingface.co/spaces",
                    "X-Title": "DocuCentric AI",
                }

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers,
                timeout=float(settings.llm.TIMEOUT)
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=extra_headers,
                timeout=float(settings.llm.TIMEOUT)
            )
            logger.info(f"LLM Service initialized: {self.provider}/{self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None
            self.async_client = None

    async def aclassify_intent(self, question: str) -> tuple[str, str]:
        """Classify query intent and depth in single call (saves tokens)"""
        prompt = f"""Classify this query. Respond with ONLY two words: INTENT DEPTH

INTENT: SUMMARY|FACTUAL|EVALUATIVE|IMPROVEMENT|GAP_ANALYSIS|SCORING|SEARCH_QUERY|GENERAL
DEPTH: shallow|deep

Query: "{question}"
Example response: FACTUAL shallow"""

        response = (await self.agenerate(prompt=prompt, max_tokens=30, temperature=0.0)).strip().upper()
        parts = response.split()
        
        intent = parts[0] if parts and parts[0] in ["SUMMARY", "FACTUAL", "EVALUATIVE", "IMPROVEMENT", "GAP_ANALYSIS", "SCORING", "SEARCH_QUERY", "GENERAL"] else "GENERAL"
        depth = parts[1].lower() if len(parts) > 1 and parts[1].lower() in ["shallow", "deep"] else "shallow"
        
        return intent, depth

    async def agenerate(
        self,
        prompt: Any,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate completion with token optimization asynchronously"""
        if not self.async_client:
            return "Error: LLM client not configured"

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self.token_optimizer.format_messages_compact(prompt)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    async def agenerate_with_context(
        self,
        question: str,
        context: str,
        conversation_history: List[Dict] = None,
        intent: str = "GENERAL",
        num_documents: int = 1,
        max_context_chars: int = 3000,
        stream: bool = False
    ):
        """
        Generate response with context asynchronously
        """
        if not self.async_client:
            yield "I'd love to help, but my brain is taking a coffee break! ☕ Please check if the API key is configured."
            return

        # Compress context to save tokens
        compressed_context = self.token_optimizer.compress_context(context, max_context_chars)
        
        # Build system prompt with personality
        system_prompt = self.personality.get_system_prompt(intent, num_documents)
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current question with context
        user_message = f"""📄 **DOCUMENT CONTEXT:**
{compressed_context}

💭 **MY QUESTION:**
{question}

Please answer based ONLY on the document context provided. If the information isn't there, tell me honestly what's missing!"""

        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            if stream:
                async for chunk in self._agenerate_stream(messages, max_tokens=2048, temperature=0.2):
                    yield chunk
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                )
                yield response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield f"Error: {str(e)}"

    async def _agenerate_stream(self, messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.2):
        """Stream generation asynchronously"""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"

    # Backward compatibility
    def generate(self, *args, **kwargs):
        import asyncio
        return asyncio.run(self.agenerate(*args, **kwargs))

    def classify_intent(self, *args, **kwargs):
        import asyncio
        return asyncio.run(self.aclassify_intent(*args, **kwargs))

    def generate_with_context(self, *args, **kwargs):
        return self.agenerate_with_context(*args, **kwargs)

    def warmup(self) -> None:
        """Verify API connection"""
        logger.info(f"LLM warmed up: {self.provider}/{self.model_name}")


# Global instance
llm_service = LLMService()
