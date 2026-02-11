import logging
import json
from typing import Type, TypeVar, Any, Dict, List, Optional
from pydantic import BaseModel
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class CustomCogneeLLMEngine:
    """
    Custom LLM Engine for Cognee that uses our local LLMService (Qwen) 
    instead of defaulting to OpenAI/LiteLLM.
    
    Implements the protocol expected by Cognee's LLMGateway.
    """
    def __init__(self):
        self.provider = "custom_local"
        self.model = "Qwen/Qwen2.5-32B-Instruct"

    async def acreate_structured_output(
        self, 
        text_input: str, 
        response_model: Type[T],
        system_prompt: str = "You are a helpful assistant."
    ) -> T:
        """
        Generate structured output matching a Pydantic model using local LLM.
        Bypasses OpenAI completely.
        """
        try:
            # 1. Create schema-enforcing prompt
            schema = response_model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            
            prompt = f"""
{text_input}

INSTRUCTION:
Extract the information into a valid JSON object matching this schema:
{schema_str}

OUTPUT ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION.
"""
            
            # 2. detailed system prompt (Cognee usually provides one in text_input, but we reinforce)
            full_system_prompt = f"{system_prompt}\nYou are a precise data extraction engine. Output valid JSON only."
            
            # 3. Call local LLM service
            # Cognee passes the 'text_input' which usually contains the user content
            # We treat 'prompt' as the user message
            
            # Ensure model is verified/loaded
            llm_service._ensure_loaded()
            
            response_text = await self._generate_async(full_system_prompt, prompt)
            
            # 4. Parse and Validate
            # Clean up markdown code blocks if present
            cleaned_json = self._clean_json(response_text)
            
            logger.debug(f"🔍 Extracted JSON: {cleaned_json[:200]}...")
            
            # Validate with Pydantic
            result = response_model.model_validate_json(cleaned_json)
            return result
            
        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}")
            logger.error(f"Response was: {response_text if 'response_text' in locals() else 'None'}")
            raise e

    async def _generate_async(self, system: str, user: str) -> str:
        """Helper to bridge sync LLMService to async Cognee"""
        import asyncio
        
        # We run the blocking generation in a thread
        # Note: llm_service.generate is blocking
        
        # Prepare the prompt string expected by LLMService
        # LLMService.generate expects a full prompt string or handles it internally?
        # LLMService.generate docstring: "prompt: The prompt to complete"
        # Qwen expects chat template. 
        # But LLMService.generate applies template if passed raw string? 
        # Looking at LLMService.generate (lines 499+), itTokenizes 'text'.
        # It expects the FULL text.
        
        # We should format it using the tokenizer's chat template if possible, 
        # but LLMService private _prepare_messages does that.
        # But generate() takes a single string 'prompt'.
        
        # Let's verify LLMService usage. It uses tokenizer(text).
        # So we must format the chat template OURSELVES here.
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        if llm_service.tokenizer:
            formatted_prompt = llm_service.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback if tokenizer not loaded (e.g. API mode)
            formatted_prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"

        return await asyncio.to_thread(
            llm_service.generate, 
            prompt=formatted_prompt,
            max_tokens=2048,
            temperature=0.1 # Low temp for extraction
        )

    def _clean_json(self, text: str) -> str:
        """Sanitize LLM output to extract JSON"""
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
