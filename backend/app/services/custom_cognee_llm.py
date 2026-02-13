import logging
import json
import os
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
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.api_key = os.getenv("LLM_API_KEY", "local")

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
            full_system_prompt = f"""{system_prompt}
You are a precise data extraction engine. Output valid JSON only.

IMPORTANT:
- Focus ONLY on non-empty data fields.
- DO NOT populate internal fields like 'id', 'created_at', 'updated_at', 'ontology_valid', 'version', 'topological_rank', 'metadata', or 'type' UNLESS they have specific values in the source text.
- Omit these internal fields entirely to save space and prevent truncation.
- Ensure the JSON is complete and valid.
"""
            
            # 3. Call local LLM service
            # Cognee passes the 'text_input' which usually contains the user content
            # We treat 'prompt' as the user message
            
            # Ensure model is verified/loaded
            llm_service._ensure_loaded()
            
            response_text = await self._generate_async(full_system_prompt, prompt)
            
            # 4. Parse and Validate
            # Clean up markdown code blocks if present
            cleaned_json = self._clean_json(response_text)
            
            # Robust JSON Check (Option 3 Fix)
            if not cleaned_json.strip().startswith(('{', '[')):
                logger.error(f"❌ LLM returned non-JSON response: {cleaned_json[:200]}...")
                raise ValueError(f"LLM API error (Non-JSON response): {cleaned_json[:100]}")
                
            logger.debug(f"🔍 Extracted JSON: {cleaned_json[:200]}...")
            
            # Partial JSON Recovery if truncated (Best effort)
            if not cleaned_json.endswith('}'):
                # Try to close open braces if it's a simple truncation
                # But pydantic validate will likely still fail if it's mid-object
                pass

            # Validate with Pydantic
            result = response_model.model_validate_json(cleaned_json)
            return result
            
        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}")
            # If truncation happened, log the end of the response to see where it stopped
            if 'response_text' in locals():
                logger.error(f"Last 100 chars of response: ...{response_text[-100:]}")
            raise e

    async def _generate_async(self, system: str, user: str) -> str:
        """Helper to bridge sync LLMService to async Cognee"""
        import asyncio
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Increase max_tokens to 4096 (safe limit for many serverless providers)
        # Previous 3072 was still hitting EOF on very long complex resumes
        return await asyncio.to_thread(
            llm_service.generate, 
            prompt=messages, 
            max_tokens=4096, 
            temperature=0.1
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
