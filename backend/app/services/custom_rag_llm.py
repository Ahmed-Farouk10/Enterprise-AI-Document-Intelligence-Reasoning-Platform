import logging
import json
import os
from typing import Type, TypeVar, Any, Dict, List, Optional
from pydantic import BaseModel
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class CustomRagLLMEngine:
    """
    Custom LLM Engine for Rag that uses our centralized llm_service.
    
    Implements the protocol expected by Rag's LLMGateway.
    """
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "huggingface")
        self.model = getattr(llm_service, "model_name", "unknown")
        self.api_key = os.getenv("LLM_API_KEY", "local")

    async def acreate_structured_output(
        self, 
        text_input: str, 
        response_model: Type[T],
        system_prompt: str = "You are a helpful assistant."
    ) -> T:
        """
        Generate structured output matching a Pydantic model using llm_service.
        """
        import asyncio
        
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        full_system_prompt = f"""{system_prompt}
You are a precise data extraction engine. Output valid JSON only.

IMPORTANT:
- Focus ONLY on non-empty data fields.
- DO NOT populate internal fields like 'id', 'created_at', 'updated_at', 'ontology_valid', 'version', 'topological_rank', 'metadata', or 'type' UNLESS they have specific values in the source text.
- Omit these internal fields entirely to save space and prevent truncation.
- Ensure the JSON is complete and valid.
INSTRUCTION:
Extract the information into a valid JSON object matching this schema:
{schema_str}

OUTPUT ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION.
"""
        message = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": text_input}
        ]
        
        logger.info(f"🔄 Attempting Structured Generation using llm_service ({self.provider})...")
        
        def _sync_gen():
            return llm_service.generate(
                prompt=message,
                max_tokens=4096,
                temperature=0.1
            )
            
        try:
            response_text = await asyncio.to_thread(_sync_gen)
            
            cleaned_json = self._clean_json(response_text)
            
            if not cleaned_json.strip().startswith(('{', '[')):
                logger.error(f"❌ LLM returned non-JSON response: {cleaned_json[:200]}...")
                raise ValueError(f"LLM API error (Non-JSON response): {cleaned_json[:100]}")
                
            logger.debug(f"🔍 Extracted JSON: {cleaned_json[:200]}...")
            
            result = response_model.model_validate_json(cleaned_json)
            return result
            
        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}")
            if 'response_text' in locals():
                logger.error(f"Last 100 chars of response: ...{response_text[-100:]}")
            raise e

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
