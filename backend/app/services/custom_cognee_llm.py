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
        Generate structured output matching a Pydantic model.
        Supports:
        1. Gemini (via Instructor + Google SDK) if LLM_PROVIDER="gemini"
        2. Local Qwen (via llm_service) otherwise
        """
        try:
            # --- GEMINI SUPPORT ---
            if os.environ.get("LLM_PROVIDER") == "gemini":
                import instructor
                import google.generativeai as genai
                from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
                from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, RetryError

                # 1. Load Keys (Primary + Fallback)
                keys = []
                primary = os.environ.get("LLM_API_KEY")
                if primary: keys.append(primary)
                
                # Check for explicit fallback env var or the one user just gave us
                fallback = os.environ.get("LLM_API_KEY_FALLBACK", "AIzaSyArJJPEvW_WhNak1NQFAfCZMY2dIrYOUWg")
                if fallback and fallback != primary: 
                    keys.append(fallback)
                
                if not keys:
                    raise ValueError("LLM_API_KEY required for Gemini")

                # State for rotation (simple index)
                # We use a mutable container to persist state across retry attempts
                key_state = {"index": 0, "keys": keys}

                def get_current_client():
                    current_key = key_state["keys"][key_state["index"] % len(key_state["keys"])]
                    # Mask key for logging
                    masked_key = current_key[:5] + "..." + current_key[-3:]
                    # logger.info(f"🔑 Using Gemini API Key #{key_state['index'] % len(key_state['keys']) + 1}: {masked_key}")
                    genai.configure(api_key=current_key)
                    return instructor.from_gemini(
                        client=genai.GenerativeModel(
                            model_name=os.environ.get("LLM_MODEL", "gemini-2.0-flash").replace("gemini/", "")
                        ),
                        mode=instructor.Mode.GEMINI_JSON
                    )

                def rotate_key(retry_state):
                    # Callback before sleep: Rotate key if 429
                    ex = retry_state.outcome.exception()
                    # Check for ResourceExhausted directly or wrapped
                    is_quota = False
                    if isinstance(ex, ResourceExhausted):
                        is_quota = True
                    elif hasattr(ex, "__cause__") and isinstance(ex.__cause__, ResourceExhausted):
                        is_quota = True
                    elif "429" in str(ex) or "Quota" in str(ex):
                        is_quota = True
                        
                    if is_quota:
                        key_state["index"] += 1
                        new_key_idx = key_state["index"] % len(key_state["keys"])
                        logger.warning(f"⚠️ Quota Exceeded (429). Rotating to API Key #{new_key_idx + 1}")
                    else:
                        logger.warning(f"⚠️ Retrying due to non-quota error: {ex}")

                # Define exceptions to retry on
                # We need to be broad because instructor might wrap exceptions
                
                @retry(
                    stop=stop_after_attempt(15), 
                    wait=wait_exponential(multiplier=1, min=2, max=60),
                    # Retry on ResourceExhausted, ServiceUnavailable, or generic Exception if it looks like a quota/network issue
                    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, GoogleAPIError, Exception)),
                    before_sleep=rotate_key,
                    reraise=True
                )
                async def _call_gemini_with_rotation():
                    # Always get fresh client in case key changed
                    client = get_current_client()
                    
                    try:
                        logger.info(f"Attempting Gemini generation (Key #{key_state['index'] % len(keys) + 1})...")
                        return await client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": text_input}
                            ],
                            response_model=response_model,
                            max_retries=0 # Disable inner retries to trigger our outer loop immediately
                        )
                    except Exception as e:
                        # Log the exact error to help debugging
                        logger.error(f"❌ Gemini call failed. Type: {type(e).__name__}, Msg: {str(e)}")
                        # If it is a 429, we want to re-raise it so tenacity catches it.
                        # If instructor wrapped it, we'll re-raise the wrapper.
                        raise e

                resp = await _call_gemini_with_rotation()
                return resp

            # --- LOCAL QWEN SUPPORT (Original Logic) ---
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
