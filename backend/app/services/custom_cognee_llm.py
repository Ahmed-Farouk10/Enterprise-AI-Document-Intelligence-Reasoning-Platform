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
        # --- 1. GEMINI (PRIMARY) ---
        try:
            if os.environ.get("LLM_PROVIDER") == "gemini":
                # ... check if we should even try Gemini (key presence) ...
                if not os.environ.get("LLM_API_KEY"):
                     logger.warning("⚠️ LLM_PROVIDER is 'gemini' but LLM_API_KEY is missing. Skipping to fallback.")
                     raise ValueError("Missing Gemini API Key")

                return await self._generate_with_gemini(text_input, system_prompt, response_model)
        except Exception as e:
            logger.warning(f"⚠️ Gemini Failed: {e}. Falling back to Ollama...")
            # Proceed to Ollama

        # --- 2. OLLAMA (FALLBACK 1) ---
        try:
            # Check if OLLAMA_HOST is set, else default to host.docker.internal
            ollama_host = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434/v1")
            ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")
            
            logger.info(f"🔄 Attempting Ollama Generation ({ollama_model} @ {ollama_host})...")
            return await self._generate_with_ollama(
                text_input, system_prompt, response_model, 
                host=ollama_host, model=ollama_model
            )
        except Exception as e:
            logger.warning(f"⚠️ Ollama Failed: {e}. Falling back to Local Qwen (Transformers)...")
            # Proceed to Local Qwen

        # --- 3. LOCAL QWEN (FALLBACK 2 - FINAL) ---
        try:
            logger.info("🛡️ Using Local Qwen (Transformers) as final fallback...")
            return await self._generate_with_local_transformers(text_input, system_prompt, response_model)
        except Exception as e:
             logger.error(f"❌ ALL LLM PROVIDERS FAILED (Gemini -> Ollama -> Local). Final Error: {e}")
             raise e
            
        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}")
            # If truncation happened, log the end of the response to see where it stopped
            if 'response_text' in locals():
                logger.error(f"Last 100 chars of response: ...{response_text[-100:]}")
            raise e

    async def _generate_with_gemini(self, text_input, system_prompt, response_model):
        import instructor
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, RetryError

        # 1. Load Keys (Primary + Fallback)
        keys = []
        primary = os.environ.get("LLM_API_KEY")
        if primary: keys.append(primary)
        
        fallback = os.environ.get("LLM_API_KEY_FALLBACK", "AIzaSyArJJPEvW_WhNak1NQFAfCZMY2dIrYOUWg")
        if fallback and fallback != primary: 
            keys.append(fallback)
        
        if not keys:
            raise ValueError("LLM_API_KEY required for Gemini")

        key_state = {"index": 0, "keys": keys}

        def get_current_client():
            current_key = key_state["keys"][key_state["index"] % len(key_state["keys"])]
            genai.configure(api_key=current_key)
            return instructor.from_gemini(
                client=genai.GenerativeModel(
                    model_name=os.environ.get("LLM_MODEL", "gemini-2.0-flash").replace("gemini/", "")
                ),
                mode=instructor.Mode.GEMINI_JSON
            )

        def rotate_key(retry_state):
            ex = retry_state.outcome.exception()
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

        @retry(
            stop=stop_after_attempt(5),  # Reduced from 15 to fail faster to Ollama
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, GoogleAPIError, Exception)),
            before_sleep=rotate_key,
            reraise=True
        )
        async def _call_gemini_with_rotation():
            client = get_current_client()
            try:
                logger.info(f"Attempting Gemini generation (Key #{key_state['index'] % len(keys) + 1})...")
                return await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_input}
                    ],
                    response_model=response_model,
                    max_retries=0 
                )
            except Exception as e:
                logger.error(f"❌ Gemini call failed. Type: {type(e).__name__}, Msg: {str(e)}")
                raise e

        return await _call_gemini_with_rotation()

    async def _generate_with_ollama(self, text_input, system_prompt, response_model, host, model):
        """
        Generate using Ollama via OpenAI-compatible API.
        Requires 'openai' package (usually installed via instructor).
        """
        import instructor
        from openai import AsyncOpenAI

        client = instructor.from_openai(
            AsyncOpenAI(
                base_url=host,
                api_key="ollama", 
            ),
            mode=instructor.Mode.JSON
        )

        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input}
            ],
            response_model=response_model
        )

    async def _generate_with_local_transformers(self, text_input, system_prompt, response_model):
        """Original Local Qwen / transformers logic"""
        
        # 1. Create schema-enforcing prompt
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        # 2. Detailed system prompt
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
        user_prompt = text_input

        # 3. Call local LLM service - DIRECT GENERATION
        # We bypass llm_service.generate() because it respects LLM_MODEL env var (which is 'gemini')
        # We want to force usage of the local model
        
        logger.info("🛡️ Invoking Local Transformer Model (Qwen/Phi3) directly...")
        
        response_text = await self._generate_local_direct(full_system_prompt, user_prompt)
        
        # 4. Parse and Validate
        cleaned_json = self._clean_json(response_text)
        
        if not cleaned_json.strip().startswith(('{', '[')):
            logger.error(f"❌ LLM returned non-JSON response: {cleaned_json[:200]}...")
            # Detect if we got an error message instead of JSON
            if "Error" in cleaned_json or "429" in cleaned_json:
                 raise ValueError(f"CRITICAL: Local Fallback failed to run model. returned error: {cleaned_json}")
            raise ValueError(f"LLM API error (Non-JSON response): {cleaned_json[:100]}")
            
        logger.debug(f"🔍 Extracted JSON: {cleaned_json[:200]}...")
        
        result = response_model.model_validate_json(cleaned_json)
        return result

    async def _generate_local_direct(self, system: str, user: str) -> str:
        """
        Directly invoke the transformers model from llm_service, ignoring config.
        """
        import asyncio
        
        def _sync_generate():
            # Check if model is loaded on the service
            if llm_service.model is None:
                # Force load the local model if it's missing (even if configured for Gemini)
                logger.info("⚠️ Local model not loaded. forcing load of Qwen/Phi3...")
                # Temporarily override model name to ensure we load a local model, not Gemini
                original_name = llm_service.model_name
                if "gemini" in original_name.lower():
                     llm_service.model_name = "Qwen/Qwen2.5-7B-Instruct" 
                
                llm_service.load_model()
                # Restore name? No, keep it loaded as Qwen for this session fallback context
            
            # Now generate using the loaded model
            # We copy logic from llm_service.generate but bypassing the "if gemini" check
            
            prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            text = llm_service.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = llm_service.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(llm_service.model.device)
            
            outputs = llm_service.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=llm_service.tokenizer.eos_token_id
            )
            
            response = llm_service.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            return response.strip()

        return await asyncio.to_thread(_sync_generate)

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
