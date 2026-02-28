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
        1. Ollama (Local Dev)
        2. HuggingFace Inference API (Spaces Native)
        3. Local Qwen (Fallback)
        """
        provider = os.environ.get("LLM_PROVIDER", "huggingface").lower()

        # --- 1. OLLAMA (LOCAL DEV) ---
        if provider == "ollama":
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
                logger.warning(f"⚠️ Ollama Failed: {e}. Falling back to Next Provider...")

        # --- 2. HUGGINGFACE INFERENCE API (SPACES NATIVE) ---
        if provider == "huggingface":
            try:
                logger.info("⚡ Using HuggingFace Inference API (Spaces Native)...")
                return await self._generate_with_hf_api(text_input, system_prompt, response_model)
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace API Failed: {e}. Falling back to Local...")

        # --- 3. LOCAL QWEN (FALLBACK) ---
        try:
            logger.info("🛡️ Using Local Qwen (Transformers) as final fallback...")
            return await self._generate_with_local_transformers(text_input, system_prompt, response_model)
        except Exception as e:
             logger.error(f"❌ ALL LLM PROVIDERS FAILED. Final Error: {e}")
             raise e
            
        except Exception as e:
            logger.error(f"❌ Structured output generation failed: {e}")
            # If truncation happened, log the end of the response to see where it stopped
            if 'response_text' in locals():
                logger.error(f"Last 100 chars of response: ...{response_text[-100:]}")
            raise e

    async def _generate_with_hf_api(self, text_input, system_prompt, response_model):
        """
        Generate structured output using HF Inference API via manual prompting + parsing.
        Reuse logic from local transformers because API just returns text.
        """
        # 1. Create schema-enforcing prompt
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        full_system_prompt = f"""{system_prompt}
You are a precise data extraction engine. Output valid JSON only.
INSTRUCTION:
Extract the information into a valid JSON object matching this schema:
{schema_str}
OUTPUT ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION.
"""
        # 2. Call API via Service
        message = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": text_input}
        ]
        
        # We use a helper from llm_service that handles the API call
        # Note: We need a dedicated method on llm_service to just return text from API
        # Fortunately _generate_via_inference_api does exactly that.
        
        response_text = llm_service._generate_via_inference_api(
            prompt=message,
            max_tokens=4096,
            temperature=0.1,
            raise_errors=True
        )
        
        # 3. Parse JSON
        cleaned_json = self._clean_json(response_text)
        
        # Check for error strings - Fail Gracefully
        if "error" in cleaned_json.lower() or "429" in cleaned_json:
             logger.error(f"❌ HF API returned error message: {cleaned_json}")
             raise ValueError(f"HF API Error: {cleaned_json[:200]}")

        if not cleaned_json.strip().startswith(('{', '[')):
             logger.error(f"❌ HF API returned non-JSON: {cleaned_json[:200]}")
             raise ValueError(f"HF API non-JSON response: {cleaned_json[:100]}")

        try:
            result = response_model.model_validate_json(cleaned_json)
            return result
        except Exception as validation_error:
            logger.error(f"❌ JSON Validation Failed: {validation_error} | Content: {cleaned_json[:200]}")
            raise validation_error

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
        Directly invoke the transformers model or HF Inference API from llm_service, ignoring config.
        """
        import asyncio
        
        def _sync_generate():
            # 1. Ensure we are targeting the Local Model (Qwen), not Gemini
            # We temporarily swap the model name so that if we use Inference API, it uses Qwen.
            original_name = llm_service.model_name
            target_local_model = "Qwen/Qwen2.5-7B-Instruct"
            
            # If currently configured for Gemini, switch to Qwen for this call
            if "gemini" in original_name.lower():
                 llm_service.model_name = target_local_model

            try:
                # 2. Check if model is loaded (or can be loaded)
                if llm_service.model is None:
                    # Attempt load (might be skipped if on HF Spaces)
                    llm_service.load_model()
                
                # 3. Branch: Local In-Memory vs. Inference API
                if llm_service.model is None:
                    # HF Spaces Mode: Model didn't load. Cannot run locally!
                    raise RuntimeError(f"Local LLM model is not loaded (HF Spaces mode). Cannot execute local fallback for {llm_service.model_name}.")
                else:
                    # Local In-Memory Mode
                    logger.info("🛡️ Using In-Memory Local Model...")
                    
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

            finally:
                # Always restore the original model name (e.g. gemini)
                llm_service.model_name = original_name

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
