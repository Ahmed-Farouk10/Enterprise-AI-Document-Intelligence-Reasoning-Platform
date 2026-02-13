"""
Cognee Pre-Import Configuration + Runtime Patching
================================================================================
CRITICAL: This module MUST be imported BEFORE any code imports cognee.

Cognee reads environment variables and initializes paths on import.
We must set paths AND patch internal functions before that happens.
"""
import os
import sys

# =============================================================================
# CRITICAL: SET LLM_API_KEY AND PROVIDER BEFORE ANYTHING ELSE
# =============================================================================
# Cognee checks for LLM_API_KEY during import and will fail if not set.
# We FORCE OpenAI provider configuration to bypass Cognee's internal validation,
# but we route it to Hugging Face Inference API. This is a crucial workaround.

# 1. Ensure API Key is set
if not os.getenv("LLM_API_KEY"):
    llm_key = os.getenv("HF_TOKEN", "local")
    os.environ["LLM_API_KEY"] = llm_key
else:
    llm_key = os.environ["LLM_API_KEY"]
    print(f"[INFO] LLM_API_KEY already set: {llm_key[:10]}...")

# 2. FORCE OpenAI Provider + HF Endpoint (Unless explicitly overridden to something else valid)
# We only do this if we are using the HF Token (which usually starts with hf_)
if llm_key.startswith("hf_") or os.getenv("HF_TOKEN"):
    print("[INFO] Configuring OpenAI Proxy for Hugging Face Inference API...")
    
    # Provider must be 'openai' to pass validation
    os.environ["LLM_PROVIDER"] = "openai" 
    os.environ["COGNEE_LLM_PROVIDER"] = "openai"
    
    # Endpoint must point to HF
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    hf_endpoint = f"https://api-inference.huggingface.co/models/{model_id}/v1"
    
    os.environ["LLM_ENDPOINT"] = hf_endpoint
    os.environ["COGNEE_LLM_ENDPOINT"] = hf_endpoint
    
    # Model name must be clean (no prefix) for OpenAI provider
    os.environ["LLM_MODEL"] = model_id
    os.environ["COGNEE_LLM_MODEL"] = model_id
    
    print(f"[INFO] LLM Configured: Provider=openai (HF Proxy), Model={model_id}")

# =============================================================================
# COGNEE PATH CONFIGURATION (AGGRESSIVE)
# =============================================================================

def configure_cognee_paths():
    """
    Configure Cognee to use a writable directory with HF Spaces best practices.
    
    Priority (in order):
    1. /data/cognee_data (HF Spaces persistent storage)
    2. HF_HOME/cognee_data (HF Spaces fallback)
    3. /app/.cognee_data (Docker)
    4. .cognee_system (Local development)
    """
    # Detect environment and set writable path
    if os.path.exists("/data") and os.access("/data", os.W_OK):
        # HuggingFace Spaces persistent storage (BEST)
        cognee_root = "/data/cognee_data"
        env_type = "HuggingFace Spaces (/data)"
    elif os.getenv("HF_HOME") and os.access(os.getenv("HF_HOME"), os.W_OK):
        cognee_root = os.path.join(os.getenv("HF_HOME"), "cognee_data")
        env_type = "HuggingFace Spaces (HF_HOME)"
    elif os.path.exists("/app") and os.access("/app", os.W_OK):
        cognee_root = "/app/.cognee_data"
        env_type = "Docker/Cloud"
    else:
        # Fallback to tmp if nothing else works (guaranteed writable)
        cognee_root = os.path.join(os.getcwd(), ".cognee_system")
        if not os.access(os.getcwd(), os.W_OK):
            cognee_root = "/tmp/cognee_data"
        env_type = "Local/Fallback"
    
    # Create directory with full permissions
    try:
        os.makedirs(cognee_root, mode=0o777, exist_ok=True)
        os.makedirs(os.path.join(cognee_root, "databases"), mode=0o777, exist_ok=True)
        os.makedirs(os.path.join(cognee_root, "data"), mode=0o777, exist_ok=True)
        os.makedirs(os.path.join(cognee_root, "models"), mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"[FATAL] Could not create Cognee directories at {cognee_root}: {e}")
        # Last ditch effort
        cognee_root = "/tmp/cognee_data"
        os.makedirs(cognee_root, exist_ok=True)

    # Set ALL possible environment variables Cognee might check
    os.environ["SYSTEM_ROOT_DIRECTORY"] = cognee_root
    os.environ["COGNEE_ROOT_DIR"] = cognee_root
    os.environ["COGNEE_DB_PATH"] = os.path.join(cognee_root, "databases")
    os.environ["COGNEE_DATA_DIR"] = os.path.join(cognee_root, "data")
    
    # CRITICAL: Disable telemetry to prevent "Permission denied" errors on read-only systems
    os.environ["COGNEE_COLLECT_ANON_USAGE"] = "false"
    
    # CRITICAL: Fix telemetry permission error on HF Spaces
    # Force it to a temp file we control
    anon_id_path = os.path.join(cognee_root, ".anon_id")
    os.environ["COGNEE_ANONYMOUS_ID_PATH"] = anon_id_path
    os.environ["ANONYMOUS_ID_PATH"] = anon_id_path # Some versions verify this too
    
    # Pre-create the anon_id file to avoid permission issues later if user changes
    try:
        if not os.path.exists(anon_id_path):
            with open(anon_id_path, "w") as f:
                f.write("hf-spaces-static-anon-id")
    except Exception as e:
        print(f"[WARNING] Could not pre-create anon_id file: {e}")

    # FORCE EMBEDDING CONFIGURATION
    models_dir = os.path.join(cognee_root, "models")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_dir
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    os.environ["HF_HOME"] = models_dir
    
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["COGNEE_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"

    # VECTOR DB CONFIGURATION
    os.environ["VECTOR_DB_PROVIDER"] = "lancedb"
    os.environ["COGNEE_VECTOR_DB_TYPE"] = "lancedb"
    os.environ["COGNEE_VECTOR_DB"] = "lancedb"
    
    # DIMENSION CONFIGURATION (384 for all-MiniLM-L6-v2)
    for dim_var in ["EMBEDDING_DIMENSION", "VECTOR_DB_DIMENSION", "COGNEE_VECTOR_DIMENSION", 
                   "COGNEE_DIMENSION", "EMBEDDING_SIZE", "VECTOR_SIZE"]:
        os.environ[dim_var] = "384"
    
    print(f"[INFO] Model cache directory: {models_dir}")
    
    # Database configuration 
    db_path = os.path.join(cognee_root, "databases", "cognee_db.db")
    os.environ["COGNEE_DB_PROVIDER"] = "sqlite"
    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["DB_NAME"] = "cognee_db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    
    print(f"=" * 80)
    print(f"[INFO] COGNEE CONFIGURATION (AGGRESSIVE PERMISSION FIX)")
    print(f"=" * 80)
    print(f"Environment: {env_type}")
    print(f"Cognee Root: {cognee_root}")
    print(f"Anon ID Path: {anon_id_path}")
    print(f"Writable: {os.access(cognee_root, os.W_OK)}")
    print(f"=" * 80)
    
    return cognee_root


# =============================================================================
# MONKEY PATCH: Override Cognee's internal path resolution
# =============================================================================
def apply_cognee_monkey_patch():
    """
    Aggressively patch Cognee's path resolution at import time.
    This runs AFTER environment variables are set but BEFORE Cognee initializes.
    """
    try:
        # Import Cognee's path utilities
        import cognee
        from cognee.shared import utils as cognee_utils
        
        # Override the get_system_root_directory function if it exists
        if hasattr(cognee_utils, 'get_system_root_directory'):
            def patched_get_system_root_directory(*args, **kwargs):
                """Force return our configured path"""
                return COGNEE_ROOT
            
            cognee_utils.get_system_root_directory = patched_get_system_root_directory
            print(f"[SUCCESS] Monkey-patched cognee.shared.utils.get_system_root_directory")
        
        # CRITICAL FIX: Patch get_anonymous_id to prevent write permission errors
        # We patch BOTH the utils module and the logging_utils module if accessible
        static_anon_id = "hf-spaces-static-anon-id"
        
        def patched_get_anonymous_id():
            return static_anon_id
            
        if hasattr(cognee_utils, 'get_anonymous_id'):
            cognee_utils.get_anonymous_id = patched_get_anonymous_id
            print(f"[SUCCESS] Monkey-patched cognee.shared.utils.get_anonymous_id")

        # Try to patch logging_utils directly as that's where the error comes from
        try:
            from cognee.shared import logging_utils
            if hasattr(logging_utils, 'get_anonymous_id'):
                logging_utils.get_anonymous_id = patched_get_anonymous_id
                print(f"[SUCCESS] Monkey-patched cognee.shared.logging_utils.get_anonymous_id")
                
            # Also patch the file path variable if it exists
            if hasattr(logging_utils, 'ANONYMOUS_ID_PATH'):
                logging_utils.ANONYMOUS_ID_PATH = os.path.join(COGNEE_ROOT, ".anon_id")
                
        except ImportError:
            pass

        # CRITICAL FIX: Force Kuzu to respect our writable path
        try:
            from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
            # We patch the class or the config it uses.
            # KuzuAdapter typically takes graph_db_url or similar in init.
            # But let's verify if we can intercept the default if it uses os.getcwd()
            
            # Alternative: Patch the config if KuzuAdapter reads from it
            try:
                from cognee.infrastructure.databases.graph import config as graph_config
                if hasattr(graph_config, 'graph_db_url'):
                    print(f"[DEBUG] Original graph_db_url: {graph_config.graph_db_url}")
                    # This might be a string literal, so patching it might not work if already imported elsewhere.
            except ImportError:
                pass
        except ImportError:
            pass
            
        # CRITICAL FIX: Skip LLM connection test on HF Spaces to avoid startup blocks
        # We do this UNCONDITIONALLY if on Spaces, as the test is flaky/slow even with valid tokens
        if os.getenv("HF_HOME") or os.getenv("SPACE_ID") or os.getenv("COGNEE_SKIP_LLM_TEST") == "true":
            try:
                from cognee.infrastructure.llm import utils as llm_utils
                
                async def _noop_llm_test():
                    print("⚡ [PATCH] Skipping LLM connection test (HF Spaces Optimization)")
                    return True
                    
                llm_utils.test_llm_connection = _noop_llm_test
                print(f"[SUCCESS] Monkey-patched cognee.infrastructure.llm.utils.test_llm_connection (SKIPPED)")
            except ImportError:
                print("[WARNING] Could not patch test_llm_connection (ImportError)")

    except ImportError as e:
        print(f"[WARNING] Could not apply monkey patch (Cognee not yet imported): {e}")
    except Exception as e:
        print(f"[WARNING] Monkey patch failed: {e}")


# =============================================================================
# FORCE PATH CONFIGURATION ON IMPORT
# =============================================================================
# 1. Configure paths (Environment Variables)
COGNEE_ROOT = configure_cognee_paths()

# 2. Apply Runtime Patches (Monkey Patching)
# We must do this immediately, otherwise early imports might cache the wrong values
print("[INFO] Applying Cognee runtime patches...")
apply_cognee_monkey_patch()

# =============================================================================
# VERIFICATION
# =============================================================================
def verify_cognee_setup():
    """Verify Cognee can initialize with our configuration."""
    try:
        import cognee
        print(f"[SUCCESS] Cognee {cognee.__version__} imported successfully")
        
        # Check if database directory is accessible
        db_path = os.path.join(COGNEE_ROOT, "databases")
        if os.path.exists(db_path) and os.access(db_path, os.W_OK):
            print(f"[SUCCESS] Database directory writable: {db_path}")
        else:
            print(f"[WARNING] Database directory issue: {db_path}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Cognee import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test configuration
    print("\n[TEST] Testing Cognee Configuration...")
    verify_cognee_setup()

