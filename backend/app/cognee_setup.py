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
# CRITICAL: SET LLM_API_KEY BEFORE ANYTHING ELSE
# =============================================================================
# Cognee checks for LLM_API_KEY during import and will fail if not set
# This MUST be the very first thing we do
if not os.getenv("LLM_API_KEY"):
    # Use HF_TOKEN if available, otherwise use 'local' placeholder
    llm_key = os.getenv("HF_TOKEN", "local")
    os.environ["LLM_API_KEY"] = llm_key
    print(f"[INFO] LLM_API_KEY set to: {llm_key[:10]}..." if len(llm_key) > 10 else f"[INFO] LLM_API_KEY set to: {llm_key}")
else:
    print(f"[INFO] LLM_API_KEY already set: {os.environ['LLM_API_KEY'][:10]}...")

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


# Execute configuration immediately on import
COGNEE_ROOT = configure_cognee_paths()


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
            original_func = cognee_utils.get_system_root_directory
            
            def patched_get_system_root_directory(*args, **kwargs):
                """Force return our configured path"""
                print(f"[DEBUG] Cognee path intercepted - forcing: {COGNEE_ROOT}")
                return COGNEE_ROOT
            
            cognee_utils.get_system_root_directory = patched_get_system_root_directory
            print(f"[SUCCESS] Monkey-patched cognee.shared.utils.get_system_root_directory")
        
        # Also try to patch config module
        try:
            from cognee.infrastructure.databases.relational import config as db_config
            if hasattr(db_config, 'get_database_url'):
                original_db_url = db_config.get_database_url
                
                def patched_get_database_url(*args, **kwargs):
                    db_path = os.path.join(COGNEE_ROOT, "databases", "cognee_db.db")
                    url = f"sqlite:///{db_path}"
                    print(f"[DEBUG] Database URL intercepted - forcing: {url}")
                    return url
                
                db_config.get_database_url = patched_get_database_url
                print(f"[SUCCESS] Monkey-patched database URL function")
        except ImportError:
            pass

        # CRITICAL FIX: Patch get_anonymous_id to prevent write permission errors
        try:
            if hasattr(cognee_utils, 'get_anonymous_id'):
                def patched_get_anonymous_id():
                    # Return static ID for HF Spaces to avoid file writes
                    return "hf-spaces-static-anon-id"
                
                cognee_utils.get_anonymous_id = patched_get_anonymous_id
                print(f"[SUCCESS] Monkey-patched get_anonymous_id to bypass file system")
        except Exception as e:
            print(f"[WARNING] Failed to patch get_anonymous_id: {e}")

    except ImportError as e:
        print(f"[WARNING] Could not apply monkey patch (Cognee not yet imported): {e}")
    except Exception as e:
        print(f"[WARNING] Monkey patch failed: {e}")


# =============================================================================
# DISABLE COGNEE ACCESS CONTROL FOR LEGACY DATA
# =============================================================================
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_cognee_setup():
    """Verify Cognee can initialize with our configuration."""
    try:
        import cognee
        print(f"[SUCCESS] Cognee {cognee.__version__} imported successfully")
        
        # Try to apply patches after import
        apply_cognee_monkey_patch()
        
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

