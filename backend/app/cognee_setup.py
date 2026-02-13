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
if llm_key.startswith("hf_") or os.getenv("HF_TOKEN"):
    print("[INFO] Configuring OpenAI Proxy for Hugging Face Inference API...")
    
    # Provider must be 'openai' to pass validation
    os.environ["LLM_PROVIDER"] = "openai" 
    os.environ["COGNEE_LLM_PROVIDER"] = "openai"
    
    # FIX: Switching to Google Gemini 2.0 Flash for stability (No 410 Errors)
    # Using local embeddings (FastEmbed) to avoid dependency on OpenAI for vectors
    
    # 1. Force Provider to 'gemini'
    os.environ["LLM_PROVIDER"] = "gemini" 
    os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
    
    # 2. Set Model ID 
    model_id = "gemini/gemini-2.0-flash"
    os.environ["LLM_MODEL"] = model_id
    os.environ["COGNEE_LLM_MODEL"] = model_id

    # 3. Set API Key (User Provided)
    # CRITICAL: If env var is missing, use the provided backup key
    if not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

    # 4. Force Local Embeddings (FastEmbed)
    # This prevents Cognee from crashing if no OpenAI key is present for embeddings
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"[INFO] LLM Configured: Provider=gemini, Model={model_id}, Embeddings=fastembed")

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
    
    # USER SUGGESTED FIX: Explicitly set COGNEE_DATA_ROOT and STORAGE_PATH
    os.environ["COGNEE_DATA_ROOT"] = os.path.join(cognee_root, "data")
    os.environ["COGNEE_STORAGE_PATH"] = os.path.join(cognee_root, "data", "storage")
    print(f"[INFO] COGNEE_DATA_ROOT set to: {os.environ['COGNEE_DATA_ROOT']}")
    
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

    # MEMORY OPTIMIZATION FOR HF SPACES
    if os.getenv("HF_HOME") or os.getenv("SPACE_ID"):
        print("[INFO] HF Spaces detected - Forcing Quantized Embeddings for Memory Efficiency")
        # Use FastEmbed's quantized model support (usually default, but let's be explicit if possible or rely on lighter loads)
        # FastEmbed models are already quantized by default, but we ensure we don't load anything else.
        # We also set a flag for other services to be careful.
        os.environ["COGNEE_LOW_MEMORY_MODE"] = "true"

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
    
    # CRITICAL: Fix for IntegrityError & Corrupted Files (Ghost Records)
    # The error 'UNIQUE constraint failed' usually means metadata is out of sync.
    # We perform a TOTAL NUCLEAR WIPE of the entire Cognee root to ensure no ghost records survive.
    # This was requested by the user to fix the deadlock.
    import shutil
    
    # User requested function structure:
    # def nuclear_wipe(): ...
    # User requested function structure:
    # def nuclear_wipe(): ...
    print(f"[INFO] 🧹 Starting Nuclear Wipe of Cognee root: {cognee_root}")
    if os.path.exists(cognee_root):
        try:
            # Wiping the whole root ensures the SQLite registry and 
            # the empty graph files are perfectly in sync (starting at zero)
            shutil.rmtree(cognee_root)
            print("[NUCLEAR] Cleared all metadata and ghost records.")
        except Exception as e:
             print(f"[WARNING] Manual cleanup failed: {e}")
    
    # Re-create the mandatory folder structure
    try:
        os.makedirs(cognee_root, mode=0o777, exist_ok=True)
        os.makedirs(os.environ["COGNEE_STORAGE_PATH"], mode=0o777, exist_ok=True)
        os.makedirs(os.environ["COGNEE_DB_PATH"], mode=0o777, exist_ok=True)
        print(f"[CONFIG] Initialized Cognee at: {cognee_root}")
    except Exception as e:
        print(f"[WARNING] Could not create Cognee directories: {e}")

    # 4. Telemetry Fix (Still needed for read-only systems)
    anon_id_path = os.path.join(cognee_root, ".anon_id")
    os.environ["COGNEE_ANONYMOUS_ID_PATH"] = anon_id_path
    try:
        if not os.path.exists(anon_id_path):
            with open(anon_id_path, "w") as f:
                f.write("hf-spaces-static-anon-id")
    except Exception as e:
        print(f"[WARNING] Could not create anon_id: {e}")

    return cognee_root


# =============================================================================
# MONKEY PATCH: Override Cognee's internal path resolution
# =============================================================================
def apply_cognee_monkey_patch():
    """
    NUCLEAR PATCH: Hijack Cognee's internal path resolution to prevent PermissionErrors.
    This intercepts os.makedirs and overrides LocalFileStorage properties.
    """
    import os
    import cognee
    from cognee.infrastructure.files.storage import LocalFileStorage
    
    print("[INFO] 🔨 Applying Nuclear Patch for PermissionError...")
    
    # 1. THE NUCLEAR OPTION: Hijack the internal path resolution
    # Cognee 0.5.2 uses an internal storage_path property. We will force it.
    target_root = os.environ.get("COGNEE_DATA_ROOT", "/app/.cache/cognee_data")
    WRITABLE_DIR = os.path.join(target_root, "data_storage")
    
    try:
        os.makedirs(WRITABLE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create WRITABLE_DIR {WRITABLE_DIR}: {e}")

    # We override the property entirely so it CANNOT point to site-packages
    def forced_storage_path(self):
        return WRITABLE_DIR

    # Attach the fix to the class itself
    LocalFileStorage.storage_path = property(forced_storage_path)
    print(f"[SUCCESS] Hijacked LocalFileStorage.storage_path -> {WRITABLE_DIR}")
    
    # 2. Patch the Ingestion Module's static variable
    try:
        from cognee.modules.ingestion import save_data_to_file
        save_data_to_file.data_root_directory = target_root
        print(f"[SUCCESS] Forced save_data_to_file.data_root_directory to {target_root}")
    except Exception as e:
        print(f"[DEBUG] save_data_to_file patch skipped: {e}")

    # 3. Patch the 'makedirs' call inside LocalFileStorage to prevent permission errors
    original_makedirs = os.makedirs
    def patched_makedirs(name, mode=0o777, exist_ok=False):
        if "site-packages/cognee" in str(name):
            try:
                if hasattr(cognee, '__file__'):
                    pkg_path = os.path.dirname(cognee.__file__)
                    if pkg_path in str(name):
                        new_path = str(name).replace(pkg_path, target_root)
                        print(f"[PATCH] os.makedirs: Redirecting {name} -> {new_path}")
                        return original_makedirs(new_path, mode, exist_ok)
            except Exception as inner:
                print(f"[WARNING] Redirect logic failed: {inner}")
                
        return original_makedirs(name, mode, exist_ok)
    
    os.makedirs = patched_makedirs
    print("[SUCCESS] Global os.makedirs patched to redirect site-packages calls")
    
    print("[INFO] Nuclear Patch application complete.")

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

