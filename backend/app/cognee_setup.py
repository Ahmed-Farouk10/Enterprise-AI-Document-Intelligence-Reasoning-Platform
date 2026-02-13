"""
Cognee Pre-Import Configuration + Runtime Patching
================================================================================
CRITICAL: This module MUST be imported BEFORE any code imports cognee.

Cognee reads environment variables and initializes paths on import.
We must set paths AND patch internal functions before that happens.
"""
import os
import shutil

# --- 1. SET ENVIRONMENT VARIABLES (The Documentation Way) ---
# We configure this BEFORE importing anything else so Cognee sees it first.

# Use Google Gemini (Official Provider)
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
os.environ["COGNEE_LLM_MODEL"] = "gemini/gemini-2.0-flash"

# YOUR GEMINI API KEY
os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

# Use FastEmbed for Vector Search (Local & Free)
os.environ["EMBEDDING_PROVIDER"] = "fastembed"
os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. PERSISTENCE CONFIGURATION ---
COGNEE_ROOT = "/app/.cache/cognee_data"
# Ensure these are set for Cognee explicit pathing
os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT
os.environ["COGNEE_STORAGE_PATH"] = os.path.join(COGNEE_ROOT, "data")
os.environ["COGNEE_DB_PATH"] = os.path.join(COGNEE_ROOT, "databases")

# CRITICAL: Disable telemetry to prevent "Permission denied" errors on read-only systems
os.environ["COGNEE_COLLECT_ANON_USAGE"] = "false"
anon_id_path = os.path.join(COGNEE_ROOT, ".anon_id")
os.environ["COGNEE_ANONYMOUS_ID_PATH"] = anon_id_path
os.environ["ANONYMOUS_ID_PATH"] = anon_id_path

# --- 3. NUCLEAR CLEANUP (Fixes the Database Integrity Error) ---
# We wipe the database folder one last time to remove the broken "HF Proxy" records.
# This must happen BEFORE Cognee initializes any DB connection.
db_path = os.path.join(COGNEE_ROOT, "databases")
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path)
        print(f"[GEMINI SETUP] Wiped {db_path} to clear old HF/OpenAI errors.")
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}")

# Re-create structure immediately
try:
    os.makedirs(db_path, mode=0o777, exist_ok=True)
    os.makedirs(os.path.join(COGNEE_ROOT, "data"), mode=0o777, exist_ok=True)
    # create anon_id if missing
    if not os.path.exists(anon_id_path):
         with open(anon_id_path, "w") as f:
             f.write("hf-spaces-static-anon-id")
except Exception as e:
    print(f"[WARNING] Could not create Cognee directories: {e}")

print(f"[GEMINI SETUP] Configuration Loaded. LLM: {os.environ.get('LLM_PROVIDER')}")

# =============================================================================
# MONKEY PATCH: Override Cognee's internal path resolution (Legacy Support)
# =============================================================================
def apply_cognee_monkey_patch():
    """
    NUCLEAR PATCH: Hijack Cognee's internal path resolution to prevent PermissionErrors.
    This intercepts os.makedirs and overrides LocalFileStorage properties.
    """
    import cognee
    from cognee.infrastructure.files.storage import LocalFileStorage
    
    print("[INFO] 🔨 Applying Nuclear Patch for PermissionError...")
    
    # 1. THE NUCLEAR OPTION: Hijack the internal path resolution
    # Cognee 0.5.2 uses an internal storage_path property. We will force it.
    target_root = os.environ.get("COGNEE_DATA_ROOT", "/app/.cache/cognee_data")
    WRITABLE_DIR = os.path.join(target_root, "data_storage") # Ensure this is unique
    
    try:
        os.makedirs(WRITABLE_DIR, exist_ok=True)
    except Exception as e:
        pass

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
        pass

    # 3. CRITICAL: Patch os.makedirs to intercept site-packages writes
    # This catches deep internal calls in Cognee that ignore our config
    import os
    original_makedirs = os.makedirs
    def patched_makedirs(name, mode=0o777, exist_ok=False):
        if "site-packages/cognee" in str(name):
            try:
                # Redirect internal hidden folder creation to our writable cache
                if ".cognee_system" in str(name):
                    new_path = str(name).replace("/usr/local/lib/python3.11/site-packages/cognee", target_root)
                    # Handle both relative and absolute path mismatches
                    if "/site-packages/" in new_path:
                         new_path = os.path.join(target_root, ".cognee_system")
                    
                    print(f"[PATCH] Redirected {name} -> {new_path}")
                    return original_makedirs(new_path, mode, exist_ok)
            except Exception as inner:
                print(f"[WARNING] Redirect logic failed: {inner}")
                
        return original_makedirs(name, mode, exist_ok)
    
    os.makedirs = patched_makedirs
    print("[SUCCESS] Deep Monkey Patch applied to os.makedirs")

# Apply Patches immediately
apply_cognee_monkey_patch()

# Export for main.py import
def configure_cognee_paths():
    return COGNEE_ROOT

def verify_cognee_setup():
    return True
