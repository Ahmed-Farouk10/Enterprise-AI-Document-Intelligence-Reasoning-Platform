import os
import shutil
import sys

# --- 1. PERSISTENCE & CLEANUP CONFIGURATION ---
# We define these globally so they are available immediately
COGNEE_ROOT = "/app/.cache/cognee_data"
DB_PATH = os.path.join(COGNEE_ROOT, "databases")

def verify_cognee_setup():
    """Helper to ensure paths exist before app starts"""
    # 1. Nuclear Wipe to fix UNIQUE Constraint / Integrity Errors
    # We wipe the DB folder on every restart to clear 'ghost' records
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"[CLEANUP] Wiped {DB_PATH} to fix IntegrityError.")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

    # 2. Re-create structure
    os.makedirs(DB_PATH, exist_ok=True)
    os.makedirs(os.path.join(COGNEE_ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(COGNEE_ROOT, "models"), exist_ok=True)

    # 3. Set Environment Variables for Cognee
    os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT
    os.environ["COGNEE_STORAGE_PATH"] = os.path.join(COGNEE_ROOT, "data")
    os.environ["COGNEE_DB_PATH"] = DB_PATH
    
    # 4. Configure Gemini (LLM)
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
    os.environ["COGNEE_LLM_MODEL"] = "gemini/gemini-2.0-flash"
    
    # Ensure Key is present (It should be in your HF Space Secrets)
    if not os.getenv("LLM_API_KEY"):
        # Fallback for testing - replace if needed, or set in Space Settings
        os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

    # 5. Configure FastEmbed (Embeddings)
    # This prevents the system from asking for an OpenAI key
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    
    print("[SETUP] Configuration Complete: Gemini + FastEmbed + Persistent Storage")

def apply_cognee_monkey_patch():
    """
    Patches Cognee internals to force writable paths on Hugging Face Spaces.
    """
    # CRITICAL: Import os inside the function to avoid UnboundLocalError
    import os
    import cognee
    from cognee.infrastructure.files.storage import LocalFileStorage
    
    print("[PATCH] Applying deep file storage patches...")

    # 1. Force the Storage Path
    # We hijack the property to always return our writable cache path
    target_root = os.environ.get("COGNEE_DATA_ROOT", "/app/.cache/cognee_data")
    writable_storage = os.path.join(target_root, "data")
    
    def forced_storage_path(self):
        return writable_storage

    LocalFileStorage.storage_path = property(forced_storage_path)
    
    # 2. Patch the Global 'os.makedirs' 
    # This catches any rogue attempts to write to /usr/local/lib
    original_makedirs = os.makedirs
    
    def patched_makedirs(name, mode=0o777, exist_ok=False):
        name_str = str(name)
        if "site-packages" in name_str and "cognee" in name_str:
            # Redirect to cache
            try:
                # Find where cognee is installed
                cognee_dir_parent = os.path.dirname(os.path.dirname(cognee.__file__))
                new_path = name_str.replace(cognee_dir_parent, target_root)
                 # Handle double mapping if target_root is already in path
                if target_root in name_str:
                    return original_makedirs(name, mode, exist_ok)
                
                return original_makedirs(new_path, mode, exist_ok)
            except Exception:
                # Fallback if logic fails
                return original_makedirs(name, mode, exist_ok)
        return original_makedirs(name, mode, exist_ok)
    
    os.makedirs = patched_makedirs
    print("[PATCH] os.makedirs redirected successfully.")

# Run setup immediately on import
verify_cognee_setup()
apply_cognee_monkey_patch()
def configure_cognee_paths():
    return COGNEE_ROOT
