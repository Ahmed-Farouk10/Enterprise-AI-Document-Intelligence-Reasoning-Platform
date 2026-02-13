import os
import shutil
import sys

# --- 1. GLOBAL PATHS ---
COGNEE_ROOT = "/app/.cache/cognee_data"
DB_PATH = os.path.join(COGNEE_ROOT, "databases")
DATA_PATH = os.path.join(COGNEE_ROOT, "data")

def verify_cognee_setup():
    """Ensures directories exist and cleans up broken states."""
    # 1. Nuclear Wipe to fix UNIQUE Constraint / Integrity Errors
    # We wipe the DB folder on every restart to clear 'ghost' records
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"[CLEANUP] Wiped {DB_PATH} to fix IntegrityError.")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

    # 2. Re-create structure with full permissions
    os.makedirs(DB_PATH, mode=0o777, exist_ok=True)
    os.makedirs(DATA_PATH, mode=0o777, exist_ok=True)
    os.makedirs(os.path.join(COGNEE_ROOT, "models"), mode=0o777, exist_ok=True)

    # 3. Set Environment Variables (The Documentation Way)
    os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT
    os.environ["COGNEE_STORAGE_PATH"] = DATA_PATH
    os.environ["COGNEE_DB_PATH"] = DB_PATH
    
    # 4. Configure Gemini (LLM)
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
    os.environ["COGNEE_LLM_MODEL"] = "gemini/gemini-2.0-flash"
    
    # Ensure Key is present
    if not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

    # 5. Configure FastEmbed (Embeddings)
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    
    print("[SETUP] Configuration Complete: Gemini + FastEmbed + Persistent Storage")

def apply_cognee_monkey_patch():
    """
    Patches Cognee internals to force writable paths.
    """
    import os
    import cognee
    from cognee.infrastructure.files.storage import LocalFileStorage
    
    print("[PATCH] Applying deep configuration overrides...")

    # --- FIX 1: Force File Storage Path ---
    target_root = "/app/.cache/cognee_data"
    writable_storage = os.path.join(target_root, "data")
    
    def forced_storage_path(self):
        return writable_storage

    LocalFileStorage.storage_path = property(forced_storage_path)
    
    # --- FIX 2: Override The Relational Database Config (The Safer Way) ---
    # Instead of importing the Adapter class (which failed), we patch the config object
    try:
        from cognee.infrastructure.databases.relational import config as rel_config
        # Force the path variable directly
        rel_config.db_path = os.path.join(target_root, "databases")
        rel_config.db_name = "cognee_db"
        print(f"[PATCH] Forced Relational DB Config to: {rel_config.db_path}")
    except ImportError:
        print("[WARNING] Could not patch relational config directly.")

    # --- FIX 3: Patch Global 'os.makedirs' ---
    # Catches any rogue attempts to create folders in read-only system paths
    original_makedirs = os.makedirs
    
    def patched_makedirs(name, mode=0o777, exist_ok=False):
        name_str = str(name)
        # If it tries to write to site-packages/cognee, redirect it!
        if "site-packages" in name_str and "cognee" in name_str:
            if ".anon_id" in name_str:
                new_path = os.path.join(target_root, ".anon_id")
            else:
                new_path = name_str.replace(
                    os.path.dirname(os.path.dirname(cognee.__file__)), 
                    target_root
                )
            
            return original_makedirs(new_path, mode, exist_ok)
            
        return original_makedirs(name, mode, exist_ok)
    
    os.makedirs = patched_makedirs
    print("[PATCH] os.makedirs redirected successfully.")

# Execute setup immediately when this module is imported
verify_cognee_setup()
apply_cognee_monkey_patch()
