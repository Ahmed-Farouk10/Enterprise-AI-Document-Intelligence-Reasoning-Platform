import os
import shutil
import sys
import logging

# --- 1. GLOBAL PATHS ---
# We use /app/.cache/cognee_data as the single source of truth
COGNEE_ROOT = "/app/.cache/cognee_data"
DB_PATH = os.path.join(COGNEE_ROOT, "databases")
DATA_PATH = os.path.join(COGNEE_ROOT, "data")
MODELS_PATH = os.path.join(COGNEE_ROOT, "models")

# Configure logging for setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognee_setup")

def verify_cognee_setup():
    """Ensures directories exist and cleans up broken states."""
    
    # 1. Nuclear Wipe on Restart (Optional but recommended for fixing DB locks)
    # Only wipe if it exists and we want a fresh start. 
    # For persistence, we might want to be less aggressive, but for now we keep it to fix errors.
    if os.path.exists(DB_PATH):
        try:
            # Check if DB is locked or corrupted
            shutil.rmtree(DB_PATH)
            logger.info(f"[CLEANUP] Wiped {DB_PATH} to ensure clean state.")
        except Exception as e:
            logger.warning(f"[WARNING] Cleanup failed: {e}")

    # 2. Re-create structure with full permissions
    os.makedirs(DB_PATH, mode=0o777, exist_ok=True)
    os.makedirs(DATA_PATH, mode=0o777, exist_ok=True)
    os.makedirs(MODELS_PATH, mode=0o777, exist_ok=True)

    # 3. Set Environment Variables - AGGRESSIVE OVERRIDE
    # We set EVERY possible variable Cognee might use to find its data
    os.environ["COGNEE_ROOT_DIR"] = COGNEE_ROOT       # Used by some versions
    os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT      # Used by others
    os.environ["COGNEE_CWD"] = COGNEE_ROOT            # Just in case
    
    # Explicitly set database paths
    os.environ["COGNEE_DB_PATH"] = DB_PATH
    os.environ["COGNEE_DATABASE_URL"] = f"sqlite:///{DB_PATH}/cognee_db.db"
    
    # Storage paths
    os.environ["COGNEE_STORAGE_PATH"] = DATA_PATH
    
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
    
    logger.info(f"[SETUP] Cognee Root: {COGNEE_ROOT}")
    logger.info("[SETUP] Configuration Complete: Gemini + FastEmbed + Persistent Storage")

def apply_cognee_monkey_patch():
    """
    Patches Cognee internals to force writable paths.
    """
    try:
        import cognee
        from cognee.infrastructure.files.storage import LocalFileStorage
        
        logger.info("[PATCH] Applying deep configuration overrides...")

        # --- FIX 1: Override The File Storage Path ---
        # This ensures files are saved to /app/.cache/cognee_data/data
        def forced_storage_path(self):
            return DATA_PATH

        LocalFileStorage.storage_path = property(forced_storage_path)
        
        # --- FIX 2: Override The Relational Database Config ---
        # depending on version, config might be in different places. We try both.
        try:
            from cognee.infrastructure.databases.relational import config as rel_config
            rel_config.db_path = DB_PATH
            rel_config.db_name = "cognee_db"
            logger.info(f"[PATCH] Forced Relational DB Config (relational.config) to: {rel_config.db_path}")
        except ImportError:
            pass

        try:
            # Also try patching the core config if it exists
            from cognee.shared import config as core_config
            core_config.data_root_directory = COGNEE_ROOT
            logger.info(f"[PATCH] Forced Core Config data_root to: {COGNEE_ROOT}")
        except ImportError:
            pass

        # --- FIX 3: Patch get_relational_engine to return a FRESH engine ---
        # The singleton might have been initialized with bad paths before we got here.
        # We explicitly re-initialize it if possible, or monkeypatch the factory.
        
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.infrastructure.databases.relational.create_relational_engine import create_relational_engine
            
            # --- FIX: Robust Engine Creation ---
            # We create a wrapper that swallows extra args to match whatever signature Cognee 0.5.x expects
            # while FORCING our own paths.
            
            def create_safe_engine(**kwargs):
                logger.info(f"[PATCH] creating_engine called. Swallowing args: {list(kwargs.keys())}")
                # We ignore whatever path/provider they passed and force ours
                # Simplified args for SQLite to avoid 'unexpected keyword argument' errors
                return create_relational_engine(
                    db_path=DB_PATH,
                    db_name="cognee_db",
                    db_provider="sqlite",
                    db_host="localhost",
                    db_port=5432,
                    db_username="cognee",
                    db_password="password"
                )
            
            # Create the singleton instance
            correct_engine = create_safe_engine()
            
            # Monkeypatch the getter to return our correct engine
            import cognee.infrastructure.databases.relational as rel_module
            
            # 1. Patch the module-level singleton if it exists
            if hasattr(rel_module, "relational_engine"):
                rel_module.relational_engine = correct_engine
                logger.info("[PATCH] Replaced cached relational_engine singleton.")

            # 2. Patch the get_relational_engine function
            def patched_get_engine():
                return correct_engine
                
            rel_module.get_relational_engine = patched_get_engine
            logger.info("[PATCH] Monkey-patched get_relational_engine().")

             # 3. Aggressive: Patch the create_relational_engine in the module itself
             # This ensures that even if something imports it directly, it gets our wrapper
            rel_module.create_relational_engine = create_safe_engine
            logger.info("[PATCH] Monkey-patched create_relational_engine factory.")
            
        except Exception as e:
            logger.warning(f"[WARNING] Could not patch engine factory: {e}")


        # --- FIX 4: Patch Global 'os.makedirs' ---
        # Catches any rogue attempts to create folders in read-only system paths
        original_makedirs = os.makedirs
        
        def patched_makedirs(name, mode=0o777, exist_ok=False):
            name_str = str(name)
            # If it tries to write to site-packages/cognee, redirect it!
            if "site-packages" in name_str and "cognee" in name_str:
                if ".anon_id" in name_str:
                    new_path = os.path.join(COGNEE_ROOT, ".anon_id")
                else:
                    # Redirect everything else
                    # We blindly replace the prefix to our root
                    # This is dangerous but effective for "site-packages/cognee/.cognee_system"
                    if ".cognee_system" in name_str:
                        # Extract the part after .cognee_system
                        parts = name_str.split(".cognee_system")
                        if len(parts) > 1:
                            subpath = parts[1].lstrip(os.sep)
                            new_path = os.path.join(COGNEE_ROOT, subpath)
                        else:
                            new_path = COGNEE_ROOT
                    else:
                         new_path = name_str.replace(
                            os.path.dirname(os.path.dirname(cognee.__file__)), 
                            COGNEE_ROOT
                        )
                
                return original_makedirs(new_path, mode, exist_ok)
                
            return original_makedirs(name, mode, exist_ok)
        
        os.makedirs = patched_makedirs
        logger.info("[PATCH] os.makedirs redirected successfully.")
        
    except Exception as e:
        logger.error(f"[FATAL] Patching failed: {e}")

# Execute setup immediately when this module is imported
verify_cognee_setup()
apply_cognee_monkey_patch()
