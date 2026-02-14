import os
import shutil
import sys
import logging
import subprocess

# --- 1. GLOBAL PATHS ---
# We use /app/.cache/cognee_data as the single source of truth
COGNEE_ROOT = "/app/.cache/cognee_data"
DB_PATH = os.path.join(COGNEE_ROOT, "databases")
DATA_PATH = os.path.join(COGNEE_ROOT, "data")
MODELS_PATH = os.path.join(COGNEE_ROOT, "models")

# Configure logging for setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognee_setup")

def log(msg, level="info"):
    """Dual logging to logger and stderr for guaranteed visibility"""
    getattr(logger, level)(msg)
    print(f"[{level.upper()}] {msg}", file=sys.stderr)

def verify_cognee_setup():
    """Ensures directories exist and cleans up broken states."""
    log(f"[SETUP] Starting verification for COGNEE_ROOT: {COGNEE_ROOT}")
    
    # 1. Recursive Permissions Fix (The "Nuclear chmod")
    if os.path.exists(COGNEE_ROOT):
        try:
            # log("[SETUP] Applying recursive 777 permissions...")
            subprocess.run(["chmod", "-R", "777", COGNEE_ROOT], check=False)
        except Exception as e:
            log(f"[WARNING] chmod failed: {e}", "warning")

    # 2. Re-create structure with full permissions
    os.makedirs(DB_PATH, mode=0o777, exist_ok=True)
    os.makedirs(DATA_PATH, mode=0o777, exist_ok=True)
    os.makedirs(MODELS_PATH, mode=0o777, exist_ok=True)

    # 3. Set Environment Variables - AGGRESSIVE OVERRIDE
    os.environ["COGNEE_ROOT_DIR"] = COGNEE_ROOT
    os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT
    os.environ["COGNEE_CWD"] = COGNEE_ROOT
    
    # Explicitly set database paths
    os.environ["COGNEE_DB_PATH"] = DB_PATH
    os.environ["COGNEE_DATABASE_URL"] = f"sqlite:///{DB_PATH}/cognee_db.db"
    
    # Storage paths
    os.environ["COGNEE_STORAGE_PATH"] = DATA_PATH
    os.environ["COGNEE_FILES_PATH"] = DATA_PATH  # Newer cognee variable
    
    # 4. Configure Gemini (LLM)
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
    
    # Ensure Key is present
    if not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    
    log(f"[SETUP] Configuration Complete. DB URL: {os.environ['COGNEE_DATABASE_URL']}")

def apply_cognee_monkey_patch():
    """
    Titanium Patch: Aggressively replaces engine creation logic in sys.modules
    """
    log("[PATCH] Starting Titanium Patch...")
    
    try:
        import cognee
        from cognee.infrastructure.files.storage import LocalFileStorage
        
        # --- FIX 1: Override File Storage ---
        def forced_storage_path(self):
            return DATA_PATH
        LocalFileStorage.storage_path = property(forced_storage_path)
        log("[PATCH] LocalFileStorage patched")

        # --- FIX 2: Define The Safe Engine Creator ---
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        
        def create_safe_engine(**kwargs):
            # log(f"[PATCH] create_safe_engine intercepted call. Args: {kwargs.keys()}")
            
            # DIRECT HACK: Always return a fresh valid engine to the specific path
            # Manual SQLAlchemy Engine Creation
            
            # --- FIX: Engine Adapter ---
            # Cognee expects the engine object to have a 'create_database' method AND 'get_async_session'.
            # Raw SQLAlchemy engines don't have these, so we wrap it.
            class RobustEngineAdapter:
                def __init__(self, real_engine):
                    self._real_engine = real_engine
                    # Create a sessionmaker for async sessions
                    self.async_session_maker = sessionmaker(
                        self._real_engine, class_=AsyncSession, expire_on_commit=False
                    )
                
                async def create_database(self):
                    logger.info("[PATCH] adapters.create_database called - ensuring directory exists.")
                    if self._real_engine.url.drivername.startswith("sqlite"):
                        # Ensure directory exists
                        db_path = self._real_engine.url.database
                        if db_path:
                            os.makedirs(os.path.dirname(db_path), mode=0o777, exist_ok=True)
                
                def get_async_session(self):
                    # Returns a new async session instance (which is an async context manager)
                    return self.async_session_maker()

                def __getattr__(self, name):
                    return getattr(self._real_engine, name)
                    
                # Proxy magic methods required by SQLAlchemy
                def connect(self, *args, **kwargs):
                    return self._real_engine.connect(*args, **kwargs)
                    
                def begin(self, *args, **kwargs):
                    return self._real_engine.begin(*args, **kwargs)
                    
                @property
                def url(self):
                    return self._real_engine.url
                    
                @property
                def driver(self):
                    return self._real_engine.driver
                    
                @property
                def dialect(self):
                    return self._real_engine.dialect

            try:
                # Ensure 4 slashes for absolute path on *nix
                # USE ASYNC DRIVER: sqlite+aiosqlite
                db_url = f"sqlite+aiosqlite:///{os.path.join(DB_PATH, 'cognee_db.db')}"
                logger.info(f"[PATCH] Manually creating ASYNC engine at: {db_url}")
                # pool_pre_ping=True helps with recovery
                raw_engine = create_async_engine(db_url, pool_pre_ping=True)
                return RobustEngineAdapter(raw_engine)

            except Exception as e:
                log(f"[FATAL] Failed to create manual engine: {e}", "error")
                raise

        # --- FIX 3: Hunt and Destroy ---
        # We replace the function in every loaded module that might have it
        
        modules_patched = 0
        target_name = "create_relational_engine"
        
        # 1. Patch the main location
        try:
            import cognee.infrastructure.databases.relational as rel_pkg
            rel_pkg.create_relational_engine = create_safe_engine
            # Initialize singleton immediately
            rel_pkg.relational_engine = create_safe_engine()
            
            # Patch getter to always return our singleton
            def safe_get_engine():
                return rel_pkg.relational_engine
            rel_pkg.get_relational_engine = safe_get_engine
            
            log("[PATCH] cognee.infrastructure.databases.relational patched")
            modules_patched += 1
        except ImportError:
            pass

        # 2. Patch the submodule definition location
        try:
            import cognee.infrastructure.databases.relational.create_relational_engine as creator_mod
            creator_mod.create_relational_engine = create_safe_engine
            log("[PATCH] definition module patched")
            modules_patched += 1
        except ImportError:
            pass

        # 3. Iterate sys.modules to find any other references
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith("cognee") and hasattr(mod, target_name):
                setattr(mod, target_name, create_safe_engine)
                # log(f"[PATCH] Patched {mod_name}")

        log(f"[PATCH] Titanium Patch Complete. Patched primary modules and singletons.")

        # --- FIX 4: Patch global makedirs just in case ---
        original_makedirs = os.makedirs
        def patched_makedirs(name, mode=0o777, exist_ok=False):
            name_str = str(name)
            if "site-packages" in name_str and "cognee" in name_str:
                # Redirect to COGNEE_ROOT
                if ".cognee_system" in name_str:
                     parts = name_str.split(".cognee_system")
                     if len(parts) > 1:
                        new_path = os.path.join(COGNEE_ROOT, parts[1].lstrip(os.sep))
                        return original_makedirs(new_path, mode, exist_ok)
                
                return original_makedirs(name, mode, exist_ok)
            return original_makedirs(name, mode, exist_ok)
        
        os.makedirs = patched_makedirs
        log("[PATCH] os.makedirs redirection active")

    except Exception as e:
        log(f"[FATAL] Patching failed: {e}", "error")
        import traceback
        traceback.print_exc()

# Execute setup immediately
verify_cognee_setup()
apply_cognee_monkey_patch()
