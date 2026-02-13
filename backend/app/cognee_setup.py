import os
import shutil
import sys

# --- 1. GLOBAL PATHS (Hugging Face Persistence) ---
COGNEE_ROOT = "/app/.cache/cognee_data"
DB_PATH = os.path.join(COGNEE_ROOT, "databases")
DATA_PATH = os.path.join(COGNEE_ROOT, "data")

def verify_cognee_setup():
    """Ensures directories exist and cleans up broken states."""
    
    # 1. NUCLEAR WIPE (Essential when switching to Multi-User Mode)
    # When enabling permissions, the directory structure changes.
    # We must wipe old data to prevent "database locked" or schema mismatches.
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"[CLEANUP] Wiped {DB_PATH} for Permission System initialization.")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

    # 2. Re-create structure with full permissions
    os.makedirs(DB_PATH, mode=0o777, exist_ok=True)
    os.makedirs(DATA_PATH, mode=0o777, exist_ok=True)
    os.makedirs(os.path.join(COGNEE_ROOT, "models"), mode=0o777, exist_ok=True)

    # ---------------------------------------------------------
    # CONFIGURATION SECTION
    # ---------------------------------------------------------

    # A. CORE PATHS
    os.environ["COGNEE_DATA_ROOT"] = COGNEE_ROOT
    os.environ["COGNEE_STORAGE_PATH"] = DATA_PATH
    os.environ["COGNEE_DB_PATH"] = DB_PATH

    # B. STRUCTURED OUTPUT (Documentation Requirement)
    # Uses 'instructor' to coerce LLM outputs into Pydantic models
    os.environ["STRUCTURED_OUTPUT_FRAMEWORK"] = "instructor"

    # C. PERMISSIONS & ACCESS CONTROL (Documentation Requirement)
    # Enables multi-user mode. Data is stored in databases/<user_uuid>/
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "true"
    os.environ["REQUIRE_AUTHENTICATION"] = "true"

    # D. LLM SETUP (Gemini)
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["COGNEE_LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
    os.environ["COGNEE_LLM_MODEL"] = "gemini/gemini-2.0-flash"
    
    # API Key Handling
    if not os.getenv("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "AIzaSyChLF3hBJXMP2S5WGgYumMrNfZK-cURvZg"

    # E. EMBEDDINGS (FastEmbed - Local)
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["COGNEE_EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

    # F. RELATIONAL DATABASE (SQLite)
    # We use SQLite for metadata, mapped to our persistent path.
    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["DB_NAME"] = "cognee_db"
    # Note: The actual path is handled by the patch below to ensure it goes to .cache

    # G. GRAPH STORE (Neo4j with Kuzu Fallback)
    # If you have a Neo4j URL set in secrets, we use it. Otherwise, we use Kuzu.
    neo4j_url = os.getenv("GRAPH_DATABASE_URL")
    
    if neo4j_url and "bolt" in neo4j_url:
        print("[CONFIG] Neo4j URL detected. Configuring Graph Store as Neo4j.")
        os.environ["GRAPH_DATABASE_PROVIDER"] = "neo4j"
        os.environ["GRAPH_DATABASE_URL"] = neo4j_url
        os.environ["GRAPH_DATABASE_USERNAME"] = os.getenv("GRAPH_DATABASE_USERNAME", "neo4j")
        os.environ["GRAPH_DATABASE_PASSWORD"] = os.getenv("GRAPH_DATABASE_PASSWORD", "password")
    else:
        print("[CONFIG] No external Neo4j URL found. Falling back to Kuzu (Local File).")
        os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
        # Kuzu database will be auto-created in .cache/cognee_data/databases/

    print("[SETUP] Configuration Complete: Permissions + Structured Output Active.")

def apply_cognee_monkey_patch():
    """
    Patches Cognee internals to force writable paths.
    """
    import os
    import cognee
    from cognee.infrastructure.files.storage import LocalFileStorage
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import SqlAlchemyAdapter

    print("[PATCH] Hijacking Cognee Database Engine...")

    target_root = "/app/.cache/cognee_data"

    # --- FIX 1: Force File Storage Path ---
    def forced_storage_path(self):
        return os.path.join(target_root, "data")
    LocalFileStorage.storage_path = property(forced_storage_path)

    # --- FIX 2: Hijack the Relational Engine Creator ---
    # This ensures even if config says 'sqlite', we force the file to live in .cache
    original_get_engine = cognee.infrastructure.databases.relational.get_relational_engine

    def patched_get_relational_engine(*args, **kwargs): # Added *args, **kwargs for safety
        # Force the writable path
        db_name = "cognee_db"
        # Ensure directory exists
        os.makedirs(os.path.join(target_root, "databases"), exist_ok=True)
        
        # Construct the connection string manually
        # Note: In Multi-user mode, Cognee might append user IDs, but this base is required
        connection_string = f"sqlite+aiosqlite:///{target_root}/databases/{db_name}.db"
        
        return SqlAlchemyAdapter(connection_string, db_name)

    # Apply the patch
    cognee.infrastructure.databases.relational.get_relational_engine = patched_get_relational_engine
    
    # Also patch if already imported in operations
    try:
        import cognee.modules.engine.operations.setup as setup_ops
        setup_ops.get_relational_engine = patched_get_relational_engine
    except ImportError:
        pass

    print(f"[PATCH] Engine redirected to persistent storage.")

    # --- FIX 3: Patch Global 'os.makedirs' ---
    original_makedirs = os.makedirs
    def patched_makedirs(name, mode=0o777, exist_ok=False):
        name_str = str(name)
        if "site-packages" in name_str and "cognee" in name_str:
            new_path = name_str.replace(
                os.path.dirname(os.path.dirname(cognee.__file__)), 
                target_root
            )
            return original_makedirs(new_path, mode, exist_ok)
        return original_makedirs(name, mode, exist_ok)
    os.makedirs = patched_makedirs

def configure_cognee_paths():
    return COGNEE_ROOT

# Run setup
verify_cognee_setup()
apply_cognee_monkey_patch()
