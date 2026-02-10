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
    print(f"🔑 LLM_API_KEY set to: {llm_key[:10]}..." if len(llm_key) > 10 else "🔑 LLM_API_KEY set to: local")
else:
    print(f"🔑 LLM_API_KEY already set: {os.environ['LLM_API_KEY'][:10]}...")

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
    if os.path.exists("/data"):
        # HuggingFace Spaces persistent storage (BEST)
        cognee_root = "/data/cognee_data"
        env_type = "HuggingFace Spaces (/data)"
    elif os.getenv("HF_HOME"):
        cognee_root = os.path.join(os.getenv("HF_HOME"), "cognee_data")
        env_type = "HuggingFace Spaces (HF_HOME)"
    elif os.path.exists("/app"):
        cognee_root = "/app/.cognee_data"
        env_type = "Docker/Cloud"
    else:
        cognee_root = os.path.join(os.getcwd(), ".cognee_system")
        env_type = "Local Development"
    
    # Create directory with full permissions
    os.makedirs(cognee_root, mode=0o777, exist_ok=True)
    
    # Create subdirectories that Cognee expects
    os.makedirs(os.path.join(cognee_root, "databases"), mode=0o777, exist_ok=True)
    os.makedirs(os.path.join(cognee_root, "data"), mode=0o777, exist_ok=True)
    
    # Set ALL possible environment variables Cognee might check
    os.environ["SYSTEM_ROOT_DIRECTORY"] = cognee_root
    os.environ["COGNEE_ROOT_DIR"] = cognee_root
    os.environ["COGNEE_DB_PATH"] = os.path.join(cognee_root, "databases")
    os.environ["COGNEE_DATA_DIR"] = os.path.join(cognee_root, "data")
    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["DB_NAME"] = "cognee_db"
    
    print(f"=" * 80)
    print(f"🧠 COGNEE CONFIGURATION (AGGRESSIVE)")
    print(f"=" * 80)
    print(f"Environment: {env_type}")
    print(f"Cognee Root: {cognee_root}")
    print(f"SYSTEM_ROOT_DIRECTORY: {os.environ.get('SYSTEM_ROOT_DIRECTORY')}")
    print(f"DB Path: {os.environ.get('COGNEE_DB_PATH')}")
    print(f"Writable: {os.access(cognee_root, os.W_OK)}")
    print(f"Exists: {os.path.exists(cognee_root)}")
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
                print(f"🔧 Cognee path intercepted - forcing: {COGNEE_ROOT}")
                return COGNEE_ROOT
            
            cognee_utils.get_system_root_directory = patched_get_system_root_directory
            print(f"✅ Monkey-patched cognee.shared.utils.get_system_root_directory")
        
        # Also try to patch config module
        try:
            from cognee.infrastructure.databases.relational import config as db_config
            if hasattr(db_config, 'get_database_url'):
                original_db_url = db_config.get_database_url
                
                def patched_get_database_url(*args, **kwargs):
                    db_path = os.path.join(COGNEE_ROOT, "databases", "cognee_db.db")
                    url = f"sqlite:///{db_path}"
                    print(f"🔧 Database URL intercepted - forcing: {url}")
                    return url
                
                db_config.get_database_url = patched_get_database_url
                print(f"✅ Monkey-patched database URL function")
        except ImportError:
            pass  # Module doesn't exist, skip
        
    except ImportError as e:
        print(f"⚠️ Could not apply monkey patch (Cognee not yet imported): {e}")
    except Exception as e:
        print(f"⚠️ Monkey patch failed: {e}")


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
        print(f"✅ Cognee {cognee.__version__} imported successfully")
        
        # Try to apply patches after import
        apply_cognee_monkey_patch()
        
        # Check if database directory is accessible
        db_path = os.path.join(COGNEE_ROOT, "databases")
        if os.path.exists(db_path) and os.access(db_path, os.W_OK):
            print(f"✅ Database directory writable: {db_path}")
        else:
            print(f"⚠️ Database directory issue: {db_path}")
        
        return True
    except Exception as e:
        print(f"❌ Cognee import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test configuration
    print("\n🧪 Testing Cognee Configuration...")
    verify_cognee_setup()

