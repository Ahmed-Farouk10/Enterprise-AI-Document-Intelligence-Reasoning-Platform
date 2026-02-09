"""
Cognee Pre-Import Configuration
================================================================================
CRITICAL: This module MUST be imported BEFORE any code imports cognee.

Cognee reads environment variables and initializes paths on import.
We must set COGNEE_ROOT_DIR before that happens.
"""
import os
import sys

# =============================================================================
# COGNEE PATH CONFIGURATION
# =============================================================================

def configure_cognee_paths():
    """
    Configure Cognee to use a writable directory.
    
    Priority (in order):
    1. HF_HOME (HuggingFace Spaces)
    2. /app/.cognee_data (Docker)
    3. .cognee_system (Local development)
    """
    # Detect environment and set writable path
    if os.getenv("HF_HOME"):
        cognee_root = os.path.join(os.getenv("HF_HOME"), "cognee_data")
        env_type = "HuggingFace Spaces"
    elif os.path.exists("/app"):
        cognee_root = "/app/.cognee_data"
        env_type = "Docker/Cloud"
    else:
        cognee_root = os.path.join(os.getcwd(), ".cognee_system")
        env_type = "Local Development"
    
    # Create directory with full permissions
    os.makedirs(cognee_root, exist_ok=True)
    
    # Set environment variables for Cognee to pick up
    # Cognee 0.5.2 uses SYSTEM_ROOT_DIRECTORY for its data directory
    os.environ["SYSTEM_ROOT_DIRECTORY"] = cognee_root
    
    # Also set legacy variables in case they're used
    os.environ["COGNEE_ROOT_DIR"] = cognee_root
    os.environ["COGNEE_DB_PATH"] = os.path.join(cognee_root, "databases")
    os.environ["COGNEE_DATA_DIR"] = cognee_root
    
    print(f"=" * 80)
    print(f"🧠 COGNEE CONFIGURATION")
    print(f"=" * 80)
    print(f"Environment: {env_type}")
    print(f"Cognee Root: {cognee_root}")
    print(f"SYSTEM_ROOT_DIRECTORY: {os.environ.get('SYSTEM_ROOT_DIRECTORY')}")
    print(f"Writable: {os.access(cognee_root, os.W_OK)}")
    print(f"=" * 80)
    
    return cognee_root


# Execute configuration immediately on import
COGNEE_ROOT = configure_cognee_paths()


# =============================================================================
# DISABLE COGNEE ACCESS CONTROL FOR LEGACY DATA
# =============================================================================
# Cognee 0.5.0+ enables multi-user access control by default
# This breaks access to data created before v0.5.0
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_cognee_setup():
    """Verify Cognee can initialize with our configuration."""
    try:
        import cognee
        print(f"✅ Cognee {cognee.__version__} imported successfully")
        print(f"✅ Configuration applied before Cognee initialization")
        return True
    except Exception as e:
        print(f"❌ Cognee import failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("\n🧪 Testing Cognee Configuration...")
    verify_cognee_setup()
