import os
import sys
import logging

# --- 1. GLOBAL PATHS ---
# We use /app/.cache/app_data as the single source of truth for our vector DB and local models
STORAGE_ROOT = "/app/.cache/app_data"
DB_PATH = os.path.join(STORAGE_ROOT, "databases")

# Configure logging for setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("storage_setup")

def log(msg, level="info"):
    """Dual logging to logger and stderr for guaranteed visibility"""
    getattr(logger, level)(msg)
    print(f"[{level.upper()}] {msg}", file=sys.stderr)

def verify_storage_setup():
    """Ensures directories exist and sets basic environment variables for LLMs."""
    log(f"[SETUP] Starting simplified verification for root: {STORAGE_ROOT}")
    
    # 1. Ensure basic working directories exist (for Vector DB and SQLite)
    os.makedirs(DB_PATH, mode=0o777, exist_ok=True)

    # 2. Configure LLM Defaults
    current_provider = os.getenv("LLM_PROVIDER", "").lower()
    if not current_provider:
        os.environ["LLM_PROVIDER"] = "huggingface"
        
    if not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"

    # Ensure Key is present (HF_TOKEN preferred)
    if not os.getenv("LLM_API_KEY") or os.getenv("LLM_API_KEY").startswith("AIza"):
        if os.getenv("HF_TOKEN"):
             os.environ["LLM_API_KEY"] = os.getenv("HF_TOKEN")
        else:
             os.environ["LLM_API_KEY"] = "local" # Fallback prevents crash
    
    log(f"[SETUP] Basic LLM Configuration Complete.")

# Execute verification on import
verify_storage_setup()
