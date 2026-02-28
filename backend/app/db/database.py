import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Configure logging
logger = logging.getLogger(__name__)

# --- 1. PERSISTENT CONFIGURATION ---
# We force the database to live in the writable cache folder provided by HF Spaces
RAG_ROOT = "/app/.cache/rag_data"
DB_DIR = os.path.join(RAG_ROOT, "databases")

# Ensure the directory exists immediately so we don't get 'unable to open database file'
os.makedirs(DB_DIR, exist_ok=True)

# Define the persistent file path
# This ensures your chat history survives restarts
DB_FILE = os.path.join(DB_DIR, "app_persistent_chat.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

# --- 2. SQLALCHEMY SETUP ---
# check_same_thread=False is required for SQLite with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 3. DEPENDENCIES ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 4. THE MISSING FUNCTION (RESTORED) ---
def wait_for_db():
    """
    Verifies database accessibility on startup.
    This function is explicitly imported by main.py, so it must exist.
    """
    try:
        logger.info(f"Checking database connection at {SQLALCHEMY_DATABASE_URL}...")
        
        # Double check directory exists
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)
        
        # Test connection
        with engine.connect() as connection:
            pass
            
        logger.info("✅ Database connection established.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise e
