import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# --- 1. DATABASE CONFIGURATION ---
# We prioritize the environment DATABASE_URL (Supabase/Postgres)
# and fallback to a local persistent SQLite file for HF Spaces/local dev.
SQLALCHEMY_DATABASE_URL = settings.database.DATABASE_URL

# Handle SQLite specific settings
connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    # Ensure directory exists for local SQLite
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
    if "/" in db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

# --- 2. SQLALCHEMY SETUP ---
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args=connect_args
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

# --- 4. STARTUP VERIFICATION ---
def wait_for_db():
    """
    Verifies database accessibility on startup.
    This function is explicitly imported by main.py, so it must exist.
    """
    try:
        logger.info(f"Checking database connection at {SQLALCHEMY_DATABASE_URL.split('@')[-1] if '@' in SQLALCHEMY_DATABASE_URL else SQLALCHEMY_DATABASE_URL}...")
        
        # Test connection
        with engine.connect() as connection:
            pass
            
        logger.info("✅ Database connection established.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        # In development, we might not want to crash immediately if DB is slow
        if settings.ENVIRONMENT == "production":
            raise e
