import os
import logging
import time
from sqlalchemy import create_engine, text
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
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300
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
def wait_for_db(max_retries: int = 3, delay: int = 2):
    """
    Verifies database accessibility on startup with retries.
    Does not crash the app if DB is unavailable to allow graceful degradation.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            logger.info(f"Checking database connection (Attempt {retry_count + 1}/{max_retries})...")
            
            # Test connection
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            logger.info("✅ Database connection established.")
            return True
        except Exception as e:
            retry_count += 1
            logger.warning(f"⚠️ Database connection attempt {retry_count} failed: {e}")
            if retry_count < max_retries:
                time.sleep(delay)
            else:
                logger.error("❌ All database connection attempts failed. App will start but DB features may fail.")
    return False
