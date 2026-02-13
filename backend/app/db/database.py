from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import logging
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Database URL - defaults to SQLite for development
# CRITICAL: On HF Spaces, use persistent storage for chat history/sessions
# We define the absolute persistent path provided by HF Spaces here
PERSISTENT_ROOT = "/app/.cache/cognee_data"

# Cognee standard environment variables (from documentation)
os.environ["COGNEE_DATA_ROOT"] = PERSISTENT_ROOT
os.environ["COGNEE_STORAGE_PATH"] = os.path.join(PERSISTENT_ROOT, "storage")
os.environ["COGNEE_DB_PATH"] = os.path.join(PERSISTENT_ROOT, "databases")

# FIX: Stop vanishing chats by forcing the main app DB into the persistent folder
# Ensure the directory exists
os.makedirs(os.path.join(PERSISTENT_ROOT, "databases"), exist_ok=True)
DATABASE_PATH = os.path.join(PERSISTENT_ROOT, "databases", "app_persistent_chat.db")

# Fallback for local dev if not in container
if not os.path.exists("/app/.cache"):
    DATABASE_PATH = "./docucentric.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
# User suggested name for SQLAlchemy
SQLALCHEMY_DATABASE_URL = DATABASE_URL

# For PostgreSQL in production, use:
# DATABASE_URL = "postgresql://user:password@localhost/dbname"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Retry logic for DB connection on startup
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OperationalError, Exception)),
    reraise=True
)
def wait_for_db():
    """Validates database connection with retries"""
    try:
        # Try to create a connection and execute a simple query
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.warning(f"Database connection failed, retrying... Error: {e}")
        raise e
