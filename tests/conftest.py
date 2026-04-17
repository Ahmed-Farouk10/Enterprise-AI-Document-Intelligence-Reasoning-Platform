"""
Pytest Configuration and Fixtures
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import settings
from app.db.database import Base, get_db


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite:///./test_data/test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db(test_engine) -> Generator:
    """Create database session for each test"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def override_db(test_db):
    """Override database dependency"""
    def _get_test_db():
        yield test_db
    
    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_document_data():
    """Sample document data for testing"""
    return {
        "filename": "test_resume.pdf",
        "content": b"Test document content"
    }


@pytest.fixture
def test_chat_message():
    """Sample chat message for testing"""
    return {
        "content": "What are the key skills mentioned in the resume?"
    }


@pytest.fixture
def test_session_data():
    """Sample session data for testing"""
    return {
        "title": "Test Session",
        "document_ids": ["test-doc-1"]
    }
