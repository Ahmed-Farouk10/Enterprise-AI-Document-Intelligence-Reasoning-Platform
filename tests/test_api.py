"""
Test API Endpoints
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Test root endpoint"""
    response = await async_client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "online"
    assert "service" in data
    assert "version" in data
    assert "features" in data


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Test health check endpoint"""
    response = await async_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data


@pytest.mark.asyncio
async def test_system_info_endpoint(async_client: AsyncClient):
    """Test system info endpoint"""
    response = await async_client.get("/system/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "app" in data
    assert "llm" in data
    assert "vector_store" in data
    assert "features" in data


@pytest.mark.asyncio
async def test_create_chat_session(async_client: AsyncClient, test_session_data):
    """Test creating a chat session"""
    response = await async_client.post(
        "/api/chat/sessions",
        json=test_session_data
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "id" in data
    assert data["title"] == test_session_data["title"]
    assert "created_at" in data


@pytest.mark.asyncio
async def test_get_chat_sessions(async_client: AsyncClient):
    """Test retrieving chat sessions list"""
    response = await async_client.get("/api/chat/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_nonexistent_session(async_client: AsyncClient):
    """Test retrieving non-existent session returns 404"""
    response = await async_client.get("/api/chat/sessions/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_document(async_client: AsyncClient, test_document_data):
    """Test document upload endpoint"""
    # This test requires actual file upload
    # Skipping for now as it requires file fixture
    pytest.skip("Requires file upload fixture")


@pytest.mark.asyncio
async def test_cache_stats_endpoint(async_client: AsyncClient):
    """Test cache statistics endpoint"""
    response = await async_client.get("/api/cache/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert "redis" in data or "cag" in data
