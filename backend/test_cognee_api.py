import os
import asyncio
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cognee Cloud Configuration from Documentation
# Default to the key provided by the user if not in .env yet for this test run
COGNEE_API_KEY = os.getenv("COGNEE_API_KEY", "7588119e6015fafdc33f8014e83d63bbce06eb5983321fa1")
API_BASE_URL = "https://api.cognee.ai"

async def test_cognee_cloud_api():
    """
    Test Cognee Cloud API as per documentation:
    curl -H "X-Api-Key: YOUR-API-KEY" -H "Content-Type: application/json" $BASE_URL/api/health
    """
    print(f"🚀 Testing Cognee Cloud API")
    print(f"📍 Base URL: {API_BASE_URL}")
    print(f"🔑 API Key: {COGNEE_API_KEY[:8]}...")

    headers = {
        "X-Api-Key": COGNEE_API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health Check
        try:
            print("\n🔍 Step 1: Health Check (/api/health)...")
            response = await client.get(f"{API_BASE_URL}/api/health", headers=headers)
            if response.status_code == 200:
                print(f"✅ Health Check Success: {response.json()}")
            else:
                print(f"❌ Health Check Failed ({response.status_code}): {response.text}")
                if response.status_code == 401:
                    print("🛑 Error: API Key is invalid.")
                    return
        except Exception as e:
            print(f"💥 Error connecting to health endpoint: {e}")

        # 2. Add Data (Example from Documentation)
        try:
            print("\n🔍 Step 2: Test Data Ingestion (/api/add)...")
            data = {"data": "Cognee API verification test."}
            response = await client.post(f"{API_BASE_URL}/api/add", json=data, headers=headers)
            if response.status_code in [200, 201]:
                print(f"✅ Data Ingestion Success: {response.json()}")
            else:
                print(f"⚠️ Data Ingestion Status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"💥 Error connecting to add endpoint: {e}")

if __name__ == "__main__":
    asyncio.run(test_cognee_cloud_api())
