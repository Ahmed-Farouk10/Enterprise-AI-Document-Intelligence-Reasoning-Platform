import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock environment if not set
os.environ["COGNEE_ROOT_DIR"] = "/tmp/cognee_debug"
os.environ["COGNEE_DB_PATH"] = "/tmp/cognee_debug/cognee.db"
os.environ["COGNEE_VECTOR_DB_URL"] = "/tmp/cognee_debug/lancedb"

# Setup Cognee (runs on import)
import app.cognee_setup

from app.services.cognee_engine import cognee_engine
from app.core.cognee_config import get_cognee_config

async def debug_ingestion():
    print("🚀 Starting Debug Ingestion...")
    
    # Sample Resume Text
    sample_text = """
    Jane Doe
    Senior Python Developer
    jane.doe@example.com
    
    EXPERIENCE
    Senior Software Engineer at TechCorp (2020-Present)
    - Led migration to FastAPI and Python 3.11
    - Managed team of 5 developers
    - Implemented RAG pipelines using LangChain and Neo4j
    
    Junior Developer at StartupInc (2018-2020)
    - Built REST APIs with Flask directly
    - Optimized PostgreSQL queries
    
    SKILLS
    Python, FastAPI, Docker, Kubernetes, AWS, Neo4j, React
    """
    
    try:
        print(f"📄 Ingesting sample text ({len(sample_text)} chars)...")
        
        # Call the professional pipeline directly
        result = await cognee_engine.ingest_document_professional(
            document_text=sample_text,
            document_id="debug_doc_001",
            document_type="resume",
            metadata={"filename": "debug_resume.txt"}
        )
        
        print("\n✅ Ingestion Result:")
        print(result)
        
        # Check graph stats
        print("\n📊 Checking Graph Stats...")
        import cognee
        from app.core.cognee_config import get_cognee_config
        
        config = get_cognee_config()
        # Force default user
        user_id = config.default_user_id
        
        print(f"Searching for data under user: {user_id}")
        
        # search for users
        users = await cognee.search("User", user_id=user_id)
        print(f"Users found: {len(users)}")
        
        # search for positions
        data = await cognee.search("Position", user_id=user_id)
        print(f"Positions found: {len(data)}")
        for item in data:
            print(f" - {item}")

    except Exception as e:
        print(f"\n❌ ERROR CAUGHT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ingestion())
