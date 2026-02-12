
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.services.hybrid_search_service import hybrid_search_service
from app.services.neo4j_service import neo4j_service

async def verify_hybrid_search():
    print("🚀 Verifying Hybrid Search Service...")
    
    # 1. Test Neo4j Connection
    print("\n[1] Testing Neo4j Connection...")
    try:
        stats = await neo4j_service.get_graph_statistics()
        print(f"✅ Neo4j Connected. Stats: {stats}")
    except Exception as e:
        print(f"❌ Neo4j Connection Failed: {e}")
        return

    # 2. Test Parametric Search (Find a node)
    print("\n[2] Testing Parametric Search...")
    nodes = await neo4j_service.parametric_search("Ahmed", limit=5)
    print(f"✅ Found {len(nodes)} candidate nodes for 'Ahmed'")
    for n in nodes:
        print(f"   - {n['name']} ({n['labels']})")

    # 3. Test Hybrid Search
    query = "What is Ahmed's experience with Python?"
    print(f"\n[3] Testing Hybrid Search for: '{query}'...")
    
    try:
        result = await hybrid_search_service.search(query)
        print(f"✅ Search Successful!")
        print(f"   - Confidence: {result['confidence']}")
        print(f"   - Sources: {result['sources']}")
        print(f"   - Entities Found: {result['entities']}")
        print(f"   - Context Length: {len(result['full_context'])} chars")
        print("\n=== Context Preview ===")
        print(result['full_context'][:500] + "...")
    except Exception as e:
        print(f"❌ Hybrid Search Failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_hybrid_search())
