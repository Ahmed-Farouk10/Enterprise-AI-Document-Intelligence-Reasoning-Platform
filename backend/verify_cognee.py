
import asyncio
import os
import sys
import logging
import cognee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_cognee")

async def verify():
    print("--- Verifying Cognee Setup ---")
    
    # Check config
    print(f"Cognee Version: {cognee.__version__}")
    
    # Check environment variables
    print(f"Graph DB Type: {os.getenv('GRAPH_DATABASE_URL', 'Not Set')}")
    print(f"LLM API Key Set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    try:
        # 1. Add document
        print("\n[1] Adding Document...")
        text = "Ahmed is a software engineer at Google DeepMind. He works on AI reasoning."
        doc_id = "test_doc_001"
        await cognee.add(text, dataset_name="verification_dataset")
        print("Success: Document Added")
        
        # 2. Cognify
        print("\n[2] Cognifying (Extracting Graph)...")
        await cognee.cognify()
        print("Success: Cognify Completed")
        
        # 3. Search
        print("\n[3] Searching Graph...")
        results = await cognee.search("Where does Ahmed work?")
        print(f"Results Found: {len(results)}")
        for r in results:
            print(f"- {r}")
            
    except Exception as e:
        print(f"\n[ERROR] Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify())
