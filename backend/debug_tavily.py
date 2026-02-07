from tavily import TavilyClient
import time
import os

key = "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" # The key from search.py
if not key:
    print("No API Key found")
    exit(1)

print(f"Testing Tavily with key: {key[:5]}...")

try:
    client = TavilyClient(api_key=key)
    start = time.time()
    # Simple search
    print("Initiating search for 'Python programming'...")
    response = client.search(query="Python programming", search_depth="basic", max_results=1)
    end = time.time()
    
    print(f"Success! Time taken: {end - start:.2f}s")
    print(f"Results: {response}")
except Exception as e:
    print(f"Error: {e}")
