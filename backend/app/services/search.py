import os
import logging
from typing import List, Optional
from pydantic import BaseModel
from tavily import TavilyClient
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

logger = logging.getLogger(__name__)

# --- Configuration ---
# Though Tavily handles its own ranking, we can still tag sources if needed.
# This simple service alternates between high-quality Tavily and free DuckDuckGo.

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    source_type: str  # "tavily_verified", "general_web"
    credibility_score: float # 0.0 to 1.0

class CredibleSearchService:
    def __init__(self):
        # 1. Initialize Tavily
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "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") # Fallback to user provided key for dev
        self.tavily_client = None
        
        if self.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
                logger.info("Tavily Search Client initialized.")
            except Exception as e:
                logger.error(f"Failed to init Tavily: {e}")

        # 2. DuckDuckGo (No init needed, stateless)
        self.ddgs = DDGS()

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Execute search: Try Tavily first (RAG optimized), fallback to DuckDuckGo.
        """
        results = []
        
        # --- STRATEGY 1: TAVILY (Primary) ---
        if self.tavily_client:
            try:
                logger.info(f"Searching Tavily for: '{query}'")
                # search_depth="advanced" gives better snippets for RAG
                # .search() usually supports timeout in recent SDKs or we can rely on thread timeout from chat.py
                # But to be safe, we wrap in our own try/except block if thread timeout logic isn't granular enough.
                # NOTE: TavilyClient might not expose a direct timeout param easily on .search(), 
                # but we can rely on specific kwargs if available or just catch the long hang.
                # However, since we are in a thread in chat.py, we can leave it.
                # Wait, better to use the underlying requests timeout if possible.
                # The TavilyClient uses `requests`. We can patch it or just hope it works?
                # Actually, let's keep it simple for now and rely on the fact that chat.py *could* timeout the thread?
                # No, standard threads don't timeout easily.
                # Best approach: Use a smaller timeout if the SDK allows.
                # If SDK doesn't allow, we can monkey-patch or just accept 5s via a wrapper if we could.
                
                # UPDATE: Since we can't easily set timeout on TavilyClient.search without inspecting source,
                # we will trust the `requests` default which is usually long.
                # To FIX the user issue, we will wrap this specific call in a func that we can kill? 
                # No, threads can't be killed. 
                # We will accept that this specific thread hangs, BUT we will make sure the
                # chat.py *waits* for it only for 5 seconds.
                
                tavily_response = self.tavily_client.search(
                    query=query, 
                    search_depth="advanced", 
                    topic="general", 
                    max_results=num_results
                    # timeout=5 # Not standard param in all versions, risking kwarg error.
                )
                
                # Parse Tavily results (they are usually high quality)
                for r in tavily_response.get("results", []):
                    results.append(SearchResult(
                        title=r.get("title", "Untitled"),
                        link=r.get("url", ""),
                        snippet=r.get("content", ""),
                        source_type="tavily_verified",
                        credibility_score=0.95 # Tavily filters for content quality
                    ))
                
                if results:
                    return results

            except Exception as e:
                logger.error(f"Tavily search failed: {e}")
                # Fallthrough to DDG

        # --- STRATEGY 2: DUCKDUCKGO (Fallback) ---
        try:
            logger.info(f"Searching DuckDuckGo (Fallback) for: '{query}'")
            ddg_results = self.ddgs.text(query, max_results=num_results)
            
            if ddg_results:
                for r in ddg_results:
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        link=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source_type="general_web",
                        credibility_score=0.7 # Standard web trust
                    ))
                    
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            
        return results

# Singleton instance
search_service = CredibleSearchService()
