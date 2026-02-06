import os
import logging
from typing import List, Optional
from pydantic import BaseModel
from tavily import TavilyClient
from duckduckgo_search import DDGS

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
                tavily_response = self.tavily_client.search(
                    query=query, 
                    search_depth="advanced", 
                    max_results=num_results
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
