import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.services.retreival import vector_store
from app.services.neo4j_service import neo4j_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SearchConfig:
    limit: int = 10
    alpha: float = 0.6  # Weight for Semantic Search (0.6) vs Keyword (0.4) inside Vector Store
    graph_weight: float = 0.3  # Weight for Graph Evidence in final fusion
    use_reranking: bool = True
    hops: int = 1

class HybridSearchService:
    """
    Orchestrates fusion of Vector Search and Knowledge Graph Traversal.
    """
    
    def __init__(self):
        self.vector_store = vector_store
        self.neo4j = neo4j_service
        
    async def search(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        config: Optional[SearchConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid search strategy:
        1. Retrieve top chunks from Vector Store (Hybrid: Semantic + Keyword)
        2. Identify 'Seed Entities' from those chunks (and query)
        3. Traverse Graph from seeds to get 'Graph Facts'
        4. Fuse results
        """
        if config is None:
            config = SearchConfig()
            
        logger.info(f"🔍 Starting Hybrid Search for: '{query}'")
        
        # Step 1: Vector Search (Parallelizable, but usually fast enough synchronous here)
        # We use the existing highly optimized retrieve_with_citations
        vector_results = await asyncio.to_thread(
            self.vector_store.retrieve_with_citations,
            query=query,
            k=config.limit,
            use_hybrid=True,
            use_reranking=config.use_reranking
        )
        
        # Extract potential entities from Vector Results to seed the graph search
        # Refinement: Also Parametric Search on the query itself
        seed_entity_ids = await self._identify_seeds(query, vector_results)
        
        # Step 2: Graph Expansion (Context via Traversal)
        graph_facts = []
        if seed_entity_ids:
            logger.info(f"🕸️ Expanding graph context from {len(seed_entity_ids)} seeds...")
            graph_facts = await self.neo4j.expand_context(
                seed_ids=list(seed_entity_ids),
                hops=config.hops,
                limit=config.limit
            )
        
        # Step 3: Result Fusion
        fused_context = self._fuse_results(vector_results, graph_facts, config)
        
        return {
            "full_context": fused_context["text"],
            "sources": fused_context["sources"],
            "entities": [f['source'] for f in graph_facts],
            "confidence": fused_context["confidence"],
            "retrieval_method": "hybrid_fusion_v1"
        }

    async def _identify_seeds(self, query: str, vector_results: List[Dict]) -> set:
        """
        Identify seed node IDs for graph traversal.
        Strategy:
        1. Parametric search on the Query (find nodes mentioned in query).
        2. (Optional/Future) Extract entities from top vector chunks.
        """
        seeds = set()
        
        # Strategy A: Query Parametric Search
        # Find nodes directly matching query terms
        query_nodes = await self.neo4j.parametric_search(query, limit=5)
        for node in query_nodes:
            seeds.add(node["id"])
            
        return seeds

    def _fuse_results(
        self, 
        vector_results: List[Dict], 
        graph_facts: List[Dict], 
        config: SearchConfig
    ) -> Dict[str, Any]:
        """
        Combine Vector Chunks and Graph Facts into a coherent context.
        """
        
        context_parts = []
        sources = []
        
        # 1. Graph Facts (High Precision / Structured)
        if graph_facts:
            facts_text = "\n".join([f"- {f['text']}" for f in graph_facts])
            context_parts.append(f"**KNOWLEDGE GRAPH VERIFIED FACTS**:\n{facts_text}")
            sources.append({"type": "graph", "count": len(graph_facts)})
            
        # 2. Vector Chunks (High Recall / Nuance)
        if vector_results:
            chunks_text = ""
            for i, res in enumerate(vector_results):
                chunks_text += f"\n[Document: {res.get('doc_name', 'Unknown')}]\n{res['text']}\n"
                
            context_parts.append(f"\n**DOCUMENT EXCERPTS**:\n{chunks_text}")
            sources.append({"type": "vector", "count": len(vector_results)})
            
        full_text = "\n\n".join(context_parts)
        
        # Simple confidence heuristic
        confidence = 0.0
        if vector_results: confidence += 0.5
        if graph_facts: confidence += 0.4
        
        # Verify if we found anything
        if not full_text.strip():
            full_text = "No relevant information found in documents or knowledge graph."
        
        return {
            "text": full_text,
            "sources": sources,
            "confidence": min(confidence, 1.0)
        }

# Singleton
hybrid_search_service = HybridSearchService()
