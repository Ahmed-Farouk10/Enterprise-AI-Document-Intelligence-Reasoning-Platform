
import logging
import asyncio
import os
import cognee
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    def __init__(self):
        self.initialized = False
        # Ensure we have a data directory for Cognee
        self.data_dir = os.path.join(os.getcwd(), "cognee_data")
        os.makedirs(self.data_dir, exist_ok=True)
        # Set Cognee configuration via environment variables or direct config if exposed
        # cognee.config.data_root = self.data_dir # Pseudo-code, depends on library version
        
    async def initialize(self):
        """Lazy initialization of Cognee resources if needed"""
        if self.initialized:
            return
        try:
            # Check if we need to run any startup logic
            # cognee.prune.prune_graph() # Optional cleanup
            self.initialized = True
            logger.info("Cognee Knowledge Graph Service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Cognee: {e}")

    async def add_document(self, content: str, doc_id: str):
        """
        Adds a document to the knowledge graph processing pipeline.
        """
        try:
            await self.initialize()
            logger.info(f"Adding document {doc_id} to Cognee Graph...")
            
            # 1. Add content to Cognee
            # Cognee add expects text or file path. We pass text.
            await cognee.add(content, dataset_name="default_dataset")
            
            # 2. Cognify (Extract entities and relationships)
            # This uses the LLM to build the graph
            logger.info(f"Cognifying document {doc_id}...")
            await cognee.cognify()
            
            logger.info(f"Document {doc_id} successfully processed into Graph.")
            return True
        except Exception as e:
            logger.error(f"Error processing document {doc_id} in Cognee: {e}")
            return False

    async def search_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Semantic Graph Search.
        Traverses the graph to find related concepts to the query.
        """
        try:
            await self.initialize()
            logger.info(f"Searching Graph for: {query}")
            
            # cognee.search returns a list of results (nodes/edges/text)
            results = await cognee.search(query)
            
            # Format results for our standard context usage
            formatted_results = []
            for res in results:
                # Default formatting, depends on Cognee's return structure
                # Typically it returns objects or dicts
                if hasattr(res, 'text'):
                     formatted_results.append({"content": res.text, "source": "knowledge_graph"})
                elif isinstance(res, dict) and 'text' in res:
                     formatted_results.append({"content": res['text'], "source": "knowledge_graph"})
                else:
                     formatted_results.append({"content": str(res), "source": "knowledge_graph"})
            
            return formatted_results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

# Singleton
kg_service = KnowledgeGraphService()
