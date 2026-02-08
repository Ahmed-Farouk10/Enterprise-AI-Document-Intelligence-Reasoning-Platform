# app/services/knowledge_graph.py
import logging
from typing import List, Dict, Any
from app.services.cognee_engine import cognee_engine, AnalysisMode

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """
    Wrapper for CogneeEngine to maintain backward compatibility 
    while upgrading to the full graph reasoning architecture.
    """
    def __init__(self):
        self.engine = cognee_engine

    async def initialize(self):
        """Initialize Cognee resources"""
        await self.engine.initialize()

    async def add_document(self, content: str, doc_id: str, document_type: str = "auto_detect"):
        """
        Adds a document to the knowledge graph processing pipeline.
        Transitioned to full Cognee architecture.
        """
        try:
            logger.info(f"Adding document {doc_id} to Cognee Engine...")
            graph_info = await self.engine.ingest_document(
                document_text=content,
                document_id=doc_id,
                document_type=document_type
            )
            logger.info(f"Document {doc_id} successfully cognified. Entities: {graph_info.entity_count}")
            return True
        except Exception as e:
            logger.error(f"Error processing document {doc_id} in Cognee: {e}")
            return False

    async def search_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Semantic Graph Search with reasoning.
        """
        try:
            logger.info(f"Searching Graph for: {query}")
            result = await self.engine.query(
                question=query,
                document_ids=[], # Searches across all if empty
                mode=AnalysisMode.SUMMARY
            )
            
            # Format results for legacy context usage
            formatted_results = []
            for entity in result.entities_involved:
                content = f"{entity['name']}: {entity['description']}" if entity['description'] else entity['name']
                formatted_results.append({"content": content, "source": "knowledge_graph"})
            
            # If no entities but we have an answer, use that
            if not formatted_results and result.answer:
                formatted_results.append({"content": result.answer, "source": "knowledge_graph"})
                
            return formatted_results
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

# Singleton
kg_service = KnowledgeGraphService()
