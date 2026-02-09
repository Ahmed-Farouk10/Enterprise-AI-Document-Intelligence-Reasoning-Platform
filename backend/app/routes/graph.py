"""
Graph API Routes
Provides endpoints for knowledge graph visualization and statistics
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

from app.services.cognee_engine import CogneeEngine, get_cognee_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


# Response Models
class GraphStats(BaseModel):
    total_entities: int
    total_relationships: int
    total_documents: int
    graph_density: float


class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # document, entity, concept
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


@router.get("/stats", response_model=GraphStats)
async def get_graph_statistics(
    cognee_engine: CogneeEngine = Depends(get_cognee_engine)
) -> GraphStats:
    """
    Get knowledge graph statistics
    
    Returns:
        GraphStats: Statistics about the knowledge graph including
                   entity count, relationship count, document count, and graph density
    """
    try:
        logger.info("Fetching graph statistics")
        
        # Get statistics from Cognee engine
        stats = await cognee_engine.get_graph_statistics()
        
        # Calculate graph density
        # Density = (actual edges) / (possible edges)
        # For directed graph: possible edges = n * (n - 1)
        n = stats.get("entity_count", 0)
        edges = stats.get("relationship_count", 0)
        
        if n > 1:
            possible_edges = n * (n - 1)
            density = edges / possible_edges if possible_edges > 0 else 0.0
        else:
            density = 0.0
        
        return GraphStats(
            total_entities=stats.get("entity_count", 0),
            total_relationships=stats.get("relationship_count", 0),
            total_documents=stats.get("document_count", 0),
            graph_density=round(density, 3)
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch graph statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch graph statistics: {str(e)}"
        )


@router.get("/nodes", response_model=GraphData)
async def get_graph_nodes(
    limit: int = 100,
    document_id: str = None,
    cognee_engine: CogneeEngine = Depends(get_cognee_engine)
) -> GraphData:
    """
    Get knowledge graph nodes and edges
    
    Args:
        limit: Maximum number of nodes to return (default: 100)
        document_id: Optional document ID to filter graph by specific document
        
    Returns:
        GraphData: Graph nodes and edges for visualization
    """
    try:
        logger.info(f"Fetching graph nodes (limit={limit}, document_id={document_id})")
        
        # Get graph data from Cognee engine
        graph_data = await cognee_engine.get_graph_data(
            limit=limit,
            document_id=document_id
        )
        
        # Transform to response format
        nodes = []
        for node in graph_data.get("nodes", []):
            nodes.append(GraphNode(
                id=node.get("id", ""),
                label=node.get("label", node.get("name", "Unknown")),
                type=node.get("type", "entity"),
                properties=node.get("properties", {})
            ))
        
        edges = []
        for edge in graph_data.get("edges", []):
            edges.append(GraphEdge(
                source=edge.get("source", ""),
                target=edge.get("target", ""),
                label=edge.get("label", edge.get("type", "related_to")),
                properties=edge.get("properties", {})
            ))
        
        logger.info(f"Returning {len(nodes)} nodes and {len(edges)} edges")
        
        return GraphData(
            nodes=nodes,
            edges=edges
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch graph nodes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch graph nodes: {str(e)}"
        )


@router.get("/document/{document_id}", response_model=GraphData)
async def get_document_graph(
    document_id: str,
    cognee_engine: CogneeEngine = Depends(get_cognee_engine)
) -> GraphData:
    """
    Get knowledge graph for a specific document
    
    Args:
        document_id: Document ID to get graph for
        
    Returns:
        GraphData: Graph nodes and edges related to the document
    """
    try:
        logger.info(f"Fetching graph for document: {document_id}")
        
        # Use the general get_graph_nodes with document filter
        return await get_graph_nodes(
            limit=200,
            document_id=document_id,
            cognee_engine=cognee_engine
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch document graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch document graph: {str(e)}"
        )
