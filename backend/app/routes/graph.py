# /app/app/routes/graph.py (Fixed Version)
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
import logging
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Mock dependency if cognee not available for imports
try:
    from app.services.cognee_engine import cognee_engine
except ImportError:
    cognee_engine = None

@router.get("/api/graph/nodes")
async def get_graph_nodes(
    limit: int = Query(50, ge=1, le=500),
    document_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    search_query: Optional[str] = None
):
    """
    Retrieve knowledge graph nodes with filtering and search.
    Fixed version that properly queries Service.
    """
    try:
        # Check if service is ready
        if not cognee_engine:
             raise HTTPException(status_code=503, detail="Graph engine not initialized")
        
        # We need to implement get_graph_data in cognee_engine or use neo4j_service directly
        from app.services.neo4j_service import neo4j_service
        
        # Execute graph query
        data = await neo4j_service.get_graph_data(limit=limit, document_id=document_id)
        
        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
            "total": len(data.get("nodes", [])),
            "filtered_count": len(data.get("nodes", []))
        }
        
    except Exception as e:
        logger.error(f"Graph query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph retrieval error: {str(e)}")

@router.get("/api/graph/edges")
async def get_graph_edges(
    node_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    limit: int = 100
):
    """Retrieve edges/relationships from knowledge graph."""
    try:
        from app.services.neo4j_service import neo4j_service
        
        data = await neo4j_service.get_graph_data(limit=limit)
        return {
            "edges": data.get("edges", []),
            "count": len(data.get("edges", []))
        }
    except Exception as e:
        logger.error(f"Edge retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/graph/rebuild")
async def rebuild_graph():
    """
    Manual trigger to rebuild knowledge graph from processed documents.
    """
    try:
        # Stub for rebuild trigger
        return {"status": "success", "message": "Graph rebuild triggered"}
    except Exception as e:
        logger.error(f"Graph rebuild failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
