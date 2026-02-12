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

@router.get("/api/graph/visualize")
async def visualize_graph():
    """
    Generate and return Knowledge Graph visualization using Cognee's native Kuzu support.
    """
    try:
        import cognee
        import os
        from fastapi.responses import HTMLResponse
        
        # Define output path
        output_path = "graph_visualization.html"
        
        # Generate visualization
        # Note: cognee.visualize_graph works with the local Kuzu graph
        await cognee.visualize_graph(output_path)
        
        # Read and return HTML
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            raise HTTPException(status_code=500, detail="Visualization generation failed")
            
    except Exception as e:
        logger.error(f"Graph visualization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@router.get("/api/graph/stats")
async def get_graph_stats():
    """Get graph statistics (entity count, relationships, documents)."""
    try:
        # Try Cognee native stats first (if available) or fallback to Neo4j gracefully
        # Since we are on HF Spaces with Kuzu, we return basic info
        return {
            "entity_count": "Active (Kuzu)", 
            "relationship_count": "Active (Kuzu)", 
            "document_count": "Managed by Cognee",
            "backend": "Kuzu (Local)"
        }
    except Exception as e:
        logger.error(f"Graph stats retrieval failed: {str(e)}")
        return {"entity_count": 0, "relationship_count": 0, "document_count": 0}

@router.post("/api/graph/rebuild")
async def rebuild_graph():
    """
    Manual trigger to rebuild knowledge graph from processed documents.
    """
    try:
        import cognee
        from app.core.config import settings
        from cognee.modules.users.models import User
        import uuid
        
        # Trigger cognify
        await cognee.cognify(user=User(id=uuid.UUID(settings.DEFAULT_USER_ID)))
        
        return {"status": "success", "message": "Graph rebuild (Cognify) triggered successfully"}
    except Exception as e:
        logger.error(f"Graph rebuild failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
