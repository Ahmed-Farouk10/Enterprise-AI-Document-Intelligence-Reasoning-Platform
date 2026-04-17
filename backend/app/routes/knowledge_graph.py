"""
Knowledge Graph API Routes
Provides document relationship visualization and statistics
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.service import DatabaseService
from app.core.rate_limiter import limiter
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/graph", tags=["knowledge-graph"])


@router.get("/stats")
async def get_graph_stats(db: Session = Depends(get_db)):
    """Get knowledge graph statistics"""
    try:
        # Get document count
        documents, total = DatabaseService.get_documents(db, skip=0, limit=1000)
        total_documents = len(documents)
        
        # Build entity/relationship counts from vector store
        from app.services.vector_store import vector_store_service
        
        # Get unique document types
        document_types = set()
        if documents:
            for doc in documents:
                doc_type = _detect_document_type(doc.filename or "")
                document_types.add(doc_type)
        
        # Calculate mock relationships (1 per document pair of same type)
        total_relationships = max(0, total_documents - 1)
        
        # Calculate graph density
        if total_documents > 1:
            max_possible_edges = total_documents * (total_documents - 1) / 2
            graph_density = total_relationships / max_possible_edges if max_possible_edges > 0 else 0
        else:
            graph_density = 0
        
        stats = {
            "total_entities": len(document_types) + total_documents,
            "total_relationships": total_relationships,
            "total_documents": total_documents,
            "graph_density": round(graph_density, 2),
            "entity_types": list(document_types),
            "vector_store_size": _get_vector_store_stats()
        }
        
        logger.info("graph_stats_retrieved", stats=stats)
        return stats
        
    except Exception as e:
        logger.error("graph_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_graph_data(db: Session = Depends(get_db)):
    """Get knowledge graph data for visualization"""
    try:
        documents, _ = DatabaseService.get_documents(db, skip=0, limit=1000)
        
        nodes = []
        edges = []
        
        # Add document type nodes
        doc_types = {}
        for doc in documents:
            doc_type = _detect_document_type(doc.filename or "")
            if doc_type not in doc_types:
                doc_types[doc_type] = {
                    "id": f"type_{doc_type}",
                    "label": doc_type.capitalize(),
                    "type": "category",
                    "size": 20,
                    "color": _get_color_for_type(doc_type)
                }
                nodes.append(doc_types[doc_type])
            
            # Add document node
            nodes.append({
                "id": doc.id,
                "label": doc.filename or "Unknown",
                "type": "document",
                "doc_type": doc_type,
                "size": 15,
                "color": "#8b5cf6",
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            })
            
            # Add edge from type to document
            edges.append({
                "source": f"type_{doc_type}",
                "target": doc.id,
                "type": "belongs_to",
                "color": "#6b7280"
            })
        
        # Add relationships between documents of same type
        for doc_type, type_docs in _group_by_type(documents).items():
            doc_list = list(type_docs)
            for i in range(len(doc_list) - 1):
                edges.append({
                    "source": doc_list[i].id,
                    "target": doc_list[i + 1].id,
                    "type": "related",
                    "color": "#10b981"
                })
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "document_types": list(doc_types.keys())
            }
        }
        
        logger.info("graph_data_retrieved", nodes=len(nodes), edges=len(edges))
        return graph_data
        
    except Exception as e:
        logger.error("graph_data_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/relationships")
async def get_document_relationships(document_id: str, db: Session = Depends(get_db)):
    """Get relationships for a specific document"""
    try:
        document = DatabaseService.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get related documents (same type)
        all_docs, _ = DatabaseService.get_documents(db, skip=0, limit=1000)
        doc_type = _detect_document_type(document.filename or "")
        
        related = []
        for doc in all_docs:
            if doc.id != document_id and _detect_document_type(doc.filename or "") == doc_type:
                related.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "relationship": "same_type",
                    "confidence": 0.8
                })
        
        return {
            "document": {
                "id": document.id,
                "filename": document.filename,
                "status": document.status,
                "type": doc_type
            },
            "relationships": related,
            "count": len(related)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_relationships_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HELPER FUNCTIONS ====================

def _detect_document_type(filename: str) -> str:
    """Detect document type from filename"""
    filename_lower = filename.lower()
    
    if any(kw in filename_lower for kw in ["resume", "cv", "curriculum"]):
        return "resume"
    elif any(kw in filename_lower for kw in ["contract", "agreement", "legal"]):
        return "contract"
    elif any(kw in filename_lower for kw in ["invoice", "receipt", "bill"]):
        return "invoice"
    elif any(kw in filename_lower for kw in ["report", "analysis", "summary"]):
        return "report"
    elif any(kw in filename_lower for kw in ["manual", "guide", "documentation"]):
        return "manual"
    else:
        return "document"


def _get_color_for_type(doc_type: str) -> str:
    """Get color for document type"""
    colors = {
        "resume": "#3b82f6",
        "contract": "#ef4444",
        "invoice": "#f59e0b",
        "report": "#8b5cf6",
        "manual": "#10b981",
        "document": "#6b7280"
    }
    return colors.get(doc_type, "#6b7280")


def _group_by_type(documents) -> Dict[str, list]:
    """Group documents by type"""
    groups = {}
    for doc in documents:
        doc_type = _detect_document_type(doc.filename or "")
        if doc_type not in groups:
            groups[doc_type] = []
        groups[doc_type].append(doc)
    return groups


def _get_vector_store_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    try:
        from app.services.vector_store import vector_store_service
        # Mock stats for now
        return {
            "collections": 1,
            "total_chunks": 0
        }
    except Exception:
        return {"collections": 0, "total_chunks": 0}
