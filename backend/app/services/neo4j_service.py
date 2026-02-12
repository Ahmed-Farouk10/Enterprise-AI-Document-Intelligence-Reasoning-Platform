"""
Neo4j Database Service
Provides direct Neo4j connection for graph queries
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
import logging
import os

logger = logging.getLogger(__name__)


class Neo4jService:
    """Direct Neo4j database service for graph queries"""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "changeme123")
        self.driver: Optional[Driver] = None
        self._available = False
        
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self._available = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            
        except Exception as e:
            self.driver = None
            self._available = False
            logger.warning(f"Neo4j connection failed: {e}. Graph features will be disabled.")
            # Do NOT raise, just degrad gracefully
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics from Neo4j
        
        Returns:
            Dict with entity_count, relationship_count, document_count
        """
        if not self.driver:
            self.connect()
            
        if not self._available or not self.driver:
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "document_count": 0
            }
        
        try:
            with self.driver.session() as session:
                # Count all nodes (entities)
                entity_result = session.run("MATCH (n) RETURN count(n) as count")
                entity_count = entity_result.single()["count"]
                
                # Count all relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                relationship_count = rel_result.single()["count"]
                
                # Count document nodes specifically
                doc_result = session.run(
                    "MATCH (d) WHERE d:Document OR d.type = 'document' RETURN count(d) as count"
                )
                document_count = doc_result.single()["count"]
                
                logger.info(f"Graph stats: {entity_count} entities, {relationship_count} relationships, {document_count} documents")
                
                return {
                    "entity_count": entity_count,
                    "relationship_count": relationship_count,
                    "document_count": document_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "document_count": 0
            }
    
    async def get_graph_data(
        self,
        limit: int = 100,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get graph nodes and edges for visualization
        
        Args:
            limit: Maximum number of nodes to return
            document_id: Optional document ID to filter by
            
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        if not self.driver:
            self.connect()
            
        if not self._available or not self.driver:
            logger.warning("Neo4j unavailable - returning empty graph")
            return {"nodes": [], "edges": []}
        
        try:
            with self.driver.session() as session:
                nodes = []
                edges = []
                
                if document_id:
                    # Get nodes related to specific document
                    node_query = """
                    MATCH (d)-[*1..2]-(n)
                    WHERE d.id = $document_id OR elementId(d) = $document_id
                    RETURN DISTINCT n
                    LIMIT $limit
                    """
                    node_result = session.run(node_query, document_id=document_id, limit=limit)
                else:
                    # Get all nodes up to limit
                    node_query = "MATCH (n) RETURN n LIMIT $limit"
                    node_result = session.run(node_query, limit=limit)
                
                # Process nodes
                node_ids = set()
                for record in node_result:
                    node = record["n"]
                    node_id = str(node.element_id)
                    node_ids.add(node_id)
                    
                    # Determine node type
                    labels = list(node.labels) if hasattr(node, 'labels') else []
                    node_type = labels[0].lower() if labels else node.get("type", "entity")
                    
                    nodes.append({
                        "id": node_id,
                        "label": node.get("name", node.get("label", node.get("title", "Unknown"))),
                        "type": node_type,
                        "properties": dict(node)
                    })
                
                # Get relationships between these nodes
                if node_ids:
                    edge_query = """
                    MATCH (n)-[r]->(m)
                    WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
                    RETURN n, r, m
                    LIMIT $limit
                    """
                    edge_result = session.run(edge_query, node_ids=list(node_ids), limit=limit)
                    
                    for record in edge_result:
                        source_id = str(record["n"].element_id)
                        target_id = str(record["m"].element_id)
                        rel = record["r"]
                        
                        edges.append({
                            "source": source_id,
                            "target": target_id,
                            "label": rel.type if hasattr(rel, 'type') else "related_to",
                            "properties": dict(rel) if hasattr(rel, '__iter__') else {}
                        })
                
                logger.info(f"Retrieved {len(nodes)} nodes and {len(edges)} edges from Neo4j")
                
                return {
                    "nodes": nodes,
                    "edges": edges
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return {
                "nodes": [],
                "edges": []
            }


    async def parametric_search(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search for nodes based on text properties (fuzzy match).
        Used to find 'seed entities' in the graph from user query.
        """
        if not self.driver:
            self.connect()
            
        if not self._available or not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                # Search across common text properties
                query = """
                MATCH (n)
                WHERE 
                    toLower(n.name) CONTAINS toLower($text) OR
                    toLower(n.title) CONTAINS toLower($text) OR
                    toLower(n.description) CONTAINS toLower($text) OR
                    toLower(n.label) CONTAINS toLower($text)
                RETURN n
                LIMIT $limit
                """
                result = session.run(query, text=query_text, limit=limit)
                
                nodes = []
                for record in result:
                    node = record["n"]
                    nodes.append({
                        "id": str(node.element_id),
                        "element_id": str(node.element_id), # Keep consistent
                        "name": node.get("name", node.get("title", "Unknown")),
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })
                return nodes
        except Exception as e:
            logger.error(f"Parametric search failed: {e}")
            return []

    async def expand_context(self, seed_ids: List[str], hops: int = 1, limit: int = 50) -> List[Dict]:
        """
        Traverse graph from seed nodes to find relevant context.
        Returns a list of 'facts' or 'triples' (Source -> Rel -> Target).
        """
        if not self.driver:
            self.connect()
            
        if not self._available or not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                # Traverse relationships from seeds
                # We use elementId for lookup if possible, or assume properties.id if that's how we stored them
                # For safety, we try both property-based ID and system elementId matching if the passed IDs look like system IDs
                
                query = """
                MATCH (source)-[r]->(target)
                WHERE 
                    (source.id IN $ids OR elementId(source) IN $ids)
                RETURN source, r, target
                LIMIT $limit
                """
                
                result = session.run(query, ids=seed_ids, limit=limit)
                
                triples = []
                for record in result:
                    s = record["source"]
                    r = record["r"]
                    t = record["target"]
                    
                    s_name = s.get("name", s.get("title", "Unknown"))
                    t_name = t.get("name", t.get("title", "Unknown"))
                    r_type = r.type
                    
                    # Create a readable 'fact'
                    triples.append({
                        "source": s_name,
                        "relationship": r_type,
                        "target": t_name,
                        "text": f"{s_name} {r_type} {t_name}",
                        "weight": 1.0  # Default weight for graph facts
                    })
                    
                return triples
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return []

# Singleton instance
neo4j_service = Neo4jService()


def get_neo4j_service() -> Neo4jService:
    """Get Neo4j service instance for dependency injection"""
    return neo4j_service
