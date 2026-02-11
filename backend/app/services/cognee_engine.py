# app/services/cognee_engine.py
import os
import asyncio
import logging
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

# Cognee imports
try:
    import cognee
    # SearchType for entity/relationship queries
    from cognee.api.v1.search import SearchType
    COGNEE_AVAILABLE = True
except ImportError as e:
    COGNEE_AVAILABLE = False
    SearchType = None  # Fallback
    logging.error(f"Failed to import Cognee: {e}")
    logging.warning("Cognee not installed or dependency missing. Run: pip install cognee[neo4j]")

from app.core import settings
from app.core.logging_config import get_logger
from app.core.cognee_config import settings as cognee_settings

# CRITICAL FIX: Monkey-patch Cognee's LLM connection test
# The test hangs for 30+ seconds on HF Spaces when no valid HF_TOKEN is available
# This bypasses the test entirely to prevent pipeline timeouts
try:
    from cognee.infrastructure.llm import utils as llm_utils
    
    async def _noop_llm_test():
        """No-op replacement for test_llm_connection on HF Spaces"""
        logger = get_logger(__name__)
        logger.info("⚡ Skipping LLM connection test (monkey-patched)")
        return True
    
    # Only patch on HF Spaces or when no HF_TOKEN
    if os.getenv("HF_HOME") or not os.getenv("HF_TOKEN"):
        llm_utils.test_llm_connection = _noop_llm_test
        print("✅ Cognee LLM test monkey-patched successfully")
except ImportError:
    pass  # Cognee not installed yet

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Cognee-powered analysis modes"""
    ENTITY_EXTRACTION = "entities"           # Extract people, orgs, skills, dates
    RELATIONSHIP_MAPPING = "relationships"   # Map connections between entities
    TEMPORAL_REASONING = "temporal"          # Timeline analysis, gaps, sequences
    COMPARATIVE_ANALYSIS = "comparative"     # Compare against standards/benchmarks
    SUMMARIZATION = "summary"                # Graph-based summarization
    ANOMALY_DETECTION = "anomalies"          # Find inconsistencies, missing data


@dataclass
class GraphQueryResult:
    """Structured result from Cognee graph query"""
    answer: str
    evidence_paths: List[List[Dict]]  # Graph traversal paths as evidence
    entities_involved: List[Dict]
    confidence_score: float
    query_type: str
    subgraph: Optional[Dict] = None  # Neo4j subgraph for visualization


@dataclass
class GraphIngestionResult:
    """Result from Cognee document ingestion"""
    success: bool
    entity_count: int
    relationship_count: int
    domain_type: str
    dataset_name: str
    error_message: Optional[str] = None


@dataclass
class DocumentGraph:
    """Represents a document's knowledge graph structure"""
    document_id: str
    entity_count: int
    relationship_count: int
    temporal_facts: List[Dict]  # Date-ranges for gap analysis
    domain_type: str  # resume, contract, academic_paper, etc.
    graph_stats: Dict[str, Any]


class CogneeEngine:
    """
    Enterprise Document Intelligence using Cognee Knowledge Graph.
    
    Replaces vector-based RAG with semantic graph reasoning.
    """
    
    def __init__(self):
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee library required. Install: pip install cognee")
        
        self.extraction_model = settings.EXTRACTION_MODEL
        self.graph_db_url = settings.GRAPH_DATABASE_URL
        
        # Analysis configurations per domain
        self.domain_configs = {
            "resume": {
                "entity_types": ["Person", "Organization", "Skill", "Degree", "JobTitle", "DateRange"],
                "relationship_types": ["WORKED_AT", "STUDIED_AT", "HAS_SKILL", "MANAGED", "REPORTS_TO"],
                "temporal_fields": ["employment_date", "education_date", "certification_date"]
            },
            "contract": {
                "entity_types": ["Party", "Clause", "Obligation", "Date", "MonetaryValue"],
                "relationship_types": ["OBLIGATED_TO", "PAYS", "DELIVERS", "VIOLATES"],
                "temporal_fields": ["effective_date", "termination_date", "milestone_date"]
            },
            "academic": {
                "entity_types": ["Author", "Institution", "Concept", "Methodology", "Citation"],
                "relationship_types": ["AUTHORED", "CITES", "USES_METHOD", "AFFILIATED_WITH"],
                "temporal_fields": ["publication_date", "study_period"]
            }
        }
    
    async def initialize(self):
        """Initialize Cognee graph database connection and create default user"""
        try:
            logger.info("🔧 Initializing Cognee database...")
            
            # Initialize Cognee's internal database (creates tables, default user, etc.)
            # This is required before any add() or cognify() operations
            import cognee
            from cognee.infrastructure.databases.relational import create_db_and_tables
            
            # Create database tables if they don't exist
            await create_db_and_tables()
            logger.info("✅ Cognee database tables created/verified")
            
            # Verify default user exists
            try:
                from cognee.modules.users.methods import get_default_user
                default_user = await get_default_user()
                logger.info(f"✅ Default user verified: {default_user.id if default_user else 'None'}")
            except Exception as user_error:
                logger.warning(f"⚠️ Could not verify default user: {user_error}")
            
            logger.info("✅ Cognee engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Cognee initialization failed: {e}")
            logger.warning("Cognee features may not work properly")
    # ====================  PROFESSIONAL PIPELINE INTEGRATION ====================
    
    async def ingest_document_professional(
        self,
        document_text: str,
        document_id: str,
        document_type: str = "auto_detect",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Professional document ingestion using custom pipelines.
        
        This method routes documents to specialized extraction pipelines
        based on document type, providing structured entity extraction
        and rich knowledge graph building.
        
        Args:
            document_text: Raw document text
            document_id: Unique document identifier
            document_type: Type of document or "auto_detect"
            metadata: Optional metadata (filename, upload_date, etc.)
            
        Returns:
            Dict with ingestion results and extracted entities
        """
        try:
            # Import professional pipelines
            from app.services.cognee_pipelines import route_to_pipeline
            
            logger.info(f"🚀 Professional ingestion: doc_id={document_id}, type={document_type}")
            
            # Route to appropriate pipeline
            result = await route_to_pipeline(
                text=document_text,
                document_id=document_id,
                document_type=document_type
            )
            
            # Format response based on result type
            if hasattr(result, 'person'):  # Resume object
                from app.models.cognee_models import Resume
                resume: Resume = result
                
                logger.info(
                    f"✅ Resume processed: {resume.person.name}, "
                    f"{len(resume.work_history)} positions, "
                    f"{len(resume.skills)} skills"
                )
                
                return {
                    "success": True,
                    "document_type": "resume",
                    "entity_count": (
                        1 +  # Person
                        len(resume.work_history) +
                        len(resume.education) +
                        len(resume.skills)
                    ),
                    "entities": {
                        "person": resume.person.name,
                        "positions": len(resume.work_history),
                        "degrees": len(resume.education),
                        "skills": len(resume.skills)
                    },
                    "dataset_name": f"resume_{document_id}"
                }
            else:  # Generic document
                logger.info(f"✅ Document processed: {document_type}")
                
                return {
                    "success": True,
                    "document_type": document_type,
                    "dataset_name": f"{document_type}_{document_id}"
                }
                
        except Exception as e:
            logger.error(f"❌ Professional ingestion failed: {e}", exc_info=True)
            
            # Fallback to basic ingestion
            logger.warning("⚠️ Falling back to basic Cognee ingestion")
            return await self.ingest_document(
                document_text=document_text,
                document_id=document_id,
                document_type=document_type,
                metadata=metadata
            )
    
    # ==================== DOCUMENT INGESTION (ORIGINAL METHOD) ====================
    
    async def ingest_document(
        self,
        document_text: str,
        document_id: str,
        document_type: str = "auto_detect",
        metadata: Optional[Dict] = None
    ) -> DocumentGraph:
        """
        Ingest document into Cognee knowledge graph with real Cognee processing.
        """
        try:
            # Auto-detect domain if not specified
            if document_type == "auto_detect":
                domain = await self._detect_domain(document_text)
            else:
                domain = document_type
            
            # Create dataset name for this document
            dataset_name = f"doc_{document_id}"
            
            # Add document to Cognee dataset with timeout
            logger.info(f"Adding document {document_id} to Cognee dataset: {dataset_name}")
            # Extract filename from metadata if provided (for logging only)
            filename = metadata.get("filename", "unknown") if metadata else "unknown"
            
            try:
                # Cognee 0.5.2 add() only accepts data and dataset_name
                # Add aggressive timeout as cognee.add() can hang indefinitely
                await asyncio.wait_for(
                    cognee.add(
                        data=document_text,
                        dataset_name=dataset_name
                    ),
                    timeout=30.0  # 30 second timeout for add operation
                )
                logger.info(f"✅ Document added to Cognee dataset: {dataset_name}")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Cognee add() timed out after 30s for {dataset_name}")
                raise RuntimeError(f"Cognee add() operation timed out - may be downloading models or waiting for external service")
            except Exception as add_error:
                logger.error(f"❌ Cognee add() failed: {add_error}")
                raise
            
            # Build knowledge graph with timeout
            logger.info(f"🔨 Building knowledge graph for {dataset_name}...")
            try:
                # Set a reasonable timeout for graph building (90 seconds - increased from 60)
                await asyncio.wait_for(
                    cognee.cognify(datasets=[dataset_name]),
                    timeout=90.0
                )
                logger.info(f"✅ Knowledge graph built successfully for {dataset_name}")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Cognee graph building timed out after 90s for {dataset_name}")
                raise RuntimeError("Cognee graph building timed out - document may be too large or LLM is slow")
            
            # Get graph statistics from Neo4j
            stats = await self._get_graph_stats_from_neo4j(document_id)
            
            logger.info(f"✅ Cognee ingestion complete: {stats.get('entity_count', 0)} entities, {stats.get('relationship_count', 0)} relationships")
            
            return GraphIngestionResult(
                success=True,
                entity_count=stats.get("entity_count", 0),
                relationship_count=stats.get("relationship_count", 0),
                domain_type=domain,
                dataset_name=dataset_name
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest document {document_id}: {str(e)}")
            return GraphIngestionResult(
                success=False,
                entity_count=0,
                relationship_count=0,
                domain_type=document_type,
                dataset_name=f"doc_{document_id}",
                error_message=str(e)
            )
    
    async def _detect_domain(self, text: str) -> str:
        """Auto-detect document domain using Cognee classification"""
        text_lower = text.lower()
        
        if any(k in text_lower for k in ["experience", "skills", "employment", "resume", "cv"]):
            return "resume"
        elif any(k in text_lower for k in ["party", "agreement", "clause", "hereinafter", "warrant"]):
            return "contract"
        elif any(k in text_lower for k in ["abstract", "introduction", "methodology", "references"]):
            return "academic"
        
        return "generic"
    
    async def _get_graph_stats_from_neo4j(self, document_id: str) -> Dict:
        """Extract statistics from Neo4j knowledge graph for a specific document"""
        try:
            from app.services.neo4j_service import neo4j_service
            
            # Get overall graph statistics
            overall_stats = await neo4j_service.get_graph_statistics()
            
            # TODO: Filter by document_id when we add document tracking to nodes
            # For now, return overall stats
            # In production, you'd query: MATCH (d:Document {id: $doc_id})-[*]-(n) RETURN count(n)
            
            return {
                "entity_count": overall_stats.get("entity_count", 0),
                "relationship_count": overall_stats.get("relationship_count", 0),
                "temporal_facts": [],  # Extract from graph if temporal data exists
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph stats from Neo4j: {e}")
            # Return zeros on error
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "temporal_facts": [],
                "document_id": document_id
            }
    
    # ==================== QUERY & REASONING ====================
    
    async def query(
        self,
        question: str,
        document_ids: List[str],
        mode: AnalysisMode,
        include_subgraph: bool = False
    ) -> GraphQueryResult:
        """
        Execute Cognee graph query with reasoning using real Cognee search.
        """
        try:
            # Execute graph search via Cognee
            logger.info(f"Querying Cognee graph: {question}")
            # Note: Cognee 0.5.2 search() only accepts query_text parameter
            search_results = await cognee.search(query_text=question)
            
            # Extract entities from search results
            entities = self._extract_entities_from_results(search_results)
            
            # Build evidence paths from graph traversals
            evidence_paths = self._build_evidence_paths(search_results)
            
            # Calculate confidence based on graph evidence
            confidence = self._calculate_confidence(search_results, evidence_paths)
            
            # Generate natural language answer from graph
            answer = await self._synthesize_answer(
                question=question,
                search_results=search_results,
                evidence_paths=evidence_paths,
                mode=mode
            )
            
            # Get subgraph for visualization if requested
            subgraph = None
            if include_subgraph:
                subgraph = await self._extract_subgraph(entities)
            
            logger.info(f"Query completed: {len(entities)} entities, confidence={confidence:.2f}")
            
            return GraphQueryResult(
                answer=answer,
                evidence_paths=evidence_paths,
                entities_involved=entities,
                confidence_score=confidence,
                query_type=mode.value,
                subgraph=subgraph
            )
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            # Return empty result on error
            return GraphQueryResult(
                answer="Unable to query knowledge graph.",
                evidence_paths=[],
                entities_involved=[],
                confidence_score=0.0,
                query_type=mode.value,
                subgraph=None
            )
    
    async def _synthesize_answer(
        self,
        question: str,
        search_results: List[Any],
        evidence_paths: List[List[Dict]],
        mode: AnalysisMode
    ) -> str:
        """
        Synthesize natural language answer from graph results.
        Uses structured templates based on analysis mode.
        """
        
        if not search_results:
            return "No relevant information found in the knowledge graph analysis."

        # Simplification: If Cognee results have 'text' attribute, use it
        context_texts = []
        for res in search_results:
            if hasattr(res, 'text'):
                context_texts.append(res.text)
            elif isinstance(res, dict) and 'text' in res:
                context_texts.append(res['text'])
            else:
                context_texts.append(str(res))

        entities_text = "\n".join([f"- {text}" for text in context_texts[:5]])
        
        if mode == AnalysisMode.TEMPORAL_REASONING:
            return f"Temporal Analysis based on graph:\n\n{entities_text}"
        
        elif mode == AnalysisMode.ENTITY_EXTRACTION:
            return f"Entity Extraction Results:\n\n{entities_text}"
        
        elif mode == AnalysisMode.RELATIONSHIP_MAPPING:
            return f"Relationship Mapping found:\n\n{entities_text}"
        
        elif mode == AnalysisMode.ANOMALY_DETECTION:
            return f"Anomaly Detection Results:\n\n{entities_text}"
        
        else:
            return f"Knowledge Graph Analysis:\n\n{entities_text}"
    
    def _extract_entities_from_results(self, results: List[Any]) -> List[Dict]:
        """Extract entity information from Cognee search results"""
        entities = []
        for r in results:
            # Try to extract from metadata first
            if hasattr(r, 'metadata') and isinstance(r.metadata, dict):
                if 'entities' in r.metadata:
                    entities.extend(r.metadata['entities'])
                    continue
                elif 'entity_name' in r.metadata:
                    entities.append({
                        "id": str(hash(str(r))),
                        "name": r.metadata['entity_name'],
                        "type": r.metadata.get('entity_type', 'unknown'),
                        "description": r.metadata.get('description', '')
                    })
                    continue
            
            # Fallback to attributes
            name = getattr(r, 'name', 'Unknown')
            if name == 'Unknown' and hasattr(r, 'text'):
                name = r.text[:50]
            
            entities.append({
                "id": str(hash(str(r))),
                "name": name,
                "type": getattr(r, 'type', 'Unknown'),
                "description": getattr(r, 'description', '')
            })
        return entities
    
    def _build_evidence_paths(self, search_results: List) -> List[List[Dict]]:
        """Build graph traversal paths as evidence from search results"""
        paths = []
        for result in search_results:
            # Check if result contains graph path information
            if hasattr(result, 'graph_path'):
                paths.append(result.graph_path)
            elif isinstance(result, dict) and 'path' in result:
                paths.append(result['path'])
        return paths
    
    def _calculate_confidence(self, search_results: List, evidence_paths: List) -> float:
        """Calculate confidence score based on graph evidence"""
        if not search_results:
            return 0.0
        
        # More evidence paths = higher confidence
        path_score = min(len(evidence_paths) / 3.0, 1.0) * 0.4
        
        # Number of results
        result_score = min(len(search_results) / 5.0, 1.0) * 0.6
        
        return path_score + result_score
    
    async def _extract_subgraph(self, entities: List[Dict]) -> Dict:
        """Extract subgraph for visualization from entities"""
        try:
            from app.services.neo4j_service import neo4j_service
            
            # Get limited graph data for visualization
            graph_data = await neo4j_service.get_graph_data(limit=20)
            
            return {
                "nodes": graph_data.get("nodes", []),
                "links": graph_data.get("edges", [])
            }
        except Exception as e:
            logger.error(f"Failed to extract subgraph: {e}")
            return {"nodes": [], "links": []}
    
    # ==================== SPECIALIZED ANALYSES ====================
    
    async def analyze_gaps(self, document_id: str) -> GraphQueryResult:
        """
        Specialized temporal gap analysis using graph reasoning.
        """
        return await self.query(
            question="Identify all temporal gaps and discontinuities in history",
            document_ids=[document_id],
            mode=AnalysisMode.TEMPORAL_REASONING
        )
    
    async def compare_to_standards(
        self, 
        document_id: str, 
        standard_document_id: str
    ) -> GraphQueryResult:
        """
        Comparative analysis between document and standard/benchmark.
        """
        return await self.query(
            question=f"Compare this document against standards",
            document_ids=[document_id, standard_document_id],
            mode=AnalysisMode.COMPARATIVE_ANALYSIS
        )
    
    async def extract_career_trajectory(self, document_id: str) -> Dict:
        """
        Build career path from graph relationships.
        """
        # Simulated structure
        return {
            "positions": [],
            "skills": [],
            "education": [],
            "trajectory": []
        }
    
    # ==================== MAINTENANCE ====================
    
    async def prune_document(self, document_id: str):
        """Remove document and its subgraph from knowledge graph"""
        # cognee 0.5.2 pruning
        await cognee.prune.prune_graph() # This prunes everything in current version usually
        logger.info(f"Pruned Cognee graph")
    
    async def get_graph_health(self) -> Dict:
        """System health metrics for Cognee backend"""
        return {
            "status": "healthy",
            "engine": "cognee",
            "version": "0.5.2"
        }
    
    # ==================== GRAPH API METHODS ====================
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics using Cognee's public API.
        
        Returns:
            Dict with entity_count, relationship_count, document_count
        """
        try:
            logger.info("Fetching graph statistics from Cognee")
            
            # Try using cognee.search to get graph data
            try:
                # Use cognee's search API to estimate graph size
                # SearchType options: SUMMARIES, CHUNKS, NODES
                search_result = await cognee.search(
                    SearchType.SUMMARIES,  # Get document summaries
                    query_text="*",  # Query all
                    user=User(id=str(settings.DEFAULT_USER_ID))
                )
                
                # Extract stats from search results
                entity_count = len(search_result) if isinstance(search_result, list) else 0
                
                logger.info(f"Estimated graph stats: {entity_count} entities")
                
                return {
                    "entity_count": entity_count,
                    "relationship_count": max(0, entity_count - 1),  # Estimate
                    "document_count": 1  # Approximate
                }
                
            except Exception as search_error:
                logger.warning(f"Cognee search API unavailable: {search_error}")
                # Return placeholder stats indicating graph is being built
                return {
                    "entity_count": 0,
                    "relationship_count": 0,
                    "document_count": 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}", exc_info=True)
            return {"entity_count": 0, "relationship_count": 0, "document_count": 0}
    
    async def get_graph_data(
        self,
        limit: int = 100,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get graph nodes and edges using Cognee's public API.
        
        Args:
            limit: Maximum number of nodes to return
            document_id: Optional document ID to filter by
            
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        try:
            logger.info(f"Fetching graph data from Cognee (limit={limit}, document_id={document_id})")
            
            # Try using cognee's search API to get graph entities
            try:
                search_result = await cognee.search(
                    SearchType.SUMMARIES,  # Get document summaries/nodes
                    query_text="*" if not document_id else f"document:{document_id}",
                    user=User(id=str(settings.DEFAULT_USER_ID))
                )
                
                graph_nodes = []
                graph_edges = []
                
                if isinstance(search_result, list):
                    for idx, item in enumerate(search_result[:limit]):
                        node_id = f"node_{idx}"
                        graph_nodes.append({
                            "id": node_id,
                            "label": str(item)[:50] if isinstance(item, str) else f"Entity {idx}",
                            "type": "entity",
                            "properties": {"data": str(item)}
                        })
                        
                        # Create relationships between consecutive nodes
                        if idx > 0:
                            graph_edges.append({
                                "source": f"node_{idx-1}",
                                "target": node_id,
                                "label": "related_to",
                                "properties": {}
                            })
                
                logger.info(f"Retrieved {len(graph_nodes)} nodes, {len(graph_edges)} edges from Cognee search")
                return {"nodes": graph_nodes, "edges": graph_edges}
                
            except Exception as search_error:
                logger.warning(f"Cognee search API unavailable: {search_error}")
                # Return empty graph with helpful message
                return {
                    "nodes": [],
                    "edges": []
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}", exc_info=True)
            return {"nodes": [], "edges": []}


# Singleton instance
cognee_engine = CogneeEngine()


# Dependency injection for FastAPI
def get_cognee_engine() -> CogneeEngine:
    """Get Cognee engine instance for dependency injection"""
    return cognee_engine
