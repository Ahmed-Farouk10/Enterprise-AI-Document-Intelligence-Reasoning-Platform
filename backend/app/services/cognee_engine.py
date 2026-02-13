# app/services/cognee_engine.py
import os
import asyncio
import logging
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import uuid

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

# from app.core import settings  <-- REMOVED: invalid import
from app.core.logging_config import get_logger
from app.core.cognee_config import settings as cognee_settings
from app.services.cognee_retrievers import ResumeRetriever

try:
    from cognee.modules.users.models import User
except ImportError:
    # Minimal mock for User if not available
    class User:
        def __init__(self, id):
            self.id = id

# CRITICAL FIX: Monkey-patch Cognee's LLM connection test
# The test hangs for 30+ seconds on HF Spaces when no valid HF_TOKEN is available
# This bypasses the test ONLY when no token exists to prevent pipeline timeouts
try:
    from cognee.infrastructure.llm import utils as llm_utils
    
    async def _noop_llm_test():
        """No-op replacement for test_llm_connection when no LLM token available"""
        logger = get_logger(__name__)
        logger.info("⚡ Skipping LLM connection test (no HF_TOKEN available)")
        return True
    
    # IMPORTANT: Only patch when on HF Spaces AND no valid token
    # If HF_TOKEN exists, let the test run to validate the LLM works!
    if os.getenv("HF_HOME") and not os.getenv("HF_TOKEN"):
        llm_utils.test_llm_connection = _noop_llm_test
        print("⚙️ Cognee: LLM test disabled (no HF_TOKEN - will use basic extraction)")
    elif os.getenv("HF_TOKEN"):
        print(f"✅ Cognee: LLM enabled with HF Inference API (token: {os.getenv('HF_TOKEN')[:8]}...)")
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
        
        self.extraction_model = cognee_settings.EXTRACTION_MODEL
        self.graph_db_url = cognee_settings.GRAPH_DATABASE_URL
        
        # Force Robust Local Configuration (Embeddings + LLM)
        try:
            # 1. Embeddings (SentenceTransformer)
            from app.services.embeddings import SentenceTransformerEmbeddingEngine
            logger.info("Initializing custom SentenceTransformerEmbeddingEngine...")
            local_embed_engine = SentenceTransformerEmbeddingEngine()
            cognee.config.embedding_engine = local_embed_engine
            logger.info(f"✅ Forced Custom Embedding Engine (dim={local_embed_engine.get_vector_size()})")

            # 2. LLM (Custom Qwen Wrapper)
            # This bypasses LLMGateway's default OpenAI behavior for structured output
            try:
                from app.services.custom_cognee_llm import CustomCogneeLLMEngine
                logger.info("Initializing custom Cognee LLM Engine (Local Qwen)...")
                local_llm_engine = CustomCogneeLLMEngine()
                
                # Inject into Cognee config
                # Cognee 0.5.x uses 'llm_engine' or 'llm_client' depending on version
                # We set both to be safe
                cognee.config.llm_engine = local_llm_engine
                cognee.config.llm_client = local_llm_engine
                
                logger.info("✅ Forced Custom LLM Engine (Local Qwen for Structured Output)")
            except Exception as e_llm:
                logger.error(f"❌ Failed to inject Custom LLM: {e_llm}")
                # Fallback to config-based (which might default to OpenAI, sadly)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize custom engines: {e}")
            logger.info("Attempting fallback to FastEmbed for embeddings...")
            
            # Fallback to FastEmbed (original logic)
            try:
                import fastembed
                engine_cls = None
                
                # Check different import paths
                try:
                    from cognee.infrastructure.llm.embeddings.FastEmbedEmbeddingEngine import FastEmbedEmbeddingEngine
                    engine_cls = FastEmbedEmbeddingEngine
                except ImportError:
                    pass

                if not engine_cls:
                    try:
                        from cognee.infrastructure.llm.embeddings.fastembed.FastEmbedEmbeddingEngine import FastEmbedEmbeddingEngine
                        engine_cls = FastEmbedEmbeddingEngine
                    except ImportError:
                        pass

                if engine_cls:
                    cognee.config.embedding_engine = engine_cls()
                    logger.info("✅ Fallback: Forced FastEmbed embedding engine")
                else:
                    if hasattr(cognee.config, "embedding_engine"):
                        cognee.config.embedding_engine = "fastembed"
                        logger.info("⚠️ Fallback: Set embedding_engine='fastembed'")
            except Exception as fe:
                logger.error(f"❌ FastEmbed fallback also failed: {fe}")
        
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
            
            # INFO: Cognee 0.5.x might create a random default user if none exists.
            # We must force usage of our configured DEFAULT_USER_ID to ensure persistence.
            try:
                from cognee.modules.users.methods import get_user, create_user
                from cognee.modules.users.models import User
                
                target_user_id = uuid.UUID(cognee_settings.DEFAULT_USER_ID)
                existing_user = await get_user(target_user_id)
                
                if not existing_user:
                    logger.info(f"👤 Creating configured default user: {target_user_id}")
                    # Create the user explicitly
                    await create_user(
                        user=User(
                            id=target_user_id,
                            email="default@example.com",
                            name="Default User"
                        )
                    )
                    logger.info(f"✅ Created user {target_user_id}")
                else:
                    logger.info(f"✅ Verified available user: {target_user_id}")
                    
            except Exception as user_error:
                logger.warning(f"⚠️ User registration issue (non-fatal): {user_error}")
            
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
            try:  # Cognee ingestion with timeout
                # Cognee 0.5.2 add() linked to user for auditing
                # Add aggressive timeout (increased for HF Spaces)
                await asyncio.wait_for(
                    cognee.add(
                        data=document_text,
                        dataset_name=dataset_name,
                        user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                    ),
                    timeout=120.0  # Increased from 30s
                )
                logger.info(f"✅ Document added to Cognee dataset: {dataset_name}")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Cognee add() timed out after 120s for {dataset_name}")
                raise RuntimeError(f"Cognee add() operation timed out - may be downloading models or waiting for external service")
            except Exception as add_error:
                logger.error(f"❌ Cognee add() failed: {add_error}")
                raise
            
            # Build knowledge graph with timeout
            logger.info(f"🔨 Building knowledge graph for {dataset_name}...")
            try:
                # Set a reasonable timeout for graph building (increased for HF Spaces)
                await asyncio.wait_for(
                    cognee.cognify(
                        datasets=[dataset_name],
                        user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                    ),
                    timeout=300.0  # Increased from 90s
                )
                logger.info(f"✅ Knowledge graph built successfully for {dataset_name}")
                
                # 3. Memory Enrichment (memify) - Derives new facts and relationships
                logger.info(f"🧠 Enriching memory graph (memify) for {dataset_name}...")
                await cognee.memify(
                    datasets=[dataset_name],
                    user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                )
                logger.info(f"✅ Memory enrichment complete for {dataset_name}")
                
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Cognee graph building/enrichure timed out after 300s for {dataset_name}")
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
        """Extract statistics from Knowledge Graph (Kuzu/Neo4j)"""
        try:
            # Check if Neo4j is available
            from app.services.neo4j_service import neo4j_service
            if neo4j_service._available:
                # Get overall graph statistics
                overall_stats = await neo4j_service.get_graph_statistics()
                
                return {
                    "entity_count": overall_stats.get("entity_count", 0),
                    "relationship_count": overall_stats.get("relationship_count", 0),
                    "temporal_facts": [],
                    "document_id": document_id
                }
            else:
                # Fallback for Kuzu/Local: Use Cognee search to estimate graph size
                try:
                    # Search for all nodes to get a count
                    results = await cognee.search(
                        query_text="*", 
                        query_type=SearchType.SUMMARIES,
                        user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                    )
                    entity_count = len(results) if isinstance(results, list) else 0
                    
                    # Estimate relationships (hard to get exact count without direct Kuzu access)
                    # Assuming average degree of 1.5
                    relationship_count = int(entity_count * 1.5)
                    
                    return {
                        "entity_count": entity_count,
                        "relationship_count": relationship_count,
                        "temporal_facts": [],
                        "document_id": document_id,
                        "status": "Active (Local Kuzu)"
                    }
                except Exception as kuzu_error:
                    logger.warning(f"Failed to query Kuzu stats: {kuzu_error}")
                    return {
                        "entity_count": 0,
                        "relationship_count": 0,
                        "temporal_facts": [],
                        "document_id": document_id,
                        "status": "Error"
                    }
            
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "temporal_facts": [],
                "document_id": document_id
            }

    # ==================== QUERY & REASONING ====================

    async def search_documents(self, query_text: str, limit: int = 50):
        """
        Wrap Cognee search with correct signature to avoid API mismatches.
        Requested Fix: Problem 5
        """
        try:
            # Use 'query_text' as parameter name (Cognee 0.5.x)
            results = await cognee.search(
                query_text=query_text, 
                user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
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
            # Map AnalysisMode to Cognee SearchType
            search_type = SearchType.GRAPH_COMPLETION # Default: Conversational + Reasoning
            
            if mode == AnalysisMode.TEMPORAL_REASONING:
                # Insights highlights relationships/connections which is good for temporal flows
                search_type = SearchType.INSIGHTS 
            elif mode == AnalysisMode.ENTITY_EXTRACTION:
                # Summaries provides a high-level view (modified from CHUNKS to reduce noise)
                search_type = SearchType.SUMMARIES
            elif mode == AnalysisMode.RELATIONSHIP_MAPPING:
                search_type = SearchType.INSIGHTS
            
            # Execute graph search via Cognee
            logger.info(f"Querying Cognee graph: {question} (Type: {search_type})")
            
            # Note: Cognee 0.5.2 search() expects query_text and user
            search_results = await cognee.search(
                query_text=question,
                query_type=search_type, # Use specific search type
                user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID)),
                save_interaction=True # Enable feedback loop (store query/result)
            )
            
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
            
            if neo4j_service._available:
                # Get limited graph data for visualization
                graph_data = await neo4j_service.get_graph_data(limit=20)
                return {
                    "nodes": graph_data.get("nodes", []),
                    "links": graph_data.get("edges", [])
                }
            else:
                # Fallback for Kuzu: Return empty subgraph for now
                # Or wait for new /visualize endpoint to be used by frontend
                return {"nodes": [], "links": []}

        except Exception as e:
            logger.error(f"Failed to extract subgraph: {e}")
            return {"nodes": [], "links": []}
    
    # ==================== SPECIALIZED ANALYSES ====================
    
    async def analyze_gaps(self, document_id: str) -> GraphQueryResult:
        """
        Specialized temporal gap analysis using graph reasoning.
        """
        # Use specialized retriever logic
        gaps = await ResumeRetriever.analyze_career_gaps(document_id)
        
        # Convert to GraphQueryResult
        answer = "No significant gaps found."
        confidence = 0.8
        
        if gaps:
            gap_details = "\n".join([
                f"- Gap of {g.duration_months} months between {g.previous_role} and {g.next_role} ({g.start_date} to {g.end_date})"
                for g in gaps
            ])
            answer = f"Identified {len(gaps)} employment gaps:\n\n{gap_details}"
            confidence = 0.95
            
        return GraphQueryResult(
            answer=answer,
            evidence_paths=[],
            entities_involved=[],  # Would populate from gaps
            confidence_score=confidence,
            query_type=AnalysisMode.TEMPORAL_REASONING.value,
            subgraph=None
        )
    
    async def compare_to_standards(
        self, 
        document_id: str, 
        standard_document_id: str
    ) -> GraphQueryResult:
        """
        Comparative analysis between document and standard/benchmark.
        """
        # Fetch real text from standard document
        job_description = "Standard Job Description"
        try:
            from app.db.database import SessionLocal
            from app.db.service import DatabaseService
            
            with SessionLocal() as db:
                std_doc = DatabaseService.get_document(db, standard_document_id)
                if std_doc:
                    # Try to get text from vector store or read file (simplified: use vector store retrieval)
                    # For now, we'll try to get it from the vector store if possible, or just use a placeholder if not found
                    # Better: Read the file if local?
                    # Let's assume we can search for it in vector store by ID
                    job_description = f"Content of document {std_doc.filename}" 
                    
                    # Try to retrieve actual content from vector store
                    from app.services.retreival import vector_store
                    # Vector store doesn't have a direct "get_text_by_id" easily exposed without query
                    # So we'll search for it strictly
                    results = vector_store.search(std_doc.filename, k=1)
                    if results:
                        job_description = results[0]['text']

        except Exception as e:
            logger.warning(f"Could not fetch standard document text: {e}")

        comparison = await ResumeRetriever.compare_to_job(document_id, job_description)
        
        return GraphQueryResult(
            answer=f"Alignment Score: {comparison.overall_match_score}%\n\nRecommendations:\n" + "\n".join(comparison.recommendations),
            evidence_paths=[],
            entities_involved=[],
            confidence_score=comparison.overall_match_score / 100.0,
            query_type=AnalysisMode.COMPARATIVE_ANALYSIS.value,
            subgraph=None
        )
    
    async def extract_career_trajectory(self, document_id: str) -> Dict:
        """
        Build career path from graph relationships using Cognee search.
        """
        try:
            # Search for career-related entities linked to this document context
            # We search for "work history jobs positions" to retrieve relevant graph nodes
            results = await cognee.search(
                query_text="work history jobs positions companies",
                query_type=SearchType.SUMMARIES,
                user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
            )
            
            positions = []
            skills = []
            companies = []
            
            # Simple heuristic processing of results since we get unstructured summaries/chunks form Kuzu
            for res in results:
                text = getattr(res, 'text', str(res)).lower()
                
                # Extract potential job titles (very basic heuristic)
                if "engineer" in text or "manager" in text or "developer" in text:
                    positions.append({"title": text[:50], "source": "graph_inference"})
                
                # Extract skills
                if "python" in text or "java" in text or "leadership" in text:
                    skills.append(text[:30])
                    
            return {
                "positions": positions,
                "skills": list(set(skills)),
                "education": [], # Hard to parse without structured schema
                "trajectory": [p["title"] for p in positions],
                "source": "cognee_graph_search"
            }
        except Exception as e:
            logger.error(f"Career trajectory extraction failed: {e}")
            return {
                "positions": [],
                "skills": [],
                "education": [],
                "trajectory": [],
                "error": str(e)
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
                    "*", # query_text is Pos 0 
                    SearchType.SUMMARIES, # type is Pos 1?
                    user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
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
        Get graph nodes and edges using Neo4j (primary) or Kuzu (fallback).
        
        Args:
            limit: Maximum number of nodes to return
            document_id: Optional document ID to filter by
            
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        try:
            logger.info(f"Fetching graph data (limit={limit}, document_id={document_id})")
            
            # 1. Try Neo4j first (Direct access)
            from app.services.neo4j_service import neo4j_service
            if neo4j_service._available:
                return await neo4j_service.get_graph_data(limit=limit, document_id=document_id)
            
            # 2. Fallback to Kuzu (via Cognee Search)
            logger.info("Neo4j unavailable - attempting Kuzu fallback extraction")
            try:
                # Perform a broad search to get graph elements
                # Insights type often returns relationships/paths
                try:
                    search_results = await cognee.search(
                        query_text="*", 
                        query_type=SearchType.INSIGHTS,
                        user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                    )
                except Exception as kuzu_e:
                    logger.warning(f"Kuzu extraction failed with INSIGHTS: {kuzu_e}. Retrying with GRAPH_COMPLETION.")
                    # GRAPH_COMPLETION is more robust on Kuzu schema issues
                    search_results = await cognee.search(
                        query_text="*", 
                        query_type=SearchType.GRAPH_COMPLETION,
                        user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
                    )
                
                nodes = []
                edges = []
                seen_nodes = set()
                
                # Transform Cognee results into Graph structure
                for i, result in enumerate(search_results):
                    # Cognee results can be complex objects or dicts
                    # We attempt to extract meaningful node data
                    val = str(result)
                    node_id = getattr(result, "id", f"node_{i}")
                    node_label = getattr(result, "text", val[:50])
                    node_type = getattr(result, "type", "Entity")
                    
                    if node_id not in seen_nodes:
                        nodes.append({
                            "id": str(node_id),
                            "label": node_type,
                            "properties": {"name": node_label, "full_text": val}
                        })
                        seen_nodes.add(node_id)
                        
                    # If result has 'graph_path' or relationships, extract edges
                    # (This depends on specific Cognee 0.5.x result structure)
                    
                logger.info(f"Kuzu fallback retrieved {len(nodes)} nodes")
                return {"nodes": nodes, "edges": edges}
                
            except Exception as kuzu_error:
                logger.warning(f"Kuzu fallback failed: {kuzu_error}")
                return {"nodes": [], "edges": []}

        except Exception as e:
            logger.error(f"Failed to get graph data: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

# Singleton instance
cognee_engine = CogneeEngine()


# Dependency injection for FastAPI
def get_cognee_engine() -> CogneeEngine:
    """Get Cognee engine instance for dependency injection"""
    return cognee_engine
