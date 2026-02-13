
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

try:
    import cognee
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.pipelines import Task, run_pipeline
    from cognee.tasks.storage import add_data_points
    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False
    logging.warning("Cognee not available - professional pipelines disabled")

from app.models.cognee_models import (
    Resume,
    Person,
    Skill,
    WorkExperience,
    Education,
    Organization,
    ResumeExtraction,
    SkillList,
    WorkHistoryList,
    EducationList
)

logger = logging.getLogger(__name__)

# Import required for user UUID generation in fallbacks
import uuid
from app.core.cognee_config import settings as cognee_settings
try:
    from cognee.modules.users.models import User
except ImportError:
    class User:
        def __init__(self, id):
            self.id = id


# ==================== ECL ARCHITECTURE (Extract-Cognify-Load) ====================

class PipelineConfig(BaseModel):
    """Configuration for the ECL pipeline behavior"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "semantic"
    graph_model: str = "default"
    include_original_text: bool = True
    cognify_timeout: float = 600.0

class ECLProcessor:
    """
    Standardized Extract-Cognify-Load Processor.
    
    Implements the professional pipeline pattern:
    1. EXTRACT: Parse document and extract entities via LLM
    2. LOAD: Store raw text (for vector search) and structured data (for graph)
    3. COGNIFY: Trigger knowledge graph construction
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
    async def process(
        self, 
        doc_id: str, 
        text_content: str, 
        metadata: Dict[str, Any],
        extraction_task: Any = None,
        dataset_name_prefix: str = "doc"
    ) -> Dict[str, Any]:
        """
        Execute the full ECL pipeline.
        
        Args:
            doc_id: Unique document identifier
            text_content: Raw text content
            metadata: Metadata dictionary (source, timestamp, etc.)
            extraction_task: Optional coroutine for structured extraction
            dataset_name: Prefix for the dataset name
            
        Returns:
            Dict containing processing status and extracted data
        """
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")

        # Create localized dataset name
        dataset_name = f"{dataset_name_prefix}_{doc_id}"
        user_uuid = uuid.UUID(cognee_settings.DEFAULT_USER_ID)
        user = User(id=user_uuid)
        
        logger.info(f"🚀 Starting ECL Pipeline for {dataset_name} (Type: {dataset_name_prefix})")
        
        try:
            # STEP 1: LOAD (Raw Text) - Indexed for vector search
            if self.config.include_original_text:
                logger.info(f"💾 [LOAD] Indexing raw text for hybrid search...")
                # In a real implementation: chunk content first? Cognee handles this if we pass text.
                await cognee.add(
                    data=text_content,
                    dataset_name=dataset_name,
                    user=user
                )

            structured_data = None
            if extraction_task:
                # STEP 2: EXTRACT (Structured Entity Extraction)
                logger.info(f"🔍 [EXTRACT] extracting structured entities...")
                
                # Await if it's a coroutine (async task)
                if asyncio.iscoroutine(extraction_task):
                    structured_data = await extraction_task
                else:
                    structured_data = extraction_task
                
                # STEP 3: LOAD (Structured Data) - Graph Nodes
                if structured_data:
                    logger.info(f"💾 [LOAD] storing {type(structured_data).__name__} entities...")
                    # Normalize to list
                    data_payload = structured_data if isinstance(structured_data, list) else [structured_data]
                    await cognee.add(
                        data=data_payload,
                        dataset_name=dataset_name,
                        user=user
                    )

            # STEP 4: COGNIFY (Graph Construction)
            logger.info(f" [COGNIFY] Building Knowledge Graph for dataset: '{dataset_name}'...")
            graph_status = "completed"
            graph_error = None
            
            try:
                # Use extended timeout for specific environments
                await asyncio.wait_for(
                    cognee.cognify(datasets=[dataset_name], user=user), 
                    timeout=self.config.cognify_timeout
                )
                logger.info(f" ECL Pipeline success: Values persisted to Graph & Vector Store")
            except asyncio.TimeoutError:
                graph_status = "partial_success"
                graph_error = "Cognify timed out - graph incomplete"
                logger.warning(f"Cognify timed out after {self.config.cognify_timeout}s. Data is saved but graph relationships may be incomplete.")
            except Exception as e:
                # CRITICAL HANDLER for "TextSummary_text collection not found"
                if "TextSummary" in str(e) or "collection not found" in str(e):
                    logger.warning(f" Vector store issue detected: {e}. Attempting recovery...")
                    graph_status = "partial_success"
                    graph_error = f"Vector store issue: {str(e)}"
                else:
                    logger.error(f" Cognify failed: {e}")
                    graph_status = "partial_success"
                    graph_error = f"Cognify failed: {str(e)}"
                    # Don't re-raise if we want to return partial results (structured data extract was successful)
            
            # STEP 5: SELF-IMPROVEMENT (Memify Registration)
            if graph_status == "completed":
                try:
                    from app.services.cognee_background import memify_service
                    memify_service.register_dataset(dataset_name)
                    logger.info(f"Registered {dataset_name} for Memify maintenance")
                except ImportError:
                    pass

            return {
                "status": graph_status,
                "dataset": dataset_name,
                "data": structured_data,
                "error": graph_error
            }

        except Exception as e:
            logger.error(f" ECL Pipeline failed for {doc_id}: {e}", exc_info=True)
            raise

# Global default processor
default_ecl = ECLProcessor()

# ==================== EXTRACTION TASKS ====================

async def extract_resume_entities(text: str) -> Resume:
    """
    Extract comprehensive structured resume data using LLM.
    
    This is the primary extraction task that converts unstructured resume text
    into a fully structured Resume object with all relationships.
    
    Args:
        text: Raw resume text from PDF/DOC
        
    Returns:
        Resume: Structured resume with Person, WorkExperience, Education, Skills
        
    Example:
        >>> text = "John Doe\\nSenior Engineer at Microsoft..."
        >>> resume = await extract_resume_entities(text)
        >>> print(resume.person.name)  # "John Doe"
        >>> print(len(resume.work_history))  # 3
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available - cannot extract entities")
    
    system_prompt = """
    Extract comprehensive structured resume information from the provided text.
    
    You must identify and extract:
    
    1. PERSON DETAILS:
       - Full name (required)
       - Current title/position
       - Contact info (email, phone, location, LinkedIn)
    
    2. PROFESSIONAL SUMMARY:
       - Career summary or objective statement
       - Key highlights
    
    3. WORK HISTORY (Extract ALL positions, do not skip any):
       - Company/organization name
       - Job title
       - Start date and end date (use 'Present' if current)
       - Calculate duration in months if dates provided
       - Location
       - Key responsibilities and achievements
       - Skills used in this role
    
    4. EDUCATION (for each degree):
       - Institution name
       - Degree type (e.g., "Bachelor of Science")
       - Field of study/major
       - Graduation date
       - GPA and honors if mentioned
    
    5. SKILLS (CRITICAL: Extract ALL skills mentioned):
       - Scan the ENTIRE text (summary, work history, skills section).
       - Extract every tool, technology, language, and soft skill.
       - If a list is provided (e.g., "Python, Java, C++"), extract EACH as a separate item.
       - Inference level: "Expert" (5+ yrs/Senior), "Advanced" (3+ yrs), "Intermediate" (1+ yr), "Beginner".
    
    6. ADDITIONAL:
       - Certifications
       - Languages spoken
       - Calculate total years of experience from work history
    
    IMPORTANT RULES:
    - Only extract information EXPLICITLY stated.
    - Use null/empty for missing info.
    - Dates: Preserve format (YYYY-MM).
    - Current roles: end_date = "Present".
    - Skills: If you see a comma-separated list, break it down!
    
    EXAMPLE SKILL OUTPUT FORMAT (for guidance):
    "skills": [
      {"name": "Python", "category": "technical", "level": "expert"}, 
      {"name": "Project Management", "category": "soft", "level": "advanced"}
    ]
    
    Return as a Resume object with all nested structures properly populated.
    """
    
    try:
        # Use Custom Engine with Strict Timeout (30s)
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        
        try:
            result = await asyncio.wait_for(
                engine.acreate_structured_output(
                    text_input=text,
                    response_model=ResumeExtraction,
                    system_prompt=system_prompt
                ),
                timeout=60.0  # Increased from 30s to allow for full 7B generation
            )
        except asyncio.TimeoutError:
            logger.warning("⏰ Resume extraction timed out (30s). Falling back to basic extraction.")
            # Basic fallback: Create minimal valid object
            from app.models.cognee_models import Person
            return Resume(
                person=Person(name="Unknown Candidate (Timeout)"),
                work_history=[],
                education=[],
                skills=[]
            )
        
        resume = result.resume
        
        logger.info(
            f" Extracted resume: "
            f"{resume.person.name}, "
            f"{len(resume.work_history)} positions, "
            f"{len(resume.education)} degrees, "
            f"{len(resume.skills)} skills"
        )
        
        return resume
        
    except Exception as e:
        logger.error(f"❌ Resume extraction failed: {e}", exc_info=True)
        raise


async def extract_skills_detailed(text: str) -> List[Skill]:
    """
    Extract skills with detailed proficiency information.
    
    This is a specialized task for deep skill extraction when resume
    contains detailed skill descriptions.
    
    Args:
        text: Resume text or skills section
        
    Returns:
        List[Skill]: Detailed skills with levels and experience
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available - cannot extract skills")
    
    system_prompt = """
    Extract ALL professional skills from this resume with maximum detail.
    
    For each skill, determine:
    1. Skill name (exact tool/technology/competency)
    2. Proficiency level: beginner, intermediate, expert, advanced
       - Infer from context clues like "proficient in", "expert", "familiar with"
    3. Years of experience
       - Calculate from work history mentions or explicit statements
    4. Category: technical, soft, language, certification
    
    Examples:
    - "5+ years Python experience" → name="Python", level="expert", years=5, category="technical"
    - "Familiar with React" → name="React", level="beginner", category="technical"
    - "Native Spanish speaker" → name="Spanish", level="expert", category="language"
    - "Strong leadership skills" → name="Leadership", level="advanced", category="soft"
    
    Be thorough - extract:
    - Programming languages and frameworks
    - Tools and platforms
    - Soft skills (leadership, communication, etc.)
    - Certifications (AWS Certified, PMP, etc.)
    - Languages
    
    Only include skills explicitly mentioned. Do not infer skills not stated.
    """
    
    try:
        logger.info("🔍 Extracting detailed skills with Custom Engine...")
        
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        
        try:
            result = await asyncio.wait_for(
                engine.acreate_structured_output(
                    text_input=text,
                    response_model=SkillList,
                    system_prompt=system_prompt
                ),
                timeout=45.0  # Increased from 20s
            )
        except asyncio.TimeoutError:
            logger.warning(" Skill extraction timed out (20s). Skipping detailed skills.")
            return []
        
        logger.info(f" Extracted {len(result.skills)} detailed skills")
        return result.skills
        
    except Exception as e:
        logger.error(f" Skill extraction failed: {e}", exc_info=True)
        return []


async def extract_work_history_timeline(text: str) -> List[WorkExperience]:
    """
    Extract work history with temporal focus for gap analysis.
    
    Specialized extraction that emphasizes dates and timeline for
    detecting employment gaps.
    
    Args:
        text: Resume text with work history
        
    Returns:
        List[WorkExperience]: Chronologically ordered work history
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available - cannot extract work history")
    
    system_prompt = """
    Extract work history with MAXIMUM temporal accuracy.
    
    For each position:
    1. Parse dates precisely (MM/YYYY or YYYY-MM format preferred)
    2. Calculate exact duration in months
    3. Note if position is current ("Present" as end_date)
    4. Order chronologically (most recent first)
    
    Pay special attention to:
    - Overlapping positions (concurrent jobs)
    - Gaps between positions
    - Short-term roles (< 6 months)
    - Career progressions within same company
    
    For each position, extract:
    - Company name
    - Job title
    - Start date (preserve format)
    - End date (or "Present")
    - Duration in months (calculate accurately)
    - Location
    - Key responsibilities (top 3-5)
    - Skills used
    
    Return positions in chronological order (newest first).
    """
    
    try:
        logger.info("📅 Extracting work history timeline with Custom Engine...")
        
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        
        try:
            result = await asyncio.wait_for(
                engine.acreate_structured_output(
                    text_input=text,
                    response_model=WorkHistoryList,
                    system_prompt=system_prompt
                ),
                timeout=45.0
            )
        except asyncio.TimeoutError:
            logger.warning(" Work history extraction timed out (20s). Skipping timeline.")
            return []
        
        logger.info(f" Extracted {len(result.experiences)} work experiences")
        return result.experiences
        
    except Exception as e:
        logger.error(f" Work history extraction failed: {e}", exc_info=True)
        return []


# ==================== CUSTOM PIPELINES ====================

# ==================== NEW DOMAIN EXTRACTION TASKS ====================

async def extract_legal_entities(text: str) -> 'Contract':
    """Extract structured data from legal contracts"""
    from app.models.ontologies import Contract
    if not COGNEE_AVAILABLE: raise RuntimeError("Cognee unavailable")
    
    system_prompt = """
    Extract structured legal data from this contract.
    Identify:
    1. Parties (Name, Role, Address)
    2. Key Dates (Effective, Termination)
    3. Jurisdiction
    4. Clauses (ID, Title, Type, Obligations)
    5. Financial Terms
    
    Be precise with Clause IDs and Obligations.
    """
    
    try:
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        result = await asyncio.wait_for(
            engine.acreate_structured_output(text, Contract, system_prompt),
            timeout=60.0
        )
        return result
    except Exception as e:
        logger.error(f"Legal extraction failed: {e}")
        raise

async def extract_financial_entities(text: str) -> 'Invoice':
    """Extract structured data from financial documents/invoices"""
    from app.models.ontologies import Invoice
    if not COGNEE_AVAILABLE: raise RuntimeError("Cognee unavailable")
    
    system_prompt = """
    Extract structured invoice data.
    Identify:
    1. Invoice Number, Dates (Issue, Due)
    2. Vendor and Customer details
    3. Line Items (Description, Qty, Price, Amount)
    4. Totals (Subtotal, Tax, Grand Total)
    5. Currency
    
    Ensure strict numerical accuracy for amounts.
    """
    
    try:
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        result = await asyncio.wait_for(
            engine.acreate_structured_output(text, Invoice, system_prompt),
            timeout=45.0
        )
        return result
    except Exception as e:
        logger.error(f"Financial extraction failed: {e}")
        raise

async def extract_education_entities(text: str) -> 'CourseMaterial':
    """Extract structured data from educational content"""
    from app.models.ontologies import CourseMaterial
    if not COGNEE_AVAILABLE: raise RuntimeError("Cognee unavailable")
    
    system_prompt = """
    Extract structured educational content.
    Identify:
    1. Course/Material Title, Subject, Level
    2. Chapters/Sections (Number, Title, Summary)
    3. Key Concepts (Term, Definition)
    4. Learning Objectives
    
    Focus on hierarchical structure (Course -> Chapter -> Concept).
    """
    
    try:
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        engine = CustomCogneeLLMEngine()
        result = await asyncio.wait_for(
            engine.acreate_structured_output(text, CourseMaterial, system_prompt),
            timeout=90.0
        )
        return result
    except Exception as e:
        logger.error(f"Education extraction failed: {e}")
        raise

# ==================== CUSTOM PIPELINES (EXPANDED) ====================

async def process_resume_document(text: str, document_id: str, document_type: str = "resume", timeout_seconds: int = 120) -> Any:
    """Direct Resume Processing Pipeline"""
    if not COGNEE_AVAILABLE: raise RuntimeError("Cognee unavailable")
    
    try:
        # Use ECL Processor
        result = await default_ecl.process(
            doc_id=document_id,
            text_content=text,
            metadata={"type": "resume"},
            extraction_task=extract_resume_entities(text),
            dataset_name_prefix="resume"
        )
        return result["data"]
    except Exception as e:
        logger.error(f"Resume pipeline failed: {e}")
        from app.models.cognee_models import Resume, Person
        return Resume(person=Person(name="Error"), work_history=[], education=[], skills=[])

async def process_legal_document(text: str, document_id: str) -> Any:
    """Direct Legal Processing Pipeline"""
    try:
        result = await default_ecl.process(
            doc_id=document_id,
            text_content=text,
            metadata={"type": "contract"},
            extraction_task=extract_legal_entities(text),
            dataset_name_prefix="contract"
        )
        return result["data"]
    except Exception as e:
        logger.error(f"Legal pipeline failed: {e}")
        return {"error": str(e)}

async def process_financial_document(text: str, document_id: str) -> Any:
    """Direct Financial Processing Pipeline"""
    try:
        result = await default_ecl.process(
            doc_id=document_id,
            text_content=text,
            metadata={"type": "invoice"},
            extraction_task=extract_financial_entities(text),
            dataset_name_prefix="invoice"
        )
        return result["data"]
    except Exception as e:
        logger.error(f"Financial pipeline failed: {e}")
        return {"error": str(e)}

async def process_education_document(text: str, document_id: str) -> Any:
    """Direct Education Processing Pipeline"""
    try:
        result = await default_ecl.process(
            doc_id=document_id,
            text_content=text,
            metadata={"type": "education"},
            extraction_task=extract_education_entities(text),
            dataset_name_prefix="course"
        )
        return result["data"]
    except Exception as e:
        logger.error(f"Education pipeline failed: {e}")
        return {"error": str(e)}



async def process_generic_document(text: str, document_id: str, document_type: str = "document") -> Dict[str, Any]:
    """Generic fall-back pipeline"""
    if not COGNEE_AVAILABLE: raise RuntimeError("Cognee unavailable")
    logger.info(f"📄 Processing GENERIC document {document_id}")
    try:
        await professional_ingestion_workflow(text, document_id)
        return {"success": True, "dataset": f"doc_{document_id}", "type": "generic"}
    except Exception as e:
        logger.error(f"Generic pipeline failed: {e}")
        raise

# ==================== HELPER FUNCTIONS ====================

def detect_document_type(text: str) -> str:
    """Auto-detect document type from content using keyword scoring"""
    text_lower = text.lower()[:5000] # Check first 5k chars
    
    scores = {
        "resume": sum(1 for kw in ["experience", "education", "skills", "resume", "cv", "career"] if kw in text_lower),
        "contract": sum(1 for kw in ["agreement", "contract", "parties", "whereas", "section", "jurisdiction"] if kw in text_lower),
        "invoice": sum(1 for kw in ["invoice", "total", "tax", "due date", "bill to", "amount"] if kw in text_lower),
        "education": sum(1 for kw in ["chapter", "syllabus", "learning objective", "course", "lecture", "student"] if kw in text_lower)
    }
    
    best_match = max(scores, key=scores.get)
    if scores[best_match] >= 2:
        return best_match
    return "document"

async def route_to_pipeline(text: str, document_id: str, document_type: Optional[str] = None) -> Any:
    """Smart routing to appropriate processing pipeline"""
    if not document_type or document_type == "auto_detect":
        document_type = detect_document_type(text)
        logger.info(f"🔍 Auto-detected type: {document_type}")
    
    if document_type == "resume":
        return await process_resume_document(text, document_id)
    elif document_type == "contract":
        return await process_legal_document(text, document_id)
    elif document_type == "invoice":
        return await process_financial_document(text, document_id)
    elif document_type == "education" or document_type == "academic":
        return await process_education_document(text, document_id)
    else:
        return await process_generic_document(text, document_id)

async def professional_ingestion_workflow(text: str, document_id: str):
    """Standard Cognee Pipeline for generic documents"""
    try:
        # Use ECL without structured extraction (Generic)
        result = await default_ecl.process(
            doc_id=document_id,
            text_content=text,
            metadata={"type": "generic"},
            extraction_task=None, # No structured extraction for generic docs yet
            dataset_name_prefix="doc"
        )
        return {"status": "completed", "document_id": document_id, "dataset": result["dataset"]}
    except Exception as e:
        logger.error(f"Professional ingestion failed: {e}")
        raise
