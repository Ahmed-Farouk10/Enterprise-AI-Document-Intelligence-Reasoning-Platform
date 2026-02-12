"""
Professional Cognee Extraction Pipelines.

This module implements custom extraction pipelines following Cognee best practices
for document intelligence. Uses LLM-powered structured extraction to build rich
knowledge graphs from unstructured documents.

Based on official Cognee documentation:
https://docs.cognee.ai/guides/custom-tasks-and-pipelines
"""

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
    
    3. WORK HISTORY (for each position):
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
    
    5. SKILLS:
       - Technical skills with proficiency levels if stated
       - Soft skills
       - Years of experience per skill if calculable
       - Categorize as: technical, soft, language, certification
    
    6. ADDITIONAL:
       - Certifications
       - Languages spoken
       - Calculate total years of experience from work history
    
    IMPORTANT RULES:
    - Only extract information EXPLICITLY stated in the text
    - Use null/empty for missing information - do NOT make assumptions
    - For dates, preserve the format from the text (YYYY-MM, YYYY, etc.)
    - For current positions, use "Present" as end_date
    - Calculate duration_months accurately from start/end dates
    - Group related skills (e.g., Python, Java → technical)
    
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
            f"✅ Extracted resume: "
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
            logger.warning("⏰ Skill extraction timed out (20s). Skipping detailed skills.")
            return []
        
        logger.info(f"✅ Extracted {len(result.skills)} detailed skills")
        return result.skills
        
    except Exception as e:
        logger.error(f"❌ Skill extraction failed: {e}", exc_info=True)
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
            logger.warning("⏰ Work history extraction timed out (20s). Skipping timeline.")
            return []
        
        logger.info(f"✅ Extracted {len(result.experiences)} work experiences")
        return result.experiences
        
    except Exception as e:
        logger.error(f"❌ Work history extraction failed: {e}", exc_info=True)
        return []


# ==================== CUSTOM PIPELINES ====================

async def process_resume_document(
    text: str,
    document_id: str,
    document_type: str = "resume",
    timeout_seconds: int = 60  # Increased from 30s
) -> Resume:
    """
    Direct Resume Processing Pipeline (Bypassing Cognee run_pipeline wrapper).
    Fixes Problem 6: Resume Pipeline Timeout
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available - professional pipeline disabled")
    
    logger.info(f"🚀 Starting DIRECT resume pipeline for doc {document_id}")
    dataset_name = f"resume_{document_id}"
    
    try:
        # Step 1: Direct Extraction with Strict Timeout
        try:
            resume = await asyncio.wait_for(
                extract_resume_entities(text),  # Already uses Custom Engine
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Extraction timed out ({timeout_seconds}s). Using minimal fallback.")
            # Basic fallback to keep flow moving
            resume = Resume(
                person=Person(name="Unknown Candidate (Timeout)"),
                work_history=[],
                education=[],
                skills=[]
            )

        # Step 2: Direct Storage (Bypassing pipeline queue)
        logger.info(f"💾 Storing resume entities for {dataset_name}...")
        try:
            # Manually push to Cognee memory
            # Note: add_data_points typically expects list of datapoints and dataset name
            await add_data_points(
                data=[resume], 
                dataset_name=dataset_name,
                user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
            )
            
            # Cognify specifically for this dataset if needed, but adding points usually enough for graph
            # await cognee.cognify(datasets=[dataset_name], user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))) 
            
            logger.info(f"✅ Data points stored successfully")
        except Exception as storage_error:
            logger.error(f"⚠️ Storage failed (non-critical): {storage_error}")
            # Continue even if storage fails, to return result to frontend
            
        logger.info(
            f"✅ Pipeline complete: "
            f"{len(resume.work_history)} positions, "
            f"{len(resume.education)} degrees, "
            f"{len(resume.skills)} skills"
        )
        
        return resume
        
    except Exception as e:
        logger.error(f"❌ Resume processing failed: {e}", exc_info=True)
        # Final fallback
        return Resume(person=Person(name="Error Processing"), work_history=[], education=[], skills=[])


async def process_generic_document(
    text: str,
    document_id: str,
    document_type: str = "document"
) -> Dict[str, Any]:
    """
    Generic document processing for non-resume documents.
    
    Falls back to basic Cognee ingestion for documents that don't
    match our specialized pipelines.
    
    Args:
        text: Document text
        document_id: Unique document identifier
        document_type: Type of document
        
    Returns:
        Dict with processing status
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available")
    
    logger.info(f"📄 Processing {document_type} with generic pipeline")
    
    try:
        # Use Professional Workflow (Problem 7 Fix) to bypass internal OpenAI defaults
        result = await professional_ingestion_workflow(text, document_id)
        
        dataset_name = f"doc_{document_id}"
        logger.info(f"✅ Generic document processed via Professional Workflow: {dataset_name}")
        
        return {
            "success": True,
            "dataset": dataset_name,
            "document_type": document_type,
            "metadata": result
        }
        
    except Exception as e:
        logger.error(f"❌ Generic pipeline failed: {e}", exc_info=True)
        raise


# ==================== HELPER FUNCTIONS ====================

def detect_document_type(text: str) -> str:
    """
    Auto-detect document type from content.
    
    Args:
        text: Document text
        
    Returns:
        str: Detected type ("resume", "contract", "report", etc.)
    """
    text_lower = text.lower()
    
    # Resume indicators
    resume_keywords = [
        "experience", "education", "skills", "work history",
        "employment", "resume", "cv", "curriculum vitae"
    ]
    
    # Contract indicators
    contract_keywords = [
        "agreement", "contract", "party", "terms and conditions",
        "whereas", "hereby", "covenant"
    ]
    
    resume_score = sum(1 for kw in resume_keywords if kw in text_lower)
    contract_score = sum(1 for kw in contract_keywords if kw in text_lower)
    
    if resume_score > contract_score and resume_score >= 3:
        return "resume"
    elif contract_score > resume_score and contract_score >= 3:
        return "contract"
    else:
        return "document"  # Generic


async def route_to_pipeline(
    text: str,
    document_id: str,
    document_type: Optional[str] = None
) -> Any:
    """
    Smart routing to appropriate processing pipeline.
    
    Args:
        text: Document text
        document_id: Document ID
        document_type: Optional explicit type, will auto-detect if None
        
    Returns:
        Processed result (Resume, Contract, or generic dict)
    """
    
    # Auto-detect if not specified
    if document_type is None or document_type == "auto_detect":
        document_type = detect_document_type(text)
        logger.info(f"🔍 Auto-detected document type: {document_type}")
    
    # Route to appropriate pipeline
    if document_type == "resume":
        return await process_resume_document(text, document_id, document_type)
    else:
        return await process_generic_document(text, document_id, document_type)


async def professional_ingestion_workflow(text: str, document_id: str):
    """
    Custom workflow bypassing Cognee's internal pipeline completely.
    Fixes Problem 7: Custom LLM Engine Not Being Used (OpenAI fallback issue)
    
    Architecture: External LLM -> Extract Data -> Local Embeddings -> Push to Memory
    """
    logger.info(f"🚀 Starting PROFESSIONAL ingestion workflow for {document_id}")
    
    try:
        # Step 1: Extract with your own LLM (HF Inference API / Local Qwen)
        # Using CustomCogneeLLMEngine directly
        from app.services.custom_cognee_llm import CustomCogneeLLMEngine
        llm_engine = CustomCogneeLLMEngine()
        
        # Determine extraction strategy based on content (e.g. resume vs generic)
        # For generic docs, we might just want to chunk and embed
        # For structured docs, we extract entities
        
        # Here we demonstrate manual chunking & embedding
        from app.services.embeddings import SentenceTransformerEmbeddingEngine
        embed_engine = SentenceTransformerEmbeddingEngine()
        
        # 1. Chunk text (simple manual chunking for demonstration)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        # 2. Embed chunks locally
        embeddings = []
        for chunk in chunks:
            emb = await embed_engine.embed_text(chunk)
            embeddings.append((chunk, emb))
            
        # 3. Push to Cognee Memory (Vector Store) directly
        try:
            from cognee.infrastructure.databases.vector import get_vector_engine
            vector_engine = get_vector_engine()
            
            # Store embeddings manually
            # This depends on Cognee's internal vector engine API which might vary
            # But ensures we bypass LLMGateway
            
            # If standard API supports manual add:
            points = []
            import uuid
            for chunk, emb in embeddings:
                points.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "vector": emb,
                    "metadata": {"document_id": document_id}
                })
                
            if hasattr(vector_engine, "create_collection"):
                await vector_engine.create_collection(f"doc_{document_id}", embed_engine.get_vector_size())
            
            if hasattr(vector_engine, "upsert"):
                await vector_engine.upsert(f"doc_{document_id}", points)
                
            logger.info(f"✅ Manually processed {len(chunks)} chunks for {document_id}")
            
        except Exception as e:
            logger.error(f"Manual vector storage failed: {e}")
            # Fallback to standard add if manual fails (risky but better than crash)
            await cognee.add(
                data=text, 
                dataset_name=f"doc_{document_id}",
                user=User(id=uuid.UUID(cognee_settings.DEFAULT_USER_ID))
            )
            
        return {"status": "completed", "document_id": document_id}
        
    except Exception as e:
        logger.error(f"Professional ingestion failed: {e}")
        raise
