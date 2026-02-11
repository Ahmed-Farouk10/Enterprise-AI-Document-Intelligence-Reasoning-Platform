"""
Professional Cognee Extraction Pipelines.

This module implements custom extraction pipelines following Cognee best practices
for document intelligence. Uses LLM-powered structured extraction to build rich
knowledge graphs from unstructured documents.

Based on official Cognee documentation:
https://docs.cognee.ai/guides/custom-tasks-and-pipelines
"""

import logging
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
        logger.info("🧠 Extracting resume entities with LLM...")
        
        # Use Cognee's structured output framework (BAML or LiteLLM+Instructor)
        result = await LLMGateway.acreate_structured_output(
            text,
            system_prompt,
            ResumeExtraction
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
        logger.info("🔍 Extracting detailed skills...")
        
        result = await LLMGateway.acreate_structured_output(
            text,
            system_prompt,
            SkillList
        )
        
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
        logger.info("📅 Extracting work history timeline...")
        
        result = await LLMGateway.acreate_structured_output(
            text,
            system_prompt,
            WorkHistoryList
        )
        
        logger.info(f"✅ Extracted {len(result.experiences)} work experiences")
        return result.experiences
        
    except Exception as e:
        logger.error(f"❌ Work history extraction failed: {e}", exc_info=True)
        return []


# ==================== CUSTOM PIPELINES ====================

async def process_resume_document(
    text: str,
    document_id: str,
    document_type: str = "resume"
) -> Resume:
    """
    Professional resume processing pipeline using Cognee best practices.
    
    This is the main entry point for resume ingestion. It:
    1. Extracts structured entities (Person, Work, Education, Skills)
    2. Builds relationships in the knowledge graph
    3. Makes data searchable via vector and graph
    4. Returns the structured Resume object
    
    Args:
        text: Raw resume text from PDF/DOCX
        document_id: Unique document identifier
        document_type: Type of document (default: "resume")
        
    Returns:
        Resume: Fully structured resume with all entities
        
    Raises:
        RuntimeError: If Cognee unavailable or extraction fails
        
    Example:
        >>> text = extract_pdf_text("resume.pdf")
        >>> resume = await process_resume_document(text, "doc_123")
        >>> print(f"Processed {resume.person.name}'s resume")
        >>> print(f"Total experience: {resume.total_years_experience} years")
    """
    
    if not COGNEE_AVAILABLE:
        raise RuntimeError("Cognee not available - professional pipeline disabled")
    
    logger.info(f"🚀 Starting professional resume pipeline for doc {document_id}")
    
    try:
        # Define pipeline tasks
        tasks = [
            # Task 1: Extract resume entities
            Task(extract_resume_entities),  # text → Resume
            
            # Task 2: Add to knowledge graph
            Task(add_data_points)            # Resume → graph storage
        ]
        
        # Dataset name for isolation
        dataset_name = f"resume_{document_id}"
        
        logger.info(f"📊 Running {len(tasks)}-step pipeline...")
        
        # Run pipeline
        final_result = None
        async for result in run_pipeline(
            tasks=tasks,
            data=text,
            datasets=[dataset_name]
        ):
            final_result = result
        
        if final_result is None:
            raise RuntimeError("Pipeline returned no result")
        
        # The final result should be our Resume object after graph insertion
        resume = final_result if isinstance(final_result, Resume) else final_result[-1]
        
        logger.info(
            f"✅ Pipeline complete: "
            f"{len(resume.work_history)} positions, "
            f"{len(resume.education)} degrees, "
            f"{len(resume.skills)} skills indexed"
        )
        
        return resume
        
    except Exception as e:
        logger.error(f"❌ Resume pipeline failed: {e}", exc_info=True)
        raise


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
        dataset_name = f"{document_type}_{document_id}"
        
        # Basic cognee ingestion
        await cognee.add(text, dataset_name=dataset_name)
        await cognee.cognify(datasets=[dataset_name])
        
        logger.info(f"✅ Generic document processed: {dataset_name}")
        
        return {
            "success": True,
            "dataset": dataset_name,
            "document_type": document_type
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
