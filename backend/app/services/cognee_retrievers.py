"""
Professional Cognee Retrievers for Domain-Specific Queries.

Custom retrieval strategies that leverage Cognee's hybrid search (graph + vector)
for domain-specific document intelligence queries.

Based on official Cognee documentation:
https://docs.cognee.ai/core-concepts/overview
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

try:
    import cognee
    from cognee.api.v1.search import SearchType
    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False
    logging.warning("Cognee not available - custom retrievers disabled")

from app.models.cognee_models import (
    CareerGap,
    SkillMatch,
    ComparisonResult
)

logger = logging.getLogger(__name__)


# ==================== RESUME RETRIEVER ====================

class ResumeRetriever:
    """
    Domain-specific retriever for resume/CV analysis.
    
    Provides high-level query methods that combine graph traversal
    with vector similarity for accurate resume intelligence.
    """
    
    @staticmethod
    async def find_candidates_with_skills(
        required_skills: List[str],
        min_experience_years: int = 0,
        datasets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find candidates matching specific skill requirements.
        
        Uses hybrid search to find people with required skills and experience level.
        
        Args:
            required_skills: List of required skills (e.g., ["Python", "Machine Learning"])
            min_experience_years: Minimum years of experience required
            datasets: Optional list of dataset names to search (defaults to all resume datasets)
            
        Returns:
            Dict with matched candidates and details
            
        Example:
            >>> results = await ResumeRetriever.find_candidates_with_skills(
            ...     required_skills=["Python", "AWS", "Docker"],
            ...     min_experience_years=3
            ... )
            >>> print(f"Found {len(results['candidates'])} matching candidates")
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        # Build query
        skills_str = ", ".join(required_skills)
        query = f"Find people who have experience with: {skills_str}"
        
        if min_experience_years > 0:
            query += f" with at least {min_experience_years} years of professional experience"
        
        logger.info(f"🔍 Searching for candidates with skills: {required_skills}")
        
        try:
            # Use GRAPH_COMPLETION for relationship traversal (Person → has_skill → Skill)
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                datasets=datasets or []  # Empty list searches all datasets
            )
            
            # Parse results
            # Note: Cognee returns LLM-generated answer + retrieved graph data
            
            logger.info(f"✅ Candidate search complete")
            
            return {
                "query": query,
                "required_skills": required_skills,
                "min_experience": min_experience_years,
                "results": results,
                "search_type": "graph_completion"
            }
            
        except Exception as e:
            logger.error(f"❌ Candidate search failed: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def analyze_career_gaps(
        document_id: str
    ) -> Dict[str, Any]:
        """
        Temporal analysis of work history to identify employment gaps.
        
        Uses graph traversal to analyze WorkExperience entities chronologically
        and detect gaps between positions.
        
        Args:
            document_id: Document ID of the resume to analyze
            
        Returns:
            Dict with gap analysis results
            
        Example:
            >>> analysis = await ResumeRetriever.analyze_career_gaps("doc_123")
            >>> if analysis['gaps_found']:
            ...     for gap in analysis['gaps']:
            ...         print(f"Gap: {gap['duration_months']} months")
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        query = """
        Analyze the work history timeline in this resume.
        
        For each employment period:
        1. Identify start and end dates
        2. Calculate duration
        3. Check for gaps between consecutive positions
        
        For any gaps found:
        - Calculate gap duration in months
        - Note the positions before and after the gap
        - Check if an explanation is provided
        
        Return a detailed timeline analysis including any employment gaps.
        """
        
        logger.info(f"📅 Analyzing career gaps for document {document_id}")
        
        try:
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                datasets=[f"resume_{document_id}"]
            )
            
            logger.info(f"✅ Career gap analysis complete")
            
            return {
                "document_id": document_id,
                "analysis": results,
                "search_type": "temporal_analysis"
            }
            
        except Exception as e:
            logger.error(f"❌ Career gap analysis failed: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def compare_to_job_requirements(
        resume_id: str,
        job_requirements: str
    ) -> Dict[str, Any]:
        """
        Comparative analysis: resume vs job posting requirements.
        
        Performs detailed skills matching and fitness scoring.
        
        Args:
            resume_id: Document ID of the resume
            job_requirements: Text of job requirements/description
            
        Returns:
            Dict with comparison results and match score
            
        Example:
            >>> job_req = "Required: Python, AWS, 5+ years experience..."
            >>> comparison = await ResumeRetriever.compare_to_job_requirements(
            ...     "doc_123",
            ...     job_req
            ... )
            >>> print(f"Match score: {comparison['match_score']}/100")
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        query = f"""
        Compare this resume against the following job requirements:
        
        {job_requirements}
        
        Perform a detailed comparative analysis:
        
        1. SKILL MATCHING:
           - Identify skills mentioned in job requirements
           - Check if candidate has each required skill
           - Note proficiency level if available
           - Calculate years of experience per skill
        
        2. EXPERIENCE MATCHING:
           - Compare required years of experience
           - Check if work history aligns with job requirements
           - Identify relevant positions
        
        3. QUALIFICATION MATCHING:
           - Check educational requirements
           - Verify certifications if required
        
        4. GAPS ANALYSIS:
           - Missing skills (required but not in resume)
           - Extra skills (in resume but not required - competitive advantage)
           - Matching skills (perfect fit)
        
        5. FIT SCORE:
           - Calculate overall match score (0-100)
           - Provide recommendation (strongly recommended, recommended, possible fit, not recommended)
        
        Return a comprehensive comparison with specific examples and recommendations.
        """
        
        logger.info(f"⚖️ Comparing resume {resume_id} to job requirements")
        
        try:
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                datasets=[f"resume_{resume_id}"]
            )
            
            logger.info(f"✅ Comparison analysis complete")
            
            return {
                "resume_id": resume_id,
                "job_requirements": job_requirements,
                "analysis": results,
                "search_type": "comparative_analysis"
            }
            
        except Exception as e:
            logger.error(f"❌ Comparison failed: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def extract_career_trajectory(
        document_id: str
    ) -> Dict[str, Any]:
        """
        Extract career progression and growth trajectory.
        
        Analyzes how the person's career has evolved over time:
        - Title progressions (Junior → Senior → Lead)
        - Industry transitions
        - Skill acquisition over time
        
        Args:
            document_id: Resume document ID
            
        Returns:
            Dict with career trajectory analysis
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        query = """
        Analyze this person's career trajectory and professional growth.
        
        Extract:
        1. Career progression (title changes, promotions)
        2. Industry transitions (if any)
        3. Skill development over time
        4. Increasing responsibilities
        5. Career highlights and achievements
        
        Present as a chronological narrative of professional growth.
        """
        
        logger.info(f"📈 Extracting career trajectory for {document_id}")
        
        try:
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                datasets=[f"resume_{document_id}"]
            )
            
            logger.info(f"✅ Career trajectory extraction complete")
            
            return {
                "document_id": document_id,
                "trajectory": results,
                "search_type": "career_progression"
            }
            
        except Exception as e:
            logger.error(f"❌ Trajectory extraction failed: {e}", exc_info=True)
            raise


# ==================== GENERIC DOCUMENT RETRIEVER ====================

class DocumentRetriever:
    """
    Generic retriever for non-resume documents.
    
    Provides flexible search across various document types.
    """
    
    @staticmethod
    async def semantic_search(
        query: str,
        document_ids: Optional[List[str]] = None,
        search_type: str = "graph_completion"
    ) -> Dict[str, Any]:
        """
        Perform semantic search across documents.
        
        Args:
            query: Natural language query
            document_ids: Optional list of document IDs to search
            search_type: "graph_completion", "graph_search", or "auto"
            
        Returns:
            Dict with search results
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        # Map search type string to Cognee SearchType enum
        type_map = {
            "graph_completion": SearchType.GRAPH_COMPLETION,
            "graph_search": SearchType.GRAPH_SEARCH,
            "auto": SearchType.GRAPH_COMPLETION  # Default
        }
        
        cognee_search_type = type_map.get(search_type, SearchType.GRAPH_COMPLETION)
        
        # Build dataset list
        datasets = []
        if document_ids:
            datasets = [f"doc_{doc_id}" for doc_id in document_ids]
        
        logger.info(f"🔍 Semantic search: {query} (type: {search_type})")
        
        try:
            results = await cognee.search(
                query_type=cognee_search_type,
                query_text=query,
                datasets=datasets
            )
            
            logger.info(f"✅ Search complete")
            
            return {
                "query": query,
                "search_type": search_type,
                "document_ids": document_ids,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def multi_document_query(
        query: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Query across multiple documents simultaneously.
        
        Useful for:
        - Comparing documents
        - Finding common themes
        - Cross-document analysis
        
        Args:
            query: Query to run across all documents
            document_ids: List of document IDs to search
            
        Returns:
            Dict with multi-document results
        """
        
        if not COGNEE_AVAILABLE:
            raise RuntimeError("Cognee not available")
        
        datasets = [f"doc_{doc_id}" for doc_id in document_ids]
        
        logger.info(f"📚 Multi-document query across {len(document_ids)} documents")
        
        try:
            results = await cognee.search(
                query_type=SearchType.GRAPH_COMPLETION,
                query_text=query,
                datasets=datasets
            )
            
            logger.info(f"✅ Multi-document query complete")
            
            return {
                "query": query,
                "document_count": len(document_ids),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-document query failed: {e}", exc_info=True)
            raise


# ==================== RETRIEVER FACTORY ====================

class RetrieverFactory:
    """
    Factory to get the appropriate retriever for a document type.
    """
    
    @staticmethod
    def get_retriever(document_type: str):
        """
        Get specialized retriever for document type.
        
        Args:
            document_type: "resume", "contract", "report", etc.
            
        Returns:
            Appropriate retriever class
        """
        
        retrievers = {
            "resume": ResumeRetriever,
            "cv": ResumeRetriever,
            "default": DocumentRetriever
        }
        
        return retrievers.get(document_type, DocumentRetriever)
