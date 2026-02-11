"""
Professional Cognee Data Models for Document Intelligence.

This module defines domain-specific DataPoint models following Cognee best practices.
Each model represents entities extracted from uploaded documents (resumes, contracts, etc.)
and their relationships in the knowledge graph.

Based on official Cognee documentation:
https://docs.cognee.ai/guides/custom-data-models
"""

from typing import List, Dict, Any, Optional
from pydantic import Field, BaseModel, model_validator
from datetime import datetime

try:
    from cognee.infrastructure.engine import DataPoint
except ImportError:
    # Fallback for type checking if Cognee not installed
    class DataPoint(BaseModel):
        pass


# ==================== CORE ENTITIES ====================

class Skill(DataPoint):
    """Skill"""
    name: str = Field(description="Skill name")
    level: Optional[str] = Field(
        default=None,
        description="Level"
    )
    years_experience: Optional[int] = Field(
        default=None,
        description="Years"
    )
    category: Optional[str] = Field(
        default=None,
        description="Category"
    )
    
    # Make skills searchable by name and category
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["name", "category", "level"]},
        description="Metadata for vector indexing"
    )

    @model_validator(mode = "before")
    @classmethod
    def convert_string_to_skill(cls, data: Any) -> Any:
        """
        Normalize skill format. 
        If LLM returns a string, convert to object with 'name' field.
        """
        if isinstance(data, str):
            return {"name": data}
        return data


class Organization(DataPoint):
    """
    Company, institution, or organization entity.
    
    Examples: Microsoft, Stanford University, Red Cross
    """
    name: str = Field(description="Organization name")
    type: Optional[str] = Field(
        default=None,
        description="Type: company, university, nonprofit, government"
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry sector (e.g., 'Technology', 'Healthcare')"
    )
    location: Optional[str] = Field(
        default=None,
        description="Primary location or headquarters"
    )
    
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["name", "industry", "type"]},
        description="Metadata for vector indexing"
    )


class Person(DataPoint):
    """
    Individual person with professional profile.
    
    Represents the main subject of a resume or profile document.
    """
    name: str = Field(description="Full name")
    title: Optional[str] = Field(
        default=None,
        description="Current professional title (e.g., 'Senior Software Engineer')"
    )
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="Current location/city")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    
    # Relationships (populated by LLM extraction)
    skills: List[Skill] = Field(
        default_factory=list,
        description="Skills possessed by this person"
    )
    
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["name", "title", "location"]},
        description="Metadata for vector indexing"
    )


# ==================== WORK HISTORY ====================

class WorkExperience(DataPoint):
    """
    Single employment history entry.
    
    Represents a position held at an organization, with dates and responsibilities.
    """
    person_name: str = Field(description="Person who held this position")
    organization: str = Field(description="Company/organization name")
    title: str = Field(description="Job title")
    
    start_date: Optional[str] = Field(
        default=None,
        description="Start date (YYYY-MM or similar format)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date (YYYY-MM, 'Present', or null if current)"
    )
    duration_months: Optional[int] = Field(
        default=None,
        description="Calculated duration in months"
    )
    
    location: Optional[str] = Field(default=None, description="Location")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Responsibilities"
    )
    skills_used: List[str] = Field(
        default_factory=list,
        description="Skills"
    )

    @model_validator(mode = "before")
    @classmethod
    def normalize_experience(cls, data: Any) -> Any:
        """Ensure responsibilities and skills_used are always lists of strings."""
        if isinstance(data, dict):
            for field in ["responsibilities", "skills_used"]:
                if field in data and isinstance(data[field], str):
                    data[field] = [data[field]]
        return data
    
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["title", "organization"]},
        description="Metadata for vector indexing"
    )


class Education(DataPoint):
    """
    Educational credential or degree.
    
    Represents academic achievement from an institution.
    """
    person_name: str = Field(description="Person who earned this degree")
    institution: str = Field(description="University or educational institution")
    degree: str = Field(description="Degree type (e.g., 'Bachelor of Science')")
    field_of_study: Optional[str] = Field(
        default=None,
        description="Major/field of study (e.g., 'Computer Science')"
    )
    graduation_date: Optional[str] = Field(
        default=None,
        description="Graduation date (YYYY or YYYY-MM)"
    )
    gpa: Optional[float] = Field(default=None, description="Grade point average")
    honors: Optional[str] = Field(
        default=None,
        description="Academic honors (e.g., 'Summa Cum Laude')"
    )
    
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["degree", "field_of_study", "institution"]},
        description="Metadata for vector indexing"
    )


# ==================== DOCUMENT MODELS ====================

class Resume(DataPoint):
    """
    Complete resume/CV document with all extracted information.
    
    This is the top-level model that aggregates all resume components:
    - Personal information
    - Work history
    - Education
    - Skills
    - Summary/objective
    """
    person: Person = Field(description="Person this resume represents")
    
    summary: Optional[str] = Field(
        default=None,
        description="Professional summary or objective statement"
    )
    
    work_history: List[WorkExperience] = Field(
        default_factory=list,
        description="Employment history, typically in reverse chronological order"
    )
    
    education: List[Education] = Field(
        default_factory=list,
        description="Educational credentials"
    )
    
    skills: List[Skill] = Field(
        default_factory=list,
        description="Professional skills and competencies"
    )
    
    certifications: List[str] = Field(
        default_factory=list,
        description="Professional certifications"
    )
    
    languages: List[str] = Field(
        default_factory=list,
        description="Languages spoken (e.g., 'English (Native)', 'Spanish (Fluent)')"
    )

    @model_validator(mode = "before")
    @classmethod
    def normalize_resume_lists(cls, data: Any) -> Any:
        """Ensure certifications and languages are always lists of strings."""
        if isinstance(data, dict):
            for field in ["certifications", "languages"]:
                if field in data and isinstance(data[field], str):
                    data[field] = [data[field]]
        return data
    
    # Metadata
    total_years_experience: Optional[int] = Field(
        default=None,
        description="Calculated total years of professional experience"
    )
    
    metadata: Dict[str, Any] = Field(
        default={"index_fields": ["summary"]},
        description="Metadata for vector indexing"
    )


# ==================== HELPER MODELS FOR LLM EXTRACTION ====================

class ResumeExtraction(BaseModel):
    """
    Container model for LLM extraction results.
    
    Used as the response schema for structured output from LLM.
    """
    resume: Resume


class SkillList(BaseModel):
    """Container for extracting multiple skills"""
    skills: List[Skill]


class WorkHistoryList(BaseModel):
    """Container for extracting work history"""
    experiences: List[WorkExperience]


class EducationList(BaseModel):
    """Container for extracting education"""
    degrees: List[Education]


# ==================== ANALYSIS MODELS ====================

class CareerGap(BaseModel):
    """
    Represents an identified gap in employment history.
    
    Used for temporal analysis of work history.
    """
    start_date: str = Field(description="When the gap started")
    end_date: str = Field(description="When the gap ended")
    duration_months: int = Field(description="Length of gap in months")
    previous_role: str = Field(description="Job before the gap")
    next_role: str = Field(description="Job after the gap")
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation if provided in resume"
    )


class SkillMatch(BaseModel):
    """
    Skill matching analysis result.
    
    Used for comparing resume skills against job requirements.
    """
    required_skill: str
    has_skill: bool
    proficiency_level: Optional[str] = None
    years_experience: Optional[int] = None
    match_confidence: float = Field(
        description="Confidence score 0-1 for the match"
    )


class ComparisonResult(BaseModel):
    """
    Result of comparing a resume against job requirements.
    """
    overall_match_score: float = Field(
        description="Overall fitness score 0-100"
    )
    matching_skills: List[SkillMatch] = Field(
        description="Skills that match requirements"
    )
    missing_skills: List[str] = Field(
        description="Required skills not found in resume"
    )
    extra_skills: List[str] = Field(
        description="Skills in resume not required by job"
    )
    recommendations: List[str] = Field(
        description="Recommendations for improvement"
    )
