
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date
try:
    from cognee.modules.graph.models import Entity
except ImportError:
    class Entity(BaseModel):
        pass

# ==================== HR ONTOLOGY (Expanded) ====================

class JobDescription(Entity):
    title: str = Field(..., description="Job title")
    department: Optional[str] = Field(None, description="Department name")
    location: Optional[str] = Field(None, description="Job location")
    employment_type: Optional[str] = Field(None, description="Full-time, Part-time, Contract")
    responsibilities: List[str] = Field(..., description="List of key responsibilities")
    required_skills: List[str] = Field(..., description="List of required technical and soft skills")
    preferred_qualifications: List[str] = Field(..., description="Nice-to-have qualifications")
    benefits: List[str] = Field(..., description="Company benefits offered")

class PerformanceReview(Entity):
    employee_name: str = Field(..., description="Name of employee being reviewed")
    review_period: str = Field(..., description="Period covered by review (e.g., Q1 2024)")
    reviewer_name: str = Field(..., description="Name of manager conducting review")
    overall_rating: str = Field(..., description="Overall performance rating (e.g., Exceeds Expectations, 4/5)")
    strengths: List[str] = Field(..., description="Areas of strong performance")
    areas_for_improvement: List[str] = Field(..., description="Areas needing development")
    goals: List[str] = Field(..., description="Goals set for next period")

# ==================== LEGAL ONTOLOGY ====================

class ContractParty(Entity):
    name: str = Field(..., description="Legal name of the party")
    role: str = Field(..., description="Role in contract (e.g., Buyer, Seller, Landlord)")
    address: Optional[str] = Field(None, description="Registered address")

class ContractClause(Entity):
    clause_id: str = Field(..., description="Clause number or identifier (e.g., 12.3)")
    title: str = Field(..., description="Title of the clause (e.g., Confidentiality)")
    text_summary: str = Field(..., description="Summary of the clause content")
    obligations: List[str] = Field(..., description="Specific obligations extracted from clause")
    type: str = Field(..., description="Type of clause (Indemnity, Liability, Term, Payment)")

class Contract(Entity):
    title: str = Field(..., description="Title of the agreement")
    parties: List[ContractParty] = Field(..., description="List of parties involved")
    effective_date: Optional[str] = Field(None, description="Date contract takes effect")
    termination_date: Optional[str] = Field(None, description="Date contract ends")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction governing contract")
    clauses: List[ContractClause] = Field(..., description="Key clauses extracted")
    financial_terms: List[str] = Field(..., description="Summary of payment/financial terms")

# ==================== FINANCIAL ONTOLOGY ====================

class LineItem(Entity):
    description: str = Field(..., description="Description of item/service")
    quantity: float = Field(..., description="Quantity provided")
    unit_price: float = Field(..., description="Price per unit")
    amount: float = Field(..., description="Total amount for line item")

class Invoice(Entity):
    invoice_number: str = Field(..., description="Unique invoice identifier")
    date: str = Field(..., description="Invoice date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    vendor_name: str = Field(..., description="Company issuing invoice")
    customer_name: str = Field(..., description="Bill to entity")
    line_items: List[LineItem] = Field(..., description="Detailed list of charges")
    subtotal: float = Field(..., description="Sum before tax")
    tax_amount: float = Field(..., description="Total tax amount")
    total_amount: float = Field(..., description="Grand total payable")
    currency: str = Field("USD", description="Currency code")

# ==================== EDUCATION ONTOLOGY ====================

class Concept(Entity):
    term: str = Field(..., description="Key concept or term")
    definition: str = Field(..., description="Definition extracted from text")

class Chapter(Entity):
    number: str = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    summary: str = Field(..., description="Summary of chapter content")
    key_concepts: List[Concept] = Field(..., description="Key concepts introduced")
    learning_objectives: List[str] = Field(..., description="What the student should learn")

class CourseMaterial(Entity):
    title: str = Field(..., description="Title of course/book")
    author: Optional[str] = Field(None, description="Author or instructor")
    subject: str = Field(..., description="Subject area (e.g., Computer Science, History)")
    level: str = Field("Intermediate", description="Beginner, Intermediate, Advanced")
    chapters: List[Chapter] = Field(..., description="Structured content breakdown")

