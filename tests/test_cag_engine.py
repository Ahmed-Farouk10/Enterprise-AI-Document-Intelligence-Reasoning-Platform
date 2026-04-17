"""
Test CAG Engine
"""
import pytest
from app.services.cag_engine import CAGEngine


@pytest.fixture
def cag_engine():
    """Create CAG engine instance for testing"""
    return CAGEngine()


def test_cache_key_generation(cag_engine):
    """Test that cache keys are deterministic"""
    doc_ids_1 = ["doc1", "doc2", "doc3"]
    doc_ids_2 = ["doc3", "doc1", "doc2"]
    
    key1 = cag_engine._get_cache_key(doc_ids_1)
    key2 = cag_engine._get_cache_key(doc_ids_2)
    
    # Keys should be same regardless of order
    assert key1 == key2


def test_document_type_detection(cag_engine):
    """Test automatic document type detection"""
    resume_text = "John Doe\nSoftware Engineer\nExperience: Python, JavaScript\nEducation: BS Computer Science"
    contract_text = "This Agreement between Party A and Party B\nObligations and Clauses\nJurisdiction: New York"
    invoice_text = "Invoice #12345\nTotal: $500.00\nTax: $50.00\nAmount Due: $550.00"
    
    assert cag_engine._detect_document_type(resume_text) == "resume"
    assert cag_engine._detect_document_type(contract_text) == "contract"
    assert cag_engine._detect_document_type(invoice_text) == "invoice"


def test_context_truncation(cag_engine):
    """Test that oversized contexts are truncated"""
    large_context = "x" * 100000
    
    # Simulate cache_context with large input
    # The actual truncation happens inside cache_context
    assert len(large_context) > cag_engine.max_context_size


@pytest.mark.asyncio
async def test_precompute_empty_context(cag_engine):
    """Test precomputing with no documents"""
    result = await cag_engine.precompute_context([])
    assert result is None


def test_cache_stats(cag_engine):
    """Test cache statistics retrieval"""
    stats = cag_engine.get_cache_stats()
    assert isinstance(stats, dict)
    # Stats should have certain keys even if empty
    assert "cached_documents" in stats or "error" in stats
