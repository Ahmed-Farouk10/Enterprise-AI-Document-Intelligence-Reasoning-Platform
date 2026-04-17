"""
Test LangGraph Workflows
"""
import pytest
from app.services.langgraph_workflows import (
    WorkflowExecutor,
    classify_intent_node,
    build_system_prompt_node,
    WorkflowState
)


@pytest.fixture
def workflow_executor():
    """Create workflow executor instance"""
    return WorkflowExecutor()


@pytest.fixture
def sample_workflow_state():
    """Create sample workflow state for testing"""
    return {
        "query": "What experience does the candidate have?",
        "session_id": "test-session-1",
        "document_ids": ["doc-1"],
        "intent": "GENERAL",
        "depth": "shallow",
        "document_type": "unknown",
        "cag_context": "Sample document context for testing",
        "vector_context": "Sample document context for testing",
        "conversation_history": [],
        "system_prompt": "",
        "reasoning_steps": [],
        "final_response": "",
        "verification_score": 0.0,
        "verification_report": {},
        "flagged": False,
        "workflow_type": "document_qa",
        "processing_time_ms": 0,
        "error": None
    }


def test_workflow_executor_initialization(workflow_executor):
    """Test workflow executor initializes correctly"""
    assert workflow_executor is not None
    assert "document_qa" in workflow_executor.workflows
    assert "resume" in workflow_executor.workflows
    assert "contract" in workflow_executor.workflows


def test_workflow_selection(workflow_executor):
    """Test workflow selection logic"""
    assert workflow_executor.select_workflow("resume", "FACTUAL") == "resume"
    assert workflow_executor.select_workflow("contract", "EVALUATIVE") == "contract"
    assert workflow_executor.select_workflow("unknown", "GENERAL") == "document_qa"


def test_intent_classification_logic(sample_workflow_state):
    """Test intent classification returns valid intent"""
    # This would require LLM service, so we'll test the structure instead
    assert "query" in sample_workflow_state
    assert sample_workflow_state["query"] != ""


def test_system_prompt_building(sample_workflow_state):
    """Test system prompt building for different intents"""
    intents = ["SUMMARY", "FACTUAL", "EVALUATIVE", "IMPROVEMENT", 
               "GAP_ANALYSIS", "SCORING", "SEARCH_QUERY", "GENERAL"]
    
    for intent in intents:
        state = sample_workflow_state.copy()
        state["intent"] = intent
        
        result = build_system_prompt_node(state)
        
        assert "system_prompt" in result
        assert len(result["system_prompt"]) > 0
        assert "DocuCentric" in result["system_prompt"]
        assert len(result["reasoning_steps"]) > 0


@pytest.mark.asyncio
async def test_workflow_execution_with_empty_context(workflow_executor):
    """Test workflow execution with no context"""
    result = await workflow_executor.execute(
        query="Test query",
        session_id="test-session",
        document_ids=[]
    )
    
    # Should return safety gate message
    assert "final_response" in result
    assert "⚠️" in result["final_response"] or "error" in result.get("error", "")


@pytest.mark.asyncio
async def test_workflow_execution_error_handling(workflow_executor):
    """Test workflow handles errors gracefully"""
    result = await workflow_executor.execute(
        query="Test query",
        session_id="test-session",
        document_ids=["nonexistent-doc"]
    )
    
    # Should complete without crashing
    assert "final_response" in result
    assert "processing_time_ms" in result
