"""
LangGraph Workflow Engine for Document Intelligence
Implements stateful multi-step reasoning workflows with CAG integration
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from typing_extensions import Literal
from datetime import datetime
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import settings
from app.services.llm_service import llm_service
from app.services.vector_store import vector_store_service
from app.services.cag_engine import cag_engine
from app.services.verification_service import get_verification_service

logger = logging.getLogger(__name__)


# ==================== WORKFLOW STATE ====================

class WorkflowState(TypedDict):
    """State schema for LangGraph workflows"""
    # Input
    query: str
    session_id: str
    document_ids: List[str]
    
    # Processing
    intent: str
    depth: str
    document_type: str
    
    # Context
    cag_context: str  # Cache-Augmented context
    vector_context: str
    conversation_history: List[Dict]
    
    # Generation
    system_prompt: str
    reasoning_steps: List[str]
    final_response: str
    
    # Verification
    verification_score: float
    verification_report: Dict[str, Any]
    flagged: bool
    
    # Metadata
    workflow_type: str
    processing_time_ms: int
    error: Optional[str]


# ==================== NODE FUNCTIONS ====================

def classify_intent_node(state: WorkflowState) -> Dict:
    """Classify query intent and route to appropriate workflow"""
    query = state["query"]
    
    # Use LLM to classify intent
    classification_prompt = f"""
    Classify the user's query intent into ONE of these categories:
    - SUMMARY: Request for document summary or overview
    - FACTUAL: Specific factual question about document content
    - EVALUATIVE: Asking for evaluation, assessment, or critique
    - IMPROVEMENT: Seeking suggestions for enhancement or optimization
    - GAP_ANALYSIS: Looking for gaps, missing information, or timeline analysis
    - SCORING: Request for scoring, rating, or quantitative assessment
    - SEARCH_QUERY: Query requiring external web search
    - GENERAL: General conversation or unclear intent
    
    Query: "{query}"
    
    Respond with ONLY the category name (e.g., "FACTUAL").
    """
    
    intent = llm_service.generate(
        prompt=[{"role": "user", "content": classification_prompt}],
        max_tokens=20,
        temperature=0.0
    ).strip().upper()
    
    # Validate intent
    valid_intents = ["SUMMARY", "FACTUAL", "EVALUATIVE", "IMPROVEMENT", 
                     "GAP_ANALYSIS", "SCORING", "SEARCH_QUERY", "GENERAL"]
    if intent not in valid_intents:
        intent = "GENERAL"
    
    # Classify depth
    depth_prompt = f"""
    Should this query require shallow (basic retrieval) or deep (comprehensive analysis) processing?
    Query: "{query}"
    Respond with ONLY "shallow" or "deep".
    """
    
    depth = llm_service.generate(
        prompt=[{"role": "user", "content": depth_prompt}],
        max_tokens=20,
        temperature=0.0
    ).strip().lower()
    
    if depth not in ["shallow", "deep"]:
        depth = "shallow"
    
    logger.info(f"intent_classified", query=query[:50], intent=intent, depth=depth)
    
    return {
        "intent": intent,
        "depth": depth,
        "reasoning_steps": [f"Intent: {intent}, Depth: {depth}"]
    }


def retrieve_cag_context_node(state: WorkflowState) -> Dict:
    """Retrieve context using Cache-Augmented Generation (CAG)"""
    query = state["query"]
    document_ids = state["document_ids"]
    
    logger.info(f"cag_retrieval", query=query[:50], documents=len(document_ids))
    
    # Check CAG cache first
    cag_result = cag_engine.get_cached_context(document_ids)
    
    if cag_result:
        logger.info("cag_cache_hit", documents=len(document_ids))
        return {
            "cag_context": cag_result["full_context"],
            "document_type": cag_result.get("document_type", "unknown"),
            "reasoning_steps": state["reasoning_steps"] + [
                f"CAG cache hit for {len(document_ids)} documents"
            ]
        }
    
    # Build context from vector store
    limit = 10 if state["depth"] == "deep" else 5
    vector_results = []
    
    for doc_id in document_ids:
        results = vector_store_service.search(
            query=query,
            limit=limit,
            document_id=doc_id
        )
        vector_results.extend(results)
    
    # Sort by relevance
    vector_results.sort(key=lambda x: x.get("distance", 0))
    
    if not vector_results:
        return {
            "cag_context": "",
            "vector_context": "",
            "reasoning_steps": state["reasoning_steps"] + [
                "No relevant context found in vector store"
            ]
        }
    
    # Format context
    context_texts = [res["text"] for res in vector_results[:10]]
    full_context = "\n---\n".join(context_texts)
    
    # Cache for future queries
    cag_engine.cache_context(
        document_ids=document_ids,
        full_context=full_context,
        metadata={
            "document_type": "unknown",
            "chunk_count": len(vector_results)
        }
    )
    
    return {
        "cag_context": full_context,
        "vector_context": full_context,
        "reasoning_steps": state["reasoning_steps"] + [
            f"Retrieved {len(vector_results)} chunks from vector store"
        ]
    }


def build_system_prompt_node(state: WorkflowState) -> Dict:
    """Build system prompt based on intent and context"""
    intent = state["intent"]
    
    # Base system prompt
    base_prompt = """You are DocuCentric, an expert AI document intelligence assistant.

CORE PRINCIPLES:
1. FACT-GROUNDED: Every claim MUST reference specific document sections. NO hallucination.
2. PRECISE: Provide exact quotes, section references, and page numbers when available.
3. TRANSPARENT: Explicitly state when information is missing or unclear.
4. PROFESSIONAL: Maintain witty but expert tone. Be helpful, not robotic.

ANSWERING RULES:
- Base responses ONLY on provided context
- Cite specific sections/paragraphs for every claim
- If information is missing, state: "The document does not specify..."
- Never invent or assume information not in context
- If uncertain, say so explicitly
"""
    
    # Intent-specific instructions
    intent_prompts = {
        "SUMMARY": """
TASK: Provide comprehensive yet concise summary
- Structure: Overview → Key Points → Details → Conclusion
- Highlight most important information
- Maintain logical flow
""",
        "FACTUAL": """
TASK: Answer specific factual question accurately
- Direct answer first, then supporting evidence
- Quote relevant sections
- Be precise and concise
""",
        "EVALUATIVE": """
TASK: Provide expert evaluation/assessment
- Identify strengths and weaknesses
- Use objective criteria
- Provide balanced perspective
- Support opinions with evidence
""",
        "IMPROVEMENT": """
TASK: Suggest actionable improvements
- Identify areas for enhancement
- Provide specific, practical recommendations
- Prioritize by impact
- Consider feasibility
""",
        "GAP_ANALYSIS": """
TASK: Identify gaps, missing information, or inconsistencies
- Systematically analyze for omissions
- Highlight temporal gaps (if applicable)
- Suggest what should be added
- Be thorough but fair
""",
        "SCORING": """
TASK: Provide quantitative scoring/assessment
- Use clear evaluation criteria
- Score each criterion (0-100)
- Explain reasoning for each score
- Provide overall score with justification
- Be objective and evidence-based
""",
        "SEARCH_QUERY": """
TASK: Answer using both document context AND external knowledge
- Integrate document info with web search results
- Clearly distinguish document content vs external info
- Synthesize into comprehensive answer
""",
        "GENERAL": """
TASK: Provide helpful response
- Be conversational but expert
- If query is unclear, ask for clarification
- Stay focused on document context
"""
    }
    
    system_prompt = base_prompt + intent_prompts.get(intent, intent_prompts["GENERAL"])
    
    return {
        "system_prompt": system_prompt,
        "reasoning_steps": state["reasoning_steps"] + [f"Built {intent}-specific system prompt"]
    }


def generate_response_node(state: WorkflowState) -> Dict:
    """Generate response using LLM with full context"""
    context = state["cag_context"]
    
    # Safety gate
    if not context or len(context.strip()) < 50:
        return {
            "final_response": "⚠️ I couldn't find relevant information in the uploaded documents. Please ensure documents have been processed successfully, or try a different question.",
            "reasoning_steps": state["reasoning_steps"] + [
                "Safety gate: Insufficient context"
            ]
        }
    
    # Build conversation history
    history = state.get("conversation_history", [])
    history_text = ""
    if history:
        history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history[-5:]  # Last 5 messages
        ])
    
    # Generate response
    messages = [
        {"role": "system", "content": state["system_prompt"]},
        {"role": "user", "content": f"""
DOCUMENT CONTEXT:
{context}

{history_text}

QUESTION: {state["query"]}

Provide a comprehensive, fact-grounded response based on the context.
"""}
    ]
    
    try:
        response = llm_service.generate(
            prompt=messages,
            max_tokens=4096,
            temperature=0.2
        )
        
        logger.info("response_generated", response_length=len(response))
        
        return {
            "final_response": response,
            "reasoning_steps": state["reasoning_steps"] + [
                f"Generated {len(response.split())} word response"
            ]
        }
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        return {
            "final_response": f"⚠️ Response generation failed: {str(e)}. Please try again.",
            "error": str(e),
            "reasoning_steps": state["reasoning_steps"] + [f"Generation error: {str(e)}"]
        }


def verify_response_node(state: WorkflowState) -> Dict:
    """Verify response for hallucination and accuracy"""
    response = state["final_response"]
    context = state["cag_context"]
    
    if state.get("error"):
        return {
            "verification_score": 0.0,
            "verification_report": {},
            "flagged": False
        }
    
    verification_service = get_verification_service()
    report = verification_service.verify_response(
        response=response,
        context=context,
        include_evidence=False
    )
    
    flagged = report.get("flagged", False)
    score = report.get("overall_score", 100)
    
    # Add warning if flagged
    final_response = response
    if flagged:
        warning = verification_service.format_warning_message(report)
        if warning:
            final_response = f"{response}\n\n{warning}"
    
    logger.info(
        "response_verified",
        score=score,
        flagged=flagged,
        hallucinated=len(report.get("hallucinated_facts", []))
    )
    
    return {
        "final_response": final_response,
        "verification_score": score,
        "verification_report": report,
        "flagged": flagged,
        "reasoning_steps": state["reasoning_steps"] + [
            f"Verification: {score}% score, {'FLAGGED' if flagged else 'PASSED'}"
        ]
    }


# ==================== WORKFLOW DEFINITIONS ====================

def create_document_qa_workflow() -> StateGraph:
    """Create main document Q&A workflow"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("retrieve_cag", retrieve_cag_context_node)
    workflow.add_node("build_prompt", build_system_prompt_node)
    workflow.add_node("generate", generate_response_node)
    workflow.add_node("verify", verify_response_node)
    
    # Define edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve_cag")
    workflow.add_edge("retrieve_cag", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)
    
    return workflow


def create_resume_analysis_workflow() -> StateGraph:
    """Specialized workflow for resume analysis"""
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("retrieve_cag", retrieve_cag_context_node)
    workflow.add_node("build_prompt", build_system_prompt_node)
    workflow.add_node("generate", generate_response_node)
    workflow.add_node("verify", verify_response_node)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve_cag")
    workflow.add_edge("retrieve_cag", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)
    
    return workflow


def create_contract_analysis_workflow() -> StateGraph:
    """Specialized workflow for contract analysis"""
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("retrieve_cag", retrieve_cag_context_node)
    workflow.add_node("build_prompt", build_system_prompt_node)
    workflow.add_node("generate", generate_response_node)
    workflow.add_node("verify", verify_response_node)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve_cag")
    workflow.add_edge("retrieve_cag", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)
    
    return workflow


# ==================== WORKFLOW EXECUTOR ====================

class WorkflowExecutor:
    """Execute LangGraph workflows with memory and caching"""
    
    def __init__(self):
        self.workflows = {
            "document_qa": create_document_qa_workflow(),
            "resume": create_resume_analysis_workflow(),
            "contract": create_contract_analysis_workflow(),
        }
        
        # Compile workflows with memory
        self.compiled_workflows = {}
        for name, workflow in self.workflows.items():
            self.compiled_workflows[name] = workflow.compile(
                checkpointer=MemorySaver()
            )
        
        logger.info("workflow_executor_initialized", workflows=list(self.workflows.keys()))
    
    def select_workflow(self, document_type: str, intent: str) -> str:
        """Select appropriate workflow based on document type and intent"""
        if document_type == "resume":
            return "resume"
        elif document_type in ["contract", "legal"]:
            return "contract"
        else:
            return "document_qa"
    
    async def execute(
        self,
        query: str,
        session_id: str,
        document_ids: List[str],
        conversation_history: Optional[List[Dict]] = None,
        workflow_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute workflow and return result"""
        import time
        start_time = time.time()
        
        # Select workflow
        if not workflow_type:
            workflow_type = "document_qa"
        
        if workflow_type not in self.compiled_workflows:
            workflow_type = "document_qa"
        
        # Initialize state
        initial_state = {
            "query": query,
            "session_id": session_id,
            "document_ids": document_ids,
            "intent": "GENERAL",
            "depth": "shallow",
            "document_type": "unknown",
            "cag_context": "",
            "vector_context": "",
            "conversation_history": conversation_history or [],
            "system_prompt": "",
            "reasoning_steps": [],
            "final_response": "",
            "verification_score": 0.0,
            "verification_report": {},
            "flagged": False,
            "workflow_type": workflow_type,
            "processing_time_ms": 0,
            "error": None
        }
        
        # Execute workflow
        config = {
            "configurable": {
                "thread_id": f"{session_id}_{int(time.time())}"
            }
        }
        
        try:
            result = await self.compiled_workflows[workflow_type].ainvoke(
                initial_state,
                config=config
            )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time
            
            logger.info(
                "workflow_completed",
                workflow=workflow_type,
                processing_time_ms=processing_time,
                verification_score=result.get("verification_score", 0)
            )
            
            return result
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(
                "workflow_failed",
                workflow=workflow_type,
                error=str(e),
                processing_time_ms=processing_time
            )
            
            return {
                **initial_state,
                "final_response": f"⚠️ Workflow error: {str(e)}. Please try again.",
                "error": str(e),
                "processing_time_ms": processing_time,
                "reasoning_steps": [f"Workflow error: {str(e)}"]
            }


# Global workflow executor instance
workflow_executor = WorkflowExecutor()
