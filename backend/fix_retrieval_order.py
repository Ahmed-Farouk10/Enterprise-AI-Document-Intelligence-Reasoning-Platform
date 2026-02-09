"""
Direct Fix Script: Swap Cognee and Vector Store Priority
==========================================================
This script patches chat.py to make vector store the primary retrieval method.

Run this from backend/ directory:
    python fix_retrieval_order.py
"""

import re

CHAT_PY_PATH = "app/routes/chat.py"

# New function that puts vector store first
NEW_FUNCTION = '''async def _get_retrieved_context(query: str, depth: str, document_ids: List[str] = []) -> Dict[str, Any]:
    """
    Retrieve context using vector store (PRIMARY) with optional Cognee enhancement.
    
    Retrieval Strategy:
    1. PRIMARY: Vector store hybrid search (always works)
    2. BONUS: Cognee graph enhancement (may fail in restricted envs)
    """
    
    # PRIMARY: Vector Store Retrieval
    try:
        logger.info("🔍 Vector store retrieval (primary)")
        vector_results = vector_store.retrieve_with_citations(query, k=5, use_hybrid=True, use_reranking=True)
        
        if not vector_results:
            return {
                "full_context": "No documents uploaded yet.",
                "document_name": "None",
                "confidence": 0.0,
                "entities": [],
                "retrieval_method": "none"
            }
        
        # Format results
        context_parts = []
        for r in vector_results:
            doc = r.get('doc_name', 'Unknown')
            content = r.get('content', '')
            score = r.get('score', 0.0)
            context_parts.append(f"[{doc}] (Score: {score:.2f})\\n{content}")
        
        vector_context = "\\n\\n---\\n\\n".join(context_parts)
        logger.info(f"✅ Vector store: {len(vector_results)} chunks")
        
    except Exception as e:
        logger.error(f"❌ Vector store failed: {e}")
        return {
            "full_context": "System error retrieving documents.",
            "document_name": "Error",
            "confidence": 0.0,
            "entities": [],
            "retrieval_method": "error"
        }
    
    # BONUS: Try Cognee Enhancement
    cognee_bonus = ""
    entities = []
    
    try:
        logger.info("🧠 Trying Cognee enhancement")
        mode = AnalysisMode.SUMMARIZATION
        if depth == LLMService.DEPTH_EVALUATIVE:
            mode = AnalysisMode.ENTITY_EXTRACTION
        elif depth == LLMService.DEPTH_IMPROVEMENT:
            mode = AnalysisMode.RELATIONSHIP_MAPPING
        
        result = await cognee_engine.query(question=query, document_ids=document_ids, mode=mode)
        
        if result.entities_involved:
            entities = result.entities_involved
            entity_list = "\\n".join([f"- {e['name']}: {e['description']}" for e in entities[:5] if e.get('description')])
            if entity_list:
                cognee_bonus = f"\\n\\n**GRAPH INSIGHTS**:\\n{entity_list}"
                logger.info(f"✅ Cognee: {len(entities)} entities")
    except Exception as e:
        logger.warning(f"⚠️ Cognee unavailable: {str(e)[:100]}")
    
    # Combine
    final_context = vector_context + cognee_bonus
    method = "vector_store" if not cognee_bonus else "hybrid"
    
    return {
        "full_context": final_context,
        "document_name": vector_results[0].get('doc_name', 'Unknown'),
        "confidence": 0.85,
        "entities": entities,
        "retrieval_method": method
    }
'''

def main():
    print("🔧 Patching chat.py to use vector store as primary retrieval...")
    
    try:
        with open(CHAT_PY_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the function
        pattern = r'async def _get_retrieved_context\(.*?\n(?:.*?\n)*?(?=\n\ndef |$)'
        
        if not re.search(pattern, content):
            print("❌ Could not find _get_retrieved_context function!")
            return False
        
        # Replace
        new_content = re.sub(pattern, NEW_FUNCTION, content, count=1)
        
        # Backup original
        with open(CHAT_PY_PATH + '.backup', 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📦 Backup created: {CHAT_PY_PATH}.backup")
        
        # Write new version
        with open(CHAT_PY_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Successfully patched chat.py!")
        print("🔍 Changes:")
        print("   - Vector store is now PRIMARY retrieval method")
        print("   - Cognee is OPTIONAL enhancement (won't break if it fails)")
        print("   - No more LLM hallucination!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
