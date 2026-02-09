"""
RAG Verification Service - Hallucination Detection
====================================================
Post-generation verification to detect facts not present in context.

Features:
- Fact extraction from LLM responses
- Context verification using multiple techniques
- Hallucination scoring and flagging
- Auto-correction triggers
"""
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class FactVerificationService:
    """Service for verifying LLM-generated facts against source context"""
    
    def __init__(self):
        # Patterns for extracting factual claims
        self.fact_patterns = {
            'organization': r'\b(?:at|for|with)\s+([A-Z][A-Za-z\s&\.,-]+(?:Inc|LLC|Corp|University|College|Ltd)?)',
            'date': r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4})',
            'degree': r'\b(Bachelor|Master|PhD|B\.S\.|M\.S\.|Ph\.D\.)(?:\s+(?:of|in)\s+([A-Za-z\s]+))?',
            'job_title': r'\b(Senior|Junior|Lead|Principal|Chief)?\s*(Software|Data|Machine Learning|AI|Full[- ]Stack)?\s*(Engineer|Developer|Scientist|Analyst|Manager)',
            'named_entity': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        }
        
        # Minimum similarity threshold for fuzzy matching
        self.similarity_threshold = 0.75
    
    def extract_facts(self, text: str) -> Dict[str, List[str]]:
        """Extract factual claims from LLM response"""
        facts = {}
        
        for fact_type, pattern in self.fact_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Clean up matches (flatten tuples from groups)
                cleaned = []
                for match in matches:
                    if isinstance(match, tuple):
                        cleaned.extend([m.strip() for m in match if m])
                    else:
                        cleaned.append(match.strip())
                facts[fact_type] = list(set(cleaned))  # Deduplicate
        
        logger.info(f"📊 Extracted facts: {sum(len(v) for v in facts.values())} claims across {len(facts)} types")
        return facts
    
    def fuzzy_match(self, text: str, context: str) -> float:
        """Calculate similarity between text and context"""
        return SequenceMatcher(None, text.lower(), context.lower()).ratio()
    
    def verify_fact(self, fact: str, context: str) -> Tuple[str, float, Optional[str]]:
        """
        Verify a single fact against context
        
        Returns:
            (status, confidence, evidence)
            status: 'verified', 'uncertain', 'hallucinated'
            confidence: 0.0-1.0
            evidence: matching snippet from context or None
        """
        # Exact match (case-insensitive)
        if fact.lower() in context.lower():
            # Find the snippet
            idx = context.lower().find(fact.lower())
            snippet = context[max(0, idx-50):min(len(context), idx+len(fact)+50)]
            return ('verified', 1.0, snippet)
        
        # Fuzzy match for partial/similar text
        # Split context into sentences for better matching
        sentences = re.split(r'[.!?\n]+', context)
        best_match = 0.0
        best_snippet = None
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            similarity = self.fuzzy_match(fact, sentence)
            if similarity > best_match:
                best_match = similarity
                best_snippet = sentence.strip()
        
        if best_match >= self.similarity_threshold:
            return ('uncertain', best_match, best_snippet)
        elif best_match >= 0.4:
            return ('uncertain', best_match, best_snippet)
        else:
            return ('hallucinated', best_match, None)
    
    def verify_response(
        self, 
        response: str, 
        context: str,
        include_evidence: bool = True
    ) -> Dict:
        """
        Verify entire LLM response against context
        
        Returns verification report with:
        - facts_extracted: All extracted claims
        - verified_facts: Facts found in context
        - uncertain_facts: Facts with partial matches
        - hallucinated_facts: Facts not in context
        - overall_score: 0-100 confidence score
        - flagged: True if significant hallucinations detected
        """
        # Extract all factual claims
        facts = self.extract_facts(response)
        
        verification_results = {
            'verified': [],
            'uncertain': [],
            'hallucinated': []
        }
        
        # Verify each fact
        for fact_type, fact_list in facts.items():
            for fact in fact_list:
                status, confidence, evidence = self.verify_fact(fact, context)
                
                result_entry = {
                    'fact': fact,
                    'type': fact_type,
                    'confidence': confidence
                }
                
                if include_evidence and evidence:
                    result_entry['evidence'] = evidence
                
                verification_results[status].append(result_entry)
        
        # Calculate overall score
        total_facts = sum(len(v) for v in verification_results.values())
        if total_facts == 0:
            overall_score = 100  # No factual claims = safe
        else:
            verified_count = len(verification_results['verified'])
            uncertain_count = len(verification_results['uncertain'])
            # Weight: verified=100%, uncertain=50%, hallucinated=0%
            overall_score = int(
                (verified_count + uncertain_count * 0.5) / total_facts * 100
            )
        
        # Flag if score is low or hallucinations detected
        flagged = (
            overall_score < 60 or 
            len(verification_results['hallucinated']) > 0
        )
        
        report = {
            'total_facts': total_facts,
            'verified_facts': verification_results['verified'],
            'uncertain_facts': verification_results['uncertain'],
            'hallucinated_facts': verification_results['hallucinated'],
            'overall_score': overall_score,
            'flagged': flagged,
            'recommendation': self._get_recommendation(overall_score, verification_results)
        }
        
        logger.info(
            f"✅ Verification complete: {overall_score}% confidence | "
            f"Verified: {len(verification_results['verified'])} | "
            f"Uncertain: {len(verification_results['uncertain'])} | "
            f"Hallucinated: {len(verification_results['hallucinated'])}"
        )
        
        return report
    
    def _get_recommendation(self, score: int, results: Dict) -> str:
        """Generate recommendation based on verification results"""
        if score >= 90:
            return "HIGH_CONFIDENCE"
        elif score >= 70:
            return "MEDIUM_CONFIDENCE"
        elif score >= 50:
            return "LOW_CONFIDENCE_REVIEW"
        else:
            return "REGENERATE_REQUIRED"
    
    def format_warning_message(self, report: Dict) -> str:
        """Format user-facing warning message for low-confidence responses"""
        if not report['flagged']:
            return ""
        
        hallucinated = report['hallucinated_facts']
        uncertain = report['uncertain_facts']
        
        warning_parts = []
        
        if hallucinated:
            facts = [f"- {h['fact']}" for h in hallucinated[:3]]
            warning_parts.append(
                f"⚠️ **Potential Hallucinations Detected**: The following information may not be in the document:\n" +
                "\n".join(facts)
            )
        
        if uncertain:
            warning_parts.append(
                f"ℹ️ **{len(uncertain)} claim(s)** have uncertain matches. Confidence: {report['overall_score']}%"
            )
        
        return "\n\n".join(warning_parts)


# Singleton instance
_verification_service = None

def get_verification_service() -> FactVerificationService:
    """Get or create the global verification service instance"""
    global _verification_service
    if _verification_service is None:
        _verification_service = FactVerificationService()
    return _verification_service
