"""
D2: Simplified Layout Parser (Heuristic-based)
Uses text patterns instead of LayoutLMv3 to avoid 1.6GB model
"""

import re
import logging

logger = logging.getLogger(__name__)

class LayoutParser:
    def __init__(self):
        logger.info("Lightweight layout parser initialized (heuristic-based)")
    
    def parse(self, text: str):
        """
        Parse document structure using heuristics
        """
        lines = text.split('\n')
        
        headers = []
        questions = []
        answers = []
        tables = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Header: short, all caps or title case, no punctuation at end
            if len(line) < 50 and (line.isupper() or line.istitle()) and not line.endswith(('.', '?', ':')):
                headers.append(line)
            
            # Question: ends with ? or has question words
            elif line.endswith('?') or any(q in line.lower() for q in ['what', 'when', 'where', 'who', 'how', 'why']):
                questions.append(line)
            
            # Answer: follows question, longer line
            elif i > 0 and lines[i-1].strip().endswith('?'):
                answers.append(line)
            
            # Table row: contains multiple | or tab-separated
            elif '|' in line or '\t' in line:
                tables.append(line)
        
        return {
            "headers": headers[:10],
            "questions": questions[:10],
            "answers": answers[:10],
            "tables": tables[:5],
            "entity_counts": {
                "headers": len(headers),
                "questions": len(questions),
                "answers": len(answers),
                "tables": len(tables)
            },
            "is_form_like": len(questions) > 0 and len(answers) > 0,
            "model": "heuristic-layout-v1",
            "dataset": "generated",
            "parsing_method": "rule_based"
        }

# Singleton
layout_parser = LayoutParser()