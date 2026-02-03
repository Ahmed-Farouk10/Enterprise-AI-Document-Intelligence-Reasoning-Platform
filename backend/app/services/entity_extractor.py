"""
D3: SROIE Entity Extraction
Deterministic regex/heuristic-based extraction for receipts
"""

import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities from receipt text using regex patterns.
    No ML model - 100% deterministic and interpretable.
    """
    
    # Currency patterns: $1,234.56 or 1,234.56 or 1234.56
    MONEY_PATTERN = r'\$\s*\d[\d,]*\.?\d{0,2}|\b\d[\d,]*\.\d{2}\b'
    
    # Date patterns: various formats
    DATE_PATTERNS = [
        r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
        r'\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
    ]
    
    # Keywords indicating totals
    TOTAL_KEYWORDS = ['total', 'amount due', 'balance', 'sum', 'grand total', 'final']
    
    # Keywords indicating tax
    TAX_KEYWORDS = ['tax', 'vat', 'gst', 'hst', 'sales tax']
    
    # Company indicators
    COMPANY_INDICATORS = ['inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co.', 'limited']
    
    def __init__(self):
        logger.info("D3 EntityExtractor initialized (regex-based)")
    
    def extract(self, text: str) -> Dict:
        """
        Extract all entities from receipt text.
        """
        text_lower = text.lower()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        entities = {
            'total_amount': self._extract_total(lines, text),
            'tax_amount': self._extract_tax(lines, text),
            'date': self._extract_date(text),
            'company_name': self._extract_company(lines),
            'items': self._extract_items(lines),
        }
        
        confidence = self._calculate_confidence(entities)
        
        return {
            'entities': entities,
            'confidence': round(confidence, 4),
            'extraction_method': 'deterministic_regex',
            'dataset': 'SROIE-pattern-based',
            'lines_processed': len(lines)
        }
    
    def _extract_total(self, lines: List[str], full_text: str) -> Dict:
        """Extract total amount using keyword proximity."""
        all_money = re.findall(self.MONEY_PATTERN, full_text)
        
        if not all_money:
            return {'value': None, 'confidence': 0.0, 'method': 'none_found'}
        
        cleaned = []
        for m in all_money:
            val = re.sub(r'[^\d.]', '', m.replace(',', ''))
            try:
                cleaned.append((m, float(val)))
            except ValueError:
                continue
        
        if not cleaned:
            return {'value': None, 'confidence': 0.0, 'method': 'parse_error'}
        
        # Find line with total keyword
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.TOTAL_KEYWORDS):
                for j in range(i, min(i+3, len(lines))):
                    for orig, val in cleaned:
                        if orig in lines[j] and val > 0:
                            return {
                                'value': orig,
                                'numeric_value': val,
                                'confidence': 0.9,
                                'method': 'keyword_proximity',
                                'keyword_found': True
                            }
        
        # Largest amount heuristic
        largest = max(cleaned, key=lambda x: x[1])
        return {
            'value': largest[0],
            'numeric_value': largest[1],
            'confidence': 0.6,
            'method': 'largest_amount_heuristic',
            'keyword_found': False
        }
    
    def _extract_tax(self, lines: List[str], full_text: str) -> Dict:
        """Extract tax amount using keyword proximity."""
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.TAX_KEYWORDS):
                money = re.findall(self.MONEY_PATTERN, line)
                if money:
                    val = re.sub(r'[^\d.]', '', money[-1].replace(',', ''))
                    try:
                        return {
                            'value': money[-1],
                            'numeric_value': float(val),
                            'confidence': 0.85,
                            'method': 'tax_keyword_line'
                        }
                    except ValueError:
                        continue
        
        return {'value': None, 'confidence': 0.0, 'method': 'not_found'}
    
    def _extract_date(self, text: str) -> Dict:
        """Extract date using multiple patterns."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    'value': match.group(),
                    'confidence': 0.8,
                    'method': 'regex_match'
                }
        
        return {'value': None, 'confidence': 0.0, 'method': 'not_found'}
    
    def _extract_company(self, lines: List[str]) -> Dict:
        """Extract company name from header or lines with company indicators."""
        for line in lines[:10]:
            line_lower = line.lower()
            if any(ind in line_lower for ind in self.COMPANY_INDICATORS):
                clean = line.strip()
                if len(clean) > 2 and len(clean) < 100:
                    return {
                        'value': clean,
                        'confidence': 0.75,
                        'method': 'company_indicator'
                    }
        
        if lines:
            first = lines[0].strip()
            if len(first) > 2 and len(first) < 50 and not any(c.isdigit() for c in first[:5]):
                return {
                    'value': first,
                    'confidence': 0.5,
                    'method': 'first_line_heuristic'
                }
        
        return {'value': None, 'confidence': 0.0, 'method': 'not_found'}
    
    def _extract_items(self, lines: List[str]) -> List[Dict]:
        """Extract line items (description + price)."""
        items = []
        
        for line in lines:
            match = re.match(r'(.+?)\s+[\$]?\s*(\d[\d,]*\.?\d{0,2})\s*$', line)
            if match:
                desc, price = match.groups()
                desc = desc.strip()
                price_clean = price.replace(',', '')
                
                desc_lower = desc.lower()
                if any(kw in desc_lower for kw in self.TOTAL_KEYWORDS + self.TAX_KEYWORDS):
                    continue
                
                if len(desc) > 2:
                    items.append({
                        'description': desc,
                        'price': f"${price}",
                        'numeric_price': float(price_clean)
                    })
            
            if len(items) >= 10:
                break
        
        return items
    
    def _calculate_confidence(self, entities: Dict) -> float:
        """Calculate overall extraction confidence."""
        confidences = []
        
        for key in ['total_amount', 'tax_amount', 'date', 'company_name']:
            if key in entities and isinstance(entities[key], dict):
                confidences.append(entities[key].get('confidence', 0))
        
        if 'total_amount' in entities and isinstance(entities['total_amount'], dict):
            confidences.append(entities['total_amount'].get('confidence', 0) * 1.5)
        
        if not confidences:
            return 0.0
        
        return min(sum(confidences) / len(confidences), 1.0)


# Singleton instance
entity_extractor = EntityExtractor()