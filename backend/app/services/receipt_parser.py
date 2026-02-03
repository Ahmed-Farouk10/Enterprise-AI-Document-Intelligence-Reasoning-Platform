import re
from typing import Dict, Any

class ReceiptParser:
    def extract(self, text: str) -> Dict[str, Any]:
        return {
            "vendor": self._extract_vendor(text),
            "date": self._extract_date(text),
            "total": self._extract_total(text)
        }

    def _extract_vendor(self, text: str) -> str:
        # First non-empty line is usually the vendor
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else None

    def _extract_date(self, text: str) -> str:
        # MM/DD/YYYY or YYYY-MM-DD
        match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        return match.group(0) if match else None

    def _extract_total(self, text: str) -> str:
        # Look for "Total" followed by number
        # Matches: Total 6.49, Total: $6.49
        match = re.search(r'(?:TOTAL|AMOUNT|DUE).*?(\d+\.\d{2})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback: largest number
        try:
            prices = re.findall(r'\d+\.\d{2}', text)
            if prices:
                return str(max(float(p) for p in prices))
        except:
            pass
        return None

receipt_parser = ReceiptParser()
