import logging
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger(__name__)

class OCRService:
    """
    Service for extracting text from various document formats.
    Currently supports: TXT, PDF (via pdfplumber with pypdf fallback).
    """

    def extract_text(self, file_path: Path, content_type: str) -> str:
        """
        Extract text from a file based on its content type.
        
        Args:
            file_path: Path to the file.
            content_type: MIME type of the file.
            
        Returns:
            Extracted text string.
            
        Raises:
            Exception: If extraction fails completely.
        """
        text = ""
        try:
            if content_type == "text/plain":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"📝 Extracted {len(text)} chars from TXT")
                
            elif content_type == "application/pdf":
                text = self._extract_from_pdf(file_path)
                
            return text
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}", exc_info=True)
            raise e

    def _extract_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF using pdfplumber with fallback to pypdf.
        """
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
            logger.info(f"📝 Extracted {len(text)} chars from PDF via pdfplumber")
        except Exception as plumber_error:
            logger.warning(f"⚠️ pdfplumber failed: {plumber_error}. Falling back to pypdf.")
            text = "" # Reset
            
        # Fallback to pypdf if plumber failed or returned empty text
        if not text or len(text.strip()) < 50:
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                logger.info(f"📝 Extracted {len(text)} chars from PDF via pypdf (Fallback)")
            except Exception as pypdf_error:
                logger.error(f"❌ pypdf fallback also failed: {pypdf_error}")
                # If both fail, let text be empty
                
        return text

# Global instance
ocr_service = OCRService()
