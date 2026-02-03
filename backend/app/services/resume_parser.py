import re
from typing import Dict, Any, List

class ResumeParser:
    def extract(self, text: str) -> Dict[str, Any]:
        return {
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "skills": self._extract_skills(text)
        }

    def _extract_email(self, text: str) -> str:
        match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> str:
        match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        return match.group(0) if match else None

    def _extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [skill for skill in self.SKILL_DB if skill in text_lower]

    @property
    def SKILL_DB(self):
        return {
            "python", "java", "javascript", "typescript", "react", "vue", "angular",
            "node", "aws", "azure", "gcp", "docker", "kubernetes", "sql", "postgresql",
            "mongodb", "redis", "fastapi", "flask", "django", "git", "linux", "html", "css",
            "machine learning", "deep learning", "nlp", "cv", "pytorch", "tensorflow"
        }

resume_parser = ResumeParser()
