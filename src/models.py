from pydantic import BaseModel
from typing import List, Optional

class ResumeParseRequest(BaseModel):
    text: str
    file_type: str = "text"  # "pdf" | "docx" | "text"

class JobDescParseRequest(BaseModel):
    text: str
    file_type: str = "text"

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description_text: str

class ResumeParsed(BaseModel):
    skills: List[str]
    experience_years: Optional[int] = None
    education: List[str] = []
    companies: List[str] = []
    raw_text: str

class JobDescriptionParsed(BaseModel):
    required_skills: List[str]
    nice_to_have: List[str] = []
    level: str  # "senior" | "mid" | "junior"
    keywords: List[str]
    raw_text: str

class Suggestion(BaseModel):
    gap: str
    suggested_phrase: str

class AnalysisResult(BaseModel):
    match_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    weak_skills: List[str] = []
    suggestions: List[Suggestion]