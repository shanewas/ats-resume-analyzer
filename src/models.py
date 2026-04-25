from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description_text: str

class ResumeParseRequest(BaseModel):
    text: str
    file_type: str = "text"

class JobDescParseRequest(BaseModel):
    text: str
    file_type: str = "text"

class Suggestion(BaseModel):
    gap: str
    suggested_phrase: str

class AnalysisResult(BaseModel):
    match_score: int
    cosine_similarity: float
    matched_skills: List[str]
    missing_skills: List[str]
    weak_skills: List[str]
    suggestions: List[Suggestion]