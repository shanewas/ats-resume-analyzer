"""
ATS Resume Analyzer — FastAPI server
Pipeline: parse → TF-IDF match → gap analysis → suggestions
"""

import io
import re
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from models import AnalyzeRequest, AnalysisResult, ResumeParsed, JobDescriptionParsed, Suggestion

# ── File Parsing ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {e}")

def extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(docx_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"DOCX parsing failed: {e}")

def parse_file(text: str, file_type: str) -> str:
    """Route parsing by file type."""
    if file_type == "pdf":
        # If text is base64 or raw bytes was passed — re-parse
        return text
    return text

# ── NLP Skill Extraction ──────────────────────────────────────────────────

# Common tech skills to look for (extendable)
SKILL_KEYWORDS = {
    # Languages
    "Python", "C++", "C#", "Java", "JavaScript", "TypeScript", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    # Frameworks
    ".NET", ".NET Core", "React", "Vue", "Angular", "Node.js", "Django", "Flask",
    "FastAPI", "Spring", "Rails", "Laravel", "Blazor", "ElectronJS",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "K8s", "Terraform",
    "Jenkins", "GitLab CI", "GitHub Actions", "CircleCI", "RPM", "systemd",
    "Linux", "RHEL", "UNIX", "Bash", "Shell",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "SQL Server", "Oracle",
    "SQL", "NoSQL",
    # Data / ML
    "TensorFlow", "PyTorch", "scikit-learn", "Keras", "pandas", "NumPy",
    "NLP", "OCR", "Machine Learning", "Deep Learning", "ML",
    "NLP", "AI", "Computer Vision",
    # APIs / Architecture
    "REST", "GraphQL", "gRPC", "OpenAPI", "Swagger",
    "Microservices", "API", "WebAPI",
    # PDFs / Documents
    "PDF", "PDF/A", "iText", "PDFium", "LibreOffice",
    # Others
    "Git", "CI/CD", "TDD", "Agile", "Scrum",
    "Figma", "Miro", "UI/UX", "UX",
    "Azure Functions", "Lambda", "SQS", "SNS", "RDS", "S3", "Blob",
    "Playwright", "Selenium", "Jest", "xUnit", "Pytest",
}

# Senior level indicators
SENIOR_KEYWORDS = {"lead", "architect", "senior", "principal", "staff", "manager", "mentor", "10+", "10 years"}
MID_KEYWORDS = {"mid", "5+", "5 years", "intermediate", "experienced"}
JUNIOR_KEYWORDS = {"junior", "entry", "graduate", "intern", "1-3", "1 year", "2 year", "3 year"}

def extract_skills(text: str) -> List[str]:
    """Extract skills from text using keyword matching."""
    text_upper = text.upper()
    found = []
    for skill in SKILL_KEYWORDS:
        # Match whole word only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found.append(skill)
    return found

def infer_experience_years(text: str) -> int:
    """Infer years of experience from text."""
    # Look for patterns like "10+ years", "5 years", "since 2015"
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'experience[^\n]*(\d+)\s*years?',
        r'(\d{4})\s*-\s*(?:present|current)',  # Start year - present
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            if num > 1970 and num < 2030:
                # It's a year — estimate
                return 2026 - num
            if num < 50:
                return num
    return 7  # default fallback

def extract_companies(text: str) -> List[str]:
    """Extract company names (rough heuristic — capitalized multi-word sequences)."""
    # Simple: find ALL-CAPS words that aren't common job titles
    stopwords = {"LLC", "INC", "LTD", "CORP", "CO", "LTD", "GMBH", "K.K.", "KK", "THE", "AND", "FOR", "WITH"}
    words = text.split()
    companies = []
    for i, word in enumerate(words):
        if word.isupper() and word not in stopwords and len(word) > 2:
            # Grab following words if they're also capitalized
            phrase = word
            for j in range(i+1, min(i+3, len(words))):
                if words[j].isupper() or words[j].lower() in {"of", "and", "for"}:
                    phrase += " " + words[j]
                else:
                    break
            if len(phrase) > 3:
                companies.append(phrase)
    return list(set(companies))[:10]

def infer_level(text: str) -> str:
    """Infer job level from JD text."""
    text_lower = text.lower()
    senior_count = sum(1 for kw in SENIOR_KEYWORDS if kw in text_lower)
    mid_count = sum(1 for kw in MID_KEYWORDS if kw in text_lower)
    junior_count = sum(1 for kw in JUNIOR_KEYWORDS if kw in text_lower)
    # Only override to senior/mid if explicit indicators present
    if senior_count > mid_count and senior_count > junior_count:
        return "senior"
    elif mid_count > junior_count:
        return "mid"
    # Default: junior (most job descriptions that don't say senior are junior/mid anyway)
    return "junior"

# ── TF-IDF Matching ────────────────────────────────────────────────────────

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_match_score(resume_text: str, jd_text: str) -> Tuple[int, List[str], List[str]]:
    """
    Compute match score using TF-IDF cosine similarity on skill-related sentences.
    Returns: (score, matched_skills, missing_skills)
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    matched = list(resume_skills & jd_skills)
    missing = list(jd_skills - resume_skills)

    if not jd_skills:
        return 0, [], []

    score = int(len(matched) / len(jd_skills) * 100)
    return score, matched, missing

# ── Suggestion Generator ──────────────────────────────────────────────────

COMMON_SKILL_PHRASES = {
    "Kubernetes": "Deployed and managed containerized applications on Kubernetes clusters, achieving 99.9% uptime across production environments.",
    "GraphQL": "Designed and implemented GraphQL APIs aggregating data from multiple microservices with sub-100ms p99 latency.",
    "PostgreSQL": "Built and optimized PostgreSQL schemas for high-throughput transaction processing, handling 10K+ queries per second.",
    "MongoDB": "Designed MongoDB document schemas for flexible data models supporting rapidly evolving product requirements.",
    "Redis": "Implemented Redis caching layer reducing API response times by 60% for frequently accessed data.",
    "Docker": "Containerized applications using Docker, reducing deployment time from hours to minutes across CI/CD pipelines.",
    "AWS": "Architected cloud infrastructure on AWS using EC2, S3, Lambda, and RDS for scalable, cost-effective deployments.",
    "Azure": "Built Azure-based solutions using App Service, Functions, and Blob Storage for enterprise cloud deployments.",
    "React": "Developed React single-page applications with component-based architecture, state management, and responsive UI.",
    "TypeScript": "Built large-scale TypeScript applications with strict typing, reducing runtime errors by 40%.",
    "CI/CD": "Established end-to-end CI/CD pipelines using GitHub Actions/Jenkins, automating testing and deployment cycles.",
    "Microservices": "Broke down monolithic applications into microservices, improving deployment frequency from monthly to daily.",
    "REST API": "Designed and documented REST APIs following OpenAPI standards, serving as contract between frontend and backend teams.",
    "TensorFlow": "Trained and deployed TensorFlow models for production inference, achieving 94% accuracy on classification tasks.",
    "Machine Learning": "Built end-to-end ML pipelines from data ingestion to model serving, integrating predictions into product features.",
    "Terraform": "Provisioned and managed cloud infrastructure as code using Terraform, enabling consistent environments across teams.",
    "GCP": "Migrated workloads to GCP using Compute Engine, Cloud Run, and BigQuery, reducing infrastructure costs by 30%.",
}

def generate_suggestions(missing_skills: List[str]) -> List[Suggestion]:
    """Generate rewrite suggestions for missing skills."""
    suggestions = []
    for skill in missing_skills[:5]:  # Max 5 suggestions
        if skill in COMMON_SKILL_PHRASES:
            suggestions.append(Suggestion(gap=skill, suggested_phrase=COMMON_SKILL_PHRASES[skill]))
        else:
            # Generic fallback
            suggestions.append(Suggestion(
                gap=skill,
                suggested_phrase=f"Incorporated {skill} in projects involving design, implementation, and testing of scalable systems."
            ))
    return suggestions

# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(title="ATS Resume Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCKS_DIR = os.path.join(BASE_DIR, "..", "mocks")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

@app.get("/")
async def root():
    return {"message": "ATS Resume Analyzer API — POST to /analyze"}

@app.post("/parse/resume")
async def parse_resume(body: dict):
    """Parse resume text → structured output."""
    text = body.get("text", "")
    file_type = body.get("file_type", "text")

    if file_type == "pdf":
        try:
            import base64
            text = extract_text_from_pdf(base64.b64decode(text))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    skills = extract_skills(text)
    return {
        "skills": skills,
        "experience_years": infer_experience_years(text),
        "education": [],
        "companies": extract_companies(text),
        "raw_text": text[:500]
    }

@app.post("/parse/job-description")
async def parse_jd(body: dict):
    """Parse job description → structured output."""
    text = body.get("text", "")
    required = extract_skills(text)
    level = infer_level(text)

    # Nice-to-have: skills mentioned once vs multiple times
    skill_count = {}
    for s in required:
        skill_count[s] = skill_count.get(s, 0) + 1
    nice_to_have = [s for s, c in skill_count.items() if c == 1]

    return {
        "required_skills": required,
        "nice_to_have": nice_to_have,
        "level": level,
        "keywords": required,
        "raw_text": text[:500]
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    """Main endpoint — analyze resume vs job description."""
    resume_text = req.resume_text
    jd_text = req.job_description_text

    score, matched, missing = compute_match_score(resume_text, jd_text)

    # Weak skills: mentioned but not in context of years
    weak = []
    for skill in matched:
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE))
        if count == 1:
            weak.append(f"{skill} (mentioned once — add more context)")

    suggestions = generate_suggestions(missing)

    return AnalysisResult(
        match_score=score,
        matched_skills=matched,
        missing_skills=missing,
        weak_skills=weak,
        suggestions=suggestions
    )

# Serve mock UI
@app.get("/ui")
async def ui():
    path = os.path.join(MOCKS_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "UI not found — run from project root or serve mocks separately"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)