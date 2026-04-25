"""
ATS Resume Analyzer — FastAPI server
Advanced pipeline: section parsing → weighted TF-IDF → cosine similarity → gap analysis → suggestions
"""

import io
import re
import os
from typing import List, Tuple, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from models import AnalyzeRequest, AnalysisResult, ResumeParsed, JobDescriptionParsed, Suggestion

# ── File Parsing ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {e}")

def extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise ValueError(f"DOCX parsing failed: {e}")

# ── Resume Section Parser ─────────────────────────────────────────────────

SECTION_HEADERS = {
    "experience": ["experience", "work history", "employment", "professional experience", "career history", "職歴", "工作经验"],
    "education": ["education", "academic", "qualifications", "학력", "教育背景"],
    "skills": ["skills", "technical skills", "technologies", "technical proficiencies", "competencies", "技能"],
    "summary": ["summary", "profile", "objective", "about", "professional summary", "自己PR", "概要"],
    "projects": ["projects", "personal projects", "open source", "portfolio", "プロジェクト"],
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "mentoring", "problem-solving",
    "collaboration", "agile", "scrum", "cross-functional", "stakeholder management",
    "presentation", "time management", "critical thinking",
}

CERTIFICATIONS = {
    "AWS Certified", "Azure Certified", "Google Cloud Certified",
    "PMP", "CISSP", "CISA", "CPA", "JLPT", "N1", "N2", "N3",
    "Microsoft Certified", "Oracle Certified", "Cisco Certified",
}

def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into sections.
    Returns: {section_name: section_content}
    """
    lines = text.split("\n")
    sections = {}
    current_section = "header"
    current_content = []

    for line in lines:
        stripped = line.strip()
        # Detect section header
        matched = False
        for section, headers in SECTION_HEADERS.items():
            for header in headers:
                # Header pattern: line starts with header keyword, possibly followed by colon or #
                if stripped.lower().startswith(header) or re.match(rf"^{re.escape(header)}\s*[:\-#]", stripped.lower()):
                    if current_content:
                        sections[current_section] = "\n".join(current_content)
                    current_section = section
                    current_content = []
                    matched = True
                    break
            if matched:
                break
        if not matched:
            current_content.append(stripped)

    if current_content:
        sections[current_section] = "\n".join(current_content)

    return sections

def extract_name(text: str) -> Optional[str]:
    """Extract name — first line that's 2-4 capitalized words, no special chars."""
    lines = text.split("\n")
    for line in lines[:10]:
        line = line.strip()
        if len(line) < 3 or len(line) > 60:
            continue
        # Skip lines with emails, phones, URLs
        if re.search(r'[\u4e00-\u9fff]', line):  # Japanese chars — skip
            continue
        if re.search(r'[<>@]|(http|www)', line.lower()):
            continue
        # Match: 2-4 words, each capitalized, no digits or special chars
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            if all(re.match(r'^[A-Z][a-z]*\.?$', w) for w in words):
                return line
    return None

# ── Skill Extraction ────────────────────────────────────────────────────────

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
    "Computer Vision",
    # APIs / Architecture
    "REST", "GraphQL", "gRPC", "OpenAPI", "Swagger",
    "Microservices", "API", "WebAPI",
    # PDFs / Documents
    "PDF", "PDF/A", "iText", "PDFium", "LibreOffice",
    # Others
    "Git", "CI/CD", "TDD", "Agile", "Scrum",
    "Figma", "Miro",
    "Azure Functions", "Lambda", "SQS", "SNS", "RDS", "S3", "Blob",
    "Playwright", "Selenium", "Jest", "xUnit", "Pytest",
}

# Priority tiers — required skills get higher weight
REQUIRED_WEIGHT = 2.0
NICE_TO_HAVE_WEIGHT = 0.5

SENIOR_KEYWORDS = {"lead", "architect", "senior", "principal", "staff", "manager", "mentor", "10+", "10 years"}
MID_KEYWORDS = {"mid", "5+", "5 years", "intermediate", "experienced"}
JUNIOR_KEYWORDS = {"junior", "entry", "graduate", "intern", "1-3", "1 year", "2 year", "3 year"}

def extract_skills_with_weight(text: str) -> Dict[str, float]:
    """Extract skills + weight based on context (section, frequency)."""
    found = {}
    text_lower = text.lower()

    for skill in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count == 0:
            continue

        # Weight multiplier by section
        section_weights = {
            "skills": 1.5,      # Skills section = strong signal
            "experience": 1.2,  # Experience = relevant context
            "projects": 1.2,
            "header": 0.8,
            "summary": 1.0,
            "education": 0.5,
        }

        base_weight = 1.0
        for section, weight in section_weights.items():
            if section in text_lower:
                base_weight = max(base_weight, weight)

        # Frequency multiplier (mentioned 2+ times = deeper knowledge signal)
        freq_mult = 1.0 if count == 1 else 1.3

        found[skill] = base_weight * freq_mult

    return found

def extract_skills(text: str) -> List[str]:
    """Simple skill list — for backward compat."""
    return list(extract_skills_with_weight(text).keys())

def extract_soft_skills(text: str) -> List[str]:
    """Extract soft skills."""
    found = []
    for skill in SOFT_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found.append(skill.title())
    return found

def extract_certifications(text: str) -> List[str]:
    """Extract professional certifications."""
    found = []
    for cert in CERTIFICATIONS:
        if re.search(r'\b' + re.escape(cert) + r'\b', text, re.IGNORECASE):
            found.append(cert)
    return found

def infer_experience_years(text: str) -> int:
    """Infer years of experience from text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'experience[^\n]*(\d+)\s*years?',
        r'(\d{4})\s*-\s*(?:present|current)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            if 1970 < num < 2030:
                return 2026 - num
            if num < 50:
                return num
    return 7

def infer_level(text: str) -> str:
    """Infer job level from JD text."""
    text_lower = text.lower()
    senior_count = sum(1 for kw in SENIOR_KEYWORDS if kw in text_lower)
    mid_count = sum(1 for kw in MID_KEYWORDS if kw in text_lower)
    junior_count = sum(1 for kw in JUNIOR_KEYWORDS if kw in text_lower)

    if senior_count > mid_count and senior_count > junior_count:
        return "senior"
    elif mid_count > junior_count:
        return "mid"
    return "junior"

# ── Weighted TF-IDF Scoring ──────────────────────────────────────────────

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_weighted_score(
    resume_skills: Dict[str, float],
    jd_skills: Dict[str, float]
) -> Tuple[int, List[str], List[str], List[str], float]:
    """
    Compute weighted match score using TF-IDF cosine similarity.
    Skills from resume/JD are used to build TF-IDF vectors.
    Returns: (score, matched, missing, weak, cosine_sim)
    """
    all_skills = sorted(set(resume_skills.keys()) | set(jd_skills.keys()))

    if not all_skills:
        return 0, [], [], [], 0.0

    # Build weighted vectors
    resume_vec = np.array([resume_skills.get(s, 0) for s in all_skills])
    jd_vec = np.array([jd_skills.get(s, 0) for s in all_skills])

    # Cosine similarity
    resume_norm = resume_vec / (np.linalg.norm(resume_vec) + 1e-9)
    jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
    cos_sim = float(np.dot(resume_norm, jd_norm))

    # Convert to 0-100 score
    score = int(cos_sim * 100)

    matched = [s for s in all_skills if resume_skills.get(s, 0) > 0 and jd_skills.get(s, 0) > 0]
    missing_jd = [s for s in all_skills if jd_skills.get(s, 0) > 0 and resume_skills.get(s, 0) == 0]
    missing = missing_jd

    # Weak = matched but mentioned only once in resume
    weak = []
    for s in matched:
        if resume_skills.get(s, 0) <= 1.0:  # base weight only = mentioned once
            weak.append(f"{s} (mentioned once — add depth/years)")

    return score, matched, missing, weak, cos_sim

def compute_match_score(resume_text: str, jd_text: str) -> Tuple[int, List[str], List[str]]:
    """
    Compute simple set-overlap match score.
    Score = (JD skills found in resume) / (total JD skills) × 100.
    Uses weighted extraction + soft equivalents.
    """
    resume_skills = extract_skills_with_weight(resume_text)
    jd_skills = extract_skills_with_weight(jd_text)

    resume_exp, jd_exp = apply_soft_matches(resume_skills, jd_skills)


    matched = [s for s in jd_exp if resume_exp.get(s, 0) > 0]
    missing = [s for s in jd_exp if resume_exp.get(s, 0) == 0]

    total_jd = len(jd_exp)
    if total_jd == 0:
        return 0, [], []
    score = int(len(matched) / total_jd * 100)
    return score, matched, missing

# ── Soft Match (fuzzy skill mapping) ─────────────────────────────────────

SOFT_EQUIVALENCES = {
    ".NET Core": ".NET",
    ".NET": ".NET Core",
    "C#": ".NET",
    "Blazor": ".NET",
    "NestJS": "Node.js",
    "Next.js": "React",
    "Vue.js": "Vue",
    "FastAPI": "Python",
    "Flask": "Python",
    "Django": "Python",
    "K8s": "Kubernetes",
    "GKE": "Kubernetes",
    "EKS": "Kubernetes",
    "Azure Functions": "Azure",
    "AWS Lambda": "AWS",
    "Google Cloud": "GCP",
    "Google Cloud Platform": "GCP",
    "Postgres": "PostgreSQL",
    "Mongo": "MongoDB",
    "Redis Cache": "Redis",
    "Selenium": "Playwright",
    "Circle CI": "CircleCI",
    "GitLab": "GitLab CI",
    "Machine Learning": "ML",
    "Deep Learning": "ML",
    "TF": "TensorFlow",
    "PyTorch": "TensorFlow",
    "scikit-learn": "Machine Learning",
    "Linux": "UNIX",
    "Shell": "Bash",
}

def apply_soft_matches(resume_skills: Dict[str, float], jd_skills: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Expand resume skills with soft equivalents so .NET Core matches .NET, etc.
    Returns modified copies.
    """
    resume_exp = dict(resume_skills)
    jd_exp = dict(jd_skills)

    for jd_skill, equivalent in SOFT_EQUIVALENCES.items():
        if jd_skill in jd_exp and equivalent not in jd_exp:
            # Add equivalent to JD at half weight
            jd_exp[equivalent] = jd_exp.get(jd_skill, 1.0) * 0.5
        if equivalent in jd_exp and jd_skill not in jd_exp:
            # Reverse: if JD has equivalent, count original as half
            resume_exp[jd_skill] = resume_skills.get(jd_skill, 0) + resume_skills.get(equivalent, 0) * 0.5

    return resume_exp, jd_exp

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
    "REST": "Designed and documented REST APIs following OpenAPI standards, serving as contract between frontend and backend teams.",
    "TensorFlow": "Trained and deployed TensorFlow models for production inference, achieving 94% accuracy on classification tasks.",
    "Machine Learning": "Built end-to-end ML pipelines from data ingestion to model serving, integrating predictions into product features.",
    "Terraform": "Provisioned and managed cloud infrastructure as code using Terraform, enabling consistent environments across teams.",
    "GCP": "Migrated workloads to GCP using Compute Engine, Cloud Run, and BigQuery, reducing infrastructure costs by 30%.",
    "Python": "Built production Python systems using FastAPI/Django, with comprehensive testing and CI/CD integration.",
    "C#": "Developed enterprise C# applications on .NET Core, shipping high-availability services to production.",
    "Docker": "Containerized applications using Docker Compose and Kubernetes, achieving zero-downtime deployments.",
    "Blazor": "Built Blazor WebAssembly applications with real-time data binding and component-based architecture.",
    "Node.js": "Developed backend services with Node.js and Express, handling high-concurrency request workloads.",
    "NestJS": "Architected NestJS backend platforms with dependency injection, GraphQL, and PostgreSQL integration.",
    "SQL": "Wrote complex SQL queries, stored procedures, and optimized database schemas for high-throughput workloads.",
    "NoSQL": "Designed non-relational data models in MongoDB/Cassandra for flexible, high-scale document storage.",
    "Playwright": "Implemented end-to-end test suites with Playwright, achieving 95% UI test coverage across critical paths.",
    "Selenium": "Built Selenium automation frameworks for regression testing, reducing QA cycle time by 70%.",
    "Linux": "Administered RHEL/Ubuntu servers, configuring systemd services, firewalls, and automation scripts.",
    "Bash": "Wrote production Bash scripts for CI/CD pipelines, deployment automation, and system monitoring.",
    "Git": "Managed Git workflows with branching strategies, code reviews, and merge policies across multi-team projects.",
    "GitHub Actions": "Configured GitHub Actions CI/CD pipelines with matrix builds, artifact publishing, and deployment gates.",
    "Jenkins": "Built Jenkins pipelines with Groovy scripts for automated testing, building, and staging deployments.",
    "Agile": "Practiced Agile/Scrum methodologies, participating in sprint planning, retrospectives, and daily standups.",
}

def generate_suggestions(missing_skills: List[str]) -> List[Suggestion]:
    """Generate rewrite suggestions for missing skills."""
    suggestions = []
    for skill in missing_skills[:5]:
        if skill in COMMON_SKILL_PHRASES:
            suggestions.append(Suggestion(gap=skill, suggested_phrase=COMMON_SKILL_PHRASES[skill]))
        else:
            suggestions.append(Suggestion(
                gap=skill,
                suggested_phrase=f"Incorporated {skill} in projects involving design, implementation, and testing of scalable systems."
            ))
    return suggestions

# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(title="ATS Resume Analyzer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCKS_DIR = os.path.join(BASE_DIR, "..", "mocks")

@app.get("/")
async def root():
    return {
        "name": "ATS Resume Analyzer",
        "version": "2.0.0",
        "endpoints": ["/analyze", "/parse/resume", "/parse/job-description", "/upload/resume", "/sections"]
    }

@app.post("/parse/resume")
async def parse_resume(body: dict):
    """Parse resume text → structured output with weighted skills."""
    text = body.get("text", "")
    file_type = body.get("file_type", "text")

    if file_type == "pdf":
        try:
            import base64
            text = extract_text_from_pdf(base64.b64decode(text))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    weighted_skills = extract_skills_with_weight(text)
    sections = parse_resume_sections(text)

    return {
        "name": extract_name(text),
        "skills": list(weighted_skills.keys()),
        "skill_weights": {k: round(v, 2) for k, v in weighted_skills.items()},
        "experience_years": infer_experience_years(text),
        "education": sections.get("education", "")[:200],
        "soft_skills": extract_soft_skills(text),
        "certifications": extract_certifications(text),
        "sections_found": list(sections.keys()),
    }

@app.post("/parse/job-description")
async def parse_jd(body: dict):
    """Parse job description → structured output."""
    text = body.get("text", "")
    weighted_skills = extract_skills_with_weight(text)
    level = infer_level(text)

    # Nice-to-have: skills mentioned once vs multiple times
    skill_count = {}
    for s in weighted_skills:
        skill_count[s] = skill_count.get(s, 0) + 1
    nice_to_have = [s for s, c in skill_count.items() if c == 1]

    return {
        "required_skills": list(weighted_skills.keys()),
        "skill_weights": {k: round(v, 2) for k, v in weighted_skills.items()},
        "nice_to_have": nice_to_have,
        "level": level,
    }

@app.post("/sections")
async def extract_sections(body: dict):
    """Extract resume sections."""
    text = body.get("text", "")
    sections = parse_resume_sections(text)
    return {k: v[:500] for k, v in sections.items()}  # truncate for readability

@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse a resume file (PDF or DOCX)."""
    content = await file.read()
    fname = file.filename.lower()

    try:
        if fname.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif fname.endswith(".docx") or fname.endswith(".doc"):
            text = extract_text_from_docx(content)
        elif fname.endswith(".txt"):
            text = content.decode("utf-8", errors="replace")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    weighted_skills = extract_skills_with_weight(text)
    sections = parse_resume_sections(text)

    return {
        "filename": file.filename,
        "name": extract_name(text),
        "skills": list(weighted_skills.keys()),
        "skill_weights": {k: round(v, 2) for k, v in weighted_skills.items()},
        "experience_years": infer_experience_years(text),
        "education": sections.get("education", "")[:200],
        "soft_skills": extract_soft_skills(text),
        "certifications": extract_certifications(text),
        "sections_found": list(sections.keys()),
        "raw_text_preview": text[:300],
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    """Main endpoint — advanced analysis with weighted TF-IDF + soft matching."""
    resume_text = req.resume_text
    jd_text = req.job_description_text

    # Weighted skill extraction
    resume_skills = extract_skills_with_weight(resume_text)
    jd_skills = extract_skills_with_weight(jd_text)

    # Apply soft equivalents (.NET Core → .NET, etc.)
    resume_exp, jd_exp = apply_soft_matches(resume_skills, jd_skills)

    # Compute weighted score
    score, matched, missing, weak, cos_sim = compute_weighted_score(resume_exp, jd_exp)

    # Also run unweighted for comparison
    raw_score, raw_matched, raw_missing = compute_match_score(resume_text, jd_text)

    suggestions = generate_suggestions(missing)

    return AnalysisResult(
        match_score=score,
        matched_skills=matched,
        missing_skills=missing,
        weak_skills=weak,
        suggestions=suggestions
    )

@app.get("/ui")
async def ui():
    path = os.path.join(MOCKS_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "UI not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
