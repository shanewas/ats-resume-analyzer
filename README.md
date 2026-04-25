# ATS Resume Analyzer

> See what the machine sees — before you apply.

An ML-powered tool that analyzes how well your resume matches a job description: keyword overlap score, cosine similarity, missing skills, gap analysis, and rewrite suggestions.

**Live demo:** https://187.127.110.92/ats/ui

```
Paste resume + job description → Get match score % + cosine similarity + actionable suggestions
```

---

## Features

- **Match Score** — percentage of job description skills present in your resume
- **Cosine Similarity** — TF-IDF weighted vector similarity (0.0–1.0)
- **Skill Gap Detection** — which required skills you're missing
- **Weak Mentions** — skills mentioned once that need more depth
- **Rewrite Suggestions** — ready-to-use phrases for each gap
- **File Upload** — PDF, DOCX, or TXT resume parsing
- **Soft Skill Detection** — leadership, communication, teamwork, etc.
- **Certification Detection** — AWS, Azure, PMP, JLPT, etc.
- **Section Parsing** — auto-detects experience, education, skills, summary sections
- **Soft Equivalences** — `.NET Core → .NET`, `NestJS → Node.js`, `K8s → Kubernetes`

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Python |
| ML | scikit-learn (TF-IDF, cosine similarity) |
| Parsing | PyPDF2, python-docx |
| UI | Vanilla HTML/CSS/JS (no build step) |
| Deploy | Docker + Nginx reverse proxy |

---

## Quick Start

### Docker (one command)

```bash
docker build -t ats-analyzer .
docker run -d -p 8001:8000 ats-analyzer
open http://localhost:8001/ui
```

### Or run locally

```bash
pip install -r requirements.txt
cd src && python main.py
# Open http://localhost:8000/ui
```

---

## API Endpoints

### Analyze resume vs job description

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python Django React AWS Docker PostgreSQL",
    "job_description_text": "Python Django React AWS Kubernetes Terraform PostgreSQL"
  }'
```

**Response:**

```json
{
  "match_score": 63,
  "cosine_similarity": 0.5891,
  "matched_skills": ["AWS", "Django", "PostgreSQL", "Python", "React"],
  "missing_skills": ["Docker", "Kubernetes", "Terraform"],
  "weak_skills": ["Python (mentioned once — add depth/years)"],
  "suggestions": [
    {
      "gap": "Kubernetes",
      "suggested_phrase": "Deployed and managed containerized applications on Kubernetes clusters, achieving 99.9% uptime across production environments."
    }
  ]
}
```

### Upload and parse a resume file

```bash
curl -X POST http://localhost:8001/upload/resume \
  -F "file=@your_resume.pdf"
```

Returns: `{ name, skills, skill_weights, experience_years, education, soft_skills, certifications }`

---

## Screenshots

### Clean UI — before analysis

![ATS Resume Analyzer — clean state](screenshots/ui-clean.png)

### Results — after analysis

![ATS Resume Analyzer — results](screenshots/results.png)

---

## Project Structure

```
ats-resume-analyzer/
├── src/
│   ├── main.py          # FastAPI app + all pipeline logic
│   └── models.py        # Pydantic request/response models
├── mocks/
│   └── index.html       # Standalone web UI (open directly or serve)
├── tests/
│   └── test_pipeline.py # 22 unit tests, all passing
├── screenshots/
│   ├── ui-clean.png    # Clean UI state
│   └── results.png      # After analysis
├── Dockerfile
├── requirements.txt
├── SPEC.md              # Full design specification
└── README.md
```

---

## Architecture

```
Resume Text
    │
    ▼
extract_skills_with_weight()
    │  ← section-aware weights (skills section = 1.5×)
    ▼
apply_soft_matches()     ← .NET Core → .NET, K8s → Kubernetes, etc.
    │
    ▼
compute_weighted_score()
    │  ← TF-IDF cosine similarity (0.0-1.0)
    │  ← set-overlap match percentage (0-100%)
    ▼
Output: { match_score, cosine_similarity, matched, missing, weak, suggestions }
```

---

## Key Algorithms

**Weighted skill extraction** — skills mentioned in the Skills section get 1.5× weight, Experience section 1.2×, Education 0.5×. Mentions 2+ times get 1.3× frequency multiplier.

**Soft skill equivalences** — maps related technologies so `.NET Core` matches `.NET`, `NestJS` matches `Node.js`, `K8s` matches `Kubernetes`.

**TF-IDF cosine similarity** — builds weighted skill vectors for both resume and JD, computes cosine similarity as a separate metric from the set-overlap percentage.

---

## Running Tests

```bash
PYTHONPATH=src python -m pytest tests/test_pipeline.py -v
```

**22 tests passing** ✅

---

*For personal use and portfolio demonstration. Not affiliated with any ATS vendor.*