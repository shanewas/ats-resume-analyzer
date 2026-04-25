# ATS Resume Analyzer

> See what the machine sees — before you apply.

An ML-powered tool that analyzes how well your resume matches a job description: keyword overlap score, missing skills, gap analysis, and rewrite suggestions.

```
Paste resume + job description → Get match score + actionable suggestions
```

## Quick start

### Docker (one command)

```bash
# Build and run
docker build -t ats-analyzer .
docker run -p 8000:8000 ats-analyzer

# Open the UI
open http://localhost:8000/ui
```

### Local (no Docker)

```bash
pip install -r requirements.txt
python -m src.main
# Open http://localhost:8000/ui
```

## What it does

**Input:**
- Resume: paste text or upload PDF/DOCX
- Job description: paste text or upload PDF

**Output:**
- Match score (0–100%)
- Matched skills (what you have that they want)
- Missing skills (gaps to fill)
- Weak mentions (skills mentioned once, need more depth)
- Rewrite suggestions (ready-to-use phrases for each gap)

## API

```bash
# Analyze
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "job_description_text": "..."}'

# Parse resume only
curl -X POST http://localhost:8000/parse/resume \
  -d '{"text": "Python, React, AWS...", "file_type": "text"}'
```

## Project structure

```
ats-resume-analyzer/
├── src/
│   ├── main.py      # FastAPI app + all pipeline logic
│   └── models.py    # Pydantic request/response models
├── mocks/
│   └── index.html   # Standalone web UI (open directly or serve)
├── tests/
│   └── test_pipeline.py
├── Dockerfile
└── requirements.txt
```

## Architecture

```
Resume → extract_skills() → TF-IDF vectorizer → cosine similarity → match score
                      ↓
              gap analysis → suggestion generator → rewrite phrases
```

## Tech stack

- FastAPI — REST API serving
- scikit-learn — TF-IDF keyword matching
- PyPDF2 + python-docx — file parsing
- spaCy — NLP skill extraction (optional, small model)
- Docker — one-command run

---

*Built as a portfolio project. For personal use and demonstration only.*