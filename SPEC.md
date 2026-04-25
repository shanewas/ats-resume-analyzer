# ATS Resume Analyzer — SPEC.md

## 1. Concept & Vision

A tool that helps job seekers understand how their resume matches a job description — by scoring keyword alignment, identifying gaps, and suggesting improvements. Built for personal use and portfolio demonstration. Not trying to replace professional CV writers — just giving engineers a data-backed view of their application fit.

**Personality:** Direct, no-nonsense, data-driven. Shows you the numbers first, explains the reasoning second. No fluff.

---

## 2. What it does

```
User pastes:
  - Job description text (or uploads PDF)
  - Resume text (or uploads PDF/DOCX)

System:
  - Parses resume → extracts: skills, experience level, education, tools/technologies
  - Parses job description → extracts: required skills, nice-to-have, experience level hints
  - Scores match: keyword overlap, skill coverage %, experience match
  - Identifies gaps: what's in the JD that you didn't mention

Output:
  - Match score (0-100%)
  - Skill gap list (what to add)
  - Keyword breakdown (matched / missing / weak)
  - Suggested rewrites for key sections
```

---

## 3. Tech Stack

- **Python 3** — core pipeline
- **FastAPI** — REST API serving layer
- **React** (or plain HTML+JS) — web UI (simple, no complex framework needed)
- **Docker** — one-command run
- **NLP:** scikit-learn TF-IDF for keyword matching + spaCy for entity extraction (skills/companies)
- **File parsing:** PyPDF2 (PDF), python-docx (DOCX)

---

## 4. Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Web UI    │────▶│  FastAPI     │────▶│  ML Pipeline │
│  (HTML/JS) │     │  /analyze    │     │  (TF-IDF +  │
└─────────────┘     │  /parse      │     │   spaCy)    │
                    └──────────────┘     └─────────────┘
                            │
                    ┌───────┴───────┐
                    ▼               ▼
              PDF/DOCX parse   Text input
```

---

## 5. Functional Requirements

### FR-1: Resume parsing
- Accept: raw text, PDF, DOCX
- Extract: name (if present), skills list, years of experience (inferred), education, work history (companies + roles)
- Output: structured JSON

### FR-2: Job description parsing
- Accept: raw text or PDF
- Extract: required skills (must-have), nice-to-have skills, experience level (senior/junior/mid), key tools mentioned
- Output: structured JSON

### FR-3: Match scoring
- TF-IDF based keyword overlap between resume and JD
- Score = (matched keywords / total JD keywords) × 100
- Normalize: skills mentioned in JD but not in resume = gaps

### FR-4: Gap analysis
- List: missing skills (high priority)
- List: weak mentions (skills mentioned vaguely vs specifically)
- Suggest: specific phrases to add to resume (actionable, not generic)

### FR-5: Simple rewrite suggestions
- For each gap: suggest a sentence/phrase you could add to your resume
- Based on the exact wording in the job description

### FR-6: File upload support
- PDF resume → text extraction
- DOCX resume → text extraction
- PDF job description → text extraction
- Raw text input (always works)

---

## 6. API Endpoints

```
POST /parse/resume
  Body: { "text": "...", "file_type": "pdf"|"docx"|"text" }
  Returns: { "skills": [...], "experience_years": N, "education": [...], "raw_text": "..." }

POST /parse/job-description
  Body: { "text": "...", "file_type": "pdf"|"text" }
  Returns: { "required_skills": [...], "nice_to_have": [...], "level": "senior|mid|junior", "keywords": [...] }

POST /analyze
  Body: { "resume_text": "...", "job_description_text": "..." }
  Returns: {
    "match_score": 73,
    "matched_skills": [...],
    "missing_skills": [...],
    "weak_skills": [...],
    "suggestions": [{ "gap": "...", "suggested_phrase": "..." }]
  }
```

---

## 7. Data Model

```python
ResumeParsed = {
    "skills": List[str],           # ["Python", "React", "Azure", ...]
    "experience_years": int,
    "education": List[str],       # ["B.Sc. CS, BRAC University"]
    "companies": List[str],
    "raw_text": str
}

JobDescriptionParsed = {
    "required_skills": List[str],
    "nice_to_have": List[str],
    "level": str,                  # "senior" | "mid" | "junior"
    "keywords": List[str],
    "raw_text": str
}

AnalysisResult = {
    "match_score": int,            # 0-100
    "matched_skills": List[str],
    "missing_skills": List[str],
    "weak_skills": List[str],      # mentioned vaguely
    "suggestions": List[{
        "gap": str,
        "suggested_phrase": str
    }]
}
```

---

## 8. Non-Goals (Out of Scope)

- User accounts / auth
- Resume hosting / storage
- Multiple file batch upload
- Cover letter analysis
- Job application automation
- Hosted SaaS (local run only for now)

---

## 9. File Structure

```
ats-resume-analyzer/
├── SPEC.md
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── main.py              # FastAPI app + CORS
│   ├── pipeline/
│   │   ├── parser.py        # PDF/DOCX/text parsing
│   │   ├── analyzer.py      # TF-IDF matching + scoring
│   │   ├── nlp.py           # spaCy skill extraction
│   │   └── suggester.py      # Rewrite suggestion logic
│   └── models.py            # Pydantic models
├── tests/
│   ├── test_parser.py
│   ├── test_analyzer.py
│   └── test_api.py
└── mocks/
    └── index.html           # Standalone UI (can be opened directly)
```

---

## 10. Success Criteria

- Can upload a PDF resume and get structured output in < 3 seconds
- Match score is meaningful (not just word count overlap)
- Gap suggestions are specific enough to be actionable
- Runs locally with one `docker-compose up`
- Code is clean, tested, and demoable to an interviewer

---

*Last updated: 2026-04-25 16:52 UTC — Initial spec*