import pytest
from src.main import extract_skills, infer_level, compute_match_score, generate_suggestions

class TestSkillExtraction:
    def test_extracts_python(self):
        text = "I have 5 years of experience with Python and Django."
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Django" in skills

    def test_extracts_cloud_skills(self):
        text = "AWS, Azure, Docker, Kubernetes, Terraform"
        skills = extract_skills(text)
        assert "AWS" in skills
        assert "Azure" in skills
        assert "Docker" in skills
        assert "Kubernetes" in skills

    def test_does_not_extract_common_words(self):
        text = "The project management system handles user authentication."
        skills = extract_skills(text)
        # management / authentication are not in skill list
        assert len(skills) == 0

    def test_case_insensitive(self):
        text = "python and AWS and react"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "AWS" in skills
        assert "React" in skills


class TestLevelInference:
    def test_senior(self):
        text = "We are looking for a senior engineer with 7+ years of Python experience."
        assert infer_level(text) == "senior"

    def test_junior(self):
        text = "Entry level role, junior developer, 1-3 years experience required."
        assert infer_level(text) == "junior"

    def test_mid(self):
        text = "Mid-level Python developer with 4+ years of experience."
        assert infer_level(text) == "mid"

    def test_default_unclear(self):
        """When no keywords present at all, default to junior."""
        text = "The role involves writing Python code."
        assert infer_level(text) == "junior"

    def test_explicit_senior_wins(self):
        text = "Lead architect with 10+ years designing systems."
        assert infer_level(text) == "senior"

    def test_mid_beats_junior(self):
        text = "Mid-level developer, 5 years experience, good Python skills."
        assert infer_level(text) == "mid"


class TestMatchScoring:
    """Tests for compute_match_score — uses set-overlap percentage."""

    def test_full_match(self):
        """Exact same skills → 100%"""
        resume = "Python Django React AWS Docker PostgreSQL Git CI/CD"
        jd = "Python Django React AWS Docker PostgreSQL Git CI/CD"
        score, matched, missing = compute_match_score(resume, jd)
        assert score == 100
        assert set(matched) == set(["Python","Django","React","AWS","Docker","PostgreSQL","Git","CI/CD"])
        assert missing == []

    def test_partial_match(self):
        """Resume has 3 of 6 JD skills → 50%"""
        resume = "Python Django React"
        jd = "Python Django React AWS Docker PostgreSQL"
        score, matched, missing = compute_match_score(resume, jd)
        assert score == 50
        assert set(matched) == set(["Python","Django","React"])
        assert set(missing) == set(["AWS","Docker","PostgreSQL"])

    def test_no_match(self):
        """No overlap → 0%"""
        resume = "Python Django"
        jd = "Rust Go Kubernetes Terraform"
        score, matched, missing = compute_match_score(resume, jd)
        assert score == 0
        assert matched == []
        assert set(missing) == set(["Rust","Go","Kubernetes","Terraform"])

    def test_empty_jd(self):
        """Empty JD → 0%"""
        resume = "Python Django React"
        jd = ""
        score, matched, missing = compute_match_score(resume, jd)
        assert score == 0
        assert matched == []
        assert missing == []

    def test_resume_has_extra_skills(self):
        """Resume has more than JD → 100% (all JD skills are covered)"""
        resume = "Python Django React AWS Docker PostgreSQL"
        jd = "Python Django React"
        score, matched, missing = compute_match_score(resume, jd)
        assert score == 100
        assert set(matched) == set(["Python","Django","React"])
        assert missing == []

    def test_identical_order_independence(self):
        """Order doesn't affect score"""
        r1, j1 = "Python React AWS", "AWS Python React"
        r2, j2 = "React AWS Python", "React AWS Python"
        s1, *_ = compute_match_score(r1, j1)
        s2, *_ = compute_match_score(r2, j2)
        assert s1 == s2


class TestSuggestions:
    def test_kubernetes_suggestion(self):
        suggestions = generate_suggestions(["Kubernetes"])
        assert len(suggestions) == 1
        assert "Kubernetes" in suggestions[0].gap
        assert len(suggestions[0].suggested_phrase) > 20

    def test_unknown_skill_suggestion(self):
        suggestions = generate_suggestions(["Cobol"])
        assert len(suggestions) == 1
        assert "Cobol" in suggestions[0].gap

    def test_max_5_suggestions(self):
        skills = ["Kubernetes", "GraphQL", "PostgreSQL", "Redis", "Kafka", "Terraform", "RabbitMQ"]
        suggestions = generate_suggestions(skills)
        assert len(suggestions) <= 5

    def test_empty_list(self):
        suggestions = generate_suggestions([])
        assert suggestions == []

    def test_aws_phrase(self):
        suggestions = generate_suggestions(["AWS"])
        assert len(suggestions) == 1
        assert "AWS" in suggestions[0].gap
        assert "AWS" in suggestions[0].suggested_phrase

    def test_multiple_suggestions_order(self):
        """First 5 skills in missing list should generate suggestions."""
        suggestions = generate_suggestions(["Docker", "AWS", "Python", "Go", "Rust", "TensorFlow"])
        assert len(suggestions) == 5
        gaps = [s.gap for s in suggestions]
        assert "Docker" in gaps
        assert "TensorFlow" not in gaps  # Should be cut off after 5
