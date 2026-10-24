import pytest

from hr_parser.schema import build_output_parser, parse_resume
from hr_parser.sections import (
    clean_resume_text,
    extract_projects,
    extract_skill_items,
    verify_grade,
)

# Text as a PDF reader typically returns it: lone bullet glyphs, wrapped lines.
PDF_TEXT = """Jane Roe
Technical Skills
Languages: Python, Java, C++, JavaScript, HTML/CSS, SQL
Frameworks & Libraries: TensorFlow, Keras, OpenCV
Developer Tools: Git, Docker
Skills: Machine Learning Algorithms, NLP, Swagger/OpenAPI Documentation
Education
B.Sc. Computer Science, Cairo University, 2024 (GPA 3.5/4.0)
Experience
ML Intern — Acme Labs, Cairo, Egypt May 2024 – Nov. 2024
●
Cleaned and prepared data for NLP projects.
●
Implemented algorithms using TensorFlow and OpenCV to
automate data-driven insights.
Data Analyst — Beta Corp 2023
●
Built reports.
Projects
Resume Parser — LangChain app
●
Extracts profiles from PDFs.
Smart Mirror | Virtual try-on
●
Runs on a local GPU.
●
Uses MediaPipe for pose
tracking.
Certifications
●
Not a project bullet.
"""

ALL_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "HTML/CSS", "SQL",
    "TensorFlow", "Keras", "OpenCV", "Git", "Docker",
    "Machine Learning Algorithms", "NLP", "Swagger/OpenAPI Documentation",
]


def wrap(body: str) -> str:
    return f"```json\n{body}\n```"


def test_lone_bullet_glyphs_are_joined_with_their_text():
    cleaned = clean_resume_text("Intern — X\n●\nDid A.\n● Did B.\n•Did C.")
    assert cleaned.splitlines() == ["Intern — X", "- Did A.", "- Did B.", "- Did C."]


def test_cleaning_is_idempotent():
    once = clean_resume_text(PDF_TEXT)
    assert clean_resume_text(once) == once


def test_skills_from_every_category_are_extracted():
    assert extract_skill_items(clean_resume_text(PDF_TEXT)) == ALL_SKILLS


def test_wrapped_skill_lines_are_joined():
    text = "Skills\nPython, Java,\nSQL, Git"
    assert extract_skill_items(text) == ["Python", "Java", "SQL", "Git"]


@pytest.mark.parametrize(
    "grade, expected",
    [("3.5/4.0", "3.5/4.0"), ("3.9/4.0", ""), ("Very Good", "")],
)
def test_grade_must_appear_in_the_resume(grade, expected):
    assert verify_grade(grade, PDF_TEXT) == expected


def test_grade_words_are_kept_when_written():
    assert verify_grade("Very Good", "Grade: Very Good") == "Very Good"


def test_model_that_only_returns_the_first_line_is_completed_from_the_resume():
    parser = build_output_parser()
    raw = wrap(
        '{"full_name": "Jane Roe", "skills": ["Python", "Invented Skill"],'
        ' "languages": [{"language": "Python", "level": ""}, {"language": "Arabic", "level": "Native"}],'
        ' "education": [{"degree": "B.Sc. Computer Science", "institution": "Cairo University", "year": 2024, "grade": "3.9/4.0"}],'
        ' "experience": ['
        '   {"role": "ML Intern", "company": "Acme Labs, Cairo, Egypt", "years": "May 2024 – Nov. 2024", "highlights": []},'
        '   {"role": "Data Analyst", "company": "Beta Corp", "years": "2023", "highlights": []}]}'
    )
    data = parse_resume(raw, parser, resume_text=PDF_TEXT)

    assert data["skills"] == ALL_SKILLS  # all categories, invented skill dropped
    assert data["languages"] == []  # Python is a skill; Arabic isn't in the resume
    assert data["education"][0]["grade"] == ""  # 3.9 is not in the resume
    assert data["experience"][0]["highlights"] == [
        "Cleaned and prepared data for NLP projects.",
        "Implemented algorithms using TensorFlow and OpenCV to automate data-driven insights.",
    ]
    assert data["experience"][1]["highlights"] == ["Built reports."]  # not the Projects bullet


def test_projects_are_read_with_name_description_and_bullets():
    assert extract_projects(clean_resume_text(PDF_TEXT)) == [
        {
            "name": "Resume Parser",
            "description": "LangChain app",
            "highlights": ["Extracts profiles from PDFs."],
        },
        {
            "name": "Smart Mirror",
            "description": "Virtual try-on",
            "highlights": ["Runs on a local GPU.", "Uses MediaPipe for pose tracking."],
        },
    ]


def test_projects_without_bullets_are_separate_entries():
    text = "Projects\nSmart Mirror — virtual try-on app\nFace Detector — CNN for masks\nEducation\nBSc"
    assert [p["name"] for p in extract_projects(text)] == ["Smart Mirror", "Face Detector"]


def test_project_heading_variants_and_subtitle_lines():
    text = "Academic Projects\nChat Bot\nTech: Python, Flask\n- Answers HR questions."
    assert extract_projects(text) == [
        {"name": "Chat Bot", "description": "Tech: Python, Flask", "highlights": ["Answers HR questions."]}
    ]


def test_a_skills_line_without_a_heading_is_found():
    text = "Education: B.Sc. CS, MIT, 2020\nSkills: Python, Machine Learning, Data Analysis\nExperience:"
    assert extract_skill_items(text) == ["Python", "Machine Learning", "Data Analysis"]


def test_model_highlights_are_kept_when_the_job_cannot_be_located():
    parser = build_output_parser()
    raw = wrap(
        '{"full_name": "A", "experience": [{"role": "Pilot", "company": "Sky", "years": "1", "highlights": ["Flew planes"]}]}'
    )
    data = parse_resume(raw, parser, resume_text=PDF_TEXT)
    assert data["experience"][0]["highlights"] == ["Flew planes"]
