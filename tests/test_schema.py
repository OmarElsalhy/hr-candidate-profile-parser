import pytest

from hr_parser.schema import (
    SAMPLE_RESUME,
    build_output_parser,
    build_prompt,
    extract_json_block,
    parse_resume,
)

EXPECTED = {
    "full_name": "John Smith",
    "email": "john.smith@email.com",
    "phone": "+1 555 123 4567",
    "location": "Boston, USA",
    "github": "github.com/johnsmith",
    "linkedin": "linkedin.com/in/johnsmith",
    "education": [
        {"degree": "B.Sc. Computer Science", "institution": "MIT", "year": 2020, "grade": "3.8/4.0"}
    ],
    "skills": ["Python", "SQL", "Git", "Docker", "Machine Learning", "Data Analysis"],
    "languages": [
        {"language": "English", "level": "Native"},
        {"language": "Arabic", "level": "Fluent"},
    ],
    "experience": [
        {
            "role": "Software Engineer",
            "company": "Google",
            "years": "2020 – 2023",
            "highlights": [
                "Built internal data pipelines in Python.",
                "Cut report generation time by 40%.",
            ],
        },
        {
            "role": "Data Scientist",
            "company": "OpenAI",
            "years": "2023 – Present",
            "highlights": ["Trained and evaluated ranking models for search."],
        },
    ],
    "projects": [
        {
            "name": "Resume Parser",
            "description": "LangChain and Streamlit app",
            "highlights": [
                "Extracts structured profiles from PDF resumes.",
                "Runs fully offline on a local GPU.",
            ],
        }
    ],
}

# Without the resume text there is nothing to read projects from.
EXPECTED_WITHOUT_TEXT = {**EXPECTED, "projects": []}

GOOD = """```json
{
  "full_name": "John Smith",
  "email": "john.smith@email.com",
  "phone": "+1 555 123 4567",
  "location": "Boston, USA",
  "github": "github.com/johnsmith",
  "linkedin": "linkedin.com/in/johnsmith",
  "education": [{"degree": "B.Sc. Computer Science", "institution": "MIT", "year": 2020, "grade": "3.8/4.0"}],
  "skills": ["Python", "SQL", "Git", "Docker", "Machine Learning", "Data Analysis"],
  "languages": [
    {"language": "English", "level": "Native"},
    {"language": "Arabic", "level": "Fluent"}
  ],
  "experience": [
    {"role": "Software Engineer", "company": "Google", "years": "2020 – 2023",
     "highlights": ["Built internal data pipelines in Python.", "Cut report generation time by 40%."]},
    {"role": "Data Scientist", "company": "OpenAI", "years": "2023 – Present",
     "highlights": ["Trained and evaluated ranking models for search."]}
  ]
}
```"""


@pytest.fixture
def parser():
    return build_output_parser()


def wrap(body: str) -> str:
    return f"```json\n{body}\n```"


def test_parses_full_profile(parser):
    assert parse_resume(GOOD, parser) == EXPECTED_WITHOUT_TEXT


def test_model_only_needs_to_return_the_lean_fields(parser):
    lean = wrap(
        '{"full_name": "John Smith", "location": "Boston, USA",'
        ' "education": [{"degree": "B.Sc. Computer Science", "institution": "MIT", "year": 2020, "grade": "3.8/4.0"}],'
        ' "skills": ["Python"],'
        ' "languages": [{"language": "English", "level": "Native"}, {"language": "Arabic", "level": "Fluent"}],'
        ' "experience": [{"role": "Software Engineer", "company": "Google", "years": "2020 – 2023"},'
        '                {"role": "Data Scientist", "company": "OpenAI", "years": "2023 – Present"}]}'
    )
    assert parse_resume(lean, parser, resume_text=SAMPLE_RESUME) == EXPECTED


def test_full_profile_is_unchanged_by_verification_against_the_sample(parser):
    assert parse_resume(GOOD, parser, resume_text=SAMPLE_RESUME) == EXPECTED


def test_ignores_text_after_the_answer(parser):
    noisy = GOOD + '\n\nHere is another example:\n```json\n{ "full_name": "Mich'
    assert parse_resume(noisy, parser) == EXPECTED_WITHOUT_TEXT


def test_skips_invalid_blocks_and_uses_first_valid_one():
    broken = '```json\n{"full_name": \n```\n' + GOOD
    assert '"John Smith"' in extract_json_block(broken)


def test_works_without_a_code_fence(parser):
    bare = GOOD.replace("```json", "").replace("```", "")
    assert parse_resume(bare, parser) == EXPECTED_WITHOUT_TEXT


def test_year_given_as_string_becomes_int(parser):
    raw = GOOD.replace('"year": 2020', '"year": "2020"')
    assert parse_resume(raw, parser)["education"][0]["year"] == 2020


def test_education_always_has_a_grade_key(parser):
    raw = wrap('{"full_name": "A", "education": [{"degree": "BSc", "institution": "X", "year": 2020}]}')
    assert parse_resume(raw, parser)["education"][0]["grade"] == ""


def test_duplicate_skills_are_removed(parser):
    raw = wrap('{"full_name": "A", "skills": ["Python", "python", " SQL ", ""]}')
    assert parse_resume(raw, parser)["skills"] == ["Python", "SQL"]


def test_only_full_name_is_required(parser):
    data = parse_resume(wrap('{"full_name": "A"}'), parser)
    assert data["full_name"] == "A"
    assert data["email"] == data["phone"] == data["location"] == data["github"] == data["linkedin"] == ""
    assert data["education"] == data["skills"] == data["experience"] == data["languages"] == []
    assert data["projects"] == []


def test_missing_full_name_raises(parser):
    with pytest.raises(ValueError):
        parse_resume(wrap('{"email": "a@b.co"}'), parser)


def test_no_json_raises_value_error(parser):
    with pytest.raises(ValueError):
        parse_resume("Sorry, I can't help with that.", parser)


def test_role_with_company_suffix_is_cleaned(parser):
    raw = wrap('{"full_name": "A", "experience": [{"role": "Software Engineer at Google", "company": "Google", "years": "2020"}]}')
    job = parse_resume(raw, parser)["experience"][0]
    assert job["role"] == "Software Engineer"
    assert job["highlights"] == []


def test_highlights_string_becomes_list(parser):
    raw = wrap('{"full_name": "A", "experience": [{"role": "Dev", "company": "X", "years": "1", "highlights": "Shipped v2"}]}')
    assert parse_resume(raw, parser)["experience"][0]["highlights"] == ["Shipped v2"]


def test_languages_given_as_strings_are_split(parser):
    raw = wrap('{"full_name": "A", "languages": ["English (Native)", "Arabic"]}')
    assert parse_resume(raw, parser)["languages"] == [
        {"language": "English", "level": "Native"},
        {"language": "Arabic", "level": ""},
    ]


def test_invented_contacts_are_removed_when_resume_text_is_given(parser):
    raw = wrap('{"full_name": "A", "email": "fake@nowhere.com", "github": "github.com/ghost", "phone": "+1 999 999 9999"}')
    data = parse_resume(raw, parser, resume_text="A\nno contact details here")
    assert data["email"] == data["github"] == data["phone"] == ""


def test_missing_contacts_are_filled_from_resume_text(parser):
    raw = wrap('{"full_name": "John Smith"}')
    data = parse_resume(raw, parser, resume_text=SAMPLE_RESUME)
    assert data["email"] == "john.smith@email.com"
    assert data["phone"] == "+1 555 123 4567"
    assert data["github"] == "github.com/johnsmith"
    assert data["linkedin"] == "linkedin.com/in/johnsmith"


def test_invented_location_is_removed_but_real_one_kept(parser):
    raw = wrap('{"full_name": "A", "location": "Paris, France"}')
    assert parse_resume(raw, parser, resume_text="A lives in Cairo, Egypt")["location"] == ""
    raw = wrap('{"full_name": "A", "location": "Cairo, Egypt"}')
    assert parse_resume(raw, parser, resume_text="A lives in cairo")["location"] == "Cairo, Egypt"


def test_prompt_contains_resume_and_schema(parser):
    prompt = build_prompt(SAMPLE_RESUME, parser)
    for needle in ("John Smith", "full_name", "location", "skills", "languages", "grade"):
        assert needle in prompt
    assert "{{" not in prompt and "}}" not in prompt  # braces were escaped correctly
