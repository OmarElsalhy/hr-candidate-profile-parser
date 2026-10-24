"""Output schema, prompt and response parsing for the resume parser."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import PromptTemplate

from hr_parser.contacts import enrich_contacts
from hr_parser.sections import clean_resume_text, enrich_sections

try:  # LangChain >= 1.0
    from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:  # pragma: no cover - older LangChain releases
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser

SAMPLE_RESUME = """John Smith
Email: john.smith@email.com | Phone: +1 555 123 4567
Location: Boston, USA
GitHub: github.com/johnsmith | LinkedIn: linkedin.com/in/johnsmith
Education
B.Sc. Computer Science, MIT, 2020 (GPA: 3.8/4.0)
Technical Skills
Languages: Python, SQL
Developer Tools: Git, Docker
Skills: Machine Learning, Data Analysis
Languages
English (Native), Arabic (Fluent)
Experience
Software Engineer — Google 2020 – 2023
● Built internal data pipelines in Python.
● Cut report generation time by 40%.
Data Scientist — OpenAI 2023 – Present
● Trained and evaluated ranking models for search.
Projects
Resume Parser — LangChain and Streamlit app
● Extracts structured profiles from PDF resumes.
● Runs fully offline on a local GPU."""

PROMPT_TEMPLATE = """You are a smart assistant that extracts structured candidate information from a resume.

Extract the candidate's name, location, education, skills, spoken languages and work experience.

Respond ONLY in JSON format as follows:
{format_instructions}

Rules:
- Use "" for a missing text field and [] for a missing list. Never invent information.
- "skills" must contain ALL technical skills from every skills category (programming languages, frameworks and libraries, tools, and any "Skills" line) as one flat list of single items.
- "languages" are spoken languages such as English. A "Languages:" line inside Technical Skills lists programming languages: those go in "skills".
- "grade" is the GPA or grade exactly as written, or "".
- "role" is the job title only, without the company name. Do not copy bullet points.

Example Input:
"Jane Doe
Location: London, UK
Education
B.Sc. Physics, Stanford, 2019 (GPA 3.9/4.0)
Technical Skills
Languages: Python, SQL
Tools: Git, Excel
Skills: Statistics, Data Visualization
Languages
English (Native)
Experience
Data Analyst — Meta Jan 2019 – Mar 2022
- Built dashboards used by 3 product teams."

Example Output:
```json
{{
  "full_name": "Jane Doe",
  "location": "London, UK",
  "education": [{{"degree": "B.Sc. Physics", "institution": "Stanford", "year": 2019, "grade": "3.9/4.0"}}],
  "skills": ["Python", "SQL", "Git", "Excel", "Statistics", "Data Visualization"],
  "languages": [{{"language": "English", "level": "Native"}}],
  "experience": [{{"role": "Data Analyst", "company": "Meta", "years": "Jan 2019 – Mar 2022"}}]
}}
```

Now extract from the following input:
"{resume_text}"
"""

STRING_FIELDS = ("full_name", "email", "phone", "location", "github", "linkedin")
LIST_FIELDS = ("education", "skills", "experience", "languages", "projects")
REQUIRED_FIELDS = ("full_name",)


def build_output_parser() -> StructuredOutputParser:
    """Create the LangChain parser for the fields the model has to generate.

    Email, phone, links, job bullets and projects are read from the resume text by
    code (see ``contacts.py`` and ``sections.py``), so the model doesn't have to write
    them out. Less to generate means faster answers and no truncated JSON.
    """
    schemas = [
        ResponseSchema(name="full_name", description="The candidate's full name as a string."),
        ResponseSchema(
            name="location",
            description="The candidate's city and country as written in the resume, or '' if missing.",
        ),
        ResponseSchema(
            name="education",
            description=(
                "A list of education entries. Each entry is an object with the keys "
                "'degree' (string), 'institution' (string), 'year' (integer) and "
                "'grade' (string: the GPA or grade as written, '' if missing)."
            ),
        ),
        ResponseSchema(
            name="skills",
            description=(
                "A flat list of strings with ALL technical skills from every skills "
                "category: programming languages, frameworks and libraries, tools and skills."
            ),
        ),
        ResponseSchema(
            name="languages",
            description=(
                "A list of spoken languages. Each entry is an object with the keys "
                "'language' (string) and 'level' (string, '' if not stated)."
            ),
        ),
        ResponseSchema(
            name="experience",
            description=(
                "A list of work experience entries. Each entry is an object with the keys "
                "'role' (string, job title only), 'company' (string) and 'years' (string, "
                "e.g. '2020-2023' or '2023-Present'). Do not include bullet points."
            ),
        ),
    ]
    return StructuredOutputParser.from_response_schemas(schemas)


def build_prompt(resume_text: str, parser: StructuredOutputParser) -> str:
    """Fill the prompt template with the resume and the parser's format instructions."""
    template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["resume_text", "format_instructions"],
    )
    return template.format(
        resume_text=clean_resume_text(resume_text),
        format_instructions=parser.get_format_instructions(),
    )


_JSON_FENCE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _extract_json_payload(text: str) -> str | None:
    """Return the first valid JSON object found in ``text`` (fenced or bare)."""
    for candidate in _JSON_FENCE.findall(text):
        if _is_valid_json(candidate):
            return candidate

    bare = _BARE_OBJECT.search(text)
    if bare and _is_valid_json(bare.group(0)):
        return bare.group(0)
    return None


def extract_json_block(text: str) -> str | None:
    """Return the first valid JSON object in ``text`` as a fenced ```json block.

    Small models sometimes keep generating after the answer or forget the fence, so
    we try fenced blocks first (skipping invalid ones) and fall back to a bare object.
    """
    payload = _extract_json_payload(text)
    return None if payload is None else f"```json\n{payload}\n```"


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def _clean_education(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {"degree": entry}
    year = entry.get("year")
    if isinstance(year, str) and year.strip().isdigit():
        year = int(year.strip())
    elif year is None:
        year = ""
    return {
        "degree": _as_text(entry.get("degree")),
        "institution": _as_text(entry.get("institution")),
        "year": year,
        "grade": _as_text(entry.get("grade") or entry.get("gpa")),
    }


def _clean_experience(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {"role": entry}

    role = _as_text(entry.get("role"))
    company = _as_text(entry.get("company"))
    suffix = f" at {company}".lower()
    if company and role.lower().endswith(suffix):  # "Engineer at Google" -> "Engineer"
        role = role[: -len(suffix)].strip()

    highlights = entry.get("highlights")
    if highlights is None:
        highlights = []
    elif not isinstance(highlights, list):
        highlights = [highlights]
    highlights = [h for h in (_as_text(h) for h in highlights) if h]

    return {
        "role": role,
        "company": company,
        "years": _as_text(entry.get("years")),
        "highlights": highlights,
    }


def _clean_project(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {"name": entry}
    highlights = entry.get("highlights")
    if highlights is None:
        highlights = []
    elif not isinstance(highlights, list):
        highlights = [highlights]
    return {
        "name": _as_text(entry.get("name")),
        "description": _as_text(entry.get("description")),
        "highlights": [h for h in (_as_text(h) for h in highlights) if h],
    }


def _clean_language(entry: Any) -> dict[str, str] | None:
    if isinstance(entry, dict):
        language = _as_text(entry.get("language") or entry.get("name"))
        level = _as_text(entry.get("level"))
    else:
        language, level = _as_text(entry), ""
        match = re.fullmatch(r"(.*?)\s*[(\-:]\s*([^()]*)\)?", language)  # "English (Native)"
        if match and match.group(1):
            language, level = match.group(1).strip(), match.group(2).strip()
    return {"language": language, "level": level} if language else None


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Make small, safe fixes so the output always matches the documented schema."""
    data = dict(data)

    for key in STRING_FIELDS:
        data[key] = _as_text(data.get(key))

    for key in LIST_FIELDS:
        value = data.get(key)
        if value is None:
            data[key] = []
        elif not isinstance(value, list):
            data[key] = [value]

    data["education"] = [_clean_education(e) for e in data["education"]]
    data["skills"] = _dedupe([s for s in (_as_text(s) for s in data["skills"]) if s])
    data["experience"] = [_clean_experience(e) for e in data["experience"]]
    data["projects"] = [_clean_project(e) for e in data["projects"]]
    languages = (_clean_language(e) for e in data["languages"])
    data["languages"] = [lang for lang in languages if lang]
    return data


def parse_resume(
    raw_response: str,
    parser: StructuredOutputParser,
    resume_text: str | None = None,
) -> dict[str, Any]:
    """Turn the raw model output into a validated candidate profile dict.

    Missing optional fields get empty defaults. If ``resume_text`` is given, the result
    is checked and completed against it: contact details are read from it, location
    is verified, skills from every skills category are added, invented grades and
    languages are dropped, and job highlights and projects are taken from the bullets
    in the resume.

    Raises ``ValueError`` if no JSON is found or ``full_name`` is missing.
    """
    payload = _extract_json_payload(raw_response)
    if payload is None:
        raise ValueError("The model did not return a valid JSON object.")

    data = json.loads(payload)
    if not isinstance(data, dict) or not all(key in data for key in REQUIRED_FIELDS):
        raise ValueError("The model output is missing the 'full_name' field.")

    for key in STRING_FIELDS:
        data.setdefault(key, "")
    for key in LIST_FIELDS:
        data.setdefault(key, [])

    parsed = normalize(parser.parse(f"```json\n{json.dumps(data)}\n```"))
    if resume_text:
        text = clean_resume_text(resume_text)
        parsed = enrich_contacts(parsed, text)
        parsed = enrich_sections(parsed, text)
    return parsed