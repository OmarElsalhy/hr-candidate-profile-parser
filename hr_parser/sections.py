"""Deterministic reading of the resume text.

A small model is good at the parts that need understanding (name, education, job titles)
but slow and unreliable at copying long lists. This module reads those parts straight
from the text instead: PDF bullets, every skills category, job bullets, projects, and
it drops values (grades, languages, skills) that the resume does not contain.
"""

from __future__ import annotations

import re
from typing import Any

BULLET_GLYPHS = set("●•▪■◦○‣∙·\uf0b7\uf0a7\uf076")

SECTION_HEADING_RE = re.compile(
    r"^(?:summary|profile|objective|education|experience|work experience|"
    r"professional experience|work history|employment|internships?|"
    r"certifications?|certificates|courses|training|awards?|honou?rs|achievements|"
    r"awards?\s*(?:&|and)\s*(?:certifications?|honou?rs)|"
    r"certifications?\s*(?:&|and)\s*(?:awards?|courses)|"
    r"publications|languages|interests|volunteer(?:ing)?(?:\s+(?:work|experience))?|"
    r"extracurricular activities|activities|references)\s*:?$",
    re.IGNORECASE,
)
PROJECTS_HEADING_RE = re.compile(
    r"^(?:(?:academic|personal|selected|key|side|technical|notable)\s+)?projects?\s*:?$",
    re.IGNORECASE,
)
SKILLS_HEADING_RE = re.compile(
    r"^(?:(?:technical|key|core|professional)\s+)?(?:skills|competencies)"
    r"(?:\s*(?:&|and|/)\s*(?:tools|technologies))?\s*:?$"
    r"|^(?:tools\s*(?:&|and)\s*)?technologies\s*:?$",
    re.IGNORECASE,
)
INLINE_SKILLS_RE = re.compile(
    r"^(?:(?:technical|key|core|professional)\s+)?(?:skills|competencies|technologies)\s*:\s*(.+)$",
    re.IGNORECASE,
)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z &/]{3,40}$")
LABELED_RE = re.compile(r"^([^:]{1,40}):\s*(.+)$")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PROJECT_SPLIT_RE = re.compile(r"\s+[|—–-]\s+")


def _is_heading(line: str) -> bool:
    return bool(
        SECTION_HEADING_RE.match(line)
        or PROJECTS_HEADING_RE.match(line)
        or SKILLS_HEADING_RE.match(line)
        or UPPER_HEADING_RE.match(line)
    )


def clean_resume_text(text: str) -> str:
    """Join PDF bullets with their text and tidy whitespace.

    PDFs often extract a bullet as a lone glyph on its own line ("●" newline "text").
    This turns every bullet into a single "- text" line. Running it twice is harmless.
    """
    out: list[str] = []
    pending = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").split("\n"):
        line = raw.strip()
        if not line:
            if not pending:
                out.append("")
            continue
        if len(line) == 1 and line in BULLET_GLYPHS:
            pending = True
            continue
        if line[0] in BULLET_GLYPHS:
            line = line[1:].strip()
            pending = True
        if pending:
            line = f"- {line}"
            pending = False
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _split_items(value: str) -> list[str]:
    """Split "a, b (c, d), e" on commas, semicolons and pipes outside brackets."""
    parts: list[str] = []
    buffer, depth = "", 0
    for char in value:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char in ",;|" and depth == 0:
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    parts.append(buffer)
    cleaned = (part.strip(" -•\t") for part in parts)
    return [part for part in cleaned if 0 < len(part) <= 60]


def _read_skill_block(lines: list[str], start: int) -> list[str]:
    items: list[str] = []
    last_was_list = False
    for raw in lines[start:]:
        line = raw.strip()
        if not line:
            continue
        if _is_heading(line) or YEAR_RE.search(line):
            break
        labeled = LABELED_RE.match(line)
        if labeled:  # "Frameworks & Libraries: TensorFlow, Keras"
            items += _split_items(labeled.group(2))
            last_was_list = True
        elif line.startswith("- "):
            items += _split_items(line[2:])
            last_was_list = True
        elif "," in line or (last_was_list and line[0].islower()):  # plain or wrapped list
            items += _split_items(line)
            last_was_list = True
        else:
            break
    return items


def extract_skill_items(text: str) -> list[str]:
    """Every item listed under a skills heading (all categories) or on a "Skills:" line."""
    lines = text.splitlines()
    items: list[str] = []
    for index, raw in enumerate(lines):
        line = raw.strip()
        inline = INLINE_SKILLS_RE.match(line)
        if inline:
            items += _split_items(inline.group(1))
        elif SKILLS_HEADING_RE.match(line):
            items += _read_skill_block(lines, index + 1)
    return _dedupe(items)


def _new_project(title: str) -> dict[str, Any]:
    parts = PROJECT_SPLIT_RE.split(title, maxsplit=1)
    return {
        "name": parts[0].strip(" :-"),
        "description": parts[1].strip() if len(parts) > 1 else "",
        "highlights": [],
    }


def _read_project_block(lines: list[str], start: int) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines[start:]:
        if not line:
            continue
        if _is_heading(line):
            break
        if line.startswith("- "):
            if current is None:
                current = _new_project("")
                projects.append(current)
            current["highlights"].append(line[2:].strip())
        elif current is not None and current["highlights"] and line[0].islower():
            current["highlights"][-1] += f" {line}"  # a bullet wrapped onto a second line
        elif (
            current is not None
            and not current["highlights"]
            and not (PROJECT_SPLIT_RE.search(line) and not LABELED_RE.match(line))
        ):
            current["description"] = f"{current['description']} {line}".strip()
        else:
            current = _new_project(line)
            projects.append(current)
    return [p for p in projects if p["name"] or p["highlights"]]


def extract_projects(text: str) -> list[dict[str, Any]]:
    """Projects listed under a "Projects" heading: name, description and bullet points."""
    lines = [line.strip() for line in text.splitlines()]
    projects: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if PROJECTS_HEADING_RE.match(line):
            projects += _read_project_block(lines, index + 1)
    return projects


def verify_grade(value: str, text: str) -> str:
    """Keep a GPA/grade only if its numbers (or the words) really appear in the resume."""
    value = (value or "").strip()
    if not value:
        return ""
    lowered = text.lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    if numbers:
        return value if all(number in lowered for number in numbers) else ""
    return value if value.lower() in lowered else ""


def _find_header(lines: list[str], job: dict[str, Any], start: int) -> int | None:
    """Index of the line that holds this job's title and company."""
    role = str(job.get("role") or "").lower().strip()
    company = str(job.get("company") or "").split(",")[0].strip().lower()
    if not role:
        return None
    for index in range(start, len(lines)):
        line = lines[index]
        if not line or line.startswith("- "):
            continue
        lowered = line.lower()
        if role in lowered and (not company or company in lowered):
            return index
    return None


def _collect_bullets(lines: list[str], start: int, end: int) -> list[str]:
    bullets: list[str] = []
    for line in lines[start:end]:
        if not line:
            continue
        if _is_heading(line):
            break
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line[0].islower():  # a bullet that wrapped onto a second line
            bullets[-1] += f" {line}"
    return bullets


def attach_highlights(experience: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Fill each job's highlights with the bullets written under its heading.

    If a job's heading or bullets can't be located, the job keeps whatever it had.
    """
    lines = [line.strip() for line in text.splitlines()]
    headers: list[int | None] = []
    cursor = 0
    for job in experience:
        index = _find_header(lines, job, cursor)
        headers.append(index)
        if index is not None:
            cursor = index + 1

    result = []
    for job, index in zip(experience, headers):
        job = dict(job)
        if index is not None:
            later = [h for h in headers if h is not None and h > index]
            end = min(later) if later else len(lines)
            bullets = _collect_bullets(lines, index + 1, end)
            if bullets:
                job["highlights"] = bullets
        result.append(job)
    return result


def enrich_sections(data: dict[str, Any], text: str) -> dict[str, Any]:
    """Complete and verify skills, languages, grades, highlights and projects from the text."""
    data = dict(data)
    lowered = text.lower()

    model_skills = [s for s in data.get("skills", []) if s.lower() in lowered]
    data["skills"] = _dedupe(extract_skill_items(text) + model_skills)

    # "Languages: Python, Java" under Technical Skills are programming languages, not spoken ones.
    skill_names = {s.lower() for s in data["skills"]}
    data["languages"] = [
        entry
        for entry in data.get("languages", [])
        if entry["language"].lower() not in skill_names and entry["language"].lower() in lowered
    ]

    data["education"] = [
        {**entry, "grade": verify_grade(entry.get("grade", ""), text)}
        for entry in data.get("education", [])
    ]
    data["experience"] = attach_highlights(data.get("experience", []), text)

    projects = extract_projects(text)
    if projects:
        data["projects"] = projects
    return data