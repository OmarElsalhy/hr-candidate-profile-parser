"""Deterministic extraction of contact details (email, phone, GitHub, LinkedIn).

Small language models often garble or invent these. Regexes are exact, so we use them
to fill missing values and to throw away anything the model made up.
"""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_-]+)", re.IGNORECASE)
LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9_%.-]+/?",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<![\w.])\(?\+?\(?\d[\d\s().-]{7,18}\d(?!\w)")
TRAILING_YEAR_RE = re.compile(r"[\s-]+(?:19|20)\d{2}$")
YEAR_RANGE_RE = re.compile(r"(?:(?:19|20)\d{2}[\s-]*){2,3}")

CONTACT_FIELDS = ("email", "phone", "github", "linkedin")


def _clean_url(value: str) -> str:
    return value.rstrip(".,;:)>]\"'")


def find_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def find_github(text: str) -> str:
    """Return the profile URL (``github.com/<user>``) of the first GitHub link."""
    match = GITHUB_RE.search(text)
    return f"github.com/{match.group(1)}" if match else ""


def find_linkedin(text: str) -> str:
    match = LINKEDIN_RE.search(text)
    return _clean_url(match.group(0)) if match else ""


def find_phone(text: str) -> str:
    """First phone-looking number with 9-15 digits (skips year ranges like 2020-2023)."""
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        # "01012345678 2020-2023": drop years glued to a complete number.
        while TRAILING_YEAR_RE.search(candidate):
            shorter = TRAILING_YEAR_RE.sub("", candidate)
            if len(re.sub(r"\D", "", shorter)) < 9:
                break
            candidate = shorter
        digits = re.sub(r"\D", "", candidate)
        if 9 <= len(digits) <= 15 and not YEAR_RANGE_RE.fullmatch(candidate):
            return candidate
    return ""


_FINDERS = {
    "email": find_email,
    "phone": find_phone,
    "github": find_github,
    "linkedin": find_linkedin,
}


def _canon_url(value: str) -> str:
    return re.sub(r"^(?:https?://)?(?:www\.)?", "", value.strip().lower()).rstrip("/")


def _appears_in(field: str, value: str, text: str) -> bool:
    """True if the model's value is really present in the resume text."""
    if field == "phone":
        digits = re.sub(r"\D", "", value)
        return len(digits) >= 7 and digits in re.sub(r"\D", "", text)
    if field in ("github", "linkedin"):
        return _canon_url(value) in text.lower()
    return value.strip().lower() in text.lower()


def verify_location(value: str, text: str) -> str:
    """Keep the model's location only if its city (the part before the first comma) is in the text."""
    value = (value or "").strip()
    city = value.split(",")[0].strip().lower()
    return value if city and city in text.lower() else ""


def enrich_contacts(data: dict, text: str) -> dict:
    """Keep model values that appear in ``text``; otherwise use the regex match (or '')."""
    data = dict(data)
    for field in CONTACT_FIELDS:
        current = (data.get(field) or "").strip()
        if current and _appears_in(field, current, text):
            data[field] = current
        else:
            data[field] = _FINDERS[field](text)
    data["location"] = verify_location(data.get("location"), text)
    return data