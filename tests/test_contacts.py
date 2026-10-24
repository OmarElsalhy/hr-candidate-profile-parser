import pytest

from hr_parser.contacts import (
    enrich_contacts,
    find_email,
    find_github,
    find_linkedin,
    find_phone,
)


def test_find_email_ignores_trailing_dot_and_mailto():
    assert find_email("Contact: mailto:omar@mail.com.") == "omar@mail.com"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Phone: +20 100 123 4567", "+20 100 123 4567"),
        ("Tel 01012345678 | Cairo", "01012345678"),
        ("(555) 123-4567", "(555) 123-4567"),
        ("Tel: (+20) 100-123-4567", "(+20) 100-123-4567"),
        ("Phone: 01012345678 2020-2023", "01012345678"),
        ("Phone:+20 100 123 4567", "+20 100 123 4567"),
    ],
)
def test_find_phone(text, expected):
    assert find_phone(text) == expected


def test_find_phone_ignores_years_and_dates():
    assert find_phone("Google (2020-2023), OpenAI (2023-Present), graduated 2020 2021 2022") == ""


def test_find_github_returns_profile_even_for_repo_links():
    assert find_github("see https://www.github.com/johnsmith/cool-repo.") == "github.com/johnsmith"
    assert find_github("nothing here") == ""


def test_find_linkedin_strips_trailing_punctuation():
    assert find_linkedin("LinkedIn: https://linkedin.com/in/john-smith/).") == "https://linkedin.com/in/john-smith/"


def test_enrich_keeps_verified_model_values_and_fixes_the_rest():
    text = "Email: a@b.co | linkedin.com/in/real"
    data = enrich_contacts(
        {"email": "a@b.co", "phone": "123456789", "github": "", "linkedin": "https://linkedin.com/in/real"},
        text,
    )
    assert data["email"] == "a@b.co"
    assert data["linkedin"] == "https://linkedin.com/in/real"  # same link, scheme added by the model
    assert data["phone"] == ""  # not in the text -> removed
