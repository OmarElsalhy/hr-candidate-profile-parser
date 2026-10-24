from pathlib import Path

from streamlit.testing.v1 import AppTest

import hr_parser.llm as llm
from hr_parser.schema import SAMPLE_RESUME
from tests.test_schema import EXPECTED, GOOD

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def open_paste_tab(at):
    at.radio[0].set_value("Paste text").run()
    return at


def test_app_starts_empty_and_parse_is_disabled(monkeypatch):
    monkeypatch.setattr(llm, "load_model", lambda *args, **kwargs: (None, None))

    at = open_paste_tab(AppTest.from_file(APP, default_timeout=30).run())
    assert not at.exception
    assert at.text_area(key="resume_text").value == ""
    assert at.button(key="parse").disabled
    assert len(at.sidebar) == 0  # no settings shown to the user


def test_app_parses_a_pasted_resume(monkeypatch):
    monkeypatch.setattr(llm, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(llm, "generate", lambda *args, **kwargs: GOOD)

    at = open_paste_tab(AppTest.from_file(APP, default_timeout=30).run())
    at.text_area(key="resume_text").set_value(SAMPLE_RESUME).run()
    at.button(key="parse").click().run()
    assert not at.exception
    assert at.session_state["result"]["data"] == EXPECTED


def test_app_shows_a_friendly_error_for_unparseable_output(monkeypatch):
    monkeypatch.setattr(llm, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(llm, "generate", lambda *args, **kwargs: "no json here")

    at = open_paste_tab(AppTest.from_file(APP, default_timeout=30).run())
    at.text_area(key="resume_text").set_value(SAMPLE_RESUME).run()
    at.button(key="parse").click().run()

    assert not at.exception
    assert at.session_state["result"]["error"]
    assert len(at.error) == 1
    assert "Traceback" not in at.error[0].value  # no stack traces for the user
