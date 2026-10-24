"""HR Candidate Profile Parser - Streamlit app."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time

import pandas as pd
import streamlit as st

from hr_parser import __version__, llm
from hr_parser.extract import SUPPORTED_TYPES, read_uploaded_file
from hr_parser.schema import build_output_parser, build_prompt, parse_resume
from hr_parser.sections import clean_resume_text

logger = logging.getLogger("hr_parser")

# Developer switch: HR_PARSER_DEBUG=1 shows the extracted text and the raw model output.
DEBUG = os.environ.get("HR_PARSER_DEBUG") == "1"

MAX_CHARS = 8000  # longer resumes are cut to keep the prompt short and fast
INPUT_UPLOAD, INPUT_PASTE = "Upload a file", "Paste text"
ERROR_MESSAGE = "We couldn't read this resume. Try a clearer PDF, or paste the text instead."

st.set_page_config(page_title="HR Candidate Profile Parser", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1100px;}
    .chip {
        display: inline-block; padding: .25rem .75rem; margin: .15rem .3rem .15rem 0;
        border-radius: 999px; font-size: .85rem;
        background: rgba(79, 70, 229, .12); border: 1px solid rgba(79, 70, 229, .35);
    }
    .field-label {
        font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
        opacity: .6; margin-bottom: .1rem;
    }
    .field-value {font-size: 1.15rem; font-weight: 600; word-break: break-all;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model... the first run downloads it, so it can take a while.")
def get_model():
    """Load the model once and keep it in memory for every parse."""
    return llm.load_model()


def esc(value) -> str:
    return html.escape(str(value)) if value not in (None, "") else "-"


def md(value) -> str:
    """Escape ``$`` so Streamlit doesn't render it as LaTeX."""
    return str(value).replace("$", "\\$")


def link_html(value: str) -> str:
    """Clickable link for http(s) URLs; '-' when empty."""
    value = (value or "").strip()
    if not value:
        return "-"
    url = value if re.match(r"^https?://", value, re.IGNORECASE) else f"https://{value}"
    label = re.sub(r"^https?://(?:www\.)?", "", url)
    return (
        f'<a href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(label)}</a>'
    )


def field(column, label: str, value_html: str) -> None:
    column.markdown(
        f'<div class="field-label">{label}</div><div class="field-value">{value_html}</div>',
        unsafe_allow_html=True,
    )


def chips(items: list[str]) -> str:
    return "".join(f'<span class="chip">{esc(item)}</span>' for item in items)


def run_parse(text: str, tokenizer, model) -> None:
    text = clean_resume_text(text)
    parser = build_output_parser()
    prompt = build_prompt(text, parser)

    start = time.perf_counter()
    with st.spinner("Reading the resume..."):
        raw = llm.generate(tokenizer, model, prompt)
    elapsed = time.perf_counter() - start

    result = {"data": None, "error": None, "elapsed": elapsed, "raw": raw, "text": text}
    try:
        result["data"] = parse_resume(raw, parser, resume_text=text)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to the user
        logger.warning("Could not parse the model output (%s). Raw output:\n%s", exc, raw)
        result["error"] = ERROR_MESSAGE
    st.session_state.result = result


def render_table(entries: list, columns: dict[str, str], empty_msg: str) -> None:
    rows = [
        {label: entry.get(key, "") for key, label in columns.items()}
        for entry in entries
        if isinstance(entry, dict)
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption(empty_msg)


def render_bullets(items: list[str]) -> None:
    if items:
        st.markdown("\n".join(f"- {md(item)}" for item in items))


def render_result(result: dict) -> None:
    st.divider()

    if result["error"]:
        st.error(result["error"])
    else:
        data = result["data"]
        st.success(f"Done in {result['elapsed']:.1f}s")

        col_name, col_email, col_phone = st.columns(3)
        field(col_name, "Full name", esc(data.get("full_name")))
        field(col_email, "Email", esc(data.get("email")))
        field(col_phone, "Phone", esc(data.get("phone")))

        st.write("")
        col_location, col_github, col_linkedin = st.columns(3)
        field(col_location, "Location", esc(data.get("location")))
        field(col_github, "GitHub", link_html(data.get("github")))
        field(col_linkedin, "LinkedIn", link_html(data.get("linkedin")))

        tab_view, tab_json = st.tabs(["Structured view", "JSON"])

        with tab_view:
            st.subheader("🎓 Education")
            render_table(
                data["education"],
                {"degree": "Degree", "institution": "Institution", "year": "Year", "grade": "GPA / Grade"},
                "No education found.",
            )

            st.subheader("🛠️ Skills")
            if data["skills"]:
                st.markdown(chips(data["skills"]), unsafe_allow_html=True)
            else:
                st.caption("No skills found.")

            st.subheader("🌐 Languages")
            languages = [
                f"{lang['language']} · {lang['level']}" if lang.get("level") else lang["language"]
                for lang in data["languages"]
            ]
            if languages:
                st.markdown(chips(languages), unsafe_allow_html=True)
            else:
                st.caption("No languages found.")

            st.subheader("💼 Experience")
            if not data["experience"]:
                st.caption("No experience found.")
            for job in data["experience"]:
                with st.container(border=True):
                    heading = " · ".join(
                        part for part in (f"**{md(job['role'] or '-')}**", md(job["company"])) if part
                    )
                    st.markdown(heading)
                    if job["years"]:
                        st.caption(job["years"])
                    render_bullets(job["highlights"])

            st.subheader("🚀 Projects")
            if not data["projects"]:
                st.caption("No projects found.")
            for project in data["projects"]:
                with st.container(border=True):
                    st.markdown(f"**{md(project['name'] or 'Project')}**")
                    if project["description"]:
                        st.caption(project["description"])
                    render_bullets(project["highlights"])

        with tab_json:
            st.json(data)
            st.download_button(
                "⬇️ Download JSON",
                json.dumps(data, indent=2, ensure_ascii=False),
                file_name="candidate_profile.json",
                mime="application/json",
            )

    if DEBUG:
        with st.expander("Debug: extracted text"):
            st.text(result["text"])
        with st.expander("Debug: raw model output"):
            st.code(result["raw"], language="markdown")


# --- Page ------------------------------------------------------------------
st.title("📄 HR Candidate Profile Parser")
st.write("Upload a resume or paste its text to get a clean, structured profile.")

tokenizer, model = get_model()

# Re-assigning keeps the text when the input method is switched.
st.session_state.resume_text = st.session_state.get("resume_text", "")

source = st.radio(
    "Input method",
    [INPUT_UPLOAD, INPUT_PASTE],
    horizontal=True,
    label_visibility="collapsed",
)

resume_text = ""
if source == INPUT_PASTE:
    st.text_area(
        "Resume text",
        key="resume_text",
        height=260,
        placeholder="Paste the resume text here...",
    )
    resume_text = st.session_state.resume_text
else:
    uploaded = st.file_uploader("Upload a resume", type=list(SUPPORTED_TYPES))
    if uploaded is not None:
        try:
            resume_text = read_uploaded_file(uploaded.name, uploaded.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read the file: {exc}")
        if DEBUG and resume_text:
            with st.expander("Debug: extracted text"):
                st.text(resume_text)

if len(resume_text) > MAX_CHARS:
    st.warning(f"The resume is long; only the first {MAX_CHARS:,} characters will be used.")
    resume_text = resume_text[:MAX_CHARS]

if st.button("Parse resume", type="primary", key="parse", disabled=not resume_text.strip()):
    run_parse(resume_text, tokenizer, model)

if st.session_state.get("result"):
    render_result(st.session_state.result)

st.caption(f"v{__version__} - Output is model-generated; verify it before relying on it.")