import pytest

from hr_parser.extract import read_uploaded_file


def test_reads_txt():
    assert read_uploaded_file("cv.txt", b"  Jane Doe  ") == "Jane Doe"


def test_rejects_unknown_type():
    with pytest.raises(ValueError):
        read_uploaded_file("cv.exe", b"x")


def test_rejects_empty_file():
    with pytest.raises(ValueError):
        read_uploaded_file("cv.txt", b"   ")


def test_pdf_hyperlinks_are_appended():
    import io

    from pypdf import PdfWriter
    from pypdf.annotations import Link

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_annotation(
        page_number=0,
        annotation=Link(rect=(10, 10, 100, 40), url="https://github.com/johnsmith"),
    )
    buffer = io.BytesIO()
    writer.write(buffer)

    text = read_uploaded_file("cv.pdf", buffer.getvalue())
    assert "https://github.com/johnsmith" in text
