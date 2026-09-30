"""Tests for document parsing and error handling."""

from __future__ import annotations

import io

import pytest

from src.config import SAMPLE_RESUMES_DIR
from src.parser import (
    DocumentParsingError,
    clean_extracted_text,
    detect_file_type,
    extract_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_safe,
)

PDF_SAMPLE = SAMPLE_RESUMES_DIR / "R002_backend_developer.pdf"
DOCX_SAMPLE = SAMPLE_RESUMES_DIR / "R003_frontend_developer.docx"
TXT_SAMPLE = SAMPLE_RESUMES_DIR / "R001_data_science_fresher.txt"


def test_extract_text_from_txt_reads_sample():
    text = extract_text_from_txt(TXT_SAMPLE)
    assert "Python" in text
    assert len(text) > 200


def test_extract_text_from_pdf_reads_sample():
    pytest.importorskip("pymupdf")
    text = extract_text_from_pdf(PDF_SAMPLE)
    assert "RAHUL MENON" in text
    assert "Django" in text


def test_extract_text_from_docx_reads_sample():
    pytest.importorskip("docx")
    text = extract_text_from_docx(DOCX_SAMPLE)
    assert "React" in text


def test_extract_text_detects_type_from_extension():
    assert "Python" in extract_text(TXT_SAMPLE)
    assert "Django" in extract_text(PDF_SAMPLE)


def test_detect_file_type_by_extension_and_magic_bytes():
    assert detect_file_type("cv.pdf") == "pdf"
    assert detect_file_type("cv.docx") == "docx"
    assert detect_file_type("cv.txt") == "txt"
    assert detect_file_type("unnamed", b"%PDF-1.7 rest") == "pdf"
    assert detect_file_type("unnamed", b"PK\x03\x04rest") == "docx"


def test_unsupported_extension_raises_useful_error():
    with pytest.raises(DocumentParsingError) as info:
        detect_file_type("resume.xyz", b"\x89PNG\r\n\x1a\n")
    assert "Unsupported file type" in str(info.value)


def test_legacy_doc_gets_specific_message():
    with pytest.raises(DocumentParsingError) as info:
        detect_file_type("resume.doc", b"\xd0\xcf\x11\xe0")
    assert ".docx" in str(info.value)


def test_empty_file_is_rejected():
    with pytest.raises(DocumentParsingError) as info:
        extract_text(b"", filename="resume.txt")
    assert "empty" in str(info.value).lower()


def test_corrupted_pdf_does_not_crash():
    pytest.importorskip("pymupdf")
    corrupted = b"%PDF-1.4\nthis is not really a pdf" + b"\x00" * 200
    with pytest.raises(DocumentParsingError):
        extract_text(corrupted, filename="broken.pdf")


def test_missing_file_raises_document_parsing_error():
    with pytest.raises(DocumentParsingError):
        extract_text("/tmp/definitely-not-here-9x8y7z.txt")


def test_extract_text_safe_never_raises():
    text, error = extract_text_safe(b"", filename="x.txt")
    assert text == ""
    assert error and isinstance(error, str)

    text, error = extract_text_safe(TXT_SAMPLE)
    assert error is None
    assert "Python" in text


def test_file_like_object_is_supported():
    stream = io.BytesIO(TXT_SAMPLE.read_bytes())
    stream.name = "R001_data_science_fresher.txt"
    assert "Python" in extract_text(stream)


def test_oversized_upload_is_rejected():
    payload = b"a" * (6 * 1024 * 1024)
    with pytest.raises(DocumentParsingError) as info:
        extract_text(payload, filename="huge.txt")
    assert "limit" in str(info.value).lower()


def test_clean_extracted_text_normalises_whitespace_and_ligatures():
    messy = "Line one   with  spaces\n\n\n\nLine two\ufb01ne\u2022bullet"
    cleaned = clean_extracted_text(messy)
    assert "  " not in cleaned
    assert "fine" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_extracted_text_handles_empty_input():
    assert clean_extracted_text("") == ""
    assert clean_extracted_text(None) == ""
