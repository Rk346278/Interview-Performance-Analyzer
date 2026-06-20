"""End-to-end resume analysis pipeline: PDF -> text -> Gemini -> structured output."""

from __future__ import annotations

from pathlib import Path

from gemini_analyzer import analyze_resume_text, extract_ml_features_from_text
from resume_parser import extract_text_from_bytes, extract_text_from_pdf


def analyze_resume_pdf(pdf_path: str | Path) -> dict[str, list[str]]:
    """Load a PDF resume, extract text, and return Gemini analysis."""
    resume_text = extract_text_from_pdf(pdf_path)
    return analyze_resume_text(resume_text)


def analyze_resume_upload(pdf_bytes: bytes) -> dict[str, list[str]]:
    """Analyze resume content from uploaded PDF bytes."""
    resume_text = extract_text_from_bytes(pdf_bytes)
    return analyze_resume_text(resume_text)


def analyze_resume_from_text(resume_text: str) -> dict[str, list[str]]:
    """Analyze resume text directly (skips PDF extraction)."""
    return analyze_resume_text(resume_text)


def extract_ml_features_upload(pdf_bytes: bytes) -> dict[str, float | int | str | None]:
    """Extract ML placement features from uploaded PDF bytes."""
    resume_text = extract_text_from_bytes(pdf_bytes)
    return extract_ml_features_from_text(resume_text)
