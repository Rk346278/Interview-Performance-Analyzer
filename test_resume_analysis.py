#!/usr/bin/env python3
"""Sample script: load a PDF resume, extract text, analyze with Gemini, print results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resume_analysis import analyze_resume_pdf
from resume_parser import extract_text_from_pdf

DEFAULT_SAMPLE_PDF = Path(__file__).resolve().parent / "samples" / "sample_resume.pdf"


def _print_analysis(analysis: dict[str, list[str]]) -> None:
    print(json.dumps(analysis, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the resume analysis pipeline end-to-end."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=str(DEFAULT_SAMPLE_PDF),
        help=f"Path to a PDF resume (default: {DEFAULT_SAMPLE_PDF})",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract and print PDF text without calling Gemini.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        return 1

    print(f"Loading resume: {pdf_path}")
    resume_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(resume_text)} characters of text.\n")

    if args.extract_only:
        print("--- Extracted Text ---")
        print(resume_text)
        return 0

    print("Sending to Gemini for analysis...\n")
    analysis = analyze_resume_pdf(pdf_path)
    print("--- Structured Analysis ---")
    _print_analysis(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
