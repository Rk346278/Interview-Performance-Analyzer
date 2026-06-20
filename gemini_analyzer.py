"""Analyze resume text using the Google Gemini API."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

ANALYSIS_KEYS = (
    "skills",
    "projects",
    "certifications",
    "strengths",
    "weaknesses",
    "alignment_issues",
    "suggestions",
)

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=list(ANALYSIS_KEYS),
    properties={
        "skills": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Technical and soft skills detected in the resume.",
        ),
        "projects": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Projects, internships, or portfolio work mentioned.",
        ),
        "certifications": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Certifications, courses, or workshops listed.",
        ),
        "strengths": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Notable resume strengths for placement interviews.",
        ),
        "weaknesses": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Gaps or weak areas visible in the resume.",
        ),
        "alignment_issues": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Mismatches between listed skills and project evidence.",
        ),
        "suggestions": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Actionable resume improvement suggestions.",
        ),
    },
)

ANALYSIS_PROMPT = """You are an expert career coach analyzing a student resume for campus placement interviews.

Analyze the resume text below and return structured JSON with these fields:
- skills: technical and soft skills explicitly or implicitly present
- projects: named projects, internships, or significant academic work
- certifications: certifications, workshops, or professional courses
- strengths: what makes this resume competitive
- weaknesses: missing elements or underdeveloped sections
- alignment_issues: skills claimed without supporting project evidence (or vice versa)
- suggestions: specific, actionable improvements for placement readiness

Be concise but specific. Use short bullet-style strings. If a category has no items, return an empty array.

RESUME TEXT:
{resume_text}
"""


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export your Gemini API key before running resume analysis."
        )
    return api_key


def _normalize_analysis(raw: dict[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key in ANALYSIS_KEYS:
        value = raw.get(key, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"Expected list for '{key}', got {type(value).__name__}")
        normalized[key] = [str(item).strip() for item in value if str(item).strip()]
    return normalized


def analyze_resume_text(
    resume_text: str,
    *,
    model: str = "gemini-2.5-flash",
) -> dict[str, list[str]]:
    """Send resume text to Gemini and return structured analysis."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty; nothing to analyze")

    client = genai.Client(api_key=_get_api_key())

    prompt = ANALYSIS_PROMPT.format(resume_text=resume_text.strip())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    response_text = response.text
    if not response_text:
        raise RuntimeError("Gemini returned an empty response")

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {response_text}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from Gemini, got {type(parsed).__name__}")

    return _normalize_analysis(parsed)


ML_FEATURE_KEYS = (
    "cgpa",
    "projects_count",
    "workshops_certifications_count",
    "extracurricular_activities",
)

ML_FEATURES_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=list(ML_FEATURE_KEYS),
    properties={
        "cgpa": types.Schema(
            type=types.Type.NUMBER,
            nullable=True,
            description="CGPA or GPA on a 10-point scale. Null if not stated.",
        ),
        "projects_count": types.Schema(
            type=types.Type.INTEGER,
            description="Count of distinct projects, capstone work, or major academic builds.",
        ),
        "workshops_certifications_count": types.Schema(
            type=types.Type.INTEGER,
            description="Count of workshops, certifications, or professional courses listed.",
        ),
        "extracurricular_activities": types.Schema(
            type=types.Type.STRING,
            enum=["Yes", "No"],
            description="Yes if clubs, sports, volunteering, or leadership outside academics appear.",
        ),
    },
)

ML_FEATURES_PROMPT = """You are extracting numeric placement-model features from a student resume.

Return JSON with:
- cgpa: CGPA/GPA on a 10-point scale. Use null if not mentioned or unclear.
- projects_count: number of distinct projects, capstone work, or portfolio builds (not internships).
- workshops_certifications_count: number of workshops, certifications, or professional courses.
- extracurricular_activities: "Yes" if clubs, sports, volunteering, cultural activities, or student leadership appear; otherwise "No".

Count conservatively from explicit resume sections. Do not invent values.

RESUME TEXT:
{resume_text}
"""


def _normalize_ml_features(raw: dict[str, Any]) -> dict[str, float | int | str | None]:
    cgpa = raw.get("cgpa")
    if cgpa is not None:
        cgpa = float(cgpa)

    projects_count = int(raw.get("projects_count", 0))
    workshops_count = int(raw.get("workshops_certifications_count", 0))

    extracurricular = raw.get("extracurricular_activities", "No")
    if extracurricular not in ("Yes", "No"):
        extracurricular = "No"

    return {
        "cgpa": cgpa,
        "projects_count": max(0, projects_count),
        "workshops_certifications_count": max(0, workshops_count),
        "extracurricular_activities": extracurricular,
    }


def extract_ml_features_from_text(
    resume_text: str,
    *,
    model: str = "gemini-2.5-flash",
) -> dict[str, float | int | str | None]:
    """Extract placement-model features from resume text using Gemini."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty; nothing to analyze")

    client = genai.Client(api_key=_get_api_key())

    prompt = ML_FEATURES_PROMPT.format(resume_text=resume_text.strip())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ML_FEATURES_SCHEMA,
            temperature=0.1,
        ),
    )

    response_text = response.text
    if not response_text:
        raise RuntimeError("Gemini returned an empty response")

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {response_text}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from Gemini, got {type(parsed).__name__}")

    return _normalize_ml_features(parsed)
