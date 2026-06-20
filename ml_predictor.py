"""Load the trained placement model and build prediction inputs."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "placement_model.pkl"

FEATURE_COLUMNS = [
    "StudentID",
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]


def load_model():
    """Return the saved logistic regression model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def yes_no_to_int(value: str) -> int:
    """Match notebook encoding: Yes -> 1, No -> 0."""
    return 1 if value == "Yes" else 0


def build_feature_row(
    *,
    cgpa: float,
    internships: int,
    projects: int,
    workshops_certifications: int,
    aptitude_test_score: float,
    soft_skills_rating: float,
    extracurricular_activities: str,
    placement_training: str,
    ssc_marks: float,
    hsc_marks: float,
    student_id: int = 0,
) -> pd.DataFrame:
    """Build a single-row DataFrame with columns in training order."""
    return pd.DataFrame(
        [
            {
                "StudentID": student_id,
                "CGPA": cgpa,
                "Internships": internships,
                "Projects": projects,
                "Workshops/Certifications": workshops_certifications,
                "AptitudeTestScore": aptitude_test_score,
                "SoftSkillsRating": soft_skills_rating,
                "ExtracurricularActivities": yes_no_to_int(extracurricular_activities),
                "PlacementTraining": yes_no_to_int(placement_training),
                "SSC_Marks": ssc_marks,
                "HSC_Marks": hsc_marks,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_placement_probability(features: pd.DataFrame) -> float:
    """Return placement probability (class 1) for one feature row."""
    model = load_model()
    probability = model.predict_proba(features)[0][1]
    return float(probability)
