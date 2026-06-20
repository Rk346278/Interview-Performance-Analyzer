"""Streamlit app — Interview Performance Analyzer (Version 1)."""

from __future__ import annotations

import html
import textwrap

import streamlit as st

from ml_predictor import build_feature_row, predict_placement_probability
from resume_analysis import analyze_resume_upload, extract_ml_features_upload

st.set_page_config(
    page_title="Interview Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-base: #0b0f19;
        --bg-panel: #111827;
        --bg-card: #151f32;
        --bg-card-hover: #1a2740;
        --border: rgba(148, 163, 184, 0.14);
        --border-accent: rgba(56, 189, 248, 0.35);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent: #38bdf8;
        --accent-2: #818cf8;
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f87171;
        --glow: rgba(56, 189, 248, 0.15);
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(56, 189, 248, 0.18), transparent),
            radial-gradient(ellipse 60% 40% at 100% 0%, rgba(129, 140, 248, 0.12), transparent),
            linear-gradient(180deg, #0b0f19 0%, #0f172a 50%, #0b0f19 100%);
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1120px;
    }

    #MainMenu, footer, header {visibility: hidden;}

    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }

    .stMarkdown p, .stMarkdown li, label, .stCaption {
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 700;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.4);
        border: none;
        color: white;
    }

    section[data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 1px dashed var(--border-accent);
        border-radius: 16px;
        padding: 1.25rem;
    }

    .stSlider [data-baseweb="slider"] div {
        color: var(--accent);
    }

    .hero-wrap {
        text-align: center;
        padding: 2.75rem 2rem 2.25rem;
        margin-bottom: 2rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(21, 31, 50, 0.95) 0%, rgba(17, 24, 39, 0.98) 100%);
        border: 1px solid var(--border-accent);
        box-shadow: 0 24px 64px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative;
        overflow: hidden;
    }

    .hero-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.12), transparent 55%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 0.75rem 0;
        line-height: 1.1;
        background: linear-gradient(135deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.65;
    }

    .input-section-label {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.75rem 0 1rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--border);
    }

    .input-section-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 800;
        color: var(--accent);
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        padding: 1.1rem 1.25rem 1.25rem !important;
        margin-bottom: 0.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    .panel-hint {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .results-shell {
        margin-top: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border);
    }

    .results-header {
        margin-bottom: 1.75rem;
    }

    .results-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }

    .results-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 0.35rem 0;
    }

    .results-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
    }

    .dashboard-section-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 1rem 0;
    }

    .prediction-dashboard-card {
        background: linear-gradient(145deg, rgba(21, 31, 50, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 22px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(56, 189, 248, 0.05) inset;
    }

    .prediction-grid {
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 2rem;
        align-items: center;
    }

    @media (max-width: 768px) {
        .prediction-grid {
            grid-template-columns: 1fr;
            justify-items: center;
            text-align: center;
        }
    }

    .circular-meter {
        position: relative;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(
            var(--meter-color, #38bdf8) calc(var(--pct, 0) * 1%),
            rgba(148, 163, 184, 0.15) 0
        );
        box-shadow: 0 0 24px var(--glow, rgba(56, 189, 248, 0.15));
    }

    .circular-meter-inner {
        width: 124px;
        height: 124px;
        border-radius: 50%;
        background: #151f32;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--border);
    }

    .circular-meter svg {
        display: none;
    }

    .meter-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
    }

    .meter-caption {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    .prediction-meta-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .prediction-headline {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        margin: 0 0 1rem 0;
        color: var(--meter-color, #38bdf8);
    }

    .progress-track {
        width: 100%;
        height: 12px;
        background: rgba(148, 163, 184, 0.12);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.75rem 0 1.25rem 0;
        border: 1px solid var(--border);
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--meter-color, #38bdf8), var(--meter-color-2, #818cf8));
        box-shadow: 0 0 16px var(--glow);
        transition: width 0.6s ease;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .status-high {
        background: rgba(52, 211, 153, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }

    .status-medium {
        background: rgba(251, 191, 36, 0.12);
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }

    .status-low {
        background: rgba(248, 113, 113, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    .insights-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
    }

    @media (max-width: 960px) {
        .insights-grid {
            grid-template-columns: 1fr;
        }
    }

    .insight-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.4rem 1.5rem;
        min-height: 220px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .insight-card:hover {
        border-color: var(--card-accent, var(--border-accent));
        transform: translateY(-2px);
    }

    .insight-card-strengths { --card-accent: rgba(52, 211, 153, 0.4); }
    .insight-card-weaknesses { --card-accent: rgba(248, 113, 113, 0.4); }
    .insight-card-suggestions { --card-accent: rgba(56, 189, 248, 0.4); }

    .insight-heading {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }

    .insight-icon {
        font-size: 1.25rem;
        line-height: 1;
    }

    .heading-strengths { color: #6ee7b7; }
    .heading-weaknesses { color: #fca5a5; }
    .heading-suggestions { color: #7dd3fc; }

    .analysis-list {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--text-secondary);
        line-height: 1.75;
        font-size: 0.92rem;
    }

    .analysis-list li {
        margin-bottom: 0.55rem;
    }

    .analysis-list li::marker {
        color: var(--text-muted);
    }

    .empty-state {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin: 0;
        font-style: italic;
    }
</style>
"""


def render_html(markup: str) -> None:
    """Render HTML using Streamlit-supported APIs (no markdown code-block parsing)."""
    content = textwrap.dedent(markup).strip()
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


def inject_app_css() -> None:
    render_html(APP_CSS)


def render_hero() -> None:
    render_html(
        """
        <div class="hero-wrap">
            <div class="hero-badge">AI · ML · Placement Analytics</div>
            <h1 class="hero-title">Interview Performance Analyzer</h1>
            <p class="hero-subtitle">
                Upload your resume, let Gemini extract key academic signals, complete the
                remaining profile details, and receive data-driven placement insights.
            </p>
        </div>
        """
    )


def section_header(number: int, title: str) -> None:
    render_html(
        f"""
        <div class="input-section-label">
            <span class="input-section-num">{number}</span>
            <span>{html.escape(title)}</span>
        </div>
        """
    )


def reset_extracted_features() -> None:
    st.session_state.extracted_features = None
    st.session_state.uploaded_file_name = None
    st.session_state.analysis_results = None


def readiness_status(probability: float) -> tuple[str, str, str, str]:
    """Return readiness label, badge class, meter color, and secondary color."""
    pct = probability * 100
    if pct >= 80:
        return "High", "status-high", "#34d399", "#10b981"
    if pct >= 50:
        return "Medium", "status-medium", "#fbbf24", "#f59e0b"
    return "Low", "status-low", "#f87171", "#ef4444"


def render_bullet_card(
    title: str,
    emoji: str,
    heading_class: str,
    card_class: str,
    items: list[str],
) -> str:
    if items:
        bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
        body = f'<ul class="analysis-list">{bullets}</ul>'
    else:
        body = '<p class="empty-state">No items identified.</p>'

    return (
        f'<div class="insight-card {card_class}">'
        f'<div class="insight-heading {heading_class}">'
        f'<span class="insight-icon">{emoji}</span>'
        f"<span>{html.escape(title)}</span>"
        f"</div>{body}</div>"
    )


def render_results_section(results: dict) -> None:
    probability = results["placement_probability"]
    pct = probability * 100
    status_label, status_class, meter_color, meter_color_2 = readiness_status(probability)
    status_emoji = {"High": "🚀", "Medium": "📈", "Low": "⚠️"}.get(status_label, "📊")

    strengths_card = render_bullet_card(
        "Strengths", "💪", "heading-strengths", "insight-card-strengths", results["strengths"]
    )
    weaknesses_card = render_bullet_card(
        "Weaknesses", "🔍", "heading-weaknesses", "insight-card-weaknesses", results["weaknesses"]
    )
    suggestions_card = render_bullet_card(
        "Suggestions", "💡", "heading-suggestions", "insight-card-suggestions", results["suggestions"]
    )

    render_html(
        f"""
        <div class="results-shell">
            <div class="results-header">
                <div class="results-kicker">Analytics Dashboard</div>
                <h2 class="results-title">Your Results</h2>
                <p class="results-subtitle">
                    Placement prediction powered by Logistic Regression and resume insights from Gemini.
                </p>
            </div>

            <p class="dashboard-section-title">Placement Prediction</p>
            <div class="prediction-dashboard-card" style="--meter-color: {meter_color}; --meter-color-2: {meter_color_2}; --glow: {meter_color}33; --pct: {pct:.1f};">
                <div class="prediction-grid">
                    <div class="circular-meter" style="--pct: {pct:.1f}; --meter-color: {meter_color}; --glow: {meter_color}33;">
                        <div class="circular-meter-inner">
                            <div class="meter-value">{pct:.1f}%</div>
                            <div class="meter-caption">Probability</div>
                        </div>
                    </div>
                    <div>
                        <div class="prediction-meta-label">Placement Probability</div>
                        <div class="prediction-headline">{pct:.1f}%</div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: {pct:.1f}%;"></div>
                        </div>
                        <div class="prediction-meta-label">Placement Readiness Status</div>
                        <span class="status-badge {status_class}">{status_emoji} {status_label}</span>
                    </div>
                </div>
            </div>

            <p class="dashboard-section-title">Resume Analysis</p>
            <div class="insights-grid">
                {strengths_card}
                {weaknesses_card}
                {suggestions_card}
            </div>
        </div>
        """
    )


inject_app_css()
render_hero()

if "extracted_features" not in st.session_state:
    st.session_state.extracted_features = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- Resume upload ---
section_header(1, "Upload Resume")
with st.container(border=True):
    uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])

if uploaded_file is None:
    reset_extracted_features()
else:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.extracted_features = None
        st.session_state.analysis_results = None

    if st.session_state.extracted_features is None:
        with st.spinner("Extracting text and analyzing resume with Gemini..."):
            try:
                st.session_state.extracted_features = extract_ml_features_upload(
                    uploaded_file.getvalue()
                )
            except EnvironmentError as exc:
                st.error(str(exc))
                st.info("Set your API key: `export GEMINI_API_KEY='your-key'`")
            except Exception as exc:
                st.error(f"Resume analysis failed: {exc}")

extracted = st.session_state.extracted_features

# --- Extracted features ---
section_header(2, "Features Extracted from Resume")
with st.container(border=True):
    if extracted is None:
        st.info("Upload a resume PDF to extract CGPA, projects, certifications, and activities.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            default_cgpa = extracted["cgpa"] if extracted["cgpa"] is not None else 0.0
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=float(default_cgpa),
                step=0.1,
                help="Extracted by Gemini. Adjust if the resume value was missed or incorrect.",
            )
            projects = st.number_input(
                "Projects Count",
                min_value=0,
                max_value=50,
                value=int(extracted["projects_count"]),
                step=1,
            )

        with col2:
            workshops_certifications = st.number_input(
                "Workshops / Certifications Count",
                min_value=0,
                max_value=50,
                value=int(extracted["workshops_certifications_count"]),
                step=1,
            )
            extracurricular = st.selectbox(
                "Extracurricular Activities",
                options=["Yes", "No"],
                index=0 if extracted["extracurricular_activities"] == "Yes" else 1,
            )

        if extracted["cgpa"] is None:
            st.warning("CGPA was not found in the resume. Please enter it manually above.")

# --- Manual inputs ---
section_header(3, "Additional Details")
with st.container(border=True):
    render_html(
        '<p class="panel-hint">Provide values that are usually not available from a resume.</p>'
    )

    manual_col1, manual_col2 = st.columns(2)

    with manual_col1:
        internships = st.number_input("Internships", min_value=0, max_value=20, value=0, step=1)
        aptitude_score = st.number_input(
            "Aptitude Test Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
        placement_training = st.selectbox("Placement Training", options=["No", "Yes"])

    with manual_col2:
        ssc_marks = st.number_input("SSC Marks", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        hsc_marks = st.number_input("HSC Marks", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# --- Soft skills questionnaire ---
section_header(4, "Soft Skills Questionnaire")
with st.container(border=True):
    render_html(
        '<p class="panel-hint">Rate each skill from 1 (low) to 5 (high). '
        "Soft Skills Rating is the average.</p>"
    )

    skills_col1, skills_col2 = st.columns(2)

    with skills_col1:
        communication = st.slider("Communication Confidence", min_value=1, max_value=5, value=3)
        presentation = st.slider("Presentation Skills", min_value=1, max_value=5, value=3)

    with skills_col2:
        teamwork = st.slider("Teamwork", min_value=1, max_value=5, value=3)
        leadership = st.slider("Leadership", min_value=1, max_value=5, value=3)

    soft_skills_rating = (communication + presentation + teamwork + leadership) / 4
    st.metric("Soft Skills Rating (calculated)", f"{soft_skills_rating:.2f}")

# --- Analyze ---
section_header(5, "Analyze")
analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

if analyze_clicked:
    if extracted is None:
        st.error("Upload and process a resume PDF before analyzing.")
    elif cgpa <= 0:
        st.error("Enter a valid CGPA greater than 0.")
    else:
        features = build_feature_row(
            cgpa=cgpa,
            internships=int(internships),
            projects=int(projects),
            workshops_certifications=int(workshops_certifications),
            aptitude_test_score=float(aptitude_score),
            soft_skills_rating=soft_skills_rating,
            extracurricular_activities=extracurricular,
            placement_training=placement_training,
            ssc_marks=float(ssc_marks),
            hsc_marks=float(hsc_marks),
        )

        try:
            placement_probability = predict_placement_probability(features)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            resume_analysis: dict[str, list[str]] = {
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
            }
            with st.spinner("Generating resume insights with Gemini..."):
                try:
                    resume_analysis = analyze_resume_upload(uploaded_file.getvalue())
                except EnvironmentError as exc:
                    st.warning(f"Resume insights unavailable: {exc}")
                except Exception as exc:
                    st.warning(f"Resume insight generation failed: {exc}")

            st.session_state.analysis_results = {
                "placement_probability": placement_probability,
                "strengths": resume_analysis.get("strengths", []),
                "weaknesses": resume_analysis.get("weaknesses", []),
                "suggestions": resume_analysis.get("suggestions", []),
            }

if st.session_state.analysis_results is not None:
    render_results_section(st.session_state.analysis_results)
