# Interview Performance Analyzer

An AI-powered Placement Prediction and Resume Analysis System built using Machine Learning, Streamlit, and Google Gemini AI.

## Project Overview

Interview Performance Analyzer helps students evaluate their placement readiness by combining Machine Learning predictions with AI-based resume analysis.

The system extracts information from a student's resume, collects additional placement-related details, predicts the probability of placement, and provides personalized feedback on strengths, weaknesses, and improvement areas.

---

## Features

### Placement Prediction
- Predicts placement probability using a trained Machine Learning model.
- Calculates placement readiness based on academic and skill-related factors.

### Resume Upload
- Upload resume in PDF format.
- Automatic text extraction from resume.

### AI-Based Feature Extraction
Google Gemini AI extracts:
- CGPA
- Project Count
- Certifications / Workshops
- Extracurricular Activities

from the uploaded resume.

### Additional Inputs
The user provides information that is generally not available in resumes:
- Internships
- Aptitude Test Score
- SSC Marks
- HSC Marks
- Placement Training Status

### Soft Skills Assessment
A questionnaire evaluates:
- Communication Skills
- Teamwork
- Presentation Skills
- Leadership

The average score is used as the Soft Skills Rating.

### Resume Analysis
Google Gemini analyzes the resume and provides:
- Strengths
- Weaknesses
- Suggestions for Improvement

---

## Machine Learning Details

### Dataset
Placement Dataset containing 10,000 student records.

### Features Used

| Feature |
|----------|
| CGPA |
| Internships |
| Projects |
| Workshops/Certifications |
| Aptitude Test Score |
| Soft Skills Rating |
| Extracurricular Activities |
| Placement Training |
| SSC Marks |
| HSC Marks |

### Target Variable

Placement Status

- Placed
- Not Placed

---

## Models Evaluated

### Logistic Regression
Accuracy: **79.45%**

### Decision Tree
Accuracy: **72.65%**

### Random Forest
Accuracy: **78.25%**

### Final Model Selected

**Logistic Regression**

Reason:
- Highest test accuracy
- Better generalization
- Lower overfitting compared to Random Forest

---

## Technology Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-Learn
- Pandas
- NumPy

### Generative AI
- Google Gemini API

### Version Control
- Git
- GitHub

---

## System Workflow

1. Upload Resume PDF
2. Extract Resume Text
3. Gemini Extracts Placement Features
4. User Enters Remaining Information
5. Soft Skills Questionnaire Completed
6. Placement Probability Predicted
7. Resume Analyzed Using Gemini
8. Results Displayed

---

## Project Structure

```text
Interview-Performance-Analyzer/
│
├── app.py
├── gemini_analyzer.py
├── ml_predictor.py
├── resume_analysis.py
├── resume_parser.py
├── placement_model.pkl
├── placementdata.csv
├── requirements.txt
├── test_resume_analysis.py
└── samples/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Rk346278/Interview-Performance-Analyzer.git
cd Interview-Performance-Analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set Gemini API Key:

### Mac/Linux

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

### Windows

```bash
set GEMINI_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```

---

## Sample Output

### Placement Prediction

- Placement Probability: 90.5%
- Placement Readiness Status: High

### Resume Analysis

#### Strengths
- Strong academic record
- Relevant AI/ML projects
- Good programming foundation

#### Weaknesses
- Limited internship experience
- Lack of quantified project impact

#### Suggestions
- Add measurable project outcomes
- Gain industry experience through internships
- Highlight leadership activities

---

## Future Enhancements

- Placement Readiness Dashboard
- Skill Gap Visualization
- Role Recommendation System
- Interview Performance Tracking
- Resume Version Comparison

---

## Learning Outcomes

Through this project:

- Built and evaluated multiple Machine Learning models.
- Performed feature engineering and model selection.
- Integrated Generative AI with traditional Machine Learning.
- Developed an end-to-end Streamlit application.
- Learned Git and GitHub workflow.
- Implemented resume parsing and analysis.

---

## Author

**Ritesh Venkatesh Kulkarni**

B.E. Artificial Intelligence & Machine Learning  
New Horizon College of Engineering, Bangalore

GitHub: https://github.com/Rk346278

---
