# app.py
import re
import json
from io import StringIO
from collections import Counter
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

st.set_page_config(page_title="Resume Analyzer", layout="wide")
# ---------------------------
# Helper data (Improved)
# ---------------------------
SKILLS = [
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "r", "go", "swift", "kotlin",
    
    # Web Development
    "html", "css", "bootstrap", "tailwind", "react", "nextjs", "vue", "angular", 
    "node", "express", "flask", "django", "fastapi", "rest", "graphql", "api",

    # Databases
    "mysql", "postgresql", "sqlite", "mongodb", "redis", "firebase",

    # Data Science / ML / AI
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
    "xgboost", "lightgbm", "catboost", "data analysis", "data cleaning", "feature engineering",
    "statistics", "machine learning", "deep learning", "ai", "computer vision", "nlp",
    "transformer", "bert", "huggingface", "opencv", "matplotlib", "seaborn", "plotly",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "linux",

    # Cybersecurity / Networks
    "networking", "firewall", "cybersecurity", "ethical hacking", "penetration testing",

    # Tools & Version Control
    "git", "github", "gitlab", "jira", "slack", "vs code", "intellij", "jupyter",

    # Other emerging tech
    "blockchain", "solidity", "web3", "iot", "arduino", "raspberry pi",

    # Soft / general tech terms
    "agile", "scrum", "teamwork", "communication", "problem solving"
]


ROLE_TEMPLATES = [
    # --- Data / AI / ML Roles ---
    {"id": "data_scientist", "title": "Data Scientist",
     "keywords": ["python", "pandas", "numpy", "scikit-learn", "data", "analysis", "ml", "model", "statistics", "sql"]},

    {"id": "ml_engineer", "title": "Machine Learning Engineer",
     "keywords": ["machine learning", "deep learning", "tensorflow", "pytorch", "keras", "deployment", "mlops", "model", "pipeline"]},

    {"id": "data_analyst", "title": "Data Analyst",
     "keywords": ["excel", "sql", "tableau", "powerbi", "pandas", "matplotlib", "data analysis", "visualization"]},

    {"id": "nlp_engineer", "title": "NLP Engineer",
     "keywords": ["nlp", "transformer", "bert", "huggingface", "tokenization", "text classification", "language model"]},

    {"id": "cv_engineer", "title": "Computer Vision Engineer",
     "keywords": ["opencv", "cnn", "image processing", "yolo", "object detection", "pytorch", "tensorflow"]},

    # --- Development Roles ---
    {"id": "frontend_dev", "title": "Frontend Developer",
     "keywords": ["javascript", "react", "html", "css", "tailwind", "ui", "ux", "frontend", "nextjs"]},

    {"id": "backend_dev", "title": "Backend Developer",
     "keywords": ["node", "express", "flask", "django", "api", "sql", "mongodb", "backend"]},

    {"id": "fullstack_dev", "title": "Full Stack Developer",
     "keywords": ["react", "node", "express", "sql", "api", "frontend", "backend", "javascript", "mongodb"]},

    # --- Cloud / DevOps ---
    {"id": "devops_engineer", "title": "DevOps Engineer",
     "keywords": ["docker", "kubernetes", "jenkins", "aws", "ci/cd", "terraform", "pipeline", "cloud", "linux"]},

    {"id": "cloud_engineer", "title": "Cloud Engineer",
     "keywords": ["aws", "azure", "gcp", "cloud", "infrastructure", "deployment", "lambda", "s3", "devops"]},

    # --- Cybersecurity ---
    {"id": "cybersecurity_analyst", "title": "Cybersecurity Analyst",
     "keywords": ["cybersecurity", "ethical hacking", "penetration testing", "firewall", "security", "networking"]},

    # --- Emerging Tech ---
    {"id": "blockchain_dev", "title": "Blockchain Developer",
     "keywords": ["blockchain", "solidity", "web3", "smart contract", "ethereum"]},

    {"id": "iot_engineer", "title": "IoT Engineer",
     "keywords": ["iot", "arduino", "raspberry pi", "embedded", "sensors", "microcontroller"]},
]


# ---------------------------
# Text extraction helpers
# ---------------------------
def read_pdf(file_bytes) -> str:
    try:
        reader = PdfReader(file_bytes)
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    else:
        # treat as text
        try:
            raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            return raw
        except Exception:
            return ""

# ---------------------------
# Parsing & heuristics
# ---------------------------
def normalize_text(t: str) -> str:
    return re.sub(r"[^a-z0-9\-\s\.]", " ", t.lower())

def find_skills(text: str, skills_list: List[str]) -> List[str]:
    norm = normalize_text(text)
    found = set()
    for skill in skills_list:
        # match word boundaries or hyphenated terms
        pat = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pat, norm):
            found.add(skill)
    return sorted(found, key=lambda s: skills_list.index(s) if s in skills_list else 999)

def estimate_experience_years(text: str) -> float:
    """
    Heuristic: look for patterns like 'X years', 'X+ years', 'since 2019', '2018-2022'
    Returns approximate years as float (0 if none)
    """
    norm = text.lower()
    # look for '(\d+(\.\d+)?)\s*(\+)?\s*years'
    m = re.findall(r"(\d+(?:\.\d+)?)\s*\+\s*years|\b(\d+(?:\.\d+)?)\s*years\b", norm)
    if m:
        # m is list of tuples: pick first non-empty group
        for tup in m:
            val = next((x for x in tup if x and x.strip()), None)
            if val:
                try:
                    return float(val)
                except:
                    pass
    # look for date ranges like 2018-2022
    yrs = re.findall(r"\b(19|20)\d{2}\b", norm)
    if yrs:
        years = sorted({int(y) for y in yrs})
        if len(years) >= 2:
            span = years[-1] - years[0]
            if span >= 0 and span < 50:
                return float(span)
    # look for 'since 2019'
    m2 = re.search(r"since\s+(19|20)\d{2}", norm)
    if m2:
        try:
            start = int(m2.group(0).split()[-1])
            import datetime
            return float(datetime.datetime.now().year - start)
        except:
            pass
    return 0.0

def score_role(text: str, role_keywords: List[str], found_skills: List[str], exp_years: float) -> Tuple[float, int]:
    """
    Returns (score_0to100, matched_keyword_count)
    Scoring idea:
      - base score: overlap ratio between role keywords and found_skills (weighted)
      - + small bonus for experience
      - clamp 0-100
    """
    norm_text = normalize_text(text)
    kw_lower = [k.lower() for k in role_keywords]
    match_count = 0
    for kw in kw_lower:
        if re.search(r"\b" + re.escape(kw) + r"\b", norm_text):
            match_count += 1
    # skill overlap (normalized)
    overlap = match_count / max(len(kw_lower), 1)
    # experience bonus: linear 0-20 for 0-10 years
    exp_bonus = min(exp_years, 10) * 2
    raw = overlap * 80 + exp_bonus  # weight overlap more
    score = max(0, min(100, round(raw, 1)))
    return score, match_count

# ---------------------------
# Suggestions
# ---------------------------
def generate_suggestions(found_skills: List[str], exp_years: float, text: str) -> List[str]:
    suggestions = []
    # Suggest adding missing commonly-seen skills
    if len(found_skills) < 4:
        suggestions.append("Add specific technical skills (tools, frameworks). Try listing 4-6 core skills at top.")
    if exp_years < 1:
        suggestions.append("If you have projects or internships, highlight them with bullets and results (metrics).")
    if "github" not in text.lower() and "gitlab" not in text.lower():
        suggestions.append("Add links: GitHub / portfolio / LinkedIn if available.")
    # formatting suggestions
    suggestions.append("Use bullet points, quantify achievements (e.g., improved X by Y%), and keep role/years clear.")
    return suggestions

# ---------------------------
# UI
# ---------------------------
st.title("🪶 Resume Analyzer")
st.markdown("Upload a resume (`.pdf` or `.txt`) or paste resume text. This is a **dummy/demo** analyzer using heuristics (keyword matching). Replace heuristics with an LLM/ML model for production.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload resume (.pdf or .txt)", type=["pdf", "txt"])
    st.write("OR paste resume text below (paste preferred for quick demo):")
    paste = st.text_area("Paste resume text", height=250)
    if st.button("Analyze"):
        # choose content
        if uploaded:
            content = extract_text_from_upload(uploaded) or paste or ""
        else:
            content = paste or ""
        if not content.strip():
            st.error("Please paste resume text or upload a file.")
            st.stop()

        # perform analysis
        skills = find_skills(content, SKILLS)
        exp = estimate_experience_years(content)
        role_results = []
        for r in ROLE_TEMPLATES:
            sc, mcount = score_role(content, r["keywords"], skills, exp)
            role_results.append({
                "id": r["id"],
                "title": r["title"],
                "score": sc,
                "matched_keywords": mcount,
                "role_keywords_sample": r["keywords"][:6]
            })
        # sort top by score
        role_results_sorted = sorted(role_results, key=lambda x: x["score"], reverse=True)[:3]
        suggestions = generate_suggestions(skills, exp, content)

        analysis = {
            "skills": skills,
            "experience_years_estimate": round(exp, 1),
            "top_roles": role_results_sorted,
            "suggestions": suggestions,
            "raw_text_snippet": content[:2000]
        }
        st.session_state["analysis"] = analysis
        st.success("Analysis complete!")

with col2:
    st.header("Preview & Quick Analysis")
    if "analysis" in st.session_state:
        a = st.session_state["analysis"]
        st.subheader("Extracted skills")
        if a["skills"]:
            st.write(", ".join(a["skills"]))
        else:
            st.write("_No known skills detected from curated list._")
        st.write(f"Estimated experience: **{a['experience_years_estimate']} years**")
        st.subheader("Top role matches")
        df = pd.DataFrame([{"Role": r["title"], "Fit %": f"{r['score']:.2f}", "Matched keywords": r["matched_keywords"]}for r in a["top_roles"]])

        st.table(df.head(6))
        st.subheader("Suggestions")
        for s in a["suggestions"]:
            st.write("•", s)

        # export options
        st.markdown("---")
        st.subheader("Export analysis")
        json_str = json.dumps(a, indent=2)
        st.download_button("Download JSON", json_str, file_name="resume_analysis.json", mime="application/json")

        # CSV summary for roles
        role_df = pd.DataFrame([{"role_id": r["id"], "role": r["title"], "score": r["score"], "matched_keywords": r["matched_keywords"]} for r in a["top_roles"]])
        csv = role_df.to_csv(index=False)
        st.download_button("Download Roles CSV", csv, file_name="resume_roles.csv", mime="text/csv")
    else:
        st.info("No analysis yet. Paste text or upload a file and press Analyze.")

st.markdown("---")
st.caption("This is a demo. For stronger results replace heuristics with an LLM prompt call or a trained classifier, and extend skill list & role templates.")
