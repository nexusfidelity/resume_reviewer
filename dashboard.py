import re
import streamlit as st
from pdf2image import convert_from_bytes
import numpy as np
import easyocr

from dotenv import load_dotenv
load_dotenv()
import os
from ollama import Client

import logging

# Disable noisy logs
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ.get("OLLAMA_API_KEY")

logging.getLogger("easyocr").setLevel(logging.ERROR)

# ----------------------
# CACHED RESOURCES
# ----------------------
@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def get_llm_client():
    return Client(
        host="https://ollama.com",
        headers={
            "Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")
        }
    )

# ----------------------
# HELPERS
# ----------------------
def parse_fit_score(response_text: str) -> int:
    """Extract the numeric fit percentage from the LLM response.
    Looks for patterns like 'Overall fit: 82%' or 'Overall fit: 82% —'
    Returns -1 if no score found.
    """
    match = re.search(r"Overall fit:\s*(\d{1,3})\s*%", response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1

# ----------------------
# SESSION STATE
# ----------------------
if 'candidates' not in st.session_state:
    st.session_state['candidates'] = []      # list of raw OCR text strings

if 'evaluations' not in st.session_state:
    st.session_state['evaluations'] = []     # list of dicts: {index, score, response}

if 'filenames' not in st.session_state:
    st.session_state['filenames'] = []       # parallel list of uploaded file names

# ----------------------
# UI CONFIG
# ----------------------
st.set_page_config(layout="wide")

# ----------------------
# SIDEBAR
# ----------------------
with st.sidebar:
    llm_model = st.selectbox(
        "LLM model",
        ("gpt-oss:20b","gpt-oss:120b","gemma3:4b-cloud","glm-4.6:cloud","minimax-m2.1:cloud"),
        index=0
    )

    user_prompt = st.text_area(
        "Prompt",
        """You are a senior HR recruiter with 10+ years of experience screening candidates.
Evaluate how well this candidate fits the job posting below.

When evaluating, consider these specific factors:

HARD FACTORS:
- Job title alignment: Did they hold a similar or relevant role?
- Years of experience: Do they meet the minimum requirement?
- Industry background: Same or adjacent industry?
- Skills & tools: Exact or near matches to what the job requires?
- Certifications & licenses: Are mandatory ones present?
- Education: Does it meet the stated requirement?

SOFT/BEHAVIORAL SIGNALS:
- Quantified achievements: Do they show numbers, percentages, revenue, team size?
- Promotion history: Were they promoted within the same company?
- Tenure: Did they stay long enough to create impact (2+ years typical)?
- Career progression: Are roles getting bigger and more senior over time?
- Job hopping: Flag if more than 3 jobs in 3 years
- Employment gaps: Flag unexplained gaps longer than 6 months
- Recency: How relevant is their most recent role?

PRESENTATION SIGNALS:
- Tailoring: Does the CV feel written for this role or is it generic?
- Clarity: Is it easy to skim and well-structured?

Fit bands:
0–25%   = Very not fit
25–50%  = Not fit
50–75%  = Fit
75–100% = Very fit

Output format (strictly follow this, nothing else):

Overall fit: [0–100%] — [Very not fit / Not fit / Fit / Very fit]

provide bullet points as to why it is within the fitband based on the score
""",
        height=400
    )

    job_posting = st.text_area(
        "Job Posting text",
        """Data Engineer (Databricks Unified Data Analytics Platform)

Location: Taguig, Philippines

Requirements:
- 5+ years Databricks
- Python, PySpark, SQL
- Azure Data Factory
""",
        height=300
    )

# ----------------------
# LAYOUT
# ----------------------
col1, col2 = st.columns([1, 2])

# ----------------------
# LEFT: OCR
# ----------------------
with col1:
    st.header("Candidates")

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF)",
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Run OCR", type="primary"):
        st.session_state['candidates'] = []
        st.session_state['filenames'] = []
        st.session_state['evaluations'] = []   # clear stale results on re-OCR

        for file in uploaded_files:
            pages = convert_from_bytes(file.read(), dpi=150)
            file_texts = []

            for page in pages:
                img = np.array(page)
                text = " ".join(get_reader().readtext(img, detail=0))
                file_texts.append(text)

            combined_text = " ".join(file_texts)
            st.session_state['candidates'].append(combined_text)
            st.session_state['filenames'].append(file.name)

    st.write(st.session_state['candidates'])

# ----------------------
# RIGHT: Evaluate & Rank
# ----------------------
with col2:
    st.header("Ranking")

    if st.button("Evaluate with LLM", type="primary"):
        client = get_llm_client()
        candidates = st.session_state['candidates']
        filenames  = st.session_state['filenames']

        collected = []   # accumulate before sorting

        with st.spinner("Evaluating candidates…"):
            for idx, candidate_data in enumerate(candidates):
                label = filenames[idx] if idx < len(filenames) else f"Candidate {idx + 1}"

                prompt = f"""
Job posting:
{job_posting}

Candidate:
{candidate_data}

{user_prompt}
"""
                full_response = ""

                for part in client.chat(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                ):
                    full_response += part['message']['content']

                score = parse_fit_score(full_response)
                collected.append({
                    "index":    idx,
                    "label":    label,
                    "score":    score,
                    "response": full_response,
                })

        # --- Sort highest → lowest ---
        collected.sort(key=lambda x: x["score"], reverse=True)
        st.session_state['evaluations'] = collected

    # --- Display ranked results (persists across reruns) ---
    if st.session_state['evaluations']:
        st.subheader("Results — highest to lowest fit")

        for rank, ev in enumerate(st.session_state['evaluations'], start=1):
            score_display = f"{ev['score']}%" if ev['score'] >= 0 else "N/A"

            # Colour-code the badge by fit band
            if ev['score'] >= 75:
                badge = f"🟢 {score_display}"
            elif ev['score'] >= 50:
                badge = f"🟡 {score_display}"
            elif ev['score'] >= 25:
                badge = f"🟠 {score_display}"
            else:
                badge = f"🔴 {score_display}" if ev['score'] >= 0 else "⚪ N/A"

            with st.expander(f"#{rank} — {ev['label']}  {badge}", expanded=(rank == 1)):
                st.markdown(ev['response'])