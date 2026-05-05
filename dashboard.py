import streamlit as st
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from llama_index.llms.ollama import Ollama

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import logging
logging.getLogger("easyocr").setLevel(logging.ERROR)

resp=None

if 'key' not in st.session_state:
    st.session_state['key'] = []

resume_text=[]

@st.cache_resource
def get_reader():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def get_llm(model_name):
    return Ollama(model=model_name, request_timeout=240, context_window=1024) #2048

# reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(layout="wide")

with st.sidebar:
    llm_model = st.selectbox(
    "LLM model",
    ("llama3.2:1b", "llama3:8b", "gemma:2b","deepseek-r1:1.5b"),
    index = 0
    )
    
    llm = get_llm(llm_model)
    
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

Key points:
+ [One-liner strength #1 — cite a specific factor above]
+ [One-liner strength #2 — cite a specific factor above]
– [One-liner gap or red flag — cite a specific factor above]
""",
        height="content"
        )
   
    job_posting = st.text_area(
   "Job Posting text",
   """Data Engineer (Databricks Unified Data Analytics Platform)
   
   Shift Schedule: Mid Shift (2PM – 12AM)
   
   Work Set up: Hybrid, 2 days per week
   
   Location: Taguig, Philippines
   
   Summary:
   
   As a Data Engineer, you will design, develop, and maintain data solutions that facilitate data generation, collection, and processing. Your typical day will involve creating data pipelines, ensuring data quality, and implementing ETL processes to migrate and deploy data across various systems. You will collaborate with cross-functional teams to enhance data accessibility and usability, contributing to the overall data strategy of the organization.
   
   Roles & Responsibilities:
   
   - Expected to be an SME.
   
   - Collaborate and manage the team to perform.
   
   - Responsible for team decisions.
   
   - Engage with multiple teams and contribute on key decisions.
   
   - Provide solutions to problems for their immediate team and across multiple teams.
   
   - Develop and optimize data pipelines to ensure efficient data flow and processing.
   
   - Monitor and troubleshoot data quality issues, implementing corrective actions as necessary.
   
   - Document data processes and workflows to ensure clarity and compliance with best practices.
   
   Professional & Technical Skills:
   
   - Required Skill: Expert proficiency in Databricks Unified Data Analytics Platform.
   
   - Additional Good To Have Skills: Experience with Python (Programming Language).
   
   - Strong understanding of data modeling and database design principles.
   
   - Experience with ETL tools and data integration techniques.
   
   - Familiarity with cloud platforms and services related to data engineering.
   
   - Proficient in data warehousing concepts and practices.
   
   Additional Information:
   
   - The candidate should have minimum 5 years of experience in Databricks Unified Data Analytics Platform.
   
   - This position is based at our Manila office.
   
   Must have and Good to have skills:
   
   Databricks (Unity Catalog, Delta Live Tables, Auto Loader)
   Python, Pyspark
   SQL
   Azure Data Factory""",
   
   height="stretch"
   
   )

col1, col2 = st.columns([1,2])


with col1:
    st.header("Candidates")
    uploaded_files = st.file_uploader(
    "Upload data", accept_multiple_files=True, type="pdf"
    )
    
    if st.button("OCR", type="primary"):
        st.session_state['key']=[]
        for file in uploaded_files:
            pages = convert_from_bytes(file.read(), dpi=150)
            
            file_texts = []
            for i, page in enumerate(pages, start=1):
                
                img = np.array(page)
                ocr_text = get_reader().readtext(img, detail=0)
                text = " ".join(ocr_text)
                file_texts.append(text)
                del img, page
            
            combined_text = " ".join(file_texts)
            st.session_state['key'].append(combined_text)
        
    st.write(st.session_state['key'])

with col2:
    st.header("Ranking")
    
    if st.button("LLM", type="primary"):
        
        for x, candidate_data in enumerate(st.session_state['key']):
        
            prompt = f"""
Job posting:
{job_posting}

Candidate:
{candidate_data}

{user_prompt}
            """
            
            resp = llm.complete(prompt)
            # resp2 = llm.complete(prompt_2)
            st.subheader("candidate "+ str(x+1))
            st.write(resp.text)
            # st.write("""
            #          0-25% = very not fit
            #          \n 25-50% = not fit
            #          \n 50-75% = fit
            #          \n 75-100% = very fit
            #          """)
            # st.write(resp2.text)
    
