import streamlit as st
from pdf2image import convert_from_bytes
import numpy as np
import easyocr
from llama_index.llms.ollama import Ollama

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

resp=None

if 'key' not in st.session_state:
    st.session_state['key'] = []

resume_text=[]

reader = easyocr.Reader(['en'])

llm = Ollama(
    # model="llama3.2:1b",
    model="llama3:8b",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Job Posting")
    job_posting = st.text_area(
    "Job posting text",
    """Position: Data Engineer (Databricks Unified Data Analytics Platform)
    
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
    
    height="content"
    
    )

with col2:
    st.header("Candidates")
    uploaded_files = st.file_uploader(
    "Upload data", accept_multiple_files=True, type="pdf"
    )
    
    if st.button("OCR", type="primary"):
        
        st.session_state['key'] = []
        
        for file in uploaded_files:
            pages = convert_from_bytes(file.read(), dpi=150)

            for i, page in enumerate(pages, start=1):

                img = np.array(page)
                text = reader.readtext(img, detail=0)
                text = " ".join(text)
                
            st.session_state['key'].append(text)
            
    st.write(st.session_state['key'])

with col3:
    st.header("Ranking")
    
    if st.button("LLM", type="primary"):
        
        for x, candidate_data in enumerate(st.session_state['key']):
        
            prompt="i have this job posting: " + job_posting + "\nand i have this resume: " + candidate_data + "\nhow qualified is the candidate? rate from 0 to 100. [output only the percentage rating and 1 paragraph reasoning to hire or not to hire]"
            #reasons why you should get this person or not
    
            resp = llm.complete(prompt)
            st.subheader("candidate "+ str(x))
            st.write(resp.text)
    
