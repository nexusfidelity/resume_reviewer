from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3:8b",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

job_posting = """Position: Data Engineer (Databricks Unified Data Analytics Platform)

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
Azure Data Factory"""

resume = """
Work Experience * Al/ML Engineer (Hitachi Payment Solutions) April 2026–present Creating and Improving Al/ML solutions with regards to secure payments of clients * Data Scientist (BSP) Aug 2025–Feb 2026 Developed Al use cases to help the Central Bank with bank supervision and business Fidel lvan C. Racines processes ML / AI Engineer * AI Software Developer(AMA university) May 2024–Nov 2024 Profession Profile Developed Al softwares to increase productivity Al and ML developer with 4+ years in technical in the University and help students upskill with and business research through data and model the help of AI. creation. Helped increase value in the company * AI Engineer (DOST FNRI) through state of the art research and development. May 2023–April 2024 Developed projects with Al/ML components to combat malnutrition in the Philippines. Contact Info * SEO Specialist_(ThinkLogic Media Group) Linkedln: fidel-ivan-racines Oct 2021–May 2022 Github: Link Successful migrated SEO metrics to new Portfolio: Link Domain. Education Technical Skills Master of Science in Machine Learning & Computer Vision – Tensorflow, Pytorch, Keras, Artificial Intelligence OpenCV, CNN, RNN Liverpool John Moores University (UK) ML Models – Linear, Logistic, Ridge, & Lasso April 2022 – April 2024 Regression, Nave Bayes, SVM, Decision Trees, Random Forest, Gradient Boosting Stats – Probability, Central Limit Theorem, Bachelor of Science in Computer Science Hypothesis Testing, Critical Value Method, Xavier University (Philippines) P-Value Method June 2015 – March 2020 Data Vis – Pie Charts, Line Charts, Bar Charts Histogram, Scatter Plots, Time Series Plots, Geo Spatial Visualization Certifications Cloud –AWS, Azure, Docker, IOT hub, : Executive Post Graduate Programme in ML Pyspark, DataBricks and Al (certificate) ML Ops – MLflow, Pycaret, Airflow, Colab . Data Science Fellowship (certificate) LLMs - Chat-gpt, LLama, Azure Al foundry, : 365 Data Science Program (certificate) AWS Bedrock, Langchain, Llama Index, RAG, . Deep Learning with Pytorch for Medical Fine-tuning Image Analysis (certificate) Web Apps - Streamlit, Flask, Quart Projects Philippine AI Lawyer The Philippine Al lawyer is an LLM chatbot trained using the knowledge base of Philippines laws. Its answers are tailored in the context of Philippine law and Supreme court decisions. Predicting malnutrition in children below 5 years using Artificial Intelligence This study leverages machine learning to predict malnutrition in children under five by analyzing a diverse set of variables, including anthropometric data, socioeconomic factors, dietary habits, and breastfeeding practices from the DOST-FNRl. By accurately classifying malnutrition risks, these models offer a powerful tool for early diagnosis, enabling healthcare providers, policymakers, and community organizations to implement timely, targeted interventions. Early predictions allow for effective allocation of resources to areas most in need, ultimately helping to reduce malnutrition's prevalence and its negative impacts on health, productivity, and poverty. Paper Chest X-ray Effusion This project allows us to detect if there is effusion in the chest x-ray. Chest X-ray exams are one of the most frequent and cost-effective medical imaging examinations available. However, clinical diagnosis of a chest X-ray can be challenging and sometimes more difficult than diagnosis via chest CT imaging. Australian Housing Market This Case Study explores the Australian Housing Market and what factors are affecting the price of homes in Australia Bike Sharing App This project aims to understand which factors are affecting the usage of a mobile app (bike sharing) at a time of a day. Bike Buyer Prediction Using the Australian dataset for bike buying, this study aims to predict if a certain individual according to his profile will most likely buy a bike. Case Study Lending Club This Case Study aims to understand what factors are affecting why applicants are likely to default on their loans. Case Study Credit Fraud This Case Study aims to understand what variables are more likely to contribute to a transaction becoming a credit fraud
"""

prompt="i have this job posting " + job_posting + "and i have this resume " + resume + "how would you rate the candidate from 0 to 100?"

resp = llm.complete(prompt)

print(resp)