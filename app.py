import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# Custom styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stTextInput, .stTextArea, .stFileUploader {
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stDataFrame {
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Ensure text is not None
            text += extracted_text
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description with resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Sidebar for input
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1946/1946488.png", width=100)
st.sidebar.title("AI Resume Screener")
st.sidebar.write("Upload resumes and enter job description to rank candidates.")

# Job description input
st.sidebar.header("Enter Job Description")
job_description = st.sidebar.text_area("Paste the job description here", height=150)

# File uploader
st.sidebar.header("Upload Resumes (PDF)")
uploaded_files = st.sidebar.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

# Main section
st.header("📄 Candidate Ranking System")

if uploaded_files and job_description:
    st.subheader("Processing Resumes...")
    
    resumes = []
    progress_bar = st.progress(0)  # Progress bar
    
    for idx, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        resumes.append(text)
        time.sleep(0.5)  # Simulate processing delay
        progress_bar.progress((idx + 1) / len(uploaded_files))  # Update progress bar

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create DataFrame
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Display results with formatting
    st.subheader("📊 Ranked Resumes")
    st.dataframe(results.style.background_gradient(cmap="Blues"))

    # Download button for results
    csv_data = results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Ranked Resumes", csv_data, "ranked_resumes.csv", "text/csv", key="download-csv")

else:
    st.info("Please upload resumes and enter a job description to proceed.")
