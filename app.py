"""
AI Resume Screener Streamlit Application.
This module handles the frontend UI for uploading, analyzing, and querying resumes.
"""
# import os
import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorestore, run_self_query

st.set_page_config(page_title="AI Resume Screener")
st.title("AI Resume Screener")
st.markdown("Uploade resume in (pdf/docx/txt) form ,App uses AI to screen resumes based on job descriptions & analyzes the resume.")

job_desc = st.text_area("Enter or paste Job Description")
uploaded_file = st.file_uploader(
    "@ Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze & Store Resume") and uploaded_file and job_desc:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Loading and analyzing resume..."):
        docs = load_resume(uploaded_file.name)
        analysis = analyze_resume(docs, job_desc)

        # Store to vector store
        text_chunks = docs
        vectordb = store_to_vectorestore(text_chunks)
        st.success("☑ Resume stored in vector store.")

        st.subheader("AI Resume Summary & Analysis")
        st.write(analysis)
        st.download_button("Download Analysis", data=analysis,
                           file_name="resume_analysis.txt", mime="text/plain")

st.divider()
st.subheader("Ask anything about stored resumes")
query = st.text_input(
    "Enter your query to fetch relevant information e.g Python developer with AWS")

if st.button("Search_resume") and query:
    with st.spinner("Running self-query on the stored resume..."):
        query_result = run_self_query(query)
        if query_result:
            for i, res in enumerate(query_result, 1):
                st.markdown(f"**Result {i}:****")
                st.write(res.page_content.strip())
        else:
            st.write("No relevant information found for the given query.")
