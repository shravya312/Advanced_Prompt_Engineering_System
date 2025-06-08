import os
import streamlit as st
from prompt_engineering_system.main import PromptEngineeringSystem
import tempfile
import shutil
import re

# ===== Load Environment Variables =====
# Handled by SpecializedApplications now

# ===== Configuration =====
# Handled by SpecializedApplications now

# Constants (moved to applications.py or removed if not needed in app.py)
# COLLECTION_NAME = "document_collection"
# VECTOR_SIZE = 384
# CHUNK_SIZE = 1000

# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("Advanced Prompt Engineering System")

system = PromptEngineeringSystem()

# Initialize session state for PDF processing for RAG
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'context_chunks' not in st.session_state:
    st.session_state.context_chunks = []
if 'text' not in st.session_state:
    st.session_state.text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

task = st.sidebar.selectbox("Select Task", [
    "summarization",
    "code_generation",
    "data_extraction",
    "question_answering"
])

reasoning_type = st.sidebar.selectbox("Select Reasoning Type", [
    "standard",
    "chain_of_thought",
    "tree_of_thought"
])

# PDF Upload Section - now in sidebar as it's optional per task
pdf_file = st.sidebar.file_uploader("Upload a PDF file (for Data Extraction/Q&A)", type=["pdf"])

if pdf_file is not None:
    if st.session_state.last_uploaded_filename != pdf_file.name:
        st.session_state.processed_pdf = False # Reset processing status for new file
        st.session_state.pdf_path = None
        st.session_state.context_chunks = []
        st.session_state.text = None
        st.session_state.chunks = []
        st.session_state.last_uploaded_filename = pdf_file.name

    if not st.session_state.processed_pdf:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "uploaded.pdf")
        try:
            with open(temp_path, "wb") as f:
                f.write(pdf_file.read())
            st.session_state.pdf_path = temp_path
            st.sidebar.info("⏳ Extracting and processing PDF...")
            text = system.applications.extract_text_from_pdf(temp_path)
            st.session_state.text = text
            if text:
                chunks = system.applications.chunk_text(text)
                st.session_state.chunks = chunks
                embeddings = system.applications.get_embeddings(chunks)
                if system.applications._setup_rag_collection() and system.applications.upload_to_qdrant(chunks, embeddings):
                    st.session_state.processed_pdf = True
                    st.sidebar.success("✅ PDF processed and indexed.")
                else:
                    st.sidebar.error("Failed to process and index PDF.")
            else:
                st.sidebar.error("Failed to extract text from PDF.")
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {str(e)}")
            if st.session_state.pdf_path:
                shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
                st.session_state.pdf_path = None

# Main input area based on selected task
input_data = {}
if task == "summarization":
    st.header("Text Summarization")
    input_data["text"] = st.text_area("Text to Summarize")
elif task == "code_generation":
    st.header("Code Generation")
    input_data["task_description"] = st.text_input("Describe the code task")
    input_data["language"] = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "Other"])
elif task == "data_extraction":
    st.header("Data Extraction")
    if st.session_state.processed_pdf:
        st.info("Using text from uploaded PDF for data extraction.")
        input_data["text"] = st.session_state.text
        st.text_area("Extracted Text (Read-only)", value=st.session_state.text, height=200, disabled=True)
    else:
        input_data["text"] = st.text_area("Text to Extract Data From")
    fields = st.text_input("Fields to Extract (comma separated)")
    input_data["fields"] = [f.strip() for f in fields.split(",") if f.strip()]
elif task == "question_answering":
    st.header("Question Answering")
    if st.session_state.processed_pdf:
        st.info("Using text from uploaded PDF for context.")
        input_data["context"] = st.session_state.text
        st.text_area("Extracted Context (Read-only)", value=st.session_state.text, height=200, disabled=True)
    else:
        input_data["context"] = st.text_area("Context")
    input_data["question"] = st.text_input("Question")

if st.button("Run Task"):
    if (task == "data_extraction" or task == "question_answering") and not (input_data.get("text") or input_data.get("context")) and not st.session_state.processed_pdf:
        st.error("Please upload a PDF or provide text input for Data Extraction/Question Answering.")
    else:
        with st.spinner("Running..."):
            result = system.run_application(task, **input_data, reasoning_type=reasoning_type)
            st.subheader("System Prompt")
            st.code(result.get("system_prompt", "N/A"))
            st.subheader("Full Prompt")
            st.code(result.get("prompt", "N/A"))
            st.subheader("Output")
            st.markdown(result.get("output", "N/A"))
            st.subheader("Prompt Score (ROUGE-L)")
            st.write(result.get("score", "N/A"))

# Cleanup when the app is closed
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    shutil.rmtree(os.path.dirname(st.session_state.pdf_path))
    st.session_state.pdf_path = None