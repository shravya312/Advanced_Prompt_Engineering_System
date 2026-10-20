import os
import streamlit as st
from prompt_engineering_system.main import PromptEngineeringSystem
from prompt_engineering_system.applications import get_qdrant_client, qdrant_config
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

# Glued labels like "Section A: ... Section B: ..." on one line (any document type).
_SECTION_LABEL_RE = re.compile(
    r"(?<!\n)\s+(?=([A-Z][A-Za-z0-9][A-Za-z0-9 /&()\-]{0,48}):)",
)


def format_output_for_display(text: str) -> str:
    """Break glued section labels onto new lines for readable markdown."""
    if not text or not isinstance(text, str):
        return text or ""
    if text.count("\n") >= 3:
        return text.strip()
    formatted = _SECTION_LABEL_RE.sub("\n\n", text.strip())
    formatted = re.sub(
        r"^([A-Z][A-Za-z0-9][A-Za-z0-9 /&()\-]{0,48}:)\s*",
        r"## \1\n",
        formatted,
        flags=re.MULTILINE,
    )
    return formatted.strip()


# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("Advanced Prompt Engineering System")

@st.cache_resource
def get_system():
    return PromptEngineeringSystem()

system = get_system()

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

_qdrant_url, _qdrant_key = qdrant_config()
if get_qdrant_client(_qdrant_url, _qdrant_key):
    st.sidebar.success("Qdrant: connected (required for retrieval-augmented tasks)")
else:
    st.sidebar.error("Qdrant: not connected — fix QDRANT_URL / QDRANT_API_KEY in .env")

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
pdf_file = st.sidebar.file_uploader("Upload a document (PDF) for extraction / Q&A", type=["pdf"])

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
            st.sidebar.info("⏳ Extracting and indexing document...")
            text = system.applications.extract_text_from_pdf(temp_path)
            st.session_state.text = text
            if text:
                chunks = system.applications.chunk_text(text)
                st.session_state.chunks = chunks
                embeddings = system.applications.get_embeddings(chunks)
                if (
                    system.applications._setup_rag_collection()
                    and embeddings is not None
                    and system.applications.upload_to_qdrant(chunks, embeddings)
                ):
                    st.session_state.processed_pdf = True
                    st.sidebar.success("Document indexed in Qdrant.")
                else:
                    st.sidebar.error("Failed to index document in Qdrant.")
            else:
                st.sidebar.error("Failed to extract text from document.")
        except Exception as e:
            st.sidebar.error(f"Error processing document: {str(e)}")
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
    input_data["task_description"] = st.text_input(
        "Describe the code task",
        value="",
        placeholder="",
        key="code_task_description",
    )
    input_data["language"] = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "Other"])
elif task == "data_extraction":
    st.header("Data Extraction")
    if st.session_state.processed_pdf:
        st.info("Using text from uploaded document for data extraction.")
        input_data["text"] = st.session_state.text
        st.text_area("Extracted Text (Read-only)", value=st.session_state.text, height=200, disabled=True)
    else:
        input_data["text"] = st.text_area("Text to Extract Data From")
    fields = st.text_input(
        "Fields to Extract (comma separated)",
        value="",
        placeholder="",
        key="extraction_fields",
    )
    input_data["fields"] = [f.strip() for f in fields.split(",") if f.strip()]
elif task == "question_answering":
    st.header("Question Answering")
    if st.session_state.processed_pdf:
        st.info("Using text from uploaded document for context.")
        input_data["context"] = st.session_state.text
        st.text_area("Extracted Context (Read-only)", value=st.session_state.text, height=200, disabled=True)
    else:
        input_data["context"] = st.text_area("Context")
    input_data["question"] = st.text_input(
        "Question",
        value="",
        placeholder="",
        key="qa_question",
    )

if st.button("Run Task"):
    if (task == "data_extraction" or task == "question_answering") and not (input_data.get("text") or input_data.get("context")) and not st.session_state.processed_pdf:
        st.error("Please upload a document or provide text input for data extraction / question answering.")
    else:
        with st.spinner("Running..."):
            result = system.run_application(task, **input_data, reasoning_type=reasoning_type)
            st.subheader("System Prompt")
            st.code(result.get("system_prompt", "N/A"))
            st.subheader("Full Prompt")
            st.code(result.get("prompt", "N/A"))
            tree = result.get("reasoning_tree")
            chain = result.get("reasoning_chain")
            if tree:
                st.subheader("Tree of Thought (reasoning)")
                st.markdown("#### Branch A")
                st.markdown(tree.get("branch_a") or "_Section not found — see full output below._")
                st.markdown("#### Branch B")
                st.markdown(tree.get("branch_b") or "_Section not found — see full output below._")
                st.markdown("#### Strategy comparison")
                st.markdown(tree.get("comparison") or "_Section not found — see full output below._")
                st.markdown("#### Final summary")
                st.markdown(tree.get("final_summary") or "_Section not found — see full output below._")
                with st.expander("Full model output (raw)"):
                    st.markdown(result.get("output", "N/A"))
            elif chain:
                st.subheader("Chain of Thought (reasoning steps)")
                for key, label in [
                    ("step_1", "Step 1 — Main topic"),
                    ("step_2", "Step 2 — Key points"),
                    ("step_3", "Step 3 — What to omit"),
                    ("step_4", "Step 4 — Draft summary"),
                ]:
                    st.markdown(f"#### {label}")
                    st.markdown(chain.get(key) or "_Section not found — see full output below._")
                st.markdown("#### Final summary")
                st.markdown(chain.get("final_summary") or "_Section not found — see full output below._")
                with st.expander("Full model output (raw)"):
                    st.markdown(result.get("output", "N/A"))
            else:
                st.subheader("Output")
                raw_output = result.get("output", "N/A")
                if task in ("data_extraction", "question_answering"):
                    st.markdown(format_output_for_display(raw_output))
                else:
                    st.markdown(raw_output)
            st.subheader("Prompt Score (ROUGE-L)")
            st.write(result.get("score", "N/A"))
