import streamlit as st
from prompt_engineering_system.main import PromptEngineeringSystem

st.set_page_config(page_title="Advanced Prompt Engineering System", layout="wide")
st.title("Advanced Prompt Engineering System")

system = PromptEngineeringSystem()

task = st.sidebar.selectbox("Select Task", [
    "summarization",
    "code_generation",
    "data_extraction",
    "question_answering"
])

input_data = {}
if task == "summarization":
    input_data["text"] = st.text_area("Text to Summarize")
elif task == "code_generation":
    input_data["task_description"] = st.text_input("Describe the code task")
    input_data["language"] = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "Other"])
elif task == "data_extraction":
    input_data["text"] = st.text_area("Text to Extract Data From")
    fields = st.text_input("Fields to Extract (comma separated)")
    input_data["fields"] = [f.strip() for f in fields.split(",") if f.strip()]
elif task == "question_answering":
    input_data["context"] = st.text_area("Context")
    input_data["question"] = st.text_input("Question")

if st.button("Run Task"):
    with st.spinner("Running..."):
        result = system.run_application(task, **input_data)
        st.subheader("System Prompt")
        st.code(result["system_prompt"])
        st.subheader("Full Prompt")
        st.code(result["prompt"])
        st.subheader("Output")
        st.code(result["output"])
        st.subheader("Prompt Score (ROUGE-L)")
        st.write(result["score"])