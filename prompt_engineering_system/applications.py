from transformers import pipeline
from .prompt_generator import PromptGenerator
from .evaluator import PromptEvaluator
import torch
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

os.environ["HF_HOME"] = "C:/Users/Shravya H Jain/huggingface_cache"

@st.cache_resource
def get_pipelines():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in environment variables.")
        return None, None

    genai.configure(api_key=gemini_api_key)
    
    # Use the gemini-1.5-flash model as requested
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.success("Successfully initialized gemini-1.5-flash model.")
    except Exception as e:
        st.error(f"Error initializing gemini-1.5-flash model: {e}")
        model = None

    return model, model

class SpecializedApplications:
    def __init__(self):
        self.device = -1
        self.code_model = "Salesforce/codegen-350M-mono"
        self.gemini_model, _ = get_pipelines()
        self.prompt_generator = PromptGenerator()
        self.evaluator = PromptEvaluator()
        self._setup_templates()

    def _setup_templates(self):
        self.prompt_generator.add_template(
            "summarization",
            "{system_prompt}\nSummarize the following text:\n{text}\n"
        )
        self.prompt_generator.add_template(
            "code_generation",
            "{system_prompt}\nWrite a {language} function to {task_description}.\n# Solution:\ndef function_name():\n"
        )
        self.prompt_generator.add_template(
            "data_extraction",
            "{system_prompt}\nExtract the following fields from the text: {fields}\nText: {text}\n"
        )
        self.prompt_generator.add_template(
            "question_answering",
            "{system_prompt}\nContext: {context}\nQuestion: {question}\nAnswer:"
        )

    def optimize_prompt(self, task_type, template_name, input_kwargs, pipe, language=None, reasoning_type=None):
        best_score = -float('inf')
        best_result = None
        model_to_use = self.gemini_model

        if model_to_use is None:
            return {"output": "Error: Gemini model not initialized.", "system_prompt": "", "prompt": "", "score": 0}

        for system_prompt in self.prompt_generator.generate_system_prompts(task_type, language):
            prompt = self.prompt_generator.generate_prompt(template_name, system_prompt, **input_kwargs)
            
            # Call Gemini API
            try:
                response = model_to_use.generate_content(prompt)
                
                # Extract output text
                # The way to extract output might vary slightly based on the model and response structure
                # For text-based models like gemini-pro, response.text is usually sufficient
                if hasattr(response, 'text'):
                    output = response.text
                else:
                    # Handle cases where response might not have a text attribute directly
                    # This might require inspecting the response object structure
                    output = str(response) # Fallback to string representation
                    st.warning(f"Gemini response did not have a .text attribute for {task_type}. Using string representation.")
                
                # Note: For Tree of Thought, a more advanced implementation might involve
                # multiple turns or specific output parsing to explore branches.
                # This current implementation relies on the prompt modifier to guide a single response.

            except Exception as e:
                output = f"Error calling Gemini API: {e}"
                st.error(output)
                continue # Skip scoring if API call failed

            # The evaluation part might need adjustment depending on Gemini's output format
            reference = input_kwargs.get("text", input_kwargs.get("task_description", ""))
            if output and isinstance(output, str):
                 reference = str(reference)
                 score = self.evaluator.rouge.score(reference, output)['rougeL'].fmeasure
            else:
                 score = 0
                 st.warning(f"Skipping evaluation for non-text output for {task_type}")

            if score > best_score:
                best_score = score
                best_result = {
                    "output": output,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "score": score
                }
        
        if best_result is None:
             return {"output": "Could not generate output using Gemini.", "system_prompt": "", "prompt": "", "score": 0}

        return best_result

    def summarize_text(self, text: str, reasoning_type: str):
        input_kwargs = {"text": text}
        return self.optimize_prompt("summarization", "summarization", input_kwargs, self.gemini_model, reasoning_type=reasoning_type)

    def generate_code(self, task_description: str, language: str, reasoning_type: str):
        input_kwargs = {"task_description": task_description, "language": language}
        return self.optimize_prompt("code_generation", "code_generation", input_kwargs, self.gemini_model, language, reasoning_type=reasoning_type)

    def extract_data(self, text: str, fields: list, reasoning_type: str):
        input_kwargs = {"text": text, "fields": ", ".join(fields)}
        return self.optimize_prompt("data_extraction", "data_extraction", input_kwargs, self.gemini_model, reasoning_type=reasoning_type)

    def answer_question(self, context: str, question: str, reasoning_type: str):
        input_kwargs = {"context": context, "question": question}
        return self.optimize_prompt("question_answering", "question_answering", input_kwargs, self.gemini_model, reasoning_type=reasoning_type)