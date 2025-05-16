from transformers import pipeline
from .prompt_generator import PromptGenerator
from .evaluator import PromptEvaluator
import torch #torch

class SpecializedApplications:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.code_model = "Salesforce/codegen-350M-mono"
        self.code_pipe = pipeline("text-generation", model=self.code_model, device=self.device)
        self.summ_model = "facebook/bart-large-cnn"
        self.summ_pipe = pipeline("summarization", model=self.summ_model, device=self.device)
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
            "{system_prompt}\nWrite a {language} function to {task_description}.\n# Solution:\n"
        )
        self.prompt_generator.add_template(
            "data_extraction",
            "{system_prompt}\nExtract the following fields from the text: {fields}\nText: {text}\n"
        )
        self.prompt_generator.add_template(
            "question_answering",
            "{system_prompt}\nContext: {context}\nQuestion: {question}\nAnswer:"
        )

    def optimize_prompt(self, task_type, template_name, input_kwargs, pipe, language=None):
        best_score = -float('inf')
        best_result = None
        for system_prompt in self.prompt_generator.generate_system_prompts(task_type, language):
            prompt = self.prompt_generator.generate_prompt(template_name, system_prompt, **input_kwargs)
            if task_type == "code_generation":
                output = pipe(prompt, max_length=128)[0]['generated_text']
            elif task_type == "summarization":
                output = pipe(prompt, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            else:
                output = pipe(prompt, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            # Use input text or task description as default reference for evaluation
            reference = input_kwargs.get("text", input_kwargs.get("task_description", ""))
            score = self.evaluator.rouge.score(reference, output)['rougeL'].fmeasure
            if score > best_score:
                best_score = score
                best_result = {
                    "output": output,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "score": score
                }
        return best_result

    def summarize_text(self, text: str):
        input_kwargs = {"text": text}
        return self.optimize_prompt("summarization", "summarization", input_kwargs, self.summ_pipe)

    def generate_code(self, task_description: str, language: str):
        input_kwargs = {"task_description": task_description, "language": language}
        return self.optimize_prompt("code_generation", "code_generation", input_kwargs, self.code_pipe, language)

    def extract_data(self, text: str, fields: list):
        input_kwargs = {"text": text, "fields": ", ".join(fields)}
        return self.optimize_prompt("data_extraction", "data_extraction", input_kwargs, self.summ_pipe)

    def answer_question(self, context: str, question: str):
        input_kwargs = {"context": context, "question": question}
        return self.optimize_prompt("question_answering", "question_answering", input_kwargs, self.summ_pipe) 