#main.py
from .prompt_generator import PromptGenerator
from .evaluator import PromptEvaluator
from .applications import SpecializedApplications

class PromptEngineeringSystem:
    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.evaluator = PromptEvaluator()
        self.applications = SpecializedApplications()

    def run_application(self, task_type, **kwargs):
        reasoning_type = kwargs.pop("reasoning_type", "standard")
        if task_type == "summarization":
            return self.applications.summarize_text(kwargs["text"], reasoning_type=reasoning_type)
        elif task_type == "code_generation":
            return self.applications.generate_code(kwargs["task_description"], kwargs["language"], reasoning_type=reasoning_type)
        elif task_type == "data_extraction":
            return self.applications.extract_data(kwargs["text"], kwargs["fields"], reasoning_type=reasoning_type)
        elif task_type == "question_answering":
            return self.applications.answer_question(kwargs["context"], kwargs["question"], reasoning_type=reasoning_type)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")