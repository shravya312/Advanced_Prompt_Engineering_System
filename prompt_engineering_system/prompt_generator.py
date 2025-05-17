from typing import List, Optional

class PromptGenerator:
    def __init__(self):
        self.templates = {}

    def add_template(self, name: str, template: str):
        self.templates[name] = template

    def generate_system_prompts(self, task_type: str, language: Optional[str] = None) -> List[str]:
        if task_type == "code_generation":
            return [
                f"You are an expert {language} programmer. Write clean, efficient, and well-documented code.",
                f"Act as a senior {language} developer. Provide optimal, readable code.",
                f"You are a helpful {language} coding assistant. Write correct and concise code."
            ]
        elif task_type == "summarization":
            return [
                "You are a world-class summarizer. Create concise, accurate summaries.",
                "Summarize the following text clearly and briefly.",
                "You are a helpful assistant. Write a short, informative summary."
            ]
        elif task_type == "data_extraction":
            return [
                "You are a precise data extraction assistant. Extract only the requested fields.",
                "Extract structured data as requested.",
                "You are a helpful assistant. Extract the required information."
            ]
        elif task_type == "question_answering":
            return [
                "You are a helpful assistant. Answer concisely and accurately.",
                "Provide a clear and correct answer.",
                "You are an expert in answering questions based on context."
            ]
        return ["You are a helpful assistant."]

    def generate_prompt(self, template_name: str, system_prompt: str, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        template = self.templates[template_name]
        return template.format(system_prompt=system_prompt, **kwargs)