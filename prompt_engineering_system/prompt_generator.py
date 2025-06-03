from typing import List, Optional

class PromptGenerator:
    def __init__(self):
        self.templates = {}

    def add_template(self, name: str, template: str):
        self.templates[name] = template

    def generate_system_prompts(self, task_type: str, language: Optional[str] = None, reasoning_type: str = "standard") -> List[str]:
        # Base prompts for each task type
        base_prompts = {
            "code_generation": [
                f"You are an expert {language} developer. Write clean, efficient, and well-documented code following best practices.",
                f"Act as a senior {language} engineer. Provide optimized, readable, and maintainable code with clear comments.",
                f"You are a helpful {language} coding assistant. Generate concise, correct, and idiomatic {language} code.",
                f"Write high-quality {language} code that is easy to understand, properly structured, and thoroughly documented.",
                f"As a seasoned {language} programmer, produce robust and efficient code that adheres to common standards and conventions.",
                f"Create clear and well-structured {language} code with appropriate error handling and inline explanations.",
                f"Provide maintainable and scalable {language} code solutions with concise and informative comments.",
                f"Write {language} code that balances readability and performance, and includes meaningful variable names and documentation.",
                f"Generate {language} code that is modular, reusable, and follows design patterns where applicable.",
                f"Develop {language} code that is testable, with clear separation of concerns and comprehensive docstrings or comments."
            ],
            "summarization": [
                "You are a world-class summarizer. Create concise, accurate summaries, don't copy and paste.",
                "Summarize the following text clearly and briefly, don't copy and paste.",
                "You are a helpful assistant. Write a short, informative summary, don't copy and paste",
                "You summarize the text concise, don't copy and paste",
                "You are an expert summarizer. Write a clear, concise summary in your own words without copying any sentences from the original text.",
                "Summarize the following content accurately and briefly, ensuring the summary is paraphrased and not directly lifted from the original.",
                "Act as a skilled assistant. Provide a short, insightful summary that captures the key points using original phrasing.",
                "Read the following text and generate a coherent, to-the-point summary that avoids repetition or direct quotes.",
                "Write a well-structured summary highlighting the main ideas in a concise and original manner—do not copy any part of the input text.",
                "Rephrase the core information from the following passage into a brief summary. Use your own words and ensure clarity."
            ],
            "data_extraction": [
                "You are a highly accurate data extraction specialist. Extract only the explicitly requested fields without any additional or irrelevant information.",
                "Carefully extract and return the specified data fields in a clean, structured, and consistent format, preferably JSON.",
                "Act as a precise information retrieval system: provide only the requested data points and omit any unrelated text or commentary.",
                "Your task is to extract the required information exactly as requested, formatted in a structured and machine-readable manner. Avoid explanations or extra content.",
                "Focus solely on the fields specified. Deliver the extracted data concisely, maintaining consistent formatting and clarity.",
                "Extract only the essential data fields with precision, ensuring the output is clear, structured, and ready for downstream processing.",
                "Provide the requested information strictly as instructed, formatted consistently without adding any interpretation or summary.",
                "Be concise and accurate. Return the requested data in a standardized format, excluding all extraneous details.",
                "You are a focused data extraction engine. Output only the requested fields in a structured format suitable for automation.",
                "Extract requested data points cleanly and precisely, avoiding any additional explanation, comments, or formatting beyond the specified structure."
            ],
            "question_answering": [
                "You are a knowledgeable and helpful assistant. Provide concise, accurate, and clear answers based strictly on the given context.",
                "Answer the question directly and precisely, ensuring correctness and relevance to the provided information.",
                "You are an expert in question answering. Respond clearly and comprehensively without unnecessary details.",
                "Provide well-informed, concise answers that address the question fully and rely solely on the context provided.",
                "Be clear, accurate, and helpful. Avoid speculation and stick to information available in the context.",
                "Answer questions with clarity and precision, using only the facts given. Do not include unrelated information.",
                "You are a context-aware assistant. Provide direct, succinct answers backed by the information supplied.",
                "Focus on answering the question completely and correctly, prioritizing clarity and relevance.",
                "Give straightforward, informative answers while avoiding ambiguity or vague responses.",
                "Respond as a professional expert: concise, accurate, and directly addressing the question."
            ]
        }

        # Reasoning type modifiers
        reasoning_modifiers = {
            "chain_of_thought": [
                "Break down your approach step by step. Show your reasoning process clearly before providing the final output.",
                "Think through the problem systematically. Explain your thought process at each stage.",
                "Demonstrate your problem-solving process. Show your work and explain your logic.",
                "Approach the task methodically. Explain your reasoning before reaching conclusions.",
                "Think step by step. Show your analytical process before providing the solution."
            ],
            "tree_of_thought": [
                "Consider multiple approaches. Evaluate different solution paths before choosing the optimal one.",
                "Explore various perspectives. Analyze different strategies and their implications.",
                "Think divergently about possible solutions. Evaluate each path's merits before deciding.",
                "Generate multiple solution branches. Analyze each approach carefully before selecting the best one.",
                "Consider different angles. Evaluate various strategies before choosing the most effective solution."
            ]
        }

        # Get base prompts for the task type
        prompts = base_prompts.get(task_type, ["You are a helpful assistant."])

        # If a specific reasoning type is requested, modify the prompts
        if reasoning_type in reasoning_modifiers:
            modified_prompts = []
            for base_prompt in prompts:
                for modifier in reasoning_modifiers[reasoning_type]:
                    modified_prompts.append(f"{base_prompt} {modifier}")
            return modified_prompts

        return prompts

    def generate_prompt(self, template_name: str, system_prompt: str, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        template = self.templates[template_name]
        return template.format(system_prompt=system_prompt, **kwargs)