from typing import List, Optional

class PromptGenerator:
    def __init__(self):
        self.templates = {}

    def add_template(self, name: str, template: str):
        self.templates[name] = template

    def generate_system_prompts(self, task_type: str, language: Optional[str] = None) -> List[str]:
        if task_type == "code_generation":
            return [
                f"You are an expert {language} developer. Write clean, efficient, and well-documented code following best practices.",
                f"Act as a senior {language} engineer. Provide optimized, readable, and maintainable code with clear comments.",
                f"You are a helpful {language} coding assistant. Generate concise, correct, and idiomatic {language} code.",
                f"Write high-quality {language} code that is easy to understand, properly structured, and thoroughly documented.",
                f"As a seasoned {language} programmer, produce robust and efficient code that adheres to common standards and conventions.",
                f"Create clear and well-structured {language} code with appropriate error handling and inline explanations.",
                f"Provide maintainable and scalable {language} code solutions with concise and informative comments.",
                f"Write {language} code that balances readability and performance, and includes meaningful variable names and documentation.",
                f"Generate {language} code that is modular, reusable, and follows design patterns where applicable.",
                f"Develop {language} code that is testable, with clear separation of concerns and comprehensive docstrings or comments.",
                f"Produce idiomatic {language} code that leverages the latest language features and libraries effectively.",
                f"Provide {language} code snippets that are ready to integrate into larger projects, with proper formatting and style.",
                f"Write efficient {language} code focusing on optimal algorithmic complexity and resource management.",
                f"Generate robust {language} code with clear input validation, exception handling, and documentation."
            ]

        elif task_type == "summarization":
            return [
                "You are a world-class summarizer. Create concise, accurate summaries,dont copy and paste.",
                "Summarize the following text clearly and briefly,dont copy and paste.",
                "You are a helpful assistant. Write a short, informative summary,dont copy and paste",
                "you summarize the text consise dont copy and paste",
                "You are an expert summarizer. Write a clear, concise summary in your own words without copying any sentences from the original text.",
                "Summarize the following content accurately and briefly, ensuring the summary is paraphrased and not directly lifted from the original.",
                "Act as a skilled assistant. Provide a short, insightful summary that captures the key points using original phrasing.",
                "Read the following text and generate a coherent, to-the-point summary that avoids repetition or direct quotes.",
                "Write a well-structured summary highlighting the main ideas in a concise and original manner—do not copy any part of the input text.",
                "Rephrase the core information from the following passage into a brief summary. Use your own words and ensure clarity."
            ]
        elif task_type == "data_extraction":
            return [
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
            ]

        elif task_type == "question_answering":
            return [
                "You are a knowledgeable and helpful assistant. Provide concise, accurate, and clear answers based strictly on the given context.",
                "Answer the question directly and precisely, ensuring correctness and relevance to the provided information.",
                "You are an expert in question answering. Respond clearly and comprehensively without unnecessary details.",
                "Provide well-informed, concise answers that address the question fully and rely solely on the context provided.",
                "Be clear, accurate, and helpful. Avoid speculation and stick to information available in the context.",
                "Answer questions with clarity and precision, using only the facts given. Do not include unrelated information.",
                "You are a context-aware assistant. Provide direct, succinct answers backed by the information supplied.",
                "Focus on answering the question completely and correctly, prioritizing clarity and relevance.",
                "Give straightforward, informative answers while avoiding ambiguity or vague responses.",
                "Respond as a professional expert: concise, accurate, and directly addressing the question.",
                "Provide answers that are fact-based, neutral, and easy to understand, grounded entirely on the provided context.",
                "Act as a reliable source of information. Deliver answers that are brief but thorough, with no extraneous content.",
                "Address the question precisely, focusing on clarity and correctness without unnecessary elaboration.",
                "Provide answers that demonstrate understanding of the question, maintaining focus on relevant details only.",
                "Deliver responses that are structured and to the point, ensuring the user gains clear insight from your answer.",
                "Be succinct and informative, ensuring each answer directly satisfies the question with no fluff.",
                "Answer with confidence and accuracy, based strictly on the data given, avoiding assumptions or guesses.",
                "Maintain an objective tone, presenting answers clearly, logically, and free from ambiguity.",
                "Provide responses that a subject matter expert would give—concise, precise, and well-informed."
            ]

        return ["You are a helpful assistant."]

    def generate_prompt(self, template_name: str, system_prompt: str, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        template = self.templates[template_name]
        return template.format(system_prompt=system_prompt, **kwargs)