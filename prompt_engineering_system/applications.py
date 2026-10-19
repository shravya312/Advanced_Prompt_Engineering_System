from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from prompt_engineering_system.prompt_generator import PromptGenerator
from prompt_engineering_system.evaluator import PromptEvaluator
from prompt_engineering_system.gemini_client import GeminiModel
import os
import streamlit as st
from dotenv import load_dotenv
from prompt_engineering_system.qdrant_rest import QdrantRestClient
from prompt_engineering_system.reasoning_output import (
    parse_chain_of_thought_output,
    parse_tree_of_thought_output,
)
import numpy as np
import uuid

TREE_OF_THOUGHT_SUMMARY_SYSTEM = (
    "You are an expert summarizer using Tree-of-Thought reasoning. "
    "Explore two different summarization strategies, compare them explicitly, "
    "then produce one final summary. Use the exact markdown section headers requested."
)
CHAIN_OF_THOUGHT_SUMMARY_SYSTEM = (
    "You are an expert summarizer using Chain-of-Thought reasoning. "
    "Work through the text in clear sequential steps, then give a final summary. "
    "Use the exact markdown section headers requested."
)

_PKG_DIR = Path(__file__).resolve().parent
load_dotenv(_PKG_DIR / ".env")

COLLECTION_NAME = "document_collection"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PROMPT_VARIANT_LIMIT = int(os.getenv("PROMPT_VARIANT_LIMIT", "3"))

os.environ.setdefault("HF_HOME", str(Path.home() / "huggingface_cache"))


@st.cache_resource
def get_gemini_model(model_name: str, api_key: str):
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables.")
        return None
    try:
        model = GeminiModel(api_key=api_key, model=model_name)
        st.success(
            f"Gemini ready (primary: {model_name}; fallbacks enabled if overloaded)."
        )
        return model
    except Exception as exc:
        st.error(f"Error initializing Gemini: {exc}")
        return None


@st.cache_resource
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        st.error(f"Failed to initialize SentenceTransformer: {exc}")
        return None


@st.cache_resource
def get_qdrant_client(qdrant_url: str, qdrant_api_key: str):
    try:
        client = QdrantRestClient(url=qdrant_url, api_key=qdrant_api_key)
        client.list_collections()
        return client
    except Exception as exc:
        st.error(f"Qdrant connection failed: {exc}")
        return None


def qdrant_config() -> Tuple[str, str]:
    url = os.getenv("QDRANT_URL", "")
    api_key = os.getenv("QDRANT_API_KEY", "")
    return url, api_key


class SpecializedApplications:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or ""
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.gemini_model = get_gemini_model(model_name, api_key)
        self.prompt_generator = PromptGenerator()
        self.evaluator = PromptEvaluator()
        self._qdrant_url, self._qdrant_api_key = qdrant_config()
        self._setup_templates()

    def _require_qdrant(self) -> QdrantRestClient:
        client = get_qdrant_client(self._qdrant_url, self._qdrant_api_key)
        if client is None:
            raise RuntimeError(
                "Qdrant is not connected. Set QDRANT_URL and QDRANT_API_KEY in .env."
            )
        return client

    def _setup_templates(self):
        self.prompt_generator.add_template(
            "summarization",
            "{system_prompt}\nSummarize the following text:\n{text}\n",
        )
        self.prompt_generator.add_template(
            "summarization_tree_of_thought",
            (
                "{system_prompt}\n\n"
                "Read the text below. Use Tree-of-Thought summarization.\n"
                "You MUST include exactly these markdown section headers in this order:\n\n"
                "## Branch A\n"
                "Name one summarization strategy (e.g. technical/method focus). "
                "Explain the angle in 1-2 sentences, then give a 2-4 sentence draft summary using that branch.\n\n"
                "## Branch B\n"
                "Name a different strategy (e.g. challenges/deployment focus). "
                "Explain the angle in 1-2 sentences, then give a 2-4 sentence draft summary using that branch.\n\n"
                "## Strategy Comparison\n"
                "Explicitly compare Branch A vs Branch B: what each emphasizes, strengths, weaknesses, "
                "and which branch fits the source text better (and why).\n\n"
                "## Final Summary\n"
                "Synthesize the best ideas into one concise paragraph (4-6 sentences). "
                "Paraphrase; do not copy sentences from the source.\n\n"
                "Text:\n{text}\n"
            ),
        )
        self.prompt_generator.add_template(
            "summarization_chain_of_thought",
            (
                "{system_prompt}\n\n"
                "Read the text below. Use Chain-of-Thought summarization.\n"
                "You MUST include exactly these markdown section headers in this order:\n\n"
                "## Step 1\n"
                "Identify the main topic and purpose of the text (1-3 sentences).\n\n"
                "## Step 2\n"
                "List the most important facts, claims, or technical points (bullet points allowed).\n\n"
                "## Step 3\n"
                "Note what to omit (minor details, repetition) and why (1-3 sentences).\n\n"
                "## Step 4\n"
                "Draft a short paraphrased summary based on Steps 1-3 (2-4 sentences).\n\n"
                "## Final Summary\n"
                "Polish Step 4 into one concise paragraph (4-6 sentences). "
                "Do not copy sentences from the source.\n\n"
                "Text:\n{text}\n"
            ),
        )
        self.prompt_generator.add_template(
            "code_generation",
            "{system_prompt}\nWrite a {language} function to {task_description}.\n# Solution:\ndef function_name():\n",
        )
        self.prompt_generator.add_template(
            "data_extraction",
            "{system_prompt}\nExtract the following fields from the text: {fields}\nText: {text}\n",
        )
        self.prompt_generator.add_template(
            "question_answering",
            "{system_prompt}\nContext: {context}\nQuestion: {question}\nAnswer:",
        )

    def _setup_rag_collection(self) -> bool:
        try:
            client = self._require_qdrant()
            client.ensure_collection(COLLECTION_NAME, VECTOR_SIZE, distance="Cosine")
            return True
        except Exception as exc:
            st.error(f"Qdrant collection setup failed: {exc}")
            return False

    def chunk_text(self, text: str) -> list:
        if not text:
            return []
        return [
            text[i : i + CHUNK_SIZE]
            for i in range(0, len(text), CHUNK_SIZE)
            if text[i : i + CHUNK_SIZE].strip()
        ]

    def get_embeddings(self, chunks: list) -> Optional[np.ndarray]:
        if not chunks:
            return None
        model = get_embedding_model()
        if model is None:
            return None
        try:
            return model.encode(chunks)
        except Exception as exc:
            st.error(f"Error generating embeddings: {exc}")
            return None

    def _replace_collection_vectors(self, chunks: list, embeddings: np.ndarray) -> None:
        client = self._require_qdrant()
        if COLLECTION_NAME in client.list_collections():
            client.delete_collection(COLLECTION_NAME)
        client.ensure_collection(COLLECTION_NAME, VECTOR_SIZE, distance="Cosine")
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": embedding.tolist(),
                "payload": {"text": chunk},
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        client.upsert_points(COLLECTION_NAME, points)

    def upload_to_qdrant(self, chunks: list, embeddings: np.ndarray) -> bool:
        if not chunks or embeddings is None or embeddings.size == 0:
            st.error("No text chunks to index in Qdrant.")
            return False
        try:
            self._replace_collection_vectors(chunks, embeddings)
            return True
        except Exception as exc:
            st.error(f"Qdrant upload failed: {exc}")
            return False

    def search_chunks(self, query: str, limit: int = 5) -> list:
        client = self._require_qdrant()
        model = get_embedding_model()
        if model is None:
            raise RuntimeError("Embedding model is not available for Qdrant search.")
        query_vector = model.encode([query])[0].tolist()
        hits = client.search(COLLECTION_NAME, query_vector, limit=limit)
        return [hit["payload"]["text"] for hit in hits if hit.get("payload")]

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as exc:
            st.error(f"Error extracting text from PDF: {exc}")
            return ""

    def optimize_prompt(
        self,
        task_type,
        template_name,
        input_kwargs,
        language=None,
        reasoning_type="standard",
    ):
        best_score = -float("inf")
        best_result = None
        model_to_use = self.gemini_model

        if model_to_use is None:
            return {
                "output": "Error: Gemini model not initialized.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        system_prompts = self.prompt_generator.generate_system_prompts(
            task_type, language, reasoning_type=reasoning_type
        )[:PROMPT_VARIANT_LIMIT]

        last_api_error = None
        for system_prompt in system_prompts:
            prompt = self.prompt_generator.generate_prompt(
                template_name, system_prompt, **input_kwargs
            )
            try:
                response = model_to_use.generate_content(prompt)
                output = response.text
            except Exception as exc:
                last_api_error = exc
                break

            reference = input_kwargs.get("text", input_kwargs.get("task_description", ""))
            if output and isinstance(output, str):
                reference = str(reference)
                score = self.evaluator.rouge.score(reference, output)["rougeL"].fmeasure
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_result = {
                    "output": output,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "score": score,
                }

        if best_result is None:
            if last_api_error:
                st.error(
                    "Gemini is temporarily unavailable (high demand or rate limits). "
                    "Wait a minute and try again, or set GEMINI_MODEL in `.env` to another model. "
                    f"Details: {last_api_error}"
                )
            return {
                "output": "Could not generate output using Gemini.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }
        return best_result

    def _summarize_tree_of_thought(self, text: str):
        if self.gemini_model is None:
            return {
                "output": "Error: Gemini model not initialized.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        system_prompt = TREE_OF_THOUGHT_SUMMARY_SYSTEM
        prompt = self.prompt_generator.generate_prompt(
            "summarization_tree_of_thought",
            system_prompt,
            text=text,
        )
        try:
            output = self.gemini_model.generate_content(prompt).text
        except Exception as exc:
            st.error(f"Gemini API error: {exc}")
            return {
                "output": "Could not generate Tree-of-Thought summary.",
                "system_prompt": system_prompt,
                "prompt": prompt,
                "score": 0,
            }

        reasoning_tree = parse_tree_of_thought_output(output)
        score_text = reasoning_tree.get("final_summary") or output
        score = self.evaluator.rouge.score(text, score_text)["rougeL"].fmeasure

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": score,
            "reasoning_tree": reasoning_tree,
            "reasoning_type": "tree_of_thought",
        }

    def _summarize_chain_of_thought(self, text: str):
        if self.gemini_model is None:
            return {
                "output": "Error: Gemini model not initialized.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        system_prompt = CHAIN_OF_THOUGHT_SUMMARY_SYSTEM
        prompt = self.prompt_generator.generate_prompt(
            "summarization_chain_of_thought",
            system_prompt,
            text=text,
        )
        try:
            output = self.gemini_model.generate_content(prompt).text
        except Exception as exc:
            st.error(f"Gemini API error: {exc}")
            return {
                "output": "Could not generate Chain-of-Thought summary.",
                "system_prompt": system_prompt,
                "prompt": prompt,
                "score": 0,
            }

        reasoning_chain = parse_chain_of_thought_output(output)
        score_text = reasoning_chain.get("final_summary") or output
        score = self.evaluator.rouge.score(text, score_text)["rougeL"].fmeasure

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": score,
            "reasoning_chain": reasoning_chain,
            "reasoning_type": "chain_of_thought",
        }

    def summarize_text(self, text: str, reasoning_type: str):
        if reasoning_type == "tree_of_thought":
            return self._summarize_tree_of_thought(text)
        if reasoning_type == "chain_of_thought":
            return self._summarize_chain_of_thought(text)
        input_kwargs = {"text": text}
        return self.optimize_prompt(
            "summarization",
            "summarization",
            input_kwargs,
            reasoning_type=reasoning_type,
        )

    def generate_code(self, task_description: str, language: str, reasoning_type: str):
        input_kwargs = {"task_description": task_description, "language": language}
        return self.optimize_prompt(
            "code_generation",
            "code_generation",
            input_kwargs,
            language=language,
            reasoning_type=reasoning_type,
        )

    def extract_data(self, text: str, fields: list, reasoning_type: str):
        chunks = self.chunk_text(text)
        embeddings = self.get_embeddings(chunks)
        if embeddings is None or not self.upload_to_qdrant(chunks, embeddings):
            return {
                "output": "Data extraction failed: could not index document in Qdrant.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        relevant_chunks = []
        for field in fields:
            relevant_chunks.extend(self.search_chunks(field))

        combined_text = text
        if relevant_chunks:
            combined_text += "\n\nRelevant context:\n" + "\n".join(dict.fromkeys(relevant_chunks))

        system_prompt = self.prompt_generator.generate_system_prompts(
            "data_extraction", reasoning_type=reasoning_type
        )[0]
        prompt = self.prompt_generator.generate_prompt(
            "data_extraction",
            system_prompt,
            text=combined_text,
            fields=", ".join(fields),
        )

        try:
            output = self.gemini_model.generate_content(prompt).text
        except Exception as exc:
            output = f"Error calling Gemini API for data extraction: {exc}"
            st.error(output)

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": 0,
        }

    def answer_question(self, context: str, question: str, reasoning_type: str):
        chunks = self.chunk_text(context)
        embeddings = self.get_embeddings(chunks)
        if embeddings is None or not self.upload_to_qdrant(chunks, embeddings):
            return {
                "output": "Question answering failed: could not index context in Qdrant.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        relevant_chunks = self.search_chunks(question)
        combined_context = context
        if relevant_chunks:
            combined_context += "\n\nRelevant context:\n" + "\n".join(dict.fromkeys(relevant_chunks))

        import re

        marks_match = re.search(r"(\d+)\s*marks", question, re.IGNORECASE)
        extra_instruction = ""
        if marks_match:
            marks = int(marks_match.group(1))
            if marks >= 8:
                extra_instruction = (
                    "This is a high-mark question. Write a long, detailed, and well-structured answer. "
                    "Include introduction, step-by-step explanation, algorithm, math, advantages, "
                    "disadvantages, applications, and examples."
                )
            elif marks >= 5:
                extra_instruction = (
                    "This is a medium-mark question. Write a moderately detailed answer with "
                    "explanation, steps, and at least one example."
                )
            else:
                extra_instruction = "This is a short-mark question. Write a concise answer."

        system_prompt = self.prompt_generator.generate_system_prompts(
            "question_answering", reasoning_type=reasoning_type
        )[0]
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions: {extra_instruction}\n\n"
            f"Answer:"
        )

        try:
            output = self.gemini_model.generate_content(prompt).text.strip()
        except Exception as exc:
            output = f"Error from Gemini: {exc}"
            st.error(output)

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": 0,
        }
