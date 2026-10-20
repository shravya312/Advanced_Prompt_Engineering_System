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
import re
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

QA_HYBRID_RAG_POLICY = (
    "Answer policy (retrieved document + Gemini knowledge):\n"
    "1) Document-bound: For questions about what the source document states (names, dates, titles, "
    "listed items, achievements), answer only from the Context. If missing, say it is "
    "not stated in the document.\n"
    "2) Elaboration allowed: When the user asks to summarize, explain, define, or describe something "
    "(terms, tools, projects, methods), answer in a clear conversational style: state what the "
    "document says, then explain ideas in plain language using general knowledge. Do not invent "
    "document facts not present in the Context.\n"
    "3) Never refuse to explain because the document omits definitions; teach the concept naturally.\n"
    "4) Do not add 'why it fits this project', design rationale, or technology justification sections "
    "unless the question explicitly asks why, reason, rationale, or purpose.\n"
)


def qa_answer_instructions(question: str) -> List[str]:
    """Pick answer style from the user's question (factual vs explain vs why vs list vs how)."""
    q = question.strip()
    if not q:
        return ["Answer mode (direct): Reply concisely using the Context."]

    instructions: List[str] = []
    marks_match = re.search(r"(\d+)\s*marks", q, re.IGNORECASE)
    if marks_match:
        marks = int(marks_match.group(1))
        if marks >= 8:
            instructions.append(
                "Exam style (high marks): long, structured answer with intro, steps, examples, "
                "and conclusion as appropriate."
            )
        elif marks >= 5:
            instructions.append(
                "Exam style (medium marks): moderate detail with explanation and at least one example."
            )
        else:
            instructions.append("Exam style (low marks): concise, to the point.")

    wants_why = bool(re.search(r"\b(why|reason|rationale|purpose)\b", q, re.IGNORECASE))
    wants_how = bool(re.search(r"\bhow\s+(does|do|is|are|can|to)\b", q, re.IGNORECASE))
    wants_list = bool(
        re.search(
            r"\b(list|enumerate|name all|what are the|which .+ are|all the)\b",
            q,
            re.IGNORECASE,
        )
    )
    wants_factual = bool(
        re.search(
            r"\b(what is my|what's my|my cgpa|my percentage|when did|when was|"
            r"where did|who is|how many|email|phone|contact)\b",
            q,
            re.IGNORECASE,
        )
    )
    wants_explain = bool(
        re.search(
            r"\b(explain|describe|summarize|summary|overview|define|tell me about|walk me through)\b",
            q,
            re.IGNORECASE,
        )
        or (
            re.search(r"\bwhat (is|are)\b", q, re.IGNORECASE) and not wants_factual
        )
    )

    if wants_why:
        instructions.append(
            "Answer mode (why): User wants rationale. Say what the document states, define the "
            "technology or approach briefly, then explain why it is used or fits (note when you "
            "are inferring beyond the document)."
        )
    elif wants_factual:
        instructions.append(
            "Answer mode (factual): Short, direct answer from the Context only. No long tutorials."
        )
    elif wants_list and not wants_explain:
        instructions.append(
            "Answer mode (list): Bullet list of items found in the Context; one line per item unless "
            "the user also asked to explain each."
        )
    elif wants_how and not wants_explain:
        instructions.append(
            "Answer mode (how): Step-by-step or procedural explanation; ground steps in the Context "
            "when describing this document or project, use general knowledge for how the method works."
        )
    elif wants_explain or wants_list:
        instructions.append(
            "Answer mode (explain): Clear conversational overview—what the document says, then "
            "plain-language explanations of key terms. No 'why it fits' section unless the question "
            "asked why."
        )
    else:
        instructions.append(
            "Answer mode (direct): Answer the question naturally; use the Context for document facts "
            "and general knowledge only when needed to make the answer understandable."
        )

    return instructions


_PKG_DIR = Path(__file__).resolve().parent
load_dotenv(_PKG_DIR / ".env")

COLLECTION_NAME = "document_collection"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
PROMPT_VARIANT_LIMIT = int(os.getenv("PROMPT_VARIANT_LIMIT", "3"))

def _configure_hf_environment() -> None:
    """Use the standard Hugging Face cache (not ~/huggingface_cache) and certifi for SSL."""
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    hf_home = Path.home() / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.pop("TRANSFORMERS_CACHE", None)


_configure_hf_environment()


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
    _configure_hf_environment()
    try:
        from sentence_transformers import SentenceTransformer

        model_id = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_id)
    except Exception as exc:
        st.error(
            "Failed to initialize SentenceTransformer. "
            "Ensure `all-MiniLM-L6-v2` can download (network/SSL) or is cached under "
            f"{Path.home() / '.cache' / 'huggingface'}. Details: {exc}"
        )
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


def _normalize_code_task_description(task_description: str, language: str) -> str:
    """Avoid 'Write a X function to Write a X function...' when users paste full sentences."""
    text = task_description.strip()
    if not text:
        return text
    lang = re.escape(language.strip())
    prefix = (
        r"^write\s+(?:a\s+)?"
        rf"(?:{lang}\s+)?"
        r"(?:function|class|script|program)\s+"
        r"(?:to|for|that|which|using)\s+"
    )
    stripped = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"^the\s+", "", stripped, flags=re.IGNORECASE).strip()
    if stripped and not re.match(
        r"^(use|implement|create|build|write|load|run|compute|define|develop|"
        r"accept|read|parse|convert|extract|generate|return|given)\b",
        stripped,
        flags=re.IGNORECASE,
    ):
        stripped = f"Use {stripped}"
    return stripped or text


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
                "Name one summarization strategy. "
                "Explain the angle in 1-2 sentences, then give a 2-4 sentence draft summary using that branch.\n\n"
                "## Branch B\n"
                "Name a different strategy. "
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
            (
                "{system_prompt}\n\n"
                "Programming language: {language}\n"
                "Task: {task_description}\n\n"
                "{reasoning_instructions}\n"
                "Output requirements (apply to the final code):\n"
                "- Follow the task literally; use file paths, arrays, or parameters the task implies.\n"
                "- Do not use webcam, live video capture, or interactive hardware unless the task explicitly asks for it.\n"
                "- Default: one clear top-level function that solves the task. Add small local helpers only if needed.\n"
                "- Use a class only if the task clearly requires persistent state or multiple coordinated methods.\n"
                "- Do not add `if __name__ == \"__main__\"`, argparse, or a demo loop unless the task asks for a script entry point.\n"
                "Put any explanation before the code. End with the complete solution in exactly one fenced code block.\n"
            ),
        )
        self.prompt_generator.add_template(
            "data_extraction",
            (
                "{system_prompt}\n\n"
                "Extract the following fields from the text: {fields}\n\n"
                "{extraction_format_notes}\n"
                "Output format (required):\n"
                "- Use one markdown heading (`## Title`) per requested field name.\n"
                "- If the source uses subheadings under a topic, mirror those subheadings as separate "
                "`##` sections instead of merging into one list.\n"
                "- One extracted value per bullet line (`- value`); do not put multiple values on one line.\n"
                "- Put a blank line between sections. Do not merge sections into one paragraph.\n"
                "- Extract only what appears in the text; do not add definitions unless a field asks for them.\n\n"
                "Text:\n{text}\n"
            ),
        )
        self.prompt_generator.add_template(
            "question_answering",
            (
                "{system_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "{hybrid_rag_policy}\n"
                "{answer_instructions}\n"
                "Answer using clear markdown: headings (`##`) per major section, bullet lists where helpful, "
                "and short paragraphs. Do not answer in one long unbroken line.\n"
            ),
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

    def _build_qa_context(self, context: str, relevant_chunks: list) -> str:
        """Avoid sending full document plus duplicate RAG chunks (reduces timeouts)."""
        max_chars = int(os.getenv("QA_CONTEXT_MAX_CHARS", "16000"))
        unique = [c for c in dict.fromkeys(relevant_chunks) if c and str(c).strip()]
        if unique:
            merged = "\n\n---\n\n".join(unique)
            if len(merged) > max_chars:
                merged = merged[:max_chars] + "\n\n[Retrieved context truncated.]"
            return merged
        text = context or ""
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[Context truncated for length.]"
        return text

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
            st.error(f"Error extracting text from document: {exc}")
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
        task_description = _normalize_code_task_description(task_description, language)
        reasoning_instructions = ""
        if reasoning_type == "chain_of_thought":
            reasoning_instructions = (
                "Reasoning: use at most 4 short bullet steps, then the final code "
                "(must satisfy the output requirements above)."
            )
        elif reasoning_type == "tree_of_thought":
            reasoning_instructions = (
                "Reasoning: compare two approaches in a short paragraph, pick one, then the final code "
                "(must satisfy the output requirements above)."
            )
        elif reasoning_type == "standard":
            reasoning_instructions = (
                "No long explanation; brief comments in code are enough."
            )
        input_kwargs = {
            "task_description": task_description,
            "language": language,
            "reasoning_instructions": reasoning_instructions,
        }
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
        extraction_format_notes = ""
        if len(fields) == 1:
            extraction_format_notes = (
                "Only one field was requested. If the document splits this topic into labeled "
                "subsections, use each subsection title from the source as its own `##` heading.\n"
            )
        prompt = self.prompt_generator.generate_prompt(
            "data_extraction",
            system_prompt,
            text=combined_text,
            fields=", ".join(fields),
            extraction_format_notes=extraction_format_notes,
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

        rag_limit = int(os.getenv("QA_RAG_CHUNK_LIMIT", "8"))
        relevant_chunks = self.search_chunks(question, limit=rag_limit)
        combined_context = self._build_qa_context(context, relevant_chunks)

        answer_instructions = qa_answer_instructions(question)

        system_prompt = self.prompt_generator.generate_system_prompts(
            "question_answering", reasoning_type=reasoning_type
        )[0]
        prompt = self.prompt_generator.generate_prompt(
            "question_answering",
            system_prompt,
            context=combined_context,
            question=question,
            hybrid_rag_policy=QA_HYBRID_RAG_POLICY,
            answer_instructions="\n".join(answer_instructions),
        )

        try:
            output = self.gemini_model.generate_content(prompt).text.strip()
        except Exception as exc:
            st.error(
                "Gemini request failed (timeout or overload). Wait a moment and try again, "
                "or ask a shorter question. "
                f"Details: {exc}"
            )
            output = f"Error from Gemini: {exc}"

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": 0,
        }
