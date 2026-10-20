from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
import hashlib
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
    "Hybrid RAG: (1) Document facts only from Context—if missing, say so. "
    "(2) For explain/define/summarize: what Context says, then plain-language teaching—no invented facts. "
    "(3) No 'why it fits' unless the question asks why/reason/purpose."
)


_COMPARE_BRANCH_RE = re.compile(
    r"\b("
    r"branch out|better fit|compare|comparison|contrast|versus|vs\.?|"
    r"two different roles?|evaluate .+ for two|for two different|"
    r"role\s*[AB]\)|both roles|which one is a better"
    r")\b",
    re.IGNORECASE,
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
    wants_compare = bool(_COMPARE_BRANCH_RE.search(q))

    if wants_compare:
        instructions.append(
            "Answer mode (compare / branch): Evaluate each role separately using only the Context. "
            "Use markdown headings: `## Role A` (or the first role name), `## Role B` (second role), "
            "`## Comparison`, then `## Conclusion` (which role is the stronger fit and why)."
        )
    elif wants_why:
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

    if re.search(
        r"\b(walk me through|thought process|step[- ]by[- ]step|prove the candidate|"
        r"which .+ (skills|projects).+ from)\b",
        q,
        re.IGNORECASE,
    ):
        instructions.append(
            "Reasoning walkthrough: use short numbered steps for your thinking, then a clear "
            "conclusion that cites only skills and projects named in the Context. "
            "Keep the full answer under about 450 words."
        )

    if wants_factual or (wants_list and not wants_explain):
        instructions.append("Be brief; do not repeat the question.")
    else:
        instructions.append(
            "Be clear and complete but not verbose; do not repeat the question or restate the policy."
        )

    return instructions


def _qa_question_flags(question: str) -> Dict[str, bool]:
    q = question.strip()
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
        or (re.search(r"\bwhat (is|are)\b", q, re.IGNORECASE) and not wants_factual)
    )
    walkthrough = bool(
        re.search(
            r"\b(walk me through|thought process|step[- ]by[- ]step|prove the candidate|"
            r"which .+ (skills|projects).+ from)\b",
            q,
            re.IGNORECASE,
        )
    )
    wants_compare = bool(_COMPARE_BRANCH_RE.search(q))
    high_marks = bool(re.search(r"\b(\d{2,})\s*marks\b", q, re.IGNORECASE))
    if not high_marks:
        marks_match = re.search(r"\b(\d+)\s*marks\b", q, re.IGNORECASE)
        high_marks = bool(marks_match and int(marks_match.group(1)) >= 8)
    return {
        "wants_why": wants_why,
        "wants_how": wants_how,
        "wants_list": wants_list,
        "wants_factual": wants_factual,
        "wants_explain": wants_explain,
        "walkthrough": walkthrough,
        "wants_compare": wants_compare,
        "high_marks": high_marks,
    }


def qa_retrieval_profile(question: str) -> Dict[str, int]:
    """Tune RAG + token budget by question type (fast factual, richer explain)."""
    flags = _qa_question_flags(question)
    if flags["wants_factual"]:
        return {
            "lead_chars": int(os.getenv("QA_FACTUAL_LEAD_CHARS", "0")),
            "rag_limit": int(os.getenv("QA_FACTUAL_RAG_CHUNK_LIMIT", "4")),
            "max_chars": int(os.getenv("QA_FACTUAL_CONTEXT_MAX_CHARS", "6500")),
            "max_output_tokens": int(os.getenv("QA_FACTUAL_MAX_OUTPUT_TOKENS", "896")),
        }
    if flags["wants_list"] and not flags["wants_explain"]:
        return {
            "lead_chars": int(os.getenv("QA_LIST_LEAD_CHARS", "0")),
            "rag_limit": int(os.getenv("QA_LIST_RAG_CHUNK_LIMIT", "7")),
            "max_chars": int(os.getenv("QA_LIST_CONTEXT_MAX_CHARS", "10000")),
            "max_output_tokens": int(os.getenv("QA_LIST_MAX_OUTPUT_TOKENS", "1536")),
        }
    if flags["wants_compare"] or flags["walkthrough"] or flags["high_marks"]:
        return {
            "lead_chars": int(os.getenv("QA_DEEP_LEAD_CHARS", "0")),
            "rag_limit": int(os.getenv("QA_DEEP_RAG_CHUNK_LIMIT", "8")),
            "max_chars": int(os.getenv("QA_DEEP_CONTEXT_MAX_CHARS", "14000")),
            "max_output_tokens": int(os.getenv("QA_DEEP_MAX_OUTPUT_TOKENS", "3072")),
        }
    if flags["wants_explain"] or flags["wants_why"] or flags["wants_how"]:
        return {
            "lead_chars": int(os.getenv("QA_EXPLAIN_LEAD_CHARS", "900")),
            "rag_limit": int(os.getenv("QA_EXPLAIN_RAG_CHUNK_LIMIT", "6")),
            "max_chars": int(os.getenv("QA_EXPLAIN_CONTEXT_MAX_CHARS", "11000")),
            "max_output_tokens": int(os.getenv("QA_EXPLAIN_MAX_OUTPUT_TOKENS", "2560")),
        }
    return {
        "lead_chars": int(os.getenv("QA_CONTEXT_LEAD_CHARS", "600")),
        "rag_limit": int(os.getenv("QA_RAG_CHUNK_LIMIT", "5")),
        "max_chars": int(os.getenv("QA_CONTEXT_MAX_CHARS", "8500")),
        "max_output_tokens": int(os.getenv("QA_DEFAULT_MAX_OUTPUT_TOKENS", "2048")),
    }


def qa_reasoning_ui_hint(reasoning_type: str, question: str = "") -> str:
    wants_compare = bool(_COMPARE_BRANCH_RE.search(question or ""))
    if reasoning_type == "chain_of_thought":
        if wants_compare:
            return (
                "Chain-of-thought: under each role heading, use 2–3 bullets of evidence from the "
                "Context, then Comparison and Conclusion."
            )
        return "Use 2–4 short numbered reasoning steps, then the final answer."
    if reasoning_type == "tree_of_thought":
        if wants_compare:
            return (
                "Tree-of-thought: treat each role as a branch (Role A vs Role B), compare strengths "
                "and gaps from the Context only, then conclude which role is the better fit."
            )
        return "Briefly note two approaches, pick one, then give the final answer."
    return ""


def _gemini_task_model_chain(
    models_env_key: str,
    primary_env_key: str,
    default_primary: str,
) -> List[str]:
    env_chain = os.getenv(models_env_key, "").strip()
    if env_chain:
        return [m.strip() for m in env_chain.split(",") if m.strip()]
    primary = os.getenv(primary_env_key, default_primary).strip()
    fallbacks = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
    seen: set[str] = set()
    chain: List[str] = []
    for name in [primary] + fallbacks:
        if name and name not in seen:
            seen.add(name)
            chain.append(name)
    return chain


def _qa_gemini_model_chain() -> List[str]:
    return _gemini_task_model_chain(
        "GEMINI_QA_MODELS", "GEMINI_QA_MODEL", "gemini-3.5-flash"
    )


def _extraction_gemini_model_chain() -> List[str]:
    return _gemini_task_model_chain(
        "GEMINI_EXTRACTION_MODELS",
        "GEMINI_EXTRACTION_MODEL",
        "gemini-3.5-flash",
    )


def _default_gemini_model_chain() -> List[str]:
    """Summarization, code generation, and standard optimize_prompt."""
    return _gemini_task_model_chain(
        "GEMINI_MODELS", "GEMINI_MODEL", "gemini-3.5-flash"
    )


def _clip_input_text(text: str, env_key: str, default_limit: int) -> str:
    limit = int(os.getenv(env_key, str(default_limit)))
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n\n[Input truncated for length.]"


def _summarization_generate_kwargs() -> Dict[str, Any]:
    return {
        "model_chain": _default_gemini_model_chain(),
        "timeout": int(os.getenv("GEMINI_SUMMARIZATION_TIMEOUT_SECONDS", "90")),
        "max_output_tokens": int(
            os.getenv("GEMINI_SUMMARIZATION_MAX_OUTPUT_TOKENS", "2048")
        ),
        "max_retries": int(os.getenv("GEMINI_TASK_MAX_RETRIES", "1")),
        "temperature": 0.3,
    }


def _code_generate_kwargs() -> Dict[str, Any]:
    return {
        "model_chain": _default_gemini_model_chain(),
        "timeout": int(os.getenv("GEMINI_CODE_TIMEOUT_SECONDS", "120")),
        "max_output_tokens": int(os.getenv("GEMINI_CODE_MAX_OUTPUT_TOKENS", "3072")),
        "max_retries": int(os.getenv("GEMINI_TASK_MAX_RETRIES", "1")),
        "temperature": 0.2,
    }


def _task_generate_kwargs(task_type: str) -> Dict[str, Any]:
    if task_type == "code_generation":
        return _code_generate_kwargs()
    return _summarization_generate_kwargs()


_PKG_DIR = Path(__file__).resolve().parent
load_dotenv(_PKG_DIR / ".env")

COLLECTION_NAME = "document_collection"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000


def pdf_document_id(pdf_bytes: bytes) -> str:
    """Stable ID from raw PDF bytes (same file => same ID regardless of filename)."""
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    return f"pdf-{digest}"


def text_document_id(text: str) -> str:
    """Stable ID for pasted text (no PDF file)."""
    normalized = (text or "").strip()
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    return f"txt-{digest}"


DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
PROMPT_VARIANT_LIMIT = int(os.getenv("PROMPT_VARIANT_LIMIT", "1"))

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
        r"accept|read|parse|convert|extract|generate|return|given|takes?|accepts?)\b",
        stripped,
        flags=re.IGNORECASE,
    ):
        stripped = f"Use {stripped}"
    return stripped or text


class SpecializedApplications:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or ""
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.gemini_model = get_gemini_model(model_name, api_key)
        self.prompt_generator = PromptGenerator()
        self.evaluator = PromptEvaluator()
        self._qdrant_url, self._qdrant_api_key = qdrant_config()
        self._indexed_content_fingerprint: Optional[str] = None
        self._indexed_document_id: Optional[str] = None
        self._last_document_index: Dict[str, Any] = {}
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
                "Answer in clear markdown (headings and bullets when useful).\n"
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

    @staticmethod
    def _document_fingerprint(text: str) -> str:
        normalized = (text or "").strip()
        return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()

    def _qdrant_index_snapshot(self) -> Dict[str, Any]:
        """Read document_id, text fingerprint, and vector count from Qdrant."""
        empty: Dict[str, Any] = {
            "document_id": None,
            "doc_fingerprint": None,
            "vector_count": 0,
        }
        try:
            client = self._require_qdrant()
        except RuntimeError:
            return empty
        if COLLECTION_NAME not in client.list_collections():
            return empty
        info = client.collection_info(COLLECTION_NAME)
        count = int(info.get("points_count") or 0)
        if count <= 0:
            return empty
        points = client.scroll_points(COLLECTION_NAME, limit=1, with_payload=True)
        if not points:
            return {**empty, "vector_count": count}
        payload = points[0].get("payload") or {}
        stored_id = payload.get("document_id")
        stored_fp = payload.get("doc_fingerprint")
        return {
            "document_id": stored_id if isinstance(stored_id, str) and stored_id else None,
            "doc_fingerprint": stored_fp if isinstance(stored_fp, str) and stored_fp else None,
            "vector_count": count,
        }

    def qdrant_document_status(
        self,
        text: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        snap = self._qdrant_index_snapshot()
        vector_count = int(snap.get("vector_count") or 0)
        stored_id = snap.get("document_id")
        stored_fp = snap.get("doc_fingerprint")
        matches_document_id = bool(
            document_id and stored_id and document_id == stored_id
        )
        matches_text = False
        if text and text.strip() and stored_fp:
            matches_text = stored_fp == self._document_fingerprint(text)
        return {
            "indexed": vector_count > 0,
            "vector_count": vector_count,
            "document_id": document_id,
            "stored_document_id": stored_id,
            "matches_document_id": matches_document_id,
            "matches_text": matches_text,
            "ready": vector_count > 0 and (matches_document_id or matches_text),
            "stored_fingerprint": stored_fp,
        }

    def index_document(
        self,
        text: str,
        *,
        document_id: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Index once per document; reuse when Qdrant already has the same document_id."""
        if not text or not str(text).strip():
            st.error("No document text to index.")
            self._last_document_index = {"ok": False, "reused": False, "vector_count": 0}
            return False

        fingerprint = self._document_fingerprint(text)
        if not document_id:
            document_id = text_document_id(text)

        snap = self._qdrant_index_snapshot()
        vector_count = int(snap.get("vector_count") or 0)
        stored_id = snap.get("document_id")
        stored_fp = snap.get("doc_fingerprint")

        def _reuse_message() -> str:
            short_id = document_id[:18] + "…" if len(document_id) > 19 else document_id
            return (
                f"Document already uploaded (ID `{short_id}`) — "
                f"reusing {vector_count} vectors in Qdrant."
            )

        if not force:
            if (
                self._indexed_document_id == document_id
                and vector_count > 0
            ):
                self._indexed_content_fingerprint = fingerprint
                self._last_document_index = {
                    "ok": True,
                    "reused": True,
                    "vector_count": vector_count,
                    "document_id": document_id,
                    "message": _reuse_message(),
                }
                return True
            if stored_id == document_id and vector_count > 0:
                self._indexed_document_id = document_id
                self._indexed_content_fingerprint = fingerprint
                self._last_document_index = {
                    "ok": True,
                    "reused": True,
                    "vector_count": vector_count,
                    "document_id": document_id,
                    "message": _reuse_message(),
                }
                return True
            if stored_fp == fingerprint and vector_count > 0 and not stored_id:
                self._indexed_document_id = document_id
                self._indexed_content_fingerprint = fingerprint
                self._last_document_index = {
                    "ok": True,
                    "reused": True,
                    "vector_count": vector_count,
                    "document_id": document_id,
                    "message": _reuse_message(),
                }
                return True

        if not self._setup_rag_collection():
            self._last_document_index = {"ok": False, "reused": False, "vector_count": 0}
            return False
        chunks = self.chunk_text(text)
        embeddings = self.get_embeddings(chunks)
        if embeddings is None or not self.upload_to_qdrant(
            chunks,
            embeddings,
            document_id=document_id,
            doc_fingerprint=fingerprint,
        ):
            self._last_document_index = {"ok": False, "reused": False, "vector_count": 0}
            return False
        self._indexed_document_id = document_id
        self._indexed_content_fingerprint = fingerprint
        vector_count = len(chunks)
        short_id = document_id[:18] + "…" if len(document_id) > 19 else document_id
        self._last_document_index = {
            "ok": True,
            "reused": False,
            "vector_count": vector_count,
            "document_id": document_id,
            "message": (
                f"Document uploaded (ID `{short_id}`) — indexed {vector_count} vectors in Qdrant."
            ),
        }
        return True

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

    def _replace_collection_vectors(
        self,
        chunks: list,
        embeddings: np.ndarray,
        document_id: str,
        doc_fingerprint: str,
    ) -> None:
        client = self._require_qdrant()
        if COLLECTION_NAME in client.list_collections():
            client.delete_collection(COLLECTION_NAME)
        client.ensure_collection(COLLECTION_NAME, VECTOR_SIZE, distance="Cosine")
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": embedding.tolist(),
                "payload": {
                    "text": chunk,
                    "document_id": document_id,
                    "doc_fingerprint": doc_fingerprint,
                    "chunk_index": idx,
                },
            }
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        client.upsert_points(COLLECTION_NAME, points)

    def upload_to_qdrant(
        self,
        chunks: list,
        embeddings: np.ndarray,
        *,
        document_id: str,
        doc_fingerprint: str,
    ) -> bool:
        if not chunks or embeddings is None or embeddings.size == 0:
            st.error("No text chunks to index in Qdrant.")
            return False
        try:
            self._replace_collection_vectors(
                chunks, embeddings, document_id, doc_fingerprint
            )
            return True
        except Exception as exc:
            st.error(f"Qdrant upload failed: {exc}")
            return False

    def _ensure_qa_index(self, context: str, document_id: Optional[str]) -> bool:
        if document_id:
            status = self.qdrant_document_status(document_id=document_id)
            if status.get("matches_document_id") and status.get("indexed"):
                return True
        return self.index_document(context, document_id=document_id)

    def _build_qa_context(
        self,
        context: str,
        relevant_chunks: list,
        profile: Optional[Dict[str, int]] = None,
    ) -> str:
        """RAG chunks plus an optional document lead-in (skipped for factual/list when profile says so)."""
        profile = profile or {}
        max_chars = profile.get("max_chars") or int(os.getenv("QA_CONTEXT_MAX_CHARS", "8500"))
        lead_chars = profile.get("lead_chars")
        if lead_chars is None:
            lead_chars = int(os.getenv("QA_CONTEXT_LEAD_CHARS", "600"))
        parts: list[str] = []
        if context and lead_chars > 0:
            lead = context[:lead_chars].strip()
            if lead:
                parts.append(lead)
        seen_normalized: set[str] = set()
        for chunk in relevant_chunks:
            text = str(chunk).strip() if chunk else ""
            if not text:
                continue
            norm = re.sub(r"\s+", " ", text)[:200]
            if norm in seen_normalized:
                continue
            if any(text in existing or existing in text for existing in parts):
                continue
            seen_normalized.add(norm)
            parts.append(text)
        if not parts:
            text = context or ""
            if len(text) > max_chars:
                return text[:max_chars] + "\n\n[Context truncated for length.]"
            return text
        merged = "\n\n---\n\n".join(parts)
        if len(merged) > max_chars:
            merged = merged[:max_chars] + "\n\n[Context truncated.]"
        return merged

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

        gen_kwargs = _task_generate_kwargs(task_type)
        last_api_error = None
        for system_prompt in system_prompts:
            prompt = self.prompt_generator.generate_prompt(
                template_name, system_prompt, **input_kwargs
            )
            try:
                response = model_to_use.generate_content(prompt, **gen_kwargs)
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

        text = _clip_input_text(text, "SUMMARIZATION_MAX_INPUT_CHARS", 14000)
        system_prompt = TREE_OF_THOUGHT_SUMMARY_SYSTEM
        prompt = self.prompt_generator.generate_prompt(
            "summarization_tree_of_thought",
            system_prompt,
            text=text,
        )
        gen_kwargs = _summarization_generate_kwargs()
        gen_kwargs["max_output_tokens"] = int(
            os.getenv("GEMINI_SUMMARIZATION_COT_MAX_OUTPUT_TOKENS", "2560")
        )
        try:
            output = self.gemini_model.generate_content(prompt, **gen_kwargs).text
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

        text = _clip_input_text(text, "SUMMARIZATION_MAX_INPUT_CHARS", 14000)
        system_prompt = CHAIN_OF_THOUGHT_SUMMARY_SYSTEM
        prompt = self.prompt_generator.generate_prompt(
            "summarization_chain_of_thought",
            system_prompt,
            text=text,
        )
        gen_kwargs = _summarization_generate_kwargs()
        gen_kwargs["max_output_tokens"] = int(
            os.getenv("GEMINI_SUMMARIZATION_COT_MAX_OUTPUT_TOKENS", "2560")
        )
        try:
            output = self.gemini_model.generate_content(prompt, **gen_kwargs).text
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
        text = _clip_input_text(text, "SUMMARIZATION_MAX_INPUT_CHARS", 14000)
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

    def extract_data(
        self,
        text: str,
        fields: list,
        reasoning_type: str,
        document_id: Optional[str] = None,
    ):
        if not self.index_document(text, document_id=document_id):
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
            output = self.gemini_model.generate_content(
                prompt,
                model_chain=_extraction_gemini_model_chain(),
            ).text
        except Exception as exc:
            output = f"Error calling Gemini API for data extraction: {exc}"
            st.error(output)

        return {
            "output": output,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "score": 0,
        }

    def answer_question(
        self,
        context: str,
        question: str,
        reasoning_type: str,
        document_id: Optional[str] = None,
    ):
        if not self._ensure_qa_index(context, document_id):
            return {
                "output": "Question answering failed: could not index context in Qdrant.",
                "system_prompt": "",
                "prompt": "",
                "score": 0,
            }

        profile = qa_retrieval_profile(question)
        max_context = profile["max_chars"]
        stripped_context = (context or "").strip()
        if stripped_context and len(stripped_context) <= max_context:
            combined_context = stripped_context
        else:
            rag_limit = profile["rag_limit"]
            relevant_chunks = self.search_chunks(question, limit=rag_limit)
            combined_context = self._build_qa_context(
                context, relevant_chunks, profile
            )

        answer_instructions = qa_answer_instructions(question)
        reasoning_hint = qa_reasoning_ui_hint(reasoning_type, question)
        if reasoning_hint:
            answer_instructions.append(reasoning_hint)

        system_prompt = self.prompt_generator.generate_system_prompts(
            "question_answering",
            reasoning_type="standard",
        )[0]
        prompt = self.prompt_generator.generate_prompt(
            "question_answering",
            system_prompt,
            context=combined_context,
            question=question,
            hybrid_rag_policy=QA_HYBRID_RAG_POLICY,
            answer_instructions="\n".join(answer_instructions),
        )

        qa_timeout = int(os.getenv("GEMINI_QA_TIMEOUT_SECONDS", "120"))
        qa_max_tokens = profile["max_output_tokens"]
        flags = _qa_question_flags(question)
        qa_temperature = 0.2 if flags["wants_factual"] else 0.35
        qa_retries = int(os.getenv("GEMINI_QA_MAX_RETRIES", "1"))
        try:
            output = self.gemini_model.generate_content(
                prompt,
                timeout=qa_timeout,
                max_output_tokens=qa_max_tokens,
                temperature=qa_temperature,
                model_chain=_qa_gemini_model_chain(),
                max_retries=qa_retries,
            ).text.strip()
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
