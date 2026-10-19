"""Parse structured Chain-of-Thought / Tree-of-Thought model outputs."""

from __future__ import annotations

import re
from typing import Dict


def _section(text: str, header_pattern: str, next_headers: str) -> str:
    pattern = (
        rf"(?is){header_pattern}\s*(.*?)(?={next_headers}|\Z)"
    )
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def parse_tree_of_thought_output(text: str) -> Dict[str, str]:
    """Extract Branch A/B, comparison, and final summary from markdown sections."""
    next_after_a = r"##\s*Branch\s*B|##\s*Strategy\s*Comparison|##\s*Final\s*Summary"
    next_after_b = r"##\s*Strategy\s*Comparison|##\s*Final\s*Summary"
    next_after_cmp = r"##\s*Final\s*Summary"

    return {
        "branch_a": _section(text, r"##\s*Branch\s*A", next_after_a),
        "branch_b": _section(text, r"##\s*Branch\s*B", next_after_b),
        "comparison": _section(text, r"##\s*Strategy\s*Comparison", next_after_cmp),
        "final_summary": _section(text, r"##\s*Final\s*Summary", r"\Z"),
    }


def parse_chain_of_thought_output(text: str) -> Dict[str, str]:
    """Extract numbered reasoning steps and final summary."""
    next_step = r"##\s*Step\s*[2-9]|##\s*Final\s*Summary"
    steps = {
        "step_1": _section(text, r"##\s*Step\s*1", next_step),
        "step_2": _section(
            text,
            r"##\s*Step\s*2",
            r"##\s*Step\s*3|##\s*Step\s*4|##\s*Final\s*Summary",
        ),
        "step_3": _section(
            text,
            r"##\s*Step\s*3",
            r"##\s*Step\s*4|##\s*Final\s*Summary",
        ),
        "step_4": _section(text, r"##\s*Step\s*4", r"##\s*Final\s*Summary"),
        "final_summary": _section(text, r"##\s*Final\s*Summary", r"\Z"),
    }
    return steps
