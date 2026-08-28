"""Text normalisation for queries, ES input and VQA answers.

Vietnamese diacritics are preserved everywhere; only casing, unicode form and
control characters are touched.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

# Control characters except tab/newline, which we collapse into spaces first.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")

# Generalized syntactic patterns for lead-in clauses emitted by chatty VLMs:
# 1. Prepositional / Participial lead-in clauses: "Based on the image,", "According to the video frame:", "Looking at the scene,"
_LEADIN_CLAUSE_RE = re.compile(
    r"^(?:based\s+on|according\s+to|looking\s+at|as\s+seen\s+in|from|in|inside|dựa\s+vào|nhìn\s+vào|trong)\s+[^,:\n]{1,80}[,:]\s*",
    re.IGNORECASE,
)

# 2. Metadata / label prefixes: "Answer:", "Final Answer -", "Đáp án là:", "Output:"
_METADATA_PREFIX_RE = re.compile(
    r"^(?:final\s+answer|answer|output|result|response|đáp\s+án(?:\s+là)?|câu\s+trả\s+lời(?:\s+là)?|trả\s+lời|kết\s+quả(?:\s+là)?)\s*[:\-]\s*",
    re.IGNORECASE,
)

# 3. Declarative / Copula lead-in phrases: "The vehicle is a...", "The answer is: ", "There are 5", "I can see a", "It appears to be"
_COPULA_LEADIN_RE = re.compile(
    r"^(?:the\s+[^,:\n]{1,35}?\s+(?:is|appears\s+to\s+be|seems\s+to\s+be|shows)|there\s+(?:are|is|appear\s+to\s+be|seems\s+to\s+be)|i\s+(?:can\s+see|see|observe|notice)|it\s+(?:is|shows|looks\s+like|appears\s+to\s+be)|có\s+thể\s+thấy|tôi\s+thấy)\s*[:\-]?\s*",
    re.IGNORECASE,
)

UNKNOWN = "UNKNOWN"


def normalize_text(text: str | None) -> str:
    """Lowercase, NFC-normalise and strip control characters.

    Diacritics are intentionally kept - Vietnamese retrieval depends on them.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFC", str(text))
    out = out.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    out = _CONTROL_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out).strip()
    return out.lower()


def normalize_query(text: str | None) -> str:
    """Normalisation applied to text going into Elasticsearch."""
    return normalize_text(text)


def clean_query_text(text: str | None) -> str:
    """Collapse whitespace but keep the original casing (LLM input)."""
    if not text:
        return ""
    out = unicodedata.normalize("NFC", str(text))
    out = _CONTROL_RE.sub(" ", out.replace("\r", "\n"))
    lines = [_WS_RE.sub(" ", line).strip() for line in out.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by the LLM."""
    if not text:
        return ""
    out = re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # An unterminated opening tag means everything after it is reasoning.
    out = re.sub(r"<think\b[^>]*>.*\Z", " ", out, flags=re.DOTALL | re.IGNORECASE)
    # A stray closing tag means everything before it was reasoning.
    if "</think>" in out.lower():
        idx = out.lower().rindex("</think>")
        out = out[idx + len("</think>"):]
    return out.strip()


def strip_code_fences(text: str) -> str:
    """Drop ```json ... ``` fences, keeping the fenced body."""
    if not text:
        return ""
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.replace("```json", " ").replace("```JSON", " ").replace("```", " ").strip()


def sanitize_answer(text: str | None, max_chars: int = 100) -> str:
    """Turn raw VLM output into an atomic competition answer.

    General syntactic peeling: strips prepositional clauses, declarative copulas,
    metadata labels, markdown, quotes and trailing punctuation without breaking
    the actual entity/number answer.
    """
    if text is None:
        return ""
    out = unicodedata.normalize("NFC", str(text))
    out = _CONTROL_RE.sub(" ", out)
    out = out.replace("\r", "\n")
    # Keep only the first non-empty line - atomic answers are single-line.
    for line in out.split("\n"):
        if line.strip():
            out = line
            break
    else:
        return ""
    out = _WS_RE.sub(" ", out).strip()

    # Iterative general syntactic peeling:
    # "Based on the provided video frame, the answer is: 5" -> "5"
    changed = True
    while changed:
        changed = False

        # 1. Peel metadata prefixes (e.g. "Answer: ", "Đáp án là: ")
        m_meta = _METADATA_PREFIX_RE.match(out)
        if m_meta:
            out = out[m_meta.end():].strip()
            changed = True

        # 2. Peel lead-in clauses (e.g. "Based on the provided frame, ", "Looking at the scene: ")
        m_clause = _LEADIN_CLAUSE_RE.match(out)
        if m_clause:
            out = out[m_clause.end():].strip()
            changed = True

        # 3. Peel copula phrases (e.g. "The color is ", "There are ", "I can see ")
        m_copula = _COPULA_LEADIN_RE.match(out)
        if m_copula:
            out = out[m_copula.end():].strip()
            changed = True

        # 4. Strip markdown formatting and quotes
        stripped = out.strip("*`_ ").rstrip(" .;!。")
        if stripped != out:
            out, changed = stripped, True
        if len(out) >= 2 and out[0] in "\"'“”‘’" and out[-1] in "\"'“”‘’":
            out, changed = out[1:-1].strip(), True

    out = out.rstrip(" .;:!。，,")
    out = _WS_RE.sub(" ", out).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def is_unknown(answer: str | None) -> bool:
    """True when the VLM declined to answer."""
    if not answer:
        return True
    return answer.strip().upper() in {UNKNOWN, "N/A", "NA", "NONE", "NULL"}


def answer_key(answer: str) -> str:
    """Normalised form used to group answers for majority voting."""
    return normalize_text(answer)


def majority_vote(answers: Iterable[str]) -> str | None:
    """Pick the most common non-empty, non-UNKNOWN answer.

    Ties break on the first-seen answer so results stay deterministic.
    """
    usable = [a for a in answers if a and not is_unknown(a)]
    if not usable:
        return None
    counts: Counter[str] = Counter(answer_key(a) for a in usable)
    order = {answer_key(a): i for i, a in enumerate(reversed(usable))}
    best_key = max(counts, key=lambda k: (counts[k], -order[k]))
    for a in usable:
        if answer_key(a) == best_key:
            return a
    return usable[0]


def truncate(text: str, max_chars: int) -> str:
    """Hard-truncate without breaking the caller's expectations."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def looks_like_exact_text(query: str | None, terms: Iterable[str] | None = None) -> bool:
    """Heuristic: did the user ask for a literal on-screen string?

    Quoted spans, ALL-CAPS tokens and explicit "chữ/dòng chữ/biển" markers all
    mean the OCR phrase match should be boosted hard.
    """
    if not query:
        return False
    if re.search(r"[\"“”'‘’]", query):
        return True
    if re.search(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*\b", query):  # VinFast, ABC
        return True
    markers = ("chữ", "dòng chữ", "biển", "ghi rõ", "có tên", "exact text", "the word")
    lowered = query.lower()
    return any(m in lowered for m in markers)
