"""LLM query decomposition into OCR / ASR / IMAGE retrieval sub-queries.

The system prompt decomposes queries into:
- OCR: Literal on-screen text (preserves original Vietnamese text and diacritics).
- ASR: Spoken words/dialogue (preserves original Vietnamese speech).
- IMAGE: Rich English visual caption for CLIP/SigLIP/BEiT-3/Qwen embeddings.
"""

from __future__ import annotations

import re
from typing import Any

from ..clients.llm import LLMClient
from ..logging_utils import get_logger
from ..schemas import (
    ALL_PATHS,
    DecomposeResult,
    LLMParseError,
    MODALITY_TO_PATH,
    PATH_ASR,
    PATH_DESCRIPTION,
    PATH_OCR,
    PATH_TO_MODALITY,
    PATH_VISUAL,
)
from ..utils.text_norm import clean_query_text, looks_like_exact_text

log = get_logger(__name__)

_VIETNAMESE_DIACRITICS_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def _is_vietnamese(text: str | None) -> bool:
    """Return True if the text contains Vietnamese accented characters."""
    if not text:
        return False
    return bool(_VIETNAMESE_DIACRITICS_RE.search(text))


SYSTEM_PROMPT = """
You are a multimodal video scene retrieval query decomposition model.
Your task is to decompose a user's natural-language search query into independent retrieval queries for four modalities:

1. OCR (On-screen text)
   - Text that may appear visibly inside video frames (signs, banners, license plates, shop names, logos, subtitles, documents, brand names).
   - RULE: MUST preserve the EXACT literal text from the query in its original language (keep Vietnamese diacritics/accents intact). NEVER translate OCR text to English.

2. ASR (Speech / Narration / Dialogue)
   - Spoken words, voiceover, dialogue, or spoken topics in the video.
   - RULE: MUST preserve original spoken terms/dialogue in original language (keep Vietnamese diacritics). NEVER translate ASR to English.

3. DESCRIPTION (Video Scene / Action Dense Caption)
   - Semantic context, actions, events, and scene descriptions matching video caption annotations in Elasticsearch.
   - RULE: MUST preserve the Vietnamese natural description from the query (keep Vietnamese diacritics).

4. IMAGE (Visual Content)
   - Visually observable content (people, vehicles, objects, actions, scene composition, colors, postures, environments).
   - RULE: MUST be translated into a concise but rich, descriptive English visual caption suitable for SigLIP/BEiT-3/Qwen embedding retrieval. The IMAGE query MUST ALWAYS be in English.

IMPORTANT RULES:
- Determine which modalities are actually useful. A query may use one, two, three, or all four modalities.
- If a modality is not needed, return null.
- Do not hallucinate information that is not present in the query.
- Output "modality_weights": Four floats in [0, 1] that sum to 1.0. Give 0.0 only to modalities set to null.
- Output ONLY valid JSON, no markdown fences, no explanation.

FEW-SHOT EXAMPLES:

Example 1 (Scene & Visual):
Input: "người đàn ông đang nấu ăn trong bếp"
Output: {
  "original_query": "người đàn ông đang nấu ăn trong bếp",
  "modalities": ["description", "image"],
  "ocr_query": null,
  "asr_query": null,
  "description_query": "người đàn ông đang nấu ăn trong bếp",
  "image_query": "a man cooking food inside a kitchen with pans and kitchenware",
  "ocr_terms": [],
  "asr_terms": [],
  "description_terms": ["nấu ăn", "bếp"],
  "image_terms": ["man cooking", "kitchen"],
  "modality_weights": {"ocr": 0.0, "asr": 0.0, "description": 0.4, "image": 0.6}
}

Example 2 (OCR + Scene + Visual):
Input: "xe buýt màu xanh lá cây có chữ Bến Thành"
Output: {
  "original_query": "xe buýt màu xanh lá cây có chữ Bến Thành",
  "modalities": ["ocr", "description", "image"],
  "ocr_query": "Bến Thành",
  "asr_query": null,
  "description_query": "xe buýt màu xanh lá cây",
  "image_query": "a green public transit bus on the city road",
  "ocr_terms": ["Bến Thành"],
  "asr_terms": [],
  "description_terms": ["xe buýt", "màu xanh"],
  "image_terms": ["green bus"],
  "modality_weights": {"ocr": 0.4, "asr": 0.0, "description": 0.3, "image": 0.3}
}

Example 3 (ASR + Scene + Visual):
Input: "phóng viên nói về dự báo thời tiết tối nay"
Output: {
  "original_query": "phóng viên nói về dự báo thời tiết tối nay",
  "modalities": ["asr", "description", "image"],
  "ocr_query": null,
  "asr_query": "dự báo thời tiết tối nay",
  "description_query": "phóng viên trường quay dự báo thời tiết",
  "image_query": "a news reporter or anchor speaking on a broadcast news set",
  "ocr_terms": [],
  "asr_terms": ["dự báo thời tiết"],
  "description_terms": ["phóng viên", "thời sự"],
  "image_terms": ["news reporter"],
  "modality_weights": {"ocr": 0.0, "asr": 0.3, "description": 0.3, "image": 0.4}
}

Output schema:
{
  "original_query": "...",
  "modalities": ["ocr", "asr", "description", "image"],
  "ocr_query": "... or null",
  "asr_query": "... or null",
  "description_query": "... or null",
  "image_query": "... or null",
  "ocr_terms": [],
  "asr_terms": [],
  "description_terms": [],
  "image_terms": [],
  "modality_weights": {"ocr": 0.0, "asr": 0.0, "description": 0.5, "image": 0.5}
}
""".strip()

SCHEMA_HINT = (
    '{"original_query": str, "modalities": [str], "ocr_query": str|null, '
    '"asr_query": str|null, "description_query": str|null, "image_query": str|null, "ocr_terms": [str], '
    '"asr_terms": [str], "description_terms": [str], "image_terms": [str], '
    '"modality_weights": {"ocr": float, "asr": float, "description": float, "image": float}}'
)


def translate_to_english_visual(llm: LLMClient, query: str, *, use_cache: bool = True) -> str:
    """Translate or expand a user query into a descriptive English visual caption."""
    system = (
        "You are an expert visual translator for video retrieval. "
        "Translate the user query into a concise, rich English visual description "
        "describing the observable scene (objects, colors, actions, context, environment). "
        "Output ONLY the English description string, no quotes, no explanations."
    )
    try:
        translated = llm.chat(system, query, use_cache=use_cache).strip().strip('"\'')
        if translated.startswith("{") and translated.endswith("}"):
            return query
        return clean_query_text(translated) or query
    except Exception as exc:
        log.warning("Visual translation fallback failed: %s", exc)
        return query


def decompose(
    llm: LLMClient,
    query: str,
    *,
    adaptive_floor: float = 0.0,
    default_weights: dict[str, float] | None = None,
    use_cache: bool = True,
) -> DecomposeResult:
    """Decompose a natural-language query into retrieval sub-queries across modalities."""
    cleaned = clean_query_text(query)
    if not cleaned:
        raise ValueError("Cannot decompose an empty query")

    try:
        payload = llm.chat_json(
            SYSTEM_PROMPT, cleaned, schema_hint=SCHEMA_HINT, use_cache=use_cache
        )
    except Exception as exc:
        log.error("Decomposition failed, falling back to the raw query: %s", exc)
        payload = {}

    result = _parse_payload(payload, cleaned)
    _apply_fallback(result, cleaned, llm, use_cache=use_cache)

    # Programmatic Guardrail: If image_query contains Vietnamese diacritics, auto-translate to English
    if result.image_query and _is_vietnamese(result.image_query):
        log.info(
            "Vietnamese text detected in image_query (%r), auto-translating to English visual caption",
            result.image_query[:40],
        )
        result.image_query = translate_to_english_visual(llm, result.image_query, use_cache=use_cache)

    _finalise_weights(result, adaptive_floor, default_weights or {})
    result_dict = result.to_dict()
    log.debug(
        "Decomposed %r -> %s",
        cleaned[:60],
        result_dict,
        extra={"progress": {
            "phase": "decompose",
            "status": "done",
            "title": "Decompose hoàn tất",
            "detail": ", ".join(result.modalities) or "raw query fallback",
            "inspector": {
                "Visual query": result.image_query or "",
                "Description query": result.description_query or "",
                "OCR terms": ", ".join(result.ocr_terms),
                "ASR terms": ", ".join(result.asr_terms),
                "Weights": str(result.modality_weights),
            },
        }},
    )
    return result


def _parse_payload(payload: dict[str, Any], original: str) -> DecomposeResult:
    """Convert the raw JSON object into a validated dataclass."""
    result = DecomposeResult(original_query=original)
    result.ocr_query = _clean_optional(payload.get("ocr_query"))
    result.asr_query = _clean_optional(payload.get("asr_query"))
    result.description_query = _clean_optional(payload.get("description_query"))
    result.image_query = _clean_optional(payload.get("image_query"))
    result.ocr_terms = _clean_terms(payload.get("ocr_terms"))
    result.asr_terms = _clean_terms(payload.get("asr_terms"))
    result.description_terms = _clean_terms(payload.get("description_terms"))
    result.image_terms = _clean_terms(payload.get("image_terms"))

    declared = payload.get("modalities")
    if isinstance(declared, list):
        result.modalities = [
            str(m).strip().lower()
            for m in declared
            if str(m).strip().lower() in MODALITY_TO_PATH
        ]
        # If description is declared but description_query was omitted, default to original
        if "description" in result.modalities and result.description_query is None:
            result.description_query = original
    else:
        if result.description_query:
            result.modalities.append("description")

    weights = payload.get("modality_weights")
    if isinstance(weights, dict):
        for modality, value in weights.items():
            key = str(modality).strip().lower()
            if key in MODALITY_TO_PATH:
                try:
                    result.modality_weights[key] = float(value)
                except (TypeError, ValueError):
                    continue

    result.exact_text = looks_like_exact_text(original, result.ocr_terms)
    return result


def _apply_fallback(
    result: DecomposeResult,
    original: str,
    llm: LLMClient | None = None,
    *,
    use_cache: bool = True,
) -> None:
    """When the LLM disables every path, retrieve with the raw query instead."""
    if any((result.ocr_query, result.asr_query, result.description_query, result.image_query)):
        # Keep ``modalities`` consistent with the queries that actually exist.
        result.modalities = [
            modality
            for modality, path in MODALITY_TO_PATH.items()
            if result.query_for(path)
        ]
        return

    log.warning(
        "Decomposition produced no usable sub-query - falling back to the raw "
        "query on all paths with English visual translation"
    )
    result.ocr_query = original
    result.asr_query = original
    result.description_query = original
    if llm is not None:
        result.image_query = translate_to_english_visual(llm, original, use_cache=use_cache)
    else:
        result.image_query = original
    result.modalities = list(MODALITY_TO_PATH)
    result.modality_weights = {}


def _finalise_weights(
    result: DecomposeResult, floor: float, defaults: dict[str, float]
) -> None:
    """Clamp / floor / normalise the weights over the paths that are active."""
    from .fusion import normalize_weights

    active = [path for path in ALL_PATHS if result.query_for(path)]
    raw = result.path_weights()
    if raw:
        # LLM-produced: cap each weight at 1.0 as the prompt specifies.
        normalised = normalize_weights(raw, active, floor=floor, clamp_max=1.0)
    else:
        # Config defaults are relative ratios, so no upper clamp.
        raw = {path: float(defaults.get(path, 1.0)) for path in active}
        normalised = normalize_weights(raw, active, floor=floor)

    result.modality_weights = {
        PATH_TO_MODALITY[path]: weight
        for path, weight in normalised.items()
        if path in PATH_TO_MODALITY
    }


def _clean_optional(value: Any) -> str | None:
    """Normalise a possibly-null string field from the model."""
    if value is None:
        return None
    text = clean_query_text(str(value))
    if not text or text.lower() in {"null", "none", "n/a", "-"}:
        return None
    return text


def _clean_terms(value: Any) -> list[str]:
    """Normalise a list-of-strings field, dropping blanks and duplicates."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = clean_query_text(str(item))
        if not text or text.lower() in {"null", "none"}:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(text)
    return out
