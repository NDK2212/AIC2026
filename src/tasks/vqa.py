"""Task 2 - Video Question Answering.

Flow: split the query into *scene description* + *question*, retrieve on the
description alone (the interrogative part only pollutes retrieval), build the
100 rows, enrich metadata from Elasticsearch, ask the LLM about representative
frames per shot using multimodal context (Description + OCR + ASR), and then
propagate those answers to the remaining rows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from ..clients.llm import LLMClient
from ..config import Config
from ..logging_utils import get_logger
from ..retrieval.pipeline import RetrievalPipeline
from ..schemas import Candidate, VQASplit
from ..submission.builder import build_rows
from ..utils.keyframe_index import KeyframeIndex
from ..utils.text_norm import clean_query_text, is_unknown, majority_vote, sanitize_answer

log = get_logger(__name__)


SPLIT_SYSTEM_PROMPT = """
You split a Vietnamese/English video search Q&A query into retrieval and question components.
Output ONLY valid JSON, no markdown, no explanation.

Schema:
{
  "scene_description": "the scene/event to be retrieved (in original language, without question words)",
  "question": "the original standalone question in original language",
  "question_en": "the standalone question translated accurately into clear, direct English",
  "expected_answer_type": "number | color | name | object | person_count | yes_no | action | other"
}

Rules:
- scene_description must describe observable visual elements for video retrieval.
- question is the question being asked.
- question_en MUST be a clear, standalone English question.
""".strip()

SPLIT_SCHEMA_HINT = (
    '{"scene_description": str, "question": str, "question_en": str, "expected_answer_type": str}'
)

VQA_LLM_SYSTEM_PROMPT = """
You are an ultra-precise video question answering assistant for a video retrieval competition.
You are given candidate video frame context (scene dense description, on-screen OCR text, and spoken ASR audio dialogue) and a question.

Your task is to deduce the answer based on the provided frame context and question.

RESPONSE RULES:
1. Answer in the SAME LANGUAGE as the question:
   - Vietnamese question -> Answer in concise Vietnamese (e.g. "màu đỏ", "cảnh sát", "nấu ăn").
   - English question    -> Answer in concise English (e.g. "red", "police officer", "cooking").
2. Number / Count / Time questions: ALWAYS output digits only (e.g. "5", "12", "0").
3. Text / Sign / OCR questions: Extract the exact text string from OCR (preserve original spelling and diacritics).
4. Yes / No questions: Output "đúng" / "sai" (or "yes" / "no" for English questions).
5. If the context does not contain enough information -> Output exactly: unknown

HARD RULES:
- Maximum 100 characters. Aim for under 20 characters.
- NO full sentences. NO explanations. NO trailing punctuation.
- NO prefixes such as "Answer:", "Đáp án:", "Câu trả lời là:", "The answer is:".
- Output ONLY the raw atomic answer string.
""".strip()


def split_query(llm: LLMClient, query: str, *, use_cache: bool = True) -> VQASplit:
    """Separate the retrievable scene description from the actual question."""
    cleaned = clean_query_text(query)
    try:
        payload = llm.chat_json(
            SPLIT_SYSTEM_PROMPT, cleaned, schema_hint=SPLIT_SCHEMA_HINT, use_cache=use_cache
        )
    except Exception as exc:
        log.error("Q&A split failed (%s) - using the whole query for both parts", exc)
        payload = {}

    scene = clean_query_text(str(payload.get("scene_description") or "")) or cleaned
    question = clean_query_text(str(payload.get("question") or "")) or cleaned
    question_en = clean_query_text(str(payload.get("question_en") or "")) or question
    answer_type = str(payload.get("expected_answer_type") or "other").strip().lower()
    split = VQASplit(
        scene_description=scene,
        question=question,
        question_en=question_en,
        expected_answer_type=answer_type,
    )
    log.info("Q&A split -> scene=%r question=%r type=%s",
             scene[:60], question[:60], answer_type)
    return split


def select_vlm_targets(
    rows: Sequence[Candidate], shot_window: int, top_n: int
) -> list[Candidate]:
    """One representative (highest-scoring) frame per shot, best shots first."""
    chosen: list[Candidate] = []
    for candidate in rows:
        if len(chosen) >= top_n:
            break
        if any(
            c.video_id == candidate.video_id
            and abs(c.frame_id - candidate.frame_id) < shot_window
            for c in chosen
        ):
            continue
        chosen.append(candidate)
    return chosen


select_qa_targets = select_vlm_targets


def answer_candidates(
    llm: LLMClient,
    targets: Sequence[Candidate],
    question: str,
    cfg: Config,
    *,
    text_searcher: Any = None,
    use_cache: bool = True,
) -> dict[tuple[str, int], str]:
    """Ask the LLM about each target frame using multimodal text context in parallel."""
    if not targets:
        return {}

    # Step 1: Enrich metadata for candidates if text_searcher is provided and enabled
    enriched_map: dict[tuple[str, int], dict[str, str]] = {}
    if cfg.vqa.enrich_context and text_searcher is not None and hasattr(text_searcher, "fetch_metadata"):
        try:
            enriched_map = text_searcher.fetch_metadata(targets, max_shot_gap=cfg.submission.shot_window)
        except Exception as exc:  # noqa: BLE001
            log.warning("Metadata enrichment failed: %s", exc)

    def ask(candidate: Candidate) -> tuple[tuple[str, int], str]:
        extra = candidate.extra or {}
        enriched = enriched_map.get(candidate.key, {})

        desc = (
            extra.get("description_matched")
            or enriched.get("description")
            or extra.get("matched_text")
            or ""
        )
        ocr = (
            extra.get("ocr_matched")
            or enriched.get("ocr")
            or ""
        )
        asr = (
            extra.get("asr_matched")
            or enriched.get("asr")
            or ""
        )

        context_items: list[str] = []
        if desc:
            context_items.append(f"Mô tả phân cảnh (Description): {str(desc).strip()[:350]}")
        if ocr:
            context_items.append(f"Chữ trên màn hình (OCR): {str(ocr).strip()[:200]}")
        if asr:
            context_items.append(f"Lời thoại âm thanh (ASR): {str(asr).strip()[:200]}")

        context_str = "\n".join(context_items) if context_items else "Không có thông tin văn bản cụ thể cho khung hình này."
        prompt = (
            f"Khung hình video [{candidate.video_id}, frame {candidate.frame_id}]:\n"
            f"{context_str}\n\n"
            f"Câu hỏi: {question}"
        )

        try:
            raw = llm.chat(VQA_LLM_SYSTEM_PROMPT, prompt, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the task
            log.warning("LLM VQA failed on %s: %s", candidate.key, exc)
            return candidate.key, ""
        return candidate.key, sanitize_answer(raw, cfg.vqa.answer_max_chars)

    workers = max(1, min(getattr(cfg.llm, "max_retries", 3) * 2, 8, len(targets)))
    answers: dict[tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vqa_llm") as pool:
        for key, answer in pool.map(ask, targets):
            if answer:
                answers[key] = answer

    known = sum(1 for a in answers.values() if not is_unknown(a))
    log.info("LLM answered %d/%d candidate frames (%d usable)", len(answers), len(targets), known)
    return answers


def answer_frames(
    vlm_or_llm: Any,
    targets: Sequence[Candidate],
    question: str,
    kf: KeyframeIndex | None = None,
    cfg: Config | None = None,
    *,
    use_cache: bool = True,
    text_searcher: Any = None,
) -> dict[tuple[str, int], str]:
    """Backward-compatible wrapper routing to answer_candidates."""
    if cfg is None:
        cfg = Config.load("config/config.yaml", no_cache=True)
    if hasattr(vlm_or_llm, "chat"):
        return answer_candidates(vlm_or_llm, targets, question, cfg, text_searcher=text_searcher, use_cache=use_cache)
    return answer_candidates(vlm_or_llm, targets, question, cfg, text_searcher=text_searcher, use_cache=use_cache)


def propagate_answers(
    rows: Sequence[Candidate],
    answers: dict[tuple[str, int], str],
    cfg: Config,
) -> list[tuple[str, int, str]]:
    """Fill in an answer for every row.

    Priority: the row's own answer, then the answer of another frame in the
    same shot, then the majority answer overall, then the configured fallback.
    """
    usable = {k: v for k, v in answers.items() if v and not is_unknown(v)}
    consensus = majority_vote(list(usable.values()))
    fallback = consensus or cfg.vqa.fallback_answer

    out: list[tuple[str, int, str]] = []
    for candidate in rows:
        answer = answers.get(candidate.key, "")
        if is_unknown(answer):
            answer = ""

        if not answer and cfg.vqa.propagate:
            answer = _nearest_shot_answer(candidate, usable, cfg.submission.shot_window)
        if not answer:
            answer = fallback
        cleaned = sanitize_answer(answer, cfg.vqa.answer_max_chars)
        if not cleaned:
            cleaned = sanitize_answer(cfg.vqa.fallback_answer, cfg.vqa.answer_max_chars)
        out.append((candidate.video_id, candidate.frame_id, cleaned))
    return out


def _nearest_shot_answer(
    candidate: Candidate,
    answers: dict[tuple[str, int], str],
    shot_window: int,
) -> str:
    """Answer of the closest answered frame within the same shot, if any."""
    best: tuple[int, str] | None = None
    for (video_id, frame_id), answer in answers.items():
        if video_id != candidate.video_id:
            continue
        distance = abs(frame_id - candidate.frame_id)
        if distance >= shot_window:
            continue
        if best is None or distance < best[0]:
            best = (distance, answer)
    return best[1] if best else ""


def run_vqa(
    query: str,
    pipeline: RetrievalPipeline,
    kf_or_vlm: Any = None,
    kf: KeyframeIndex | None = None,
    cfg: Config | None = None,
    *,
    trace_name: str | None = None,
    use_cache: bool = True,
) -> list[tuple[str, int, str]]:
    """Run the whole Q&A task and return ``(video_id, frame_id, answer)`` rows.

    Supports signatures:
      run_vqa(query, pipeline, kf, cfg)
      run_vqa(query, pipeline, vlm, kf, cfg)
    """
    resolved_cfg = cfg or getattr(pipeline, "cfg", None)
    if resolved_cfg is None and isinstance(kf_or_vlm, Config):
        resolved_cfg = kf_or_vlm
        resolved_kf = None
    elif isinstance(kf_or_vlm, KeyframeIndex):
        resolved_kf = kf_or_vlm
    else:
        resolved_kf = kf

    if resolved_cfg is None:
        resolved_cfg = Config.load("config/config.yaml", no_cache=True)

    split = split_query(pipeline.llm, query, use_cache=use_cache)

    candidates, _ = pipeline.run(
        split.scene_description,
        topk=resolved_cfg.submission.max_rows * 5,
        trace_name=trace_name,
        use_cache=use_cache,
    )
    rows = build_rows(candidates, resolved_cfg.submission, resolved_kf)[: resolved_cfg.submission.max_rows]
    if not rows:
        log.error("Q&A retrieval produced no candidates for %r", query[:60])
        return []

    targets = select_vlm_targets(rows, resolved_cfg.submission.shot_window, resolved_cfg.vqa.llm_top_n)
    
    qa_question = split.question if split.question else query
    answers = answer_candidates(
        pipeline.llm,
        targets,
        qa_question,
        resolved_cfg,
        text_searcher=getattr(pipeline, "text", None),
        use_cache=use_cache,
    )

    out = propagate_answers(rows, answers, resolved_cfg)
    log.info("Q&A produced %d rows for %r", len(out), query[:60])
    return out
