"""Task 2 - Video Question Answering.

Flow: split the query into *scene description* + *question*, retrieve on the
description alone (the interrogative part only pollutes retrieval), build the
100 rows, enrich metadata from Elasticsearch, ask the LLM about representative
frames per shot using multimodal context (Description + OCR + ASR), and then
propagate those answers to the remaining rows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
  "expected_answer_type": "number | color | name | object | person_count | yes_no | action | other",
  "evidence_query": "an observable scene likely to reveal the answer elsewhere in the same video"
}

Rules:
- scene_description must describe observable visual elements for video retrieval.
- question is the question being asked.
- question_en MUST be a clear, standalone English question.
- evidence_query is NOT another wording of the action scene. Describe shots that
  may visually or textually reveal the requested answer elsewhere in the same
  video: close-ups, ingredient/product views, labels, title cards, diagrams, or
  spoken introductions. Do not guess the answer itself.
- Example: if the action is stuffing four fish and the question asks the fish
  species, evidence_query should seek close-up/full-body fish, ingredient cards,
  recipe titles, labels, or narration naming the fish.
""".strip()

SPLIT_SCHEMA_HINT = (
    '{"scene_description": str, "question": str, "question_en": str, '
    '"expected_answer_type": str, "evidence_query": str}'
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

VQA_VLM_SYSTEM_PROMPT = """
You are an ultra-precise multimodal video question answering assistant.
You receive several keyframes from the SAME candidate video. Each image is
preceded by text metadata for that frame: Description, OCR and ASR. Combine all
visual and textual evidence across the frames; an action can appear in one
frame while a title, label, close-up, count or object identity appears in another.

Return the shortest atomic answer in the same language as the question.
- Counts/numbers: digits only.
- OCR/sign text: preserve exact spelling.
- Yes/no: đúng/sai for Vietnamese, yes/no for English.
- If neither images nor metadata support an answer, return exactly: unknown
- Maximum 100 characters. No explanation, prefix or punctuation.
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
    evidence_query = clean_query_text(str(payload.get("evidence_query") or ""))
    if not evidence_query:
        evidence_query = _fallback_evidence_query(scene, question)
    split = VQASplit(
        scene_description=scene,
        question=question,
        question_en=question_en,
        expected_answer_type=answer_type,
        evidence_query=evidence_query,
    )
    log.info(
        "Q&A split -> scene=%r question=%r evidence=%r type=%s",
        scene[:60], question[:60], evidence_query[:60], answer_type,
        extra={"progress": {
            "phase": "split",
            "status": "done",
            "title": "Đã tách cảnh và câu hỏi",
            "detail": f"Kiểu đáp án: {answer_type}",
            "inspector": {
                "Scene query": scene,
                "Question": question,
                "Evidence query": evidence_query,
                "Answer type": answer_type,
            },
        }},
    )
    return split


def _fallback_evidence_query(scene: str, question: str) -> str:
    """Safe second-stage query when the splitter omits ``evidence_query``."""
    return clean_query_text(
        "Các phân cảnh khác trong cùng video có cận cảnh rõ đối tượng chính, "
        "cảnh giới thiệu thành phần, nhãn, tiêu đề hoặc lời thoại cung cấp tên "
        f"đối tượng. Ngữ cảnh: {scene}. Cần trả lời: {question}"
    )


def select_evidence_videos(
    rows: Sequence[Candidate], video_top_n: int
) -> list[str]:
    """Return the best unique video IDs to scope the second-stage search."""
    chosen: list[str] = []
    for candidate in rows:
        if candidate.video_id not in chosen:
            chosen.append(candidate.video_id)
            if len(chosen) >= max(1, video_top_n):
                break
    return chosen


def retrieve_cross_shot_evidence(
    pipeline: RetrievalPipeline,
    split: VQASplit,
    rows: Sequence[Candidate],
    cfg: Config,
    *,
    use_cache: bool = True,
    pipeline_options: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Find answer-bearing shots inside videos identified by scene retrieval."""
    if not cfg.vqa.cross_shot_evidence or not split.evidence_query:
        return []
    video_ids = select_evidence_videos(rows, cfg.vqa.evidence_video_top_n)
    if not video_ids:
        return []
    try:
        candidates, _ = pipeline.run(
            split.evidence_query,
            topk=max(cfg.vqa.evidence_top_n * 3, cfg.vqa.evidence_top_n),
            write_trace=False,
            use_cache=use_cache,
            video_ids=video_ids,
            **(pipeline_options or {}),
        )
    except Exception as exc:  # noqa: BLE001 - primary scene results remain usable
        log.warning("Cross-shot evidence retrieval failed: %s", exc)
        return []
    scoped = [candidate for candidate in candidates if candidate.video_id in video_ids]
    return select_vlm_targets(
        scoped, cfg.submission.shot_window, cfg.vqa.evidence_top_n
    )


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
    """Answer once per video from all selected shots, then attach it to each shot.

    Grouping is important for questions whose identifying evidence appears in a
    different shot from the queried action (for example a recipe title card or
    an earlier close-up of a fish).
    """
    if not targets:
        return {}

    # Step 1: Enrich metadata for candidates if text_searcher is provided and enabled
    enriched_map: dict[tuple[str, int], dict[str, str]] = {}
    if cfg.vqa.enrich_context and text_searcher is not None and hasattr(text_searcher, "fetch_metadata"):
        try:
            enriched_map = text_searcher.fetch_metadata(targets, max_shot_gap=cfg.submission.shot_window)
        except Exception as exc:  # noqa: BLE001
            log.warning("Metadata enrichment failed: %s", exc)

    def frame_context(candidate: Candidate) -> str:
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
        return f"Khung hình {candidate.frame_id}:\n{context_str}"

    grouped: dict[str, list[Candidate]] = {}
    for candidate in targets:
        grouped.setdefault(candidate.video_id, []).append(candidate)

    def ask(group: tuple[str, list[Candidate]]) -> tuple[list[tuple[str, int]], str]:
        video_id, candidates = group
        context_str = "\n\n".join(frame_context(candidate) for candidate in candidates)
        prompt = (
            f"Các khung hình thuộc cùng video [{video_id}]:\n"
            f"{context_str}\n\n"
            "Hãy kết hợp bằng chứng giữa các phân cảnh trong video này. "
            "Một khung hình có thể chứa hành động được hỏi, khung khác có thể "
            "cho thấy cận cảnh, tên, nhãn hoặc lời giới thiệu của đối tượng.\n\n"
            f"Câu hỏi: {question}"
        )

        try:
            raw = llm.chat(VQA_LLM_SYSTEM_PROMPT, prompt, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not stop the task
            log.warning("LLM VQA failed on video %s: %s", video_id, exc)
            return [candidate.key for candidate in candidates], ""
        return (
            [candidate.key for candidate in candidates],
            sanitize_answer(raw, cfg.vqa.answer_max_chars),
        )

    groups = list(grouped.items())
    workers = max(1, min(getattr(cfg.llm, "max_retries", 3) * 2, 8, len(groups)))
    answers: dict[tuple[str, int], str] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vqa_llm") as pool:
        for keys, answer in pool.map(ask, groups):
            if answer:
                for key in keys:
                    answers[key] = answer

    known = sum(1 for a in answers.values() if not is_unknown(a))
    log.info("LLM answered %d/%d candidate frames (%d usable)", len(answers), len(targets), known)
    return answers


def answer_candidates_vlm(
    vlm: Any,
    targets: Sequence[Candidate],
    question: str,
    cfg: Config,
    kf: KeyframeIndex,
    *,
    text_searcher: Any = None,
    use_cache: bool = True,
) -> dict[tuple[str, int], str]:
    """Answer per video from multiple images plus Description/OCR/ASR context."""
    if not targets:
        return {}

    enriched_map: dict[tuple[str, int], dict[str, str]] = {}
    if (
        cfg.vqa.enrich_context
        and text_searcher is not None
        and hasattr(text_searcher, "fetch_metadata")
    ):
        try:
            enriched_map = text_searcher.fetch_metadata(
                targets, max_shot_gap=cfg.submission.shot_window
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("VLM metadata enrichment failed: %s", exc)

    grouped: dict[str, list[Candidate]] = {}
    for candidate in targets:
        grouped.setdefault(candidate.video_id, []).append(candidate)

    answers: dict[tuple[str, int], str] = {}
    images_used = 0
    for video_id, candidates in grouped.items():
        selected = _select_multimodal_frames(
            candidates, cfg.vqa.vlm_images_per_video
        )
        image_paths: list[Path] = []
        contexts: list[str] = []
        for candidate in sorted(selected, key=lambda item: item.frame_id):
            image_path = _resolve_vlm_image(kf, candidate)
            if image_path is None:
                continue
            image_paths.append(image_path)
            contexts.append(_vlm_frame_context(candidate, enriched_map))

        if not image_paths:
            log.warning("VLM has no readable image for candidate video %s", video_id)
            continue

        prompt = (
            f"Candidate video: {video_id}. The following frames may include both "
            "the queried action and separate answer-bearing evidence.\n"
            f"Question: {question}"
        )
        try:
            if hasattr(vlm, "ask_many"):
                raw = vlm.ask_many(
                    image_paths,
                    VQA_VLM_SYSTEM_PROMPT,
                    prompt,
                    contexts=contexts,
                    use_cache=use_cache,
                )
            else:  # compatibility with older/custom VLM clients
                raw = vlm.ask(
                    image_paths[0],
                    VQA_VLM_SYSTEM_PROMPT,
                    "\n\n".join([prompt, *contexts]),
                    use_cache=use_cache,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("VLM answering failed on video %s: %s", video_id, exc)
            continue

        answer = sanitize_answer(raw, cfg.vqa.answer_max_chars)
        images_used += len(image_paths)
        if answer:
            for candidate in candidates:
                answers[candidate.key] = answer

    known = sum(1 for answer in answers.values() if not is_unknown(answer))
    model = getattr(getattr(vlm, "cfg", None), "model", type(vlm).__name__)
    log.info(
        "VLM %s answered %d/%d candidate frames (%d usable) from %d images across %d videos",
        model,
        len(answers),
        len(targets),
        known,
        images_used,
        len(grouped),
        extra={"progress": {
            "phase": "answer",
            "status": "done",
            "title": "VLM đã trả lời",
            "detail": (
                f"{model}: {images_used} ảnh, {len(grouped)} video, "
                f"{known} frame có đáp án"
            ),
        }},
    )
    return answers


def _select_multimodal_frames(
    candidates: Sequence[Candidate], limit: int
) -> list[Candidate]:
    """Keep strong action frames and tail evidence frames under one image cap."""
    unique: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        if candidate.key not in seen:
            unique.append(candidate)
            seen.add(candidate.key)
    limit = max(1, int(limit))
    if len(unique) <= limit:
        return unique
    head_count = (limit + 1) // 2
    tail_count = limit - head_count
    chosen = unique[:head_count]
    if tail_count:
        chosen.extend(unique[-tail_count:])
    return chosen


def _vlm_frame_context(
    candidate: Candidate,
    enriched_map: dict[tuple[str, int], dict[str, str]],
) -> str:
    extra = candidate.extra or {}
    enriched = enriched_map.get(candidate.key, {})
    description = str(
        extra.get("description_matched")
        or enriched.get("description")
        or extra.get("matched_text")
        or ""
    ).strip()
    ocr = str(extra.get("ocr_matched") or enriched.get("ocr") or "").strip()
    asr = str(extra.get("asr_matched") or enriched.get("asr") or "").strip()
    parts = [f"Frame {candidate.frame_id} metadata:"]
    if description:
        parts.append(f"Description: {description[:500]}")
    if ocr:
        parts.append(f"OCR: {ocr[:300]}")
    if asr:
        parts.append(f"ASR: {asr[:300]}")
    if len(parts) == 1:
        parts.append("No text metadata available; inspect the image directly.")
    return "\n".join(parts)


def _resolve_vlm_image(
    kf: KeyframeIndex, candidate: Candidate
) -> Path | None:
    image_path = kf.path_of(candidate.video_id, candidate.frame_id)
    if image_path is not None and image_path.is_file():
        return image_path

    cache_dir = getattr(kf, "cache_dir", None)
    if cache_dir is None:
        return None
    image_path = Path(cache_dir) / candidate.video_id / f"{candidate.frame_id}.jpg"
    if image_path.is_file():
        return image_path
    image = kf.get_image(candidate.video_id, candidate.frame_id)
    if image is None:
        return None
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="JPEG", quality=90)
    return image_path


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
    same shot, then same-video consensus, global consensus, and fallback.
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
        if not answer and cfg.vqa.propagate:
            answer = majority_vote([
                value for (video_id, _), value in usable.items()
                if video_id == candidate.video_id
            ])
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
    resolved_vlm: Any = None
    if resolved_cfg is None and isinstance(kf_or_vlm, Config):
        resolved_cfg = kf_or_vlm
        resolved_kf = None
    elif isinstance(kf_or_vlm, KeyframeIndex):
        resolved_kf = kf_or_vlm
    else:
        resolved_vlm = kf_or_vlm
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
    scene_target_count = len(targets)
    evidence_targets = retrieve_cross_shot_evidence(
        pipeline, split, rows, resolved_cfg, use_cache=use_cache
    )
    seen_target_keys = {candidate.key for candidate in targets}
    targets.extend(
        candidate for candidate in evidence_targets
        if candidate.key not in seen_target_keys
    )
    log.info(
        "VQA context: %d scene targets + %d cross-shot evidence targets",
        scene_target_count, len(targets) - scene_target_count,
    )
    
    qa_question = split.question if split.question else query
    if resolved_cfg.vqa.mode == "vision" and resolved_vlm is not None and resolved_kf is not None:
        answers = answer_candidates_vlm(
            resolved_vlm,
            targets,
            qa_question,
            resolved_cfg,
            resolved_kf,
            text_searcher=getattr(pipeline, "text", None),
            use_cache=use_cache,
        )
    else:
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
