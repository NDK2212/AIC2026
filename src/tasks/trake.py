"""Task 3 - Temporal Retrieval and Alignment of Key Events.

Pipeline:

1. an LLM decomposes the query into ``N`` chronologically ordered key moments;
2. every step is retrieved independently through the normal three-path pipeline;
3. videos are scored by how well they cover *all* steps, and the best ones go
   into the alignment stage;
4. inside each video an **anchored bidirectional DP** picks one frame per step,
   subject to the temporal constraints, maximising the summed step score.

The DP is general in ``N`` (any ``N >= 1``), runs in ``O(N² · K log K)`` per
video, and can *skip* a step - charging ``miss_penalty`` and interpolating the
frame afterwards - because TRAKE scores a fraction of matched moments: a video
with 3 of 4 moments right still earns 0.75, while discarding the correct video
earns nothing at all.
"""

from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from ..clients.llm import LLMClient
from ..config import Config, TrakeConfig
from ..logging_utils import get_logger
from ..retrieval.decompose import decompose
from ..retrieval.pipeline import RetrievalPipeline
from ..schemas import (
    Candidate,
    DecomposeResult,
    LLMParseError,
    TrakePlan,
    TrakeSequence,
    TrakeStep,
)
from ..utils.keyframe_index import KeyframeIndex
from ..utils.text_norm import clean_query_text

log = get_logger(__name__)

NEG_INF = -math.inf

PLAN_SYSTEM_PROMPT = """
You are a temporal event-sequence decomposition model for video retrieval.

You are given a query that describes ONE structured action composed of a
SEQUENCE of key moments (events) happening in chronological order inside a
single video.

Your job:
1. Determine N = the exact number of key moments the query asks for.
   - If the query explicitly lists or numbers them, N is that count.
   - If the query names an action but does not enumerate the moments,
     infer the canonical decomposition of that action.
2. For each moment, write a short, self-contained, visually-groundable
   description of THAT SINGLE INSTANT (not the whole action).

RULES:
- Each step description must describe a FROZEN INSTANT that could be
  identified in one single video frame.
- Use concrete visual cues: body position, contact with ground/object,
  highest point, first contact, object leaving hand, etc.
- Do NOT describe duration or process ("is running", "keeps jumping").
  Describe the instant ("the moment the foot first leaves the ground").
- Steps MUST be in strict chronological order.
- Do NOT invent moments that the query does not imply.
- Keep the sport/action context in EVERY step so each step is retrievable
  on its own (e.g. "high jump athlete ..." in every step).
- Write step descriptions in English.
- Also copy the original query language version in "description_local".
- Output ONLY valid JSON.

Output schema:
{
  "original_query": "...",
  "action": "short name of the overall action",
  "num_events": 4,
  "steps": [
    {"index": 0, "description": "...", "description_local": "..."},
    {"index": 1, "description": "...", "description_local": "..."}
  ]
}
""".strip()

PLAN_SCHEMA_HINT = (
    '{"original_query": str, "action": str, "num_events": int, '
    '"steps": [{"index": int, "description": str, "description_local": str}]}'
)


# ---------------------------------------------------------------------------
# step A - temporal decomposition
# ---------------------------------------------------------------------------
def plan_trake(
    llm: LLMClient,
    query: str,
    cfg: Config,
    *,
    use_cache: bool = True,
) -> TrakePlan:
    """Split a TRAKE query into its ordered key moments."""
    cleaned = clean_query_text(query)
    if not cleaned:
        raise ValueError("Cannot plan an empty TRAKE query")

    payload: dict = {}
    try:
        payload = llm.chat_json(
            PLAN_SYSTEM_PROMPT, cleaned, schema_hint=PLAN_SCHEMA_HINT, use_cache=use_cache
        )
    except Exception as exc:
        log.error("TRAKE planning failed: %s", exc)

    steps = _parse_steps(payload)
    if not steps:
        log.error("TRAKE planner returned no steps - retrying once without cache")
        try:
            payload = llm.chat_json(
                PLAN_SYSTEM_PROMPT, cleaned, schema_hint=PLAN_SCHEMA_HINT, use_cache=False
            )
            steps = _parse_steps(payload)
        except Exception as exc:
            log.error("TRAKE planning retry failed: %s", exc)
    if not steps:
        log.warning("TRAKE planner returned no steps from LLM - activating heuristic fallback")
        num_events_heuristic = _events_stated_in_query(cleaned) or 3
        steps = [
            TrakeStep(index=i, description=cleaned, description_local=cleaned)
            for i in range(num_events_heuristic)
        ]

    declared = payload.get("num_events")
    num_events = int(declared) if isinstance(declared, (int, float)) and declared else len(steps)

    stated = _events_stated_in_query(cleaned)
    if stated and stated != num_events:
        log.warning(
            "Query states %d key moments but the planner said %d - trusting the query",
            stated, num_events,
        )
        num_events = stated

    steps = _fit_steps(steps, num_events)
    anchor = _clamp_anchor(cfg.trake.anchor_index, len(steps))

    plan = TrakePlan(
        original_query=cleaned,
        num_events=len(steps),
        steps=steps,
        anchor_index=anchor,
        action=str(payload.get("action") or "").strip(),
    )
    log.info(
        "TRAKE plan: %d events (anchor=%d) - %s",
        plan.num_events, plan.anchor_index,
        " | ".join(s.description[:40] for s in plan.steps),
    )
    return plan


def _parse_steps(payload: dict) -> list[TrakeStep]:
    """Validate and renumber the step list coming out of the planner."""
    raw = payload.get("steps")
    if not isinstance(raw, list):
        return []

    parsed: list[tuple[int, TrakeStep]] = []
    for fallback_index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        description = clean_query_text(str(item.get("description") or ""))
        if not description:
            continue
        try:
            order = int(item.get("index", fallback_index))
        except (TypeError, ValueError):
            order = fallback_index
        local = clean_query_text(str(item.get("description_local") or "")) or None
        parsed.append((order, TrakeStep(index=order, description=description,
                                        description_local=local)))

    if not parsed:
        return []
    parsed.sort(key=lambda pair: pair[0])
    # Renumber to a contiguous 0..N-1 range regardless of what the model emitted.
    return [
        TrakeStep(index=i, description=step.description,
                  description_local=step.description_local)
        for i, (_, step) in enumerate(parsed)
    ]


def _fit_steps(steps: list[TrakeStep], num_events: int) -> list[TrakeStep]:
    """Force the step list to exactly ``num_events`` entries."""
    if num_events <= 0 or num_events == len(steps):
        return steps
    if num_events < len(steps):
        log.warning("Trimming %d planned steps down to %d", len(steps), num_events)
        return [
            TrakeStep(index=i, description=s.description, description_local=s.description_local)
            for i, s in enumerate(steps[:num_events])
        ]

    log.warning(
        "Planner returned %d steps but %d are required - padding by repeating the last",
        len(steps), num_events,
    )
    out = list(steps)
    last = steps[-1]
    while len(out) < num_events:
        out.append(
            TrakeStep(
                index=len(out),
                description=last.description,
                description_local=last.description_local,
            )
        )
    return out


def _events_stated_in_query(query: str) -> int | None:
    """Read an explicit moment count out of the query text, if it has one."""
    import re

    words = {
        "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8,
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    }
    lowered = query.lower()
    unit = (
        r"(?:khoảnh khắc|khoanh khac|bước|buoc|sự kiện|su kien|giai đoạn"
        r"|moments?|events?|steps?|stages?|frames?|phases?)"
    )
    # Adjectives that commonly sit between the count and the noun.
    modifier = r"(?:key|main|important|chính|quan trọng|then chốt)\s+"

    digits = re.search(rf"\b(\d+)\s+(?:{modifier})?{unit}", lowered)
    if digits:
        value = int(digits.group(1))
        if 1 <= value <= 20:
            return value

    for word, value in words.items():
        if re.search(rf"\b{word}\s+(?:{modifier})?{unit}", lowered):
            return value

    # "(1) ... (2) ... (3) ..." style enumerations
    numbered = re.findall(r"\((\d+)\)", lowered)
    if len(numbered) >= 2:
        values = sorted({int(n) for n in numbered})
        if values == list(range(1, len(values) + 1)) and len(values) <= 20:
            return len(values)
    return None


def _clamp_anchor(anchor: int, num_steps: int) -> int:
    """Keep the anchor inside ``[0, N-1]``, preferring the second step."""
    if num_steps <= 0:
        return 0
    if 0 <= anchor < num_steps:
        return anchor
    clamped = min(1, num_steps - 1)
    log.warning("anchor_index %d is out of range for %d steps - using %d",
                anchor, num_steps, clamped)
    return clamped


# ---------------------------------------------------------------------------
# step B - per-step retrieval
# ---------------------------------------------------------------------------
def retrieve_steps(
    plan: TrakePlan,
    pipeline: RetrievalPipeline,
    cfg: Config,
    *,
    trace_name: str | None = None,
    use_cache: bool = True,
) -> list[list[Candidate]]:
    """Retrieve candidates for every step, running the steps in parallel."""

    def run_step(step: TrakeStep) -> list[Candidate]:
        try:
            desc_local = step.description_local or step.description
            desc_en = step.description or desc_local
            w_vis = cfg.trake.step_weights.get("visual", 0.6)
            w_desc = cfg.trake.step_weights.get("description", 0.4)
            step.decompose = DecomposeResult(
                original_query=desc_local,
                modalities=["image", "description"],
                ocr_query=None,
                asr_query=None,
                description_query=desc_local,
                image_query=desc_en,
                ocr_terms=[],
                asr_terms=[],
                description_terms=[],
                image_terms=[],
                modality_weights={"image": w_vis, "description": w_desc},
                exact_text=False,
            )

            candidates, _ = pipeline.run(
                desc_local,
                topk=cfg.trake.per_step_topk,
                decompose_result=step.decompose,
                trace_name=f"{trace_name or 'trake'}-step{step.index}",
                use_cache=use_cache,
            )
            return candidates
        except Exception as exc:  # noqa: BLE001 - one bad step must not kill the task
            log.error("TRAKE step %d retrieval failed: %s", step.index, exc, exc_info=True)
            return []

    # Preload visual encoders once before launching concurrent thread pool
    if hasattr(pipeline, "visual") and hasattr(pipeline.visual, "encoders"):
        try:
            pipeline.visual.encoders()
        except Exception as exc:
            log.warning("Preloading visual encoders: %s", exc)

    workers = max(1, min(len(plan.steps), 4))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trake") as pool:
        results = list(pool.map(run_step, plan.steps))

    for step, candidates in zip(plan.steps, results):
        log.info("TRAKE step %d -> %d candidates", step.index, len(candidates))
    return results


def normalize_step_scores(
    step_candidates: Sequence[Sequence[Candidate]],
) -> list[dict[tuple[str, int], float]]:
    """Min-max normalise every step's scores into ``[0, 1]``.

    Different steps come back on different score scales; without this a single
    step could dominate the DP objective and the ``miss_penalty`` constant would
    mean something different for each step.
    """
    out: list[dict[tuple[str, int], float]] = []
    for candidates in step_candidates:
        if not candidates:
            out.append({})
            continue
        scores = [c.score for c in candidates]
        low, high = min(scores), max(scores)
        span = high - low
        table: dict[tuple[str, int], float] = {}
        for candidate in candidates:
            value = 1.0 if span <= 0 else (candidate.score - low) / span
            # A key can repeat across expansion; keep its best score.
            if value > table.get(candidate.key, NEG_INF):
                table[candidate.key] = value
        out.append(table)
    return out


# ---------------------------------------------------------------------------
# step C - candidate videos
# ---------------------------------------------------------------------------
def rank_videos(
    normalized: Sequence[dict[tuple[str, int], float]],
    cfg: TrakeConfig,
) -> list[str]:
    """Score videos by summed best-per-step score plus a coverage bonus."""
    best: dict[str, list[float]] = {}
    for step_index, table in enumerate(normalized):
        for (video_id, _frame), score in table.items():
            row = best.setdefault(video_id, [NEG_INF] * len(normalized))
            if score > row[step_index]:
                row[step_index] = score

    scored: list[tuple[float, int, str]] = []
    for video_id, row in best.items():
        coverage = sum(1 for value in row if value > NEG_INF)
        total = sum(value for value in row if value > NEG_INF)
        scored.append((total + cfg.coverage_bonus * coverage, coverage, video_id))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    videos = [video_id for _, _, video_id in scored[: cfg.max_videos]]
    log.info(
        "TRAKE video shortlist: %d of %d videos (top: %s)",
        len(videos), len(scored), ", ".join(videos[:5]),
    )
    return videos


# ---------------------------------------------------------------------------
# step D - anchored bidirectional DP
# ---------------------------------------------------------------------------
def _video_steps(
    video_id: str, normalized: Sequence[dict[tuple[str, int], float]]
) -> list[tuple[list[int], list[float]]]:
    """Per-step ``(frames ascending, scores)`` restricted to one video."""
    out: list[tuple[list[int], list[float]]] = []
    for table in normalized:
        rows = sorted(
            ((frame, score) for (vid, frame), score in table.items() if vid == video_id),
            key=lambda pair: pair[0],
        )
        out.append(([f for f, _ in rows], [s for _, s in rows]))
    return out


def _best_before(
    prev_frames: list[int],
    prev_values: list[float],
    cur_frames: list[int],
    min_gap: int,
    max_gap: int | None,
) -> list[tuple[float, int]]:
    """For each ``cur`` frame, the best ``prev`` value in the allowed window.

    Sliding-window maximum: both sequences are ascending, so the window
    ``[f - max_gap, f - min_gap]`` only ever moves right.  Returns
    ``(value, index)`` with ``index == -1`` when the window is empty.
    """
    window: deque[int] = deque()
    out: list[tuple[float, int]] = []
    head = 0
    for frame in cur_frames:
        upper = frame - min_gap
        while head < len(prev_frames) and prev_frames[head] <= upper:
            while window and prev_values[window[-1]] <= prev_values[head]:
                window.pop()
            window.append(head)
            head += 1
        if max_gap is not None:
            lower = frame - max_gap
            while window and prev_frames[window[0]] < lower:
                window.popleft()
        out.append((prev_values[window[0]], window[0]) if window else (NEG_INF, -1))
    return out


def _best_after(
    next_frames: list[int],
    next_values: list[float],
    cur_frames: list[int],
    min_gap: int,
    max_gap: int | None,
) -> list[tuple[float, int]]:
    """Mirror of :func:`_best_before`, scanning frames in descending order."""
    window: deque[int] = deque()
    out: list[tuple[float, int]] = [(NEG_INF, -1)] * len(cur_frames)
    head = len(next_frames) - 1
    for position in range(len(cur_frames) - 1, -1, -1):
        lower = cur_frames[position] + min_gap
        while head >= 0 and next_frames[head] >= lower:
            while window and next_values[window[-1]] <= next_values[head]:
                window.pop()
            window.append(head)
            head -= 1
        if max_gap is not None:
            upper = cur_frames[position] + max_gap
            while window and next_frames[window[0]] > upper:
                window.popleft()
        out[position] = (next_values[window[0]], window[0]) if window else (NEG_INF, -1)
    return out


def _forward_dp(
    steps: list[tuple[list[int], list[float]]],
    cfg: TrakeConfig,
) -> tuple[list[list[float]], list[list[tuple[int, int] | None]]]:
    """``prefix[j][i]`` = best score covering steps ``0..j`` with step ``j`` at ``i``.

    Steps before ``j`` may be skipped at ``miss_penalty`` each, which is what
    makes a video with a missing moment still competitive.
    """
    n = len(steps)
    penalty = cfg.miss_penalty if cfg.allow_fill else NEG_INF
    prefix: list[list[float]] = [[] for _ in range(n)]
    parent: list[list[tuple[int, int] | None]] = [[] for _ in range(n)]

    for j in range(n):
        frames, scores = steps[j]
        if not frames:
            continue
        base = penalty * j if j else 0.0
        best = [base] * len(frames)
        back: list[tuple[int, int] | None] = [None] * len(frames)

        for prev in range(j):
            prev_frames, _ = steps[prev]
            if not prev_frames or not prefix[prev]:
                continue
            skipped = j - prev - 1
            if skipped and penalty == NEG_INF:
                continue           # skipping is forbidden, so this jump is illegal
            gap_cost = penalty * skipped if skipped else 0.0
            transitions = _best_before(
                prev_frames, prefix[prev], frames, cfg.min_gap, cfg.max_gap
            )
            for i, (value, index) in enumerate(transitions):
                if index < 0:
                    continue
                total = value + gap_cost
                if total > best[i]:
                    best[i] = total
                    back[i] = (prev, index)

        prefix[j] = [scores[i] + best[i] for i in range(len(frames))]
        parent[j] = back
    return prefix, parent


def _backward_dp(
    steps: list[tuple[list[int], list[float]]],
    cfg: TrakeConfig,
) -> tuple[list[list[float]], list[list[tuple[int, int] | None]]]:
    """``suffix[j][i]`` = best score covering steps ``j..N-1`` with step ``j`` at ``i``."""
    n = len(steps)
    penalty = cfg.miss_penalty if cfg.allow_fill else NEG_INF
    suffix: list[list[float]] = [[] for _ in range(n)]
    parent: list[list[tuple[int, int] | None]] = [[] for _ in range(n)]

    for j in range(n - 1, -1, -1):
        frames, scores = steps[j]
        if not frames:
            continue
        base = penalty * (n - 1 - j) if j < n - 1 else 0.0
        best = [base] * len(frames)
        forward: list[tuple[int, int] | None] = [None] * len(frames)

        for nxt in range(j + 1, n):
            next_frames, _ = steps[nxt]
            if not next_frames or not suffix[nxt]:
                continue
            skipped = nxt - j - 1
            if skipped and penalty == NEG_INF:
                continue           # skipping is forbidden, so this jump is illegal
            gap_cost = penalty * skipped if skipped else 0.0
            transitions = _best_after(
                next_frames, suffix[nxt], frames, cfg.min_gap, cfg.max_gap
            )
            for i, (value, index) in enumerate(transitions):
                if index < 0:
                    continue
                total = value + gap_cost
                if total > best[i]:
                    best[i] = total
                    forward[i] = (nxt, index)

        suffix[j] = [scores[i] + best[i] for i in range(len(frames))]
        parent[j] = forward
    return suffix, parent


def align_video(
    video_id: str,
    normalized: Sequence[dict[tuple[str, int], float]],
    cfg: TrakeConfig,
    kf: KeyframeIndex | None = None,
    max_paths: int | None = None,
    anchor_index: int = 1,
) -> list[TrakeSequence]:
    """Best temporal alignments of the N steps inside one video.

    Runs the forward and backward DPs once, then reads off the best sequence
    through every candidate of the anchor step - giving both the optimum and a
    set of genuinely distinct runner-ups.
    """
    steps = _video_steps(video_id, normalized)
    n = len(steps)
    if n == 0 or all(not frames for frames, _ in steps):
        return []

    anchor = _clamp_anchor(anchor_index, n)
    if not steps[anchor][0]:
        # The anchor step has nothing in this video: fall back to the closest
        # step that does, so the DP still has something to enumerate over.
        options = [j for j in range(n) if steps[j][0]]
        anchor = min(options, key=lambda j: (abs(j - anchor), j))
        log.debug("%s: anchor step empty, using step %d instead", video_id, anchor)

    prefix, pparent = _forward_dp(steps, cfg)
    suffix, sparent = _backward_dp(steps, cfg)

    anchor_frames, anchor_scores = steps[anchor]
    totals = [
        (prefix[anchor][i] + suffix[anchor][i] - anchor_scores[i], i)
        for i in range(len(anchor_frames))
    ]
    totals.sort(key=lambda pair: (-pair[0], anchor_frames[pair[1]]))

    wanted = max_paths or cfg.paths_per_video
    sequences: list[TrakeSequence] = []
    seen: set[tuple[int, ...]] = set()

    for total, index in totals:
        if len(sequences) >= wanted:
            break
        if total <= NEG_INF:
            continue
        assigned = _backtrack(anchor, index, steps, pparent, sparent)
        sequence = _materialise(
            video_id, assigned, n, total, cfg, kf, anchor_frames[index]
        )
        if sequence is None:
            continue
        marker = tuple(sequence.frame_ids)
        if marker in seen:
            continue
        seen.add(marker)
        sequences.append(sequence)

    return sequences


def _backtrack(
    anchor: int,
    index: int,
    steps: list[tuple[list[int], list[float]]],
    pparent: list[list[tuple[int, int] | None]],
    sparent: list[list[tuple[int, int] | None]],
) -> dict[int, tuple[int, float]]:
    """Walk both parent tables out from the anchor into ``{step: (frame, score)}``."""
    assigned: dict[int, tuple[int, float]] = {}

    step, pos = anchor, index
    while True:
        frames, scores = steps[step]
        assigned[step] = (frames[pos], scores[pos])
        link = pparent[step][pos] if pparent[step] else None
        if link is None:
            break
        step, pos = link

    step, pos = anchor, index
    while True:
        link = sparent[step][pos] if sparent[step] else None
        if link is None:
            break
        step, pos = link
        frames, scores = steps[step]
        assigned[step] = (frames[pos], scores[pos])

    return assigned


def _materialise(
    video_id: str,
    assigned: dict[int, tuple[int, float]],
    num_steps: int,
    total: float,
    cfg: TrakeConfig,
    kf: KeyframeIndex | None,
    anchor_frame: int,
) -> TrakeSequence | None:
    """Turn a partial assignment into a complete, ordered frame sequence."""
    if not assigned:
        return None

    known = sorted(assigned)
    missing = [j for j in range(num_steps) if j not in assigned]
    if missing and not cfg.allow_fill:
        return None

    frames = {j: assigned[j][0] for j in known}
    scores = {j: assigned[j][1] for j in known}

    if missing:
        step_gap = _average_step_gap(known, frames, video_id, kf)
        for j in missing:
            before = max((k for k in known if k < j), default=None)
            after = min((k for k in known if k > j), default=None)
            if before is not None and after is not None:
                ratio = (j - before) / (after - before)
                value = frames[before] + (frames[after] - frames[before]) * ratio
            elif before is not None:
                value = frames[before] + step_gap * (j - before)
            elif after is not None:
                value = frames[after] - step_gap * (after - j)
            else:  # pragma: no cover - guarded by the empty-assignment check
                return None
            estimate = int(round(value))
            if kf is not None:
                estimate = kf.nearest_frame(video_id, estimate)
            frames[j] = max(0, estimate)
            scores[j] = cfg.miss_penalty

    ordered = _enforce_monotonic([frames[j] for j in range(num_steps)], cfg.min_gap)
    return TrakeSequence(
        video_id=video_id,
        frame_ids=ordered,
        total_score=total,
        per_step_score=[scores[j] for j in range(num_steps)],
        filled_steps=missing,
        anchor_frame=anchor_frame,
    )


def _average_step_gap(
    known: list[int],
    frames: dict[int, int],
    video_id: str,
    kf: KeyframeIndex | None,
) -> int:
    """Guess a sensible step-to-step frame distance for missing steps."""
    if len(known) >= 2:
        span = frames[known[-1]] - frames[known[0]]
        steps = known[-1] - known[0]
        if steps > 0 and span > 0:
            return max(1, span // steps)
    if kf is not None:
        return max(1, kf.median_gap(video_id))
    return 25


def _enforce_monotonic(frames: list[int], min_gap: int, min_frame: int = 0) -> list[int]:
    """Guarantee strictly increasing non-negative frame ids."""
    if not frames:
        return []
    gap = max(1, min_gap)
    out = list(frames)
    if out[0] < min_frame:
        out[0] = min_frame
    for i in range(1, len(out)):
        if out[i] < out[i - 1] + gap:
            out[i] = out[i - 1] + gap
    return out


# ---------------------------------------------------------------------------
# step E - ranking and output
# ---------------------------------------------------------------------------
def refine_frame(video_id: str, frame_id: int, step_description: str) -> int:
    """Hook for a future dense local search around a chosen frame.

    Identity by default.  When denser frames become available around a
    candidate, replace this with a local re-scoring pass - the rest of the
    pipeline needs no changes.
    """
    return frame_id


def rank_sequences(
    sequences: Sequence[TrakeSequence], cfg: TrakeConfig, max_rows: int
) -> list[TrakeSequence]:
    """Sort sequences and diversify the head across videos.

    A wrong video scores exactly zero, so spending the first five rows on one
    video is the single most expensive mistake available in this task.
    """
    ordered = sorted(
        sequences,
        key=lambda s: (-s.total_score, len(s.filled_steps), s.video_id, s.frame_ids),
    )
    if not ordered:
        return []

    window = max(1, min(cfg.head_window, len(ordered)))
    need = max(1, min(cfg.head_min_videos, window))

    head: list[TrakeSequence] = []
    videos: set[str] = set()
    used = [False] * len(ordered)

    while len(head) < window:
        slots_left = window - len(head)
        # Force a new video once the remaining slots are exactly what is still
        # needed to reach ``need`` distinct videos.
        must_be_new = len(videos) < need and slots_left <= (need - len(videos))

        pick = _first_unused(ordered, used, videos if must_be_new else None)
        if pick is None and must_be_new:
            pick = _first_unused(ordered, used, None)   # no new video exists
        if pick is None:
            break

        used[pick] = True
        head.append(ordered[pick])
        videos.add(ordered[pick].video_id)

    if len(videos) < need:
        log.warning(
            "Only %d distinct videos available for the TRAKE head (wanted %d)",
            len(videos), need,
        )
    rest = [seq for i, seq in enumerate(ordered) if not used[i]]
    return (head + rest)[:max_rows]


def _first_unused(
    ordered: Sequence[TrakeSequence],
    used: list[bool],
    exclude_videos: set[str] | None,
) -> int | None:
    """Index of the best unused sequence, optionally from an unseen video."""
    for i, sequence in enumerate(ordered):
        if used[i]:
            continue
        if exclude_videos is not None and sequence.video_id in exclude_videos:
            continue
        return i
    return None


def save_plan(cfg: Config, plan: TrakePlan, query_stem: str) -> Path | None:
    """Persist the plan so ``validate`` can check the per-row event count."""
    try:
        cfg.runs.dir.mkdir(parents=True, exist_ok=True)
        path = cfg.runs.dir / f"{query_stem}_plan.json"
        payload = plan.to_dict()
        payload["query_stem"] = query_stem
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path
    except OSError as exc:  # pragma: no cover
        log.warning("Could not save the TRAKE plan: %s", exc)
        return None


def run_trake(
    query: str,
    pipeline: RetrievalPipeline,
    kf: KeyframeIndex | None,
    cfg: Config,
    *,
    trace_name: str | None = None,
    use_cache: bool = True,
) -> tuple[list[list[object]], TrakePlan]:
    """Run the whole TRAKE task.

    Returns ``(rows, plan)`` where each row is ``[video_id, f_0, ..., f_{N-1}]``
    with exactly ``plan.num_events`` frames.
    """
    plan = plan_trake(pipeline.llm, query, cfg, use_cache=use_cache)
    step_candidates = retrieve_steps(
        plan, pipeline, cfg, trace_name=trace_name, use_cache=use_cache
    )
    normalized = normalize_step_scores(step_candidates)

    if not any(normalized):
        log.error("TRAKE retrieval returned nothing for every step")
        return [], plan

    videos = rank_videos(normalized, cfg.trake)
    sequences: list[TrakeSequence] = []
    for video_id in videos:
        sequences.extend(
            align_video(
                video_id,
                normalized,
                cfg.trake,
                kf,
                max_paths=cfg.trake.paths_per_video,
                anchor_index=plan.anchor_index,
            )
        )

    ranked = rank_sequences(sequences, cfg.trake, cfg.submission.max_rows)
    rows: list[list[object]] = []
    for sequence in ranked:
        refined = [
            refine_frame(sequence.video_id, frame, plan.steps[i].description)
            for i, frame in enumerate(sequence.frame_ids)
        ]
        refined = _enforce_monotonic(refined, cfg.trake.min_gap)
        if len(refined) != plan.num_events:  # pragma: no cover - defensive
            log.error(
                "Dropping a TRAKE row with %d frames (expected %d)",
                len(refined), plan.num_events,
            )
            continue
        rows.append([sequence.video_id, *refined])

    log.info(
        "TRAKE produced %d rows over %d videos (%d sequences considered)",
        len(rows), len({r[0] for r in rows}), len(sequences),
    )
    return rows, plan
