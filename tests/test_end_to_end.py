"""Whole-task smoke tests: query in, valid competition CSV out.

Nothing here touches a network service - the LLM, VLM, Elasticsearch and Qdrant
are all replaced by fakes - but every other layer is the real one, including
the fusion, the row builder, the CSV writer and the validator.
"""

from __future__ import annotations

import json

import pytest

from src.config import Config
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas import Candidate, PATH_ASR, PATH_DESCRIPTION, PATH_OCR, PATH_VISUAL
from src.submission.validator import validate
from src.submission.writer import write_kis, write_qa, write_trake
from src.tasks.kis import run_kis
from src.tasks.trake import run_trake
from src.tasks.vqa import run_vqa
from src.utils.keyframe_index import KeyframeIndex

VIDEOS = ["L01_V001", "L01_V002", "L02_V003", "L02_V004"]
FRAMES = list(range(0, 2000, 25))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dataset(tmp_path):
    """A real keyframe tree with real (tiny) JPEG files and map CSVs."""
    from PIL import Image

    root = tmp_path / "keyframes"
    map_dir = tmp_path / "map-keyframes"
    map_dir.mkdir(parents=True)

    for video in VIDEOS:
        directory = root / video
        directory.mkdir(parents=True)
        for i, _frame in enumerate(FRAMES):
            Image.new("RGB", (32, 24), (i % 255, 40, 90)).save(directory / f"{i:04d}.jpg")
        rows = ["n,pts_time,fps,frame_idx"]
        rows += [f"{i + 1},{i * 0.04},25,{frame}" for i, frame in enumerate(FRAMES)]
        (map_dir / f"{video}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    return KeyframeIndex(root, map_dir, None)


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config.load("config/config.yaml", no_cache=True)
    config.runs.dir = tmp_path / "runs"
    config.runs.dir.mkdir(parents=True, exist_ok=True)
    return config


class ScriptedLLM:
    """Answers the decomposition, Q&A-split and TRAKE-plan prompts."""

    def __init__(self, num_events: int = 4):
        self.num_events = num_events

    def chat_json(self, system, user, schema_hint=None, use_cache=True):
        if "temporal event-sequence decomposition" in system:
            return {
                "original_query": user,
                "action": "high jump",
                "num_events": self.num_events,
                "steps": [
                    {"index": i,
                     "description": f"high jump athlete, key moment {i + 1}",
                     "description_local": f"khoảnh khắc {i + 1}"}
                    for i in range(self.num_events)
                ],
            }
        if "split a Vietnamese/English video search query" in system:
            return {
                "scene_description": "lễ trao giải thưởng âm nhạc",
                "question": "có bao nhiêu người lên sân khấu?",
                "expected_answer_type": "person_count",
            }
        return {
            "original_query": user,
            "modalities": ["ocr", "asr", "image"],
            "ocr_query": "VinFast",
            "asr_query": "trao giải",
            "image_query": f"english description of: {user[:40]}",
            "ocr_terms": ["VinFast"],
            "asr_terms": ["giải thưởng"],
            "image_terms": ["stage", "award"],
            "modality_weights": {"ocr": 0.2, "asr": 0.2, "image": 0.6},
        }

    def chat(self, system, user, use_cache=True):
        if "video question answering" in system.lower():
            return "5"
        return json.dumps(self.chat_json(system, user))


def scripted_candidates(source: str, offset: int) -> list[Candidate]:
    """A deterministic, plausible ranking spread over videos and frames."""
    out: list[Candidate] = []
    rank = 1
    for v, video in enumerate(VIDEOS):
        for i in range(12):
            frame = FRAMES[(offset + v * 7 + i * 5) % len(FRAMES)]
            out.append(
                Candidate(video_id=video, frame_id=frame,
                          score=100.0 - rank, source=source, rank=rank)
            )
            rank += 1
    return out


class ScriptedText:
    def search_ocr(self, query, terms=None, size=None, exact_text=False):
        return scripted_candidates(PATH_OCR, 3)

    def search_asr(self, query, terms=None, size=None):
        return scripted_candidates(PATH_ASR, 11)

    def search_description(self, query, terms=None, size=None):
        return scripted_candidates(PATH_DESCRIPTION, 17)


class ScriptedVisual:
    def search(self, image_query, limit=None):
        # Vary by sub-query so different TRAKE steps rank different frames.
        offset = sum(ord(c) for c in (image_query or "")) % 17
        return scripted_candidates(PATH_VISUAL, offset)


class ScriptedVLM:
    """Returns a stable answer, plus an UNKNOWN and a messy one to exercise cleanup."""

    def __init__(self):
        self.calls = 0

    def ask(self, image_path, system, question, use_cache=True):
        self.calls += 1
        if self.calls == 1:
            return "UNKNOWN"
        if self.calls == 2:
            return 'Đáp án: "5".'
        return "5"


@pytest.fixture
def pipeline(cfg):
    return RetrievalPipeline(
        cfg=cfg, llm=ScriptedLLM(),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )


# ---------------------------------------------------------------------------
# KIS
# ---------------------------------------------------------------------------
def test_kis_end_to_end(cfg, pipeline, dataset, tmp_path):
    rows = run_kis("Tìm cảnh một người mở laptop", pipeline, dataset, cfg,
                   trace_name="query-1-kis")

    assert len(rows) == cfg.submission.max_rows
    assert len(set(rows)) == len(rows)                     # no duplicate rows
    assert all(v in VIDEOS and isinstance(f, int) for v, f in rows)

    out = tmp_path / "submission" / "query-1-kis.csv"
    assert write_kis(out, rows, cfg.submission.max_rows) == 100
    assert validate(out.parent) == []


def test_kis_head_is_spread_across_shots(cfg, pipeline, dataset):
    rows = run_kis("Tìm cảnh một người mở laptop", pipeline, dataset, cfg)
    head = rows[:5]
    # R@1 and R@5 must not be spent on five frames of one shot.
    assert len({v for v, _ in head}) >= 2


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
def test_qa_end_to_end(cfg, pipeline, dataset, tmp_path):
    rows = run_vqa(
        "Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu?",
        pipeline, dataset, cfg, trace_name="query-3-qa",
    )

    assert len(rows) == cfg.submission.max_rows
    for _video, _frame, answer in rows:
        assert answer                                     # never empty
        assert len(answer) <= cfg.vqa.answer_max_chars
        assert answer.upper() != "UNKNOWN"                # UNKNOWN is replaced
    # The messy answer was cleaned into its atomic form and propagated.
    assert "5" in {a for _, _, a in rows}

    out = tmp_path / "submission" / "query-3-qa.csv"
    write_qa(out, rows, cfg.submission.max_rows,
             cfg.vqa.answer_max_chars, cfg.vqa.fallback_answer)
    assert validate(out.parent) == []


def test_qa_falls_back_when_the_vlm_is_useless(cfg, dataset, tmp_path):
    class DeadLLM(ScriptedLLM):
        def chat(self, *a, **k):
            raise RuntimeError("LLM endpoint is down")

    broken_pipe = RetrievalPipeline(
        cfg=cfg, llm=DeadLLM(),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )
    rows = run_vqa("Trong video ... có bao nhiêu người?", broken_pipe, dataset, cfg)

    assert len(rows) == cfg.submission.max_rows
    assert all(answer == cfg.vqa.fallback_answer for _, _, answer in rows)

    out = tmp_path / "submission" / "query-3-qa.csv"
    write_qa(out, rows, cfg.submission.max_rows,
             cfg.vqa.answer_max_chars, cfg.vqa.fallback_answer)
    assert validate(out.parent) == []


# ---------------------------------------------------------------------------
# TRAKE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_events", [2, 3, 4, 6])
def test_trake_end_to_end(num_events, cfg, dataset, tmp_path):
    pipeline = RetrievalPipeline(
        cfg=cfg, llm=ScriptedLLM(num_events),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )
    rows, plan = run_trake(
        f"Tìm {num_events} khoảnh khắc chính khi vận động viên thực hiện cú nhảy",
        pipeline, dataset, cfg, trace_name="query-4-trake",
    )

    assert plan.num_events == num_events
    assert rows
    for row in rows:
        assert len(row) == num_events + 1                 # video + N frames
        frames = list(row[1:])
        assert all(b > a for a, b in zip(frames, frames[1:])), frames
        assert row[0] in VIDEOS

    out = tmp_path / "submission" / "query-4-trake.csv"
    written = write_trake(out, rows, num_events, cfg.submission.max_rows)
    assert written == len(rows[: cfg.submission.max_rows])
    assert validate(out.parent, {"query-4-trake": num_events}) == []


def test_trake_head_covers_several_videos(cfg, dataset):
    pipeline = RetrievalPipeline(
        cfg=cfg, llm=ScriptedLLM(4),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )
    rows, _ = run_trake("Tìm 4 khoảnh khắc chính", pipeline, dataset, cfg)

    head_videos = {row[0] for row in rows[:5]}
    assert len(head_videos) >= min(cfg.trake.head_min_videos, len(VIDEOS))


def test_trake_plan_is_saved_for_the_validator(cfg, dataset, tmp_path):
    from src.tasks.trake import save_plan

    pipeline = RetrievalPipeline(
        cfg=cfg, llm=ScriptedLLM(3),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )
    _, plan = run_trake("Tìm 3 khoảnh khắc", pipeline, dataset, cfg)
    save_plan(cfg, plan, "query-4-trake")

    from src.submission.validator import load_expected_events

    assert load_expected_events(cfg.runs.dir) == {"query-4-trake": 3}


# ---------------------------------------------------------------------------
# all three together
# ---------------------------------------------------------------------------
def test_a_full_submission_directory_validates_and_packs(cfg, dataset, tmp_path):
    from src.submission.packer import pack, verify_zip

    out_dir = tmp_path / "submission"
    pipeline = RetrievalPipeline(
        cfg=cfg, llm=ScriptedLLM(4),
        text_searcher=ScriptedText(), visual_searcher=ScriptedVisual(),
    )

    write_kis(out_dir / "query-1-kis.csv",
              run_kis("q1", pipeline, dataset, cfg), cfg.submission.max_rows)
    write_kis(out_dir / "query-2-kis.csv",
              run_kis("q2", pipeline, dataset, cfg), cfg.submission.max_rows)
    write_qa(out_dir / "query-3-qa.csv",
             run_vqa("q3?", pipeline, ScriptedVLM(), dataset, cfg),
             cfg.submission.max_rows, cfg.vqa.answer_max_chars, cfg.vqa.fallback_answer)
    rows, plan = run_trake("Tìm 4 khoảnh khắc", pipeline, dataset, cfg)
    write_trake(out_dir / "query-4-trake.csv", rows, plan.num_events,
                cfg.submission.max_rows)

    assert validate(out_dir, {"query-4-trake": plan.num_events}) == []

    archive = pack(out_dir, tmp_path / "teamABC.zip", {"query-4-trake": plan.num_events})
    assert verify_zip(archive) == []
