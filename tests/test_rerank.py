from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.config import BGERerankConfig
from src.retrieval.rerank import BGEReranker
from src.schemas import Candidate


def make_cand(video_id: str, frame_id: int, score: float, source: str, extra: dict | None = None) -> Candidate:
    return Candidate(
        video_id=video_id,
        frame_id=frame_id,
        score=score,
        source=source,
        rank=1,
        extra=extra or {},
    )


def test_bge_extract_candidate_text():
    cfg = BGERerankConfig(enabled=True)
    reranker = BGEReranker(cfg)

    # 1. Pure visual candidate -> returns empty string
    c_visual = make_cand("L01_V001", 100, 0.8, "visual", {})
    assert reranker._extract_candidate_text(c_visual) == ""

    # 2. OCR candidate with matched_text -> returns matched_text
    c_ocr = make_cand("L01_V001", 100, 0.8, "ocr", {"matched_text": "VINFAST"})
    assert reranker._extract_candidate_text(c_ocr) == "VINFAST"

    # 3. Candidate with ocr_matched and asr_matched -> returns combined string
    c_both = make_cand("L01_V001", 100, 0.8, "fusion", {
        "ocr_matched": "Biển báo",
        "asr_matched": "xe đang chạy",
    })
    assert reranker._extract_candidate_text(c_both) == "OCR: Biển báo | ASR: xe đang chạy"


def test_bge_selective_reranking_pure_visual():
    cfg = BGERerankConfig(enabled=True, weight=0.8, top_n=10)
    reranker = BGEReranker(cfg)
    reranker._model = MagicMock()
    reranker._compute_scores = MagicMock(return_value=[])

    # Three visual candidates with no text
    c1 = make_cand("L01_V001", 100, 0.9, "visual")
    c2 = make_cand("L01_V002", 200, 0.8, "visual")
    c3 = make_cand("L01_V003", 300, 0.7, "visual")

    results = reranker.rerank("cảnh người đi bộ", [c1, c2, c3])

    # _compute_scores should NOT be called at all because no candidates have text
    reranker._compute_scores.assert_not_called()

    # Scores and ranks must be perfectly preserved
    assert len(results) == 3
    assert [c.score for c in results] == [0.9, 0.8, 0.7]
    assert [c.video_id for c in results] == ["L01_V001", "L01_V002", "L01_V003"]
    assert [c.rank for c in results] == [1, 2, 3]


def test_bge_selective_reranking_mixed():
    cfg = BGERerankConfig(enabled=True, weight=0.5, top_n=10)
    reranker = BGEReranker(cfg)
    reranker._model = MagicMock()
    
    # Mock _compute_scores to return 0.9 for the text candidate
    reranker._compute_scores = MagicMock(return_value=[0.9])

    # Visual candidate has score 0.85
    c_visual = make_cand("L01_V001", 100, 0.85, "visual")
    # Text candidate has score 0.60, but OCR matches query strongly
    c_text = make_cand("L01_V002", 200, 0.60, "ocr", {"ocr_matched": "xe đạp điện"})

    results = reranker.rerank("xe đạp điện", [c_visual, c_text])

    # Only 1 pair passed to compute_scores (the text candidate)
    reranker._compute_scores.assert_called_once_with([("xe đạp điện", "OCR: xe đạp điện")])

    # c_text score = 0.60 + 0.5 * 0.9 = 1.05 -> jumps to rank 1!
    # c_visual score = 0.85 (preserved!) -> rank 2
    assert results[0].video_id == "L01_V002"
    assert pytest.approx(results[0].score) == 1.05
    assert results[0].rank == 1
    assert results[0].extra["bge_score"] == 0.9

    assert results[1].video_id == "L01_V001"
    assert pytest.approx(results[1].score) == 0.85
    assert results[1].rank == 2


def test_bge_reranker_disabled_or_empty():
    cfg = BGERerankConfig(enabled=False)
    reranker = BGEReranker(cfg)
    c1 = make_cand("L01_V001", 100, 0.9, "visual")

    assert reranker.rerank("query", [c1]) == [c1]
    assert reranker.rerank("", [c1]) == [c1]
    assert reranker.rerank("query", []) == []
