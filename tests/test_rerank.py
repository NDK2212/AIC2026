from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import torch

from src.config import BGERerankConfig, Qwen3VLRerankConfig
from src.retrieval.rerank import BGEReranker, Qwen3VLReranker
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


def test_qwen3_vl_reranks_fused_images_once_with_metadata():
    cfg = Qwen3VLRerankConfig(enabled=True, device="cuda", top_n=2, weight=1.0)
    kf = MagicMock()
    image_a, image_b = object(), object()
    kf.batch_get_images.return_value = {
        ("L01_V001", 100): image_a,
        ("L01_V002", 200): image_b,
    }
    reranker = Qwen3VLReranker(cfg, kf)
    reranker._model = MagicMock()  # bypass heavyweight lazy loading
    reranker._score_pair = MagicMock(side_effect=[0.1, 0.95])
    candidates = [
        make_cand(
            "L01_V001", 100, 0.8, "fused",
            {"description_matched": "người đi bộ"},
        ),
        make_cand(
            "L01_V002", 200, 0.3, "fused",
            {"description_matched": "đầu bếp sơ chế cá"},
        ),
    ]

    results = reranker.rerank("đầu bếp chuẩn bị cá", candidates)

    assert results[0].video_id == "L01_V002"
    assert results[0].extra["qwen3_vl_rerank_score"] == 0.95
    assert reranker._score_pair.call_args_list[1].kwargs == {
        "text": "Description: đầu bếp sơ chế cá",
        "image": image_b,
    }


def test_qwen3_vl_tokenize_converts_multimodal_token_types_to_tensor():
    class FakeTokenizer:
        all_special_ids = [99]
        padding_side = "left"

        @staticmethod
        def pad(payload, **_kwargs):
            rows = payload["input_ids"]
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
            }

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return ["rendered"]

        @staticmethod
        def __call__(**_kwargs):
            return {
                "input_ids": [[1, 2, 3, 4, 5, 6]],
                "mm_token_type_ids": [[0, 0, 1, 1, 0, 0]],
            }

    cfg = Qwen3VLRerankConfig(enabled=True, max_length=16)
    reranker = Qwen3VLReranker(cfg)
    reranker._torch = torch
    reranker._processor = FakeProcessor()
    reranker._process_vision_info = lambda *_args, **_kwargs: (None, None, {})

    inputs = reranker._tokenize([{"role": "user", "content": []}])

    assert isinstance(inputs["mm_token_type_ids"], torch.Tensor)
    assert inputs["mm_token_type_ids"].dtype == torch.long
    assert inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape


def test_embedding_devices_are_cpu_and_qwen_reranker_is_cuda():
    from src.config import Config

    cfg = Config.load("config/config.yaml", no_cache=True)

    assert cfg.embedding.siglip.device == "cpu"
    assert cfg.embedding.beit3.device == "cpu"
    assert cfg.embedding.qwen is not None and cfg.embedding.qwen.device == "cpu"
    assert cfg.embedding.qwen.enabled is True
    assert cfg.rerank.qwen3_vl.enabled is True
    assert cfg.rerank.qwen3_vl.device == "cuda"
    assert cfg.rerank.qwen3_vl.torch_dtype == "float16"
    assert cfg.rerank.qwen3_vl.max_pixels == 401408
    assert cfg.rerank.blip2.enabled is False
    assert cfg.rerank.bge.enabled is False
