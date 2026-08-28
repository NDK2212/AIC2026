"""Exhaustive, rigorous test suite for the 4-Path Symmetric Architecture.

Covers edge cases, failure isolation, multi-threading race conditions,
BGE & BLIP-2 symmetric reranking, VLM context enrichment, and 4-way WRRF fusion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
import pytest

from src.config import BGERerankConfig, Blip2RerankConfig, Config
from src.retrieval.decompose import decompose
from src.retrieval.fusion import fuse, normalize_weights, weighted_rrf
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.rerank import BGEReranker, BLIP2Reranker
from src.schemas import (
    ALL_PATHS,
    Candidate,
    DecomposeResult,
    PATH_ASR,
    PATH_DESCRIPTION,
    PATH_OCR,
    PATH_VISUAL,
)
from src.tasks.vqa import answer_candidates, answer_frames, sanitize_answer
from tests.conftest import make_candidate


# ---------------------------------------------------------------------------
# Test Doubles & Fakes
# ---------------------------------------------------------------------------
class Fake4PathLLM:
    """Simulates LLM decomposition for 4 paths."""

    def __init__(self, response_dict: dict):
        self.response_dict = response_dict

    def chat_json(self, system, user, schema_hint=None, use_cache=True):
        return self.response_dict


class Fake4PathTextSearcher:
    """Mock text searcher supporting OCR, ASR, and Description."""

    def __init__(self, ocr=None, asr=None, description=None, fail_paths=()):
        self.ocr = ocr or []
        self.asr = asr or []
        self.description = description or []
        self.fail_paths = set(fail_paths)
        self.calls: list[str] = []

    def search_ocr(self, query, terms=None, size=None, exact_text=False):
        self.calls.append(PATH_OCR)
        if PATH_OCR in self.fail_paths:
            raise RuntimeError("Elasticsearch OCR cluster node timeout")
        return self.ocr

    def search_asr(self, query, terms=None, size=None):
        self.calls.append(PATH_ASR)
        if PATH_ASR in self.fail_paths:
            raise RuntimeError("Elasticsearch ASR cluster node timeout")
        return self.asr

    def search_description(self, query, terms=None, size=None):
        self.calls.append(PATH_DESCRIPTION)
        if PATH_DESCRIPTION in self.fail_paths:
            raise RuntimeError("Elasticsearch Description cluster node timeout")
        return self.description


class FakeVisualSearcher:
    def __init__(self, results=None, fail=False):
        self.results = results or []
        self.fail = fail
        self.calls: list[str] = []

    def search(self, image_query, limit=None):
        self.calls.append(image_query)
        if self.fail:
            raise RuntimeError("Qdrant gRPC connection refused")
        return self.results


@pytest.fixture
def base_cfg() -> Config:
    return Config.load("config/config.yaml", no_cache=True)


# ==============================================================================
# 1. 4-PATH CONCURRENCY & FAILURE ISOLATION TESTS
# ==============================================================================
def test_4path_single_failure_does_not_affect_other_paths(base_cfg):
    """When Description path fails with 500 error, OCR, ASR and Visual must still fuse."""
    decomp_payload = {
        "original_query": "xe cứu hỏa đang chạy trên đường",
        "modalities": ["ocr", "asr", "description", "image"],
        "ocr_query": "Cứu Hỏa",
        "asr_query": "tiếng còi hú",
        "description_query": "xe cứu hỏa màu đỏ đang làm nhiệm vụ",
        "image_query": "a red fire truck rushing on the city road",
        "modality_weights": {"ocr": 0.2, "asr": 0.2, "description": 0.3, "image": 0.3},
    }

    text = Fake4PathTextSearcher(
        ocr=[make_candidate("L01_V001", 100, 10.0, PATH_OCR, 1)],
        asr=[make_candidate("L01_V002", 200, 8.0, PATH_ASR, 1)],
        description=[],
        fail_paths=[PATH_DESCRIPTION],  # Description fails
    )
    visual = FakeVisualSearcher([make_candidate("L01_V003", 300, 0.95, PATH_VISUAL, 1)])

    pipeline = RetrievalPipeline(
        cfg=base_cfg,
        llm=Fake4PathLLM(decomp_payload),
        text_searcher=text,
        visual_searcher=visual,
    )

    ranked, _ = pipeline.run("xe cứu hỏa đang chạy trên đường", topk=10, write_trace=False)

    assert len(ranked) == 3
    video_ids = {c.video_id for c in ranked}
    assert video_ids == {"L01_V001", "L01_V002", "L01_V003"}


def test_4path_multiple_failures_graceful_degradation(base_cfg):
    """When OCR and Visual fail simultaneously, ASR and Description must survive."""
    decomp_payload = {
        "original_query": "tin tức thời sự",
        "modalities": ["ocr", "asr", "description", "image"],
        "ocr_query": "VTV1",
        "asr_query": "chào buổi sáng",
        "description_query": "trường quay thời sự",
        "image_query": "news anchor set",
        "modality_weights": {"ocr": 0.25, "asr": 0.25, "description": 0.25, "image": 0.25},
    }

    text = Fake4PathTextSearcher(
        ocr=[],
        asr=[make_candidate("L02_V010", 150, 6.0, PATH_ASR, 1)],
        description=[make_candidate("L02_V020", 250, 7.0, PATH_DESCRIPTION, 1)],
        fail_paths=[PATH_OCR],  # OCR fails
    )
    visual = FakeVisualSearcher([], fail=True)  # Qdrant fails

    pipeline = RetrievalPipeline(
        cfg=base_cfg,
        llm=Fake4PathLLM(decomp_payload),
        text_searcher=text,
        visual_searcher=visual,
    )

    ranked, _ = pipeline.run("tin tức thời sự", topk=10, write_trace=False)

    assert len(ranked) == 2
    assert {c.video_id for c in ranked} == {"L02_V010", "L02_V020"}


def test_4path_all_failures_returns_empty_gracefully(base_cfg):
    """When all 4 backends fail, pipeline must return empty list without raising."""
    decomp_payload = {
        "original_query": "query",
        "modalities": ["ocr", "asr", "description", "image"],
        "ocr_query": "q", "asr_query": "q", "description_query": "q", "image_query": "q",
    }
    text = Fake4PathTextSearcher(fail_paths=[PATH_OCR, PATH_ASR, PATH_DESCRIPTION])
    visual = FakeVisualSearcher(fail=True)

    pipeline = RetrievalPipeline(
        cfg=base_cfg,
        llm=Fake4PathLLM(decomp_payload),
        text_searcher=text,
        visual_searcher=visual,
    )

    ranked, decomp = pipeline.run("query", topk=10, write_trace=False)
    assert ranked == []


def test_4path_heavy_multithreading_concurrency(base_cfg):
    """Stress test: 20 simultaneous concurrent pipeline queries."""
    decomp_payload = {
        "original_query": "concurrent query",
        "modalities": ["ocr", "asr", "description", "image"],
        "ocr_query": "q_ocr",
        "asr_query": "q_asr",
        "description_query": "q_desc",
        "image_query": "q_img",
        "modality_weights": {"ocr": 0.25, "asr": 0.25, "description": 0.25, "image": 0.25},
    }

    text = Fake4PathTextSearcher(
        ocr=[make_candidate("L01_V001", 10, 1.0, PATH_OCR, 1)],
        asr=[make_candidate("L01_V002", 20, 2.0, PATH_ASR, 1)],
        description=[make_candidate("L01_V003", 30, 3.0, PATH_DESCRIPTION, 1)],
    )
    visual = FakeVisualSearcher([make_candidate("L01_V004", 40, 4.0, PATH_VISUAL, 1)])

    pipeline = RetrievalPipeline(
        cfg=base_cfg,
        llm=Fake4PathLLM(decomp_payload),
        text_searcher=text,
        visual_searcher=visual,
    )

    def run_query(idx: int):
        ranked, _ = pipeline.run(f"query {idx}", topk=10, write_trace=False)
        return len(ranked)

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(run_query, range(20)))

    assert len(counts) == 20
    assert all(c == 4 for c in counts)


# ==============================================================================
# 2. SYMMETRIC BGE & BLIP-2 RERANKING MECHANICS
# ==============================================================================
def test_bge_reranks_description_path_independently():
    """Verify BGE cross-encoder rescores Description candidates with description_query."""
    cfg = BGERerankConfig(enabled=True, weight=0.5, top_n=10)
    reranker = BGEReranker(cfg)
    reranker._model = MagicMock()
    # Mock scores: cand2 has strong match 0.95, cand1 has weak match 0.20
    reranker._compute_scores = MagicMock(return_value=[0.20, 0.95])

    c1 = make_candidate("L01_V001", 100, 0.80, PATH_DESCRIPTION, 1)
    c1 = c1.replace(extra={"description_matched": "người đi bộ qua đường"})

    c2 = make_candidate("L01_V002", 200, 0.50, PATH_DESCRIPTION, 2)
    c2 = c2.replace(extra={"description_matched": "vụ va chạm xe máy nghiêm trọng"})

    results = reranker.rerank("vụ va chạm xe máy", [c1, c2])

    # c2 blended = 0.50 + 0.5 * 0.95 = 0.975 -> jumps to rank 1!
    # c1 blended = 0.80 + 0.5 * 0.20 = 0.900 -> rank 2
    assert results[0].video_id == "L01_V002"
    assert results[0].rank == 1
    assert results[0].extra["bge_score"] == 0.95

    assert results[1].video_id == "L01_V001"
    assert results[1].rank == 2


def test_blip2_reranker_parallel_batching_and_error_tolerance():
    """Verify BLIP2 handles missing images, batched chunks, and invalid paths smoothly."""
    cfg = Blip2RerankConfig(enabled=True, weight=1.0, top_n=10, batch_size=2)
    mock_kf = MagicMock()

    # resolve_image returns None for some, path for others
    def fake_resolve(video_id, frame_id):
        if frame_id == 999:
            return None  # Missing image
        return f"/fake/path/{video_id}_{frame_id}.jpg"

    mock_kf.resolve_image.side_effect = fake_resolve

    reranker = BLIP2Reranker(cfg, kf=mock_kf)
    reranker._model = MagicMock()
    reranker._processor = MagicMock()
    reranker._torch = MagicMock()

    # Create 5 candidates
    cands = [make_candidate("L01_V001", i * 100, float(10 - i), PATH_VISUAL, i + 1) for i in range(5)]
    # Mark frame 2 as missing
    cands[2] = make_candidate("L01_V001", 999, 8.0, PATH_VISUAL, 3)

    with patch("PIL.Image.open", side_effect=lambda p: MagicMock()):
        # Mock forward pass
        mock_out = MagicMock()
        mock_logits = MagicMock()
        mock_out.__getitem__.return_value = mock_logits
        reranker._torch.inference_mode.return_value.__enter__.return_value = None
        reranker._torch.softmax.return_value.__getitem__.return_value.cpu.return_value.tolist.return_value = [0.8, 0.9]

        reranked = reranker.rerank("a car on road", cands)

    assert len(reranked) == 5
    assert all(isinstance(c, Candidate) for c in reranked)


# ==============================================================================
# 3. VQA VLM CONTEXT ENRICHMENT FROM METADATA
# ==============================================================================
def test_vqa_context_enrichment_with_all_modalities():
    """When candidate contains Description, OCR, and ASR, VLM prompt is fully enriched."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "5"

    candidate = Candidate(
        video_id="L01_V001",
        frame_id=120,
        score=10.0,
        extra={
            "description_matched": "Lễ hội ẩm thực đường phố Hà Nội",
            "ocr_matched": "Gian hàng Phở Thìn",
            "asr_matched": "Xin mời quý khách dùng thử",
        },
    )

    base_cfg = Config.load("config/config.yaml", no_cache=True)
    answers = answer_candidates(mock_llm, [candidate], "Có bao nhiêu người?", base_cfg, use_cache=False)

    assert ("L01_V001", 120) in answers
    assert answers[("L01_V001", 120)] == "5"

    # Verify LLM received the enriched context prompt
    mock_llm.chat.assert_called_once()
    called_prompt = mock_llm.chat.call_args[0][1]
    assert "Mô tả phân cảnh (Description): Lễ hội ẩm thực đường phố Hà Nội" in called_prompt
    assert "Chữ trên màn hình (OCR): Gian hàng Phở Thìn" in called_prompt
    assert "Lời thoại âm thanh (ASR): Xin mời quý khách dùng thử" in called_prompt
    assert "Câu hỏi: Có bao nhiêu người?" in called_prompt


def test_vqa_context_enrichment_partial_metadata():
    """When candidate only has description_matched, only scene context is passed."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "màu đỏ"

    candidate = Candidate(
        video_id="L01_V002",
        frame_id=300,
        score=5.0,
        extra={"description_matched": "Người đàn ông lái xe máy"},
    )

    base_cfg = Config.load("config/config.yaml", no_cache=True)
    answers = answer_candidates(mock_llm, [candidate], "Chiếc xe màu gì?", base_cfg, use_cache=False)

    assert answers[("L01_V002", 300)] == "màu đỏ"
    called_prompt = mock_llm.chat.call_args[0][1]
    assert "Mô tả phân cảnh (Description): Người đàn ông lái xe máy" in called_prompt
    assert "Chữ trên màn hình (OCR)" not in called_prompt
    assert "Lời thoại âm thanh (ASR)" not in called_prompt


# ==============================================================================
# 4. 4-PATH WRRF FUSION & WEIGHTING MATHEMATICS
# ==============================================================================
def test_4path_wrrf_fusion_mathematics():
    """Verify exact Weighted RRF formula with 4 active paths: sum(w_p / (k + rank_p))."""
    k = 60
    weights = {PATH_OCR: 0.1, PATH_ASR: 0.2, PATH_DESCRIPTION: 0.3, PATH_VISUAL: 0.4}

    # Video 1 appears at rank 1 in all 4 paths
    c_ocr = make_candidate("L01_V001", 100, 1.0, PATH_OCR, 1)
    c_asr = make_candidate("L01_V001", 100, 1.0, PATH_ASR, 1)
    c_desc = make_candidate("L01_V001", 100, 1.0, PATH_DESCRIPTION, 1)
    c_vis = make_candidate("L01_V001", 100, 1.0, PATH_VISUAL, 1)

    paths = {
        PATH_OCR: [c_ocr],
        PATH_ASR: [c_asr],
        PATH_DESCRIPTION: [c_desc],
        PATH_VISUAL: [c_vis],
    }

    fused = weighted_rrf(paths, weights, k=k)

    expected_score = (0.1 / (k + 1)) + (0.2 / (k + 1)) + (0.3 / (k + 1)) + (0.4 / (k + 1))
    assert len(fused) == 1
    assert pytest.approx(fused[0].score, 1e-6) == expected_score
    assert fused[0].rank == 1


def test_4path_weight_normalization_and_clamping():
    """Verify weight clamping and floor over 4 active modalities."""
    raw_weights = {"ocr": 99.0, "asr": 0.0, "description": 1.0, "image": 50.0}
    active = [PATH_OCR, PATH_ASR, PATH_DESCRIPTION, PATH_VISUAL]

    # With clamp_max=1.0, 99.0 and 50.0 are clamped to 1.0
    # floor=0.1 brings asr from 0.0 to 0.1
    # normalized sums to 1.0
    norm = normalize_weights(raw_weights, active, floor=0.1, clamp_max=1.0)

    assert pytest.approx(sum(norm.values()), 1e-6) == 1.0
    assert norm[PATH_ASR] >= 0.03  # Has non-zero share due to floor


# ==============================================================================
# 5. REVIEWER EDGE CASE TESTS: PROMPT, AUTO-POPULATION & FIELD RESILIENCE
# ==============================================================================
def test_decompose_system_prompt_instructs_all_4_modalities():
    """Verify SYSTEM_PROMPT in decompose.py explicitly defines all 4 modalities."""
    from src.retrieval.decompose import SYSTEM_PROMPT

    assert "OCR (On-screen text)" in SYSTEM_PROMPT
    assert "ASR (Speech / Narration / Dialogue)" in SYSTEM_PROMPT
    assert "DESCRIPTION (Video Scene / Action Dense Caption)" in SYSTEM_PROMPT
    assert "IMAGE (Visual Content)" in SYSTEM_PROMPT
    assert '"description_query"' in SYSTEM_PROMPT
    assert '"description_terms"' in SYSTEM_PROMPT


def test_decompose_auto_populates_description_when_omitted_by_llm():
    """When an LLM declares description modality but omits description_query string, it auto-populates original Vietnamese query."""
    mock_llm = MagicMock()
    # LLM payload that declares description & image modalities but left description_query null
    mock_llm.chat_json.return_value = {
        "original_query": "người phụ nữ mặc áo tím đang tưới cây",
        "modalities": ["description", "image"],
        "ocr_query": None,
        "asr_query": None,
        "description_query": None,
        "image_query": "a woman in purple shirt watering green plants in garden",
        "ocr_terms": [],
        "asr_terms": [],
        "image_terms": ["woman", "watering plants"],
        "modality_weights": {"description": 0.4, "image": 0.6},
    }

    result = decompose(mock_llm, "người phụ nữ mặc áo tím đang tưới cây")

    # description_query must be auto-populated to original Vietnamese query!
    assert result.description_query == "người phụ nữ mặc áo tím đang tưới cây"
    assert result.image_query == "a woman in purple shirt watering green plants in garden"
    assert "description" in result.modalities


def test_search_text_resilient_field_extraction():
    """Verify TextSearcher recovers description text whether stored as caption, dense_caption or description."""
    from src.clients.elastic import ElasticWrapper
    from src.config import ElasticConfig
    from src.retrieval.search_text import TextSearcher

    mock_client = MagicMock(spec=ElasticWrapper)
    cfg = ElasticConfig(
        hosts=["http://localhost:9200"],
        index="test_index",
        fields={
            "video_id": "video_id",
            "frame_id": "frame_id",
            "ocr": "ocr_text",
            "asr": "asr_text",
            "description": "description_text",
        },
    )

    searcher = TextSearcher(mock_client, cfg)

    # Document 1 has 'caption' instead of 'description_text'
    hits = [
        {
            "_score": 4.5,
            "_source": {
                "video_id": "L01_V001",
                "frame_id": 100,
                "caption": "đoàn xe mô tô hộ tống",
            },
        },
        {
            "_score": 3.8,
            "_source": {
                "video_id": "L01_V002",
                "frame_id": 200,
                "dense_caption": "khung cảnh bờ sông Sài Gòn",
            },
        },
    ]

    candidates = searcher._to_candidates(hits, "description_text", PATH_DESCRIPTION)
    assert len(candidates) == 2
    assert candidates[0].extra["description_matched"] == "đoàn xe mô tô hộ tống"
    assert candidates[1].extra["description_matched"] == "khung cảnh bờ sông Sài Gòn"


def test_vqa_context_enrichment_truncation_limits():
    """Verify VQA answer_candidates truncates long descriptions cleanly."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "có"

    long_description = "A" * 1000
    candidate = Candidate(
        video_id="L01_V001",
        frame_id=50,
        score=10.0,
        extra={"description_matched": long_description},
    )

    base_cfg = Config.load("config/config.yaml", no_cache=True)
    answers = answer_candidates(mock_llm, [candidate], "Trời có đang mưa không?", base_cfg, use_cache=False)

    assert answers[("L01_V001", 50)] == "có"
    called_prompt = mock_llm.chat.call_args[0][1]
    # Check that context length was capped cleanly to <= 350
    assert "Mô tả phân cảnh (Description): " + "A" * 350 in called_prompt
    assert "A" * 351 not in called_prompt


def test_vqa_fetch_metadata_for_visual_candidates():
    """Verify TextSearcher.fetch_metadata enriches pure visual candidates from Elasticsearch."""
    from src.clients.elastic import ElasticWrapper
    from src.config import ElasticConfig
    from src.retrieval.search_text import TextSearcher

    mock_client = MagicMock(spec=ElasticWrapper)
    cfg = ElasticConfig(
        hosts=["http://localhost:9200"],
        index="test_index",
        fields={
            "video_id": "video_id",
            "frame_id": "frame_id",
            "ocr": "ocr_text",
            "asr": "clean_text",
            "description": "description_text",
        },
    )
    searcher = TextSearcher(mock_client, cfg)
    mock_client.search.return_value = [
        {
            "_source": {
                "video_id": "L01_V001",
                "frame_id": 100,
                "description_text": "Lễ trao giải âm nhạc",
                "ocr_text": "CÚP VÀNG",
                "clean_text": "xin chúc mừng",
            }
        }
    ]

    c_visual = Candidate(video_id="L01_V001", frame_id=105, score=0.9, source="visual")
    meta = searcher.fetch_metadata([c_visual], max_shot_gap=60)

    assert ("L01_V001", 105) in meta
    assert meta[("L01_V001", 105)]["description"] == "Lễ trao giải âm nhạc"
    assert meta[("L01_V001", 105)]["ocr"] == "CÚP VÀNG"
    assert meta[("L01_V001", 105)]["asr"] == "xin chúc mừng"

