"""End-to-end pipeline behaviour with every backend mocked out."""

from __future__ import annotations

import json

import pytest

from src.config import Config
from src.retrieval.decompose import decompose
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas import Candidate, PATH_ASR, PATH_OCR, PATH_VISUAL
from tests.conftest import make_candidate


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeLLM:
    """Returns a canned JSON payload and records the prompts it was given."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system, user, schema_hint=None, use_cache=True):
        self.calls.append((system, user))
        payload = self.payload(user) if callable(self.payload) else self.payload
        if isinstance(payload, str):
            from src.clients.llm import extract_json_object

            return extract_json_object(payload)
        return payload

    def chat(self, system, user, use_cache=True):
        return json.dumps(self.payload)


class FakeTextSearcher:
    def __init__(self, ocr=None, asr=None, description=None, fail=()):
        self.ocr = ocr or []
        self.asr = asr or []
        self.description = description or []
        self.fail = set(fail)
        self.seen: list[str] = []

    def search_ocr(self, query, terms=None, size=None, exact_text=False):
        self.seen.append("ocr")
        if "ocr" in self.fail:
            raise RuntimeError("elasticsearch is down")
        return self.ocr

    def search_asr(self, query, terms=None, size=None):
        self.seen.append("asr")
        if "asr" in self.fail:
            raise RuntimeError("elasticsearch is down")
        return self.asr

    def search_description(self, query, terms=None, size=None):
        self.seen.append("description")
        if "description" in self.fail:
            raise RuntimeError("elasticsearch is down")
        return self.description


class FakeVisualSearcher:
    def __init__(self, results=None, fail=False):
        self.results = results or []
        self.fail = fail
        self.seen: list[str] = []

    def search(self, image_query, limit=None):
        self.seen.append(image_query or "")
        if self.fail:
            raise RuntimeError("qdrant is down")
        return self.results


DECOMPOSITION = {
    "original_query": "q",
    "modalities": ["ocr", "asr", "image"],
    "ocr_query": "VinFast",
    "asr_query": "tai nạn giao thông",
    "image_query": "a person in a red shirt",
    "ocr_terms": ["VinFast"],
    "asr_terms": ["tai nạn"],
    "image_terms": ["red shirt"],
    "modality_weights": {"ocr": 0.3, "asr": 0.2, "image": 0.5},
}


@pytest.fixture
def cfg() -> Config:
    return Config.load("config/config.yaml", no_cache=True)


def build_pipeline(cfg, llm, text, visual):
    return RetrievalPipeline(cfg=cfg, llm=llm, text_searcher=text, visual_searcher=visual)


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------
def test_decomposition_normalises_weights_to_one():
    result = decompose(FakeLLM(DECOMPOSITION), "q", adaptive_floor=0.2)
    assert sum(result.modality_weights.values()) == pytest.approx(1.0)
    assert min(result.modality_weights.values()) >= 0.2 - 1e-9


def test_a_null_modality_gets_no_weight():
    payload = dict(DECOMPOSITION, ocr_query=None, asr_query=None,
                   modality_weights={"ocr": 0.4, "asr": 0.4, "image": 0.2})
    result = decompose(FakeLLM(payload), "q", adaptive_floor=0.2)

    assert result.ocr_query is None
    assert "ocr" not in result.modality_weights
    assert result.modality_weights == {"image": pytest.approx(1.0)}
    assert result.modalities == ["image"]


def test_all_null_falls_back_to_the_raw_query():
    payload = dict(DECOMPOSITION, ocr_query=None, asr_query=None, image_query=None)
    result = decompose(FakeLLM(payload), "tìm người áo đỏ", adaptive_floor=0.0,
                       default_weights={"ocr": 1.0, "asr": 1.0, "visual": 1.5})

    assert result.ocr_query == result.asr_query == result.image_query == "tìm người áo đỏ"
    assert sum(result.modality_weights.values()) == pytest.approx(1.0)


def test_decomposition_survives_a_think_wrapped_response():
    raw = f'<think>hmm</think>```json\n{json.dumps(DECOMPOSITION)}\n```'
    result = decompose(FakeLLM(raw), "q", adaptive_floor=0.0)
    assert result.image_query == "a person in a red shirt"


def test_the_string_null_is_treated_as_null():
    payload = dict(DECOMPOSITION, ocr_query="null", asr_query="None")
    result = decompose(FakeLLM(payload), "q")
    assert result.ocr_query is None and result.asr_query is None


def test_exact_text_is_detected_from_the_query():
    result = decompose(FakeLLM(DECOMPOSITION), 'biển ghi "VinFast"')
    assert result.exact_text is True


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def test_all_three_paths_run_and_fuse(cfg):
    text = FakeTextSearcher(
        ocr=[make_candidate("L01_V001", 10, 5.0, PATH_OCR, 1)],
        asr=[make_candidate("L01_V002", 20, 4.0, PATH_ASR, 1)],
    )
    visual = FakeVisualSearcher([make_candidate("L01_V003", 30, 0.9, PATH_VISUAL, 1)])
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION), text, visual)

    candidates, result = pipeline.run("q", topk=10, write_trace=False)

    assert sorted(text.seen) == ["asr", "ocr"]
    assert visual.seen == ["a person in a red shirt"]
    assert {c.key for c in candidates} == {
        ("L01_V001", 10), ("L01_V002", 20), ("L01_V003", 30)
    }
    assert result.modality_weights


def test_a_disabled_path_is_skipped_cleanly(cfg):
    payload = dict(DECOMPOSITION, ocr_query=None, asr_query=None)
    text = FakeTextSearcher(ocr=[make_candidate("L01_V001", 10, 5.0)])
    visual = FakeVisualSearcher([make_candidate("L01_V003", 30, 0.9)])
    pipeline = build_pipeline(cfg, FakeLLM(payload), text, visual)

    candidates, _ = pipeline.run("q", topk=10, write_trace=False)

    assert text.seen == []                       # never called at all
    assert [c.key for c in candidates] == [("L01_V003", 30)]


def test_one_failing_path_does_not_kill_the_query(cfg):
    text = FakeTextSearcher(asr=[make_candidate("L01_V002", 20, 4.0)], fail=["ocr"])
    visual = FakeVisualSearcher([make_candidate("L01_V003", 30, 0.9)])
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION), text, visual)

    candidates, _ = pipeline.run("q", topk=10, write_trace=False)

    assert {c.key for c in candidates} == {("L01_V002", 20), ("L01_V003", 30)}


def test_every_path_failing_yields_an_empty_result(cfg):
    text = FakeTextSearcher(fail=["ocr", "asr"])
    visual = FakeVisualSearcher(fail=True)
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION), text, visual)

    candidates, _ = pipeline.run("q", topk=10, write_trace=False)
    assert candidates == []


def test_adaptive_weights_come_from_the_llm(cfg):
    cfg.fusion.adaptive = True
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION),
                              FakeTextSearcher(), FakeVisualSearcher())
    _, result = pipeline.run("q", topk=1, write_trace=False)

    weights = pipeline.resolve_weights(result)
    assert weights[PATH_VISUAL] > weights[PATH_OCR] > weights[PATH_ASR]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_static_weights_are_used_when_adaptive_is_off(cfg):
    cfg.fusion.adaptive = False
    cfg.fusion.weights = {"ocr": 1.0, "asr": 1.0, "visual": 2.0}
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION),
                              FakeTextSearcher(), FakeVisualSearcher())
    _, result = pipeline.run("q", topk=1, write_trace=False)

    weights = pipeline.resolve_weights(result)
    assert weights[PATH_VISUAL] == pytest.approx(0.5)
    assert weights[PATH_OCR] == pytest.approx(0.25)


def test_the_trace_file_records_the_weights_actually_used(cfg, tmp_path):
    cfg.runs.dir = tmp_path
    text = FakeTextSearcher(ocr=[make_candidate("L01_V001", 10, 5.0)])
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION), text, FakeVisualSearcher())

    pipeline.run("q", topk=10, trace_name="query-1-kis", write_trace=True)

    traces = list(tmp_path.glob("*query-1-kis.json"))
    assert len(traces) == 1
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert set(payload["fusion"]["weights_used"]) == {PATH_OCR, PATH_ASR, PATH_VISUAL}
    assert payload["decompose"]["image_query"] == "a person in a red shirt"
    assert "fused_top50" in payload


def test_topk_truncates_the_result(cfg):
    text = FakeTextSearcher(ocr=[make_candidate("L01_V001", i, 10.0 - i) for i in range(50)])
    pipeline = build_pipeline(cfg, FakeLLM(DECOMPOSITION), text, FakeVisualSearcher())

    candidates, _ = pipeline.run("q", topk=7, write_trace=False)
    assert len(candidates) == 7


def test_a_supplied_decomposition_skips_the_llm(cfg):
    llm = FakeLLM(DECOMPOSITION)
    pipeline = build_pipeline(cfg, llm, FakeTextSearcher(), FakeVisualSearcher())
    prepared = decompose(FakeLLM(DECOMPOSITION), "q")

    llm.calls.clear()
    pipeline.run("q", topk=1, decompose_result=prepared, write_trace=False)
    assert llm.calls == []


class FakeReranker:
    def __init__(self, enabled=True, tag="reranked"):
        self.enabled = enabled
        self.tag = tag
        self.queries = []

    def rerank(self, query, candidates):
        self.queries.append(query)
        if not self.enabled:
            return list(candidates)
        # Reverse order as mock rerank effect
        return [
            c.replace(
                score=c.score + 10.0,
                extra={**c.extra, self.tag: True},
            )
            for c in reversed(candidates)
        ]


def test_blip2_reranker_inside_visual_path(cfg):
    visual = FakeVisualSearcher([
        make_candidate("L01_V001", 10, 0.5, PATH_VISUAL, 1),
        make_candidate("L01_V002", 20, 0.8, PATH_VISUAL, 2),
    ])
    blip2 = FakeReranker(enabled=True, tag="blip2_called")
    pipeline = RetrievalPipeline(
        cfg=cfg,
        llm=FakeLLM(DECOMPOSITION),
        text_searcher=FakeTextSearcher(),
        visual_searcher=visual,
        blip2_reranker=blip2,
    )

    candidates, _ = pipeline.run("q", topk=5, write_trace=False)

    assert blip2.queries == ["a person in a red shirt"]
    assert any(c.extra.get("blip2_called") for c in candidates)


def test_bge_reranker_per_text_path(cfg):
    text = FakeTextSearcher(ocr=[make_candidate("L01_V001", 10, 5.0, PATH_OCR, 1)])
    visual = FakeVisualSearcher([make_candidate("L01_V002", 20, 0.9, PATH_VISUAL, 1)])
    bge = FakeReranker(enabled=True, tag="bge_called")
    pipeline = RetrievalPipeline(
        cfg=cfg,
        llm=FakeLLM(DECOMPOSITION),
        text_searcher=text,
        visual_searcher=visual,
        bge_reranker=bge,
    )

    candidates, _ = pipeline.run("test original query", topk=5, write_trace=False)

    # In symmetric 4-path architecture, BGE reranks text paths with their specialized queries
    assert "VinFast" in bge.queries
    assert any(c.extra.get("bge_called") for c in candidates)

