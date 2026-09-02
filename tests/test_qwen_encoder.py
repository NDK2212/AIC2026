import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.config import Config, EncoderConfig
from src.embedding import get_encoder, reset_encoders
from src.embedding.qwen import QwenTextEncoder
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.search_visual import VisualSearcher
from src.schemas import Candidate, EncoderUnavailable, PATH_VISUAL
from tests.conftest import make_candidate


# ==============================================================================
# 1. Config Parsing & Registration Tests
# ==============================================================================
def test_qwen_config_parsing():
    """Verify that config.yaml correctly loads qwen under qdrant and embedding."""
    cfg = Config.load("config/config.yaml")
    assert "qwen" in cfg.qdrant.vector_names
    assert cfg.qdrant.vector_names["qwen"] == "qwen"
    assert cfg.embedding.qwen is not None
    assert cfg.embedding.qwen.dim == 2048
    assert cfg.embedding.qwen.backend == "sentence_transformers"
    assert cfg.embedding.qwen.model_id == "Qwen/Qwen3-VL-Embedding-2B"


def test_qwen_get_encoder_factory():
    """Verify that get_encoder returns a QwenTextEncoder singleton."""
    reset_encoders()
    enc_cfg = EncoderConfig(
        name="qwen",
        enabled=True,
        backend="sentence_transformers",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        dim=2048,
        device="cpu",
    )
    encoder = get_encoder(enc_cfg)
    assert isinstance(encoder, QwenTextEncoder)
    assert encoder.dim == 2048
    assert encoder.name == "qwen"


# ==============================================================================
# 2. Math & Normalization Tests
# ==============================================================================
def test_qwen_encode_and_l2_normalization():
    """Verify that QwenTextEncoder produces float32 arrays with exact L2 unit length."""
    enc_cfg = EncoderConfig(
        name="qwen",
        enabled=True,
        backend="sentence_transformers",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        dim=2048,
        device="cpu",
    )
    encoder = QwenTextEncoder(enc_cfg)

    # Mock SentenceTransformer output
    mock_model = MagicMock()
    dummy_vec = np.random.randn(3, 2048).astype(np.float32)
    mock_model.encode.return_value = dummy_vec
    mock_model.get_embedding_dimension.return_value = 2048

    encoder._model = mock_model
    encoder._loaded = True

    texts = ["person in red", "car at night", "cooking food"]
    res = encoder.encode(texts)

    assert isinstance(res, np.ndarray)
    assert res.shape == (3, 2048)
    assert res.dtype == np.float32

    # L2 norm of every row must be strictly 1.0
    norms = np.linalg.norm(res, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-5)


# ==============================================================================
# 3. VisualSearcher 3-Vector End-to-End Integration Tests
# ==============================================================================
def test_visual_searcher_encodes_all_three_vectors():
    """Verify VisualSearcher generates all 3 vectors (SigLIP, BEiT3, Qwen) and passes them to Qdrant."""
    cfg = Config.load("config/config.yaml")
    cfg.embedding.qwen.enabled = True
    mock_qdrant = MagicMock()

    # Canned Qdrant response with 3 points
    mock_qdrant.hybrid_search.return_value = [
        {"id": "p1", "score": 0.85, "payload": {"video_name": "L24_V001", "frame_id": 120}},
        {"id": "p2", "score": 0.72, "payload": {"video_name": "L24_V002", "frame_id": 450}},
    ]

    searcher = VisualSearcher(cfg, mock_qdrant)

    # Mock get_encoder for siglip (1152), beit3 (1024), qwen (2048)
    def fake_get_encoder(enc_cfg, cache=None):
        mock_enc = MagicMock()
        mock_enc.dim = enc_cfg.dim
        mock_enc.encode_one.return_value = np.zeros(enc_cfg.dim, dtype=np.float32)
        return mock_enc

    with patch("src.embedding.get_encoder", side_effect=fake_get_encoder):
        candidates = searcher.search("a dog playing in the park", limit=50)

        # 1. Verify Qdrant was called with all 3 named vectors
        mock_qdrant.hybrid_search.assert_called_once()
        call_args = mock_qdrant.hybrid_search.call_args
        vectors_sent = call_args[0][0]

        assert set(vectors_sent.keys()) == {"siglip", "beit3", "qwen"}
        assert vectors_sent["siglip"].shape == (1152,)
        assert vectors_sent["beit3"].shape == (1024,)
        assert vectors_sent["qwen"].shape == (2048,)

        # 2. Verify returned candidates
        assert len(candidates) == 2
        assert candidates[0].video_id == "L24_V001"
        assert candidates[0].frame_id == 120
        assert candidates[0].source == PATH_VISUAL
        assert candidates[1].video_id == "L24_V002"
        assert candidates[1].frame_id == 450


def test_visual_searcher_graceful_degradation_when_qwen_fails():
    """If Qwen fails to load, visual search should continue with SigLIP and BEiT-3 without crashing."""
    cfg = Config.load("config/config.yaml")
    cfg.embedding.qwen.enabled = True
    mock_qdrant = MagicMock()
    mock_qdrant.hybrid_search.return_value = [
        {"id": "p1", "score": 0.8, "payload": {"video_name": "L24_V001", "frame_id": 100}}
    ]

    searcher = VisualSearcher(cfg, mock_qdrant)

    def fake_get_encoder(enc_cfg, cache=None):
        if enc_cfg.name == "qwen":
            raise EncoderUnavailable("Qwen weights not found on disk")
        mock_enc = MagicMock()
        mock_enc.dim = enc_cfg.dim
        mock_enc.encode_one.return_value = np.zeros(enc_cfg.dim, dtype=np.float32)
        return mock_enc

    with patch("src.embedding.get_encoder", side_effect=fake_get_encoder):
        candidates = searcher.search("a scene description", limit=50)

        # Must not raise, and must search with the remaining 2 vectors
        assert len(candidates) == 1
        assert "qwen" in searcher._degraded
        vectors_sent = mock_qdrant.hybrid_search.call_args[0][0]
        assert set(vectors_sent.keys()) == {"siglip", "beit3"}


def test_visual_searcher_keeps_other_vectors_when_qwen_fails_during_encode():
    """A runtime failure in one loaded encoder must not zero the visual path."""
    cfg = Config.load("config/config.yaml")
    mock_qdrant = MagicMock()
    mock_qdrant.hybrid_search.return_value = [
        {"id": "p1", "score": 0.8, "payload": {"video_name": "L24_V001", "frame_id": 100}}
    ]
    searcher = VisualSearcher(cfg, mock_qdrant)

    def fake_get_encoder(enc_cfg, cache=None):
        mock_enc = MagicMock()
        mock_enc.dim = enc_cfg.dim
        if enc_cfg.name == "qwen":
            mock_enc.encode_one.side_effect = AttributeError(
                "BaseModelOutputWithPooling has no attribute detach"
            )
        else:
            mock_enc.encode_one.return_value = np.zeros(enc_cfg.dim, dtype=np.float32)
        return mock_enc

    with patch("src.embedding.get_encoder", side_effect=fake_get_encoder):
        candidates = searcher.search("a scene description", limit=50)

    assert len(candidates) == 1
    assert "qwen" in searcher._degraded
    vectors_sent = mock_qdrant.hybrid_search.call_args[0][0]
    assert "siglip" in vectors_sent
    assert "qwen" not in vectors_sent


# ==============================================================================
# 4. Full Pipeline Multi-Path Execution Test
# ==============================================================================
def test_full_pipeline_multi_vector_fusion():
    """Verify full RetrievalPipeline runs OCR, ASR, and 3-Vector Visual search in parallel and fuses."""
    cfg = Config.load("config/config.yaml")
    cfg.embedding.qwen.enabled = True

    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = {
        "original_query": "người áo đỏ VinFast",
        "modalities": ["ocr", "image"],
        "ocr_query": "VinFast",
        "asr_query": None,
        "image_query": "a person in red shirt near a car",
        "ocr_terms": ["VinFast"],
        "asr_terms": [],
        "image_terms": ["red shirt"],
        "modality_weights": {"ocr": 0.4, "asr": 0.0, "image": 0.6},
    }

    mock_text = MagicMock()
    mock_text.search_ocr.return_value = [
        make_candidate(video="L24_V001", frame=120, score=5.0, source="ocr")
    ]
    mock_text.search_asr.return_value = []

    mock_qdrant = MagicMock()
    mock_qdrant.hybrid_search.return_value = [
        {"id": "p1", "score": 0.9, "payload": {"video_name": "L24_V001", "frame_id": 120}},
        {"id": "p2", "score": 0.7, "payload": {"video_name": "L24_V005", "frame_id": 300}},
    ]

    visual_searcher = VisualSearcher(cfg, mock_qdrant)

    def fake_get_encoder(enc_cfg, cache=None):
        mock_enc = MagicMock()
        mock_enc.dim = enc_cfg.dim
        mock_enc.encode_one.return_value = np.zeros(enc_cfg.dim, dtype=np.float32)
        return mock_enc

    with patch("src.embedding.get_encoder", side_effect=fake_get_encoder):
        pipeline = RetrievalPipeline(
            cfg=cfg,
            llm=mock_llm,
            text_searcher=mock_text,
            visual_searcher=visual_searcher,
        )

        ranked, decomp = pipeline.run("người áo đỏ VinFast", topk=10, write_trace=False)

        assert len(ranked) >= 2
        # Candidate L24_V001/120 was matched by BOTH OCR and Visual (3-vector), so it should rank #1
        assert ranked[0].video_id == "L24_V001"
        assert ranked[0].frame_id == 120
        assert "ocr" in ranked[0].extra["per_path_rank"]
        assert "visual" in ranked[0].extra["per_path_rank"]
