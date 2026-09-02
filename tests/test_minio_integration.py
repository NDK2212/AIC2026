"""Tests for MinIO Keyframe integration, 2-tier caching, and BLIP-1 ITM reranking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from src.clients.minio_client import MinioKeyframeClient
from src.config import MinioConfig, Blip2RerankConfig
from src.retrieval.rerank import BLIP2Reranker
from src.schemas import Candidate
from src.utils.keyframe_index import KeyframeIndex


def test_minio_config_defaults(tmp_path):
    cfg = MinioConfig.parse({}, tmp_path)
    assert cfg.enabled is True
    assert cfg.endpoint == "bucket.viettech.fit"
    assert cfg.bucket == "aic2026"
    assert cfg.prefix == "keyframes/batch_1/"
    assert cfg.secure is True


def test_minio_client_image_fetch_and_retry():
    cfg = MinioConfig(
        enabled=True,
        endpoint="dummy.endpoint",
        access_key="admin",
        secret_key="admin",
        bucket="test-bucket",
        prefix="keyframes/",
        max_retries=2,
    )
    client = MinioKeyframeClient(cfg)

    # Mock MinIO SDK
    mock_minio = MagicMock()
    client._client = mock_minio

    # Case 1: successful image stream
    import io
    test_img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.stream.return_value = [img_bytes]
    mock_minio.get_object.return_value = mock_resp

    img = client.get_image("keyframes/test.jpg")
    assert img is not None
    assert img.size == (64, 64)

    # Case 2: error retry failure
    mock_minio.get_object.side_effect = Exception("Connection refused")
    failed_img = client.get_image("keyframes/bad.jpg")
    assert failed_img is None


def test_keyframe_index_with_minio(tmp_path):
    cache_dir = tmp_path / "cache_kf"
    cfg = MinioConfig(
        enabled=True,
        cache_dir=cache_dir,
    )
    mock_minio = MagicMock(spec=MinioKeyframeClient)
    mock_minio.enabled = True
    mock_minio.discover_video_frames.return_value = {
        2: "keyframes/batch_1/L21_V001/scene_0000_frame_000002_0.jpg",
        4: "keyframes/batch_1/L21_V001/scene_0000_frame_000004_1.jpg",
    }
    test_img = Image.new("RGB", (32, 32), color="blue")
    mock_minio.get_image.return_value = test_img

    kf = KeyframeIndex(
        root=tmp_path / "non_existent_local_dir",
        cache=None,
        minio=mock_minio,
        cache_dir=cache_dir,
    )

    # Video existence through MinIO discovery
    assert kf.has_video("L21_V001") is True
    assert kf.all_frames("L21_V001") == [2, 4]
    assert kf.nearest_frame("L21_V001", 3) == 2

    # Fetch image and check caching
    loaded = kf.get_image("L21_V001", 2)
    assert loaded is not None
    assert loaded.size == (32, 32)
    assert (cache_dir / "L21_V001" / "2.jpg").is_file()

    # Second read should hit disk cache without calling minio
    mock_minio.get_image.reset_mock()
    loaded_cached = kf.get_image("L21_V001", 2)
    assert loaded_cached is not None
    assert mock_minio.get_image.call_count == 0

    # Batch get images
    items = [("L21_V001", 2), ("L21_V001", 4)]
    batch_res = kf.batch_get_images(items)
    assert len(batch_res) == 2
    assert ("L21_V001", 2) in batch_res
    assert ("L21_V001", 4) in batch_res


def test_minio_fills_an_exact_frame_missing_from_a_local_video(tmp_path):
    root = tmp_path / "keyframes" / "L21_V001"
    root.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="red").save(
        root / "scene_0000_frame_000002_0.jpg"
    )
    mock_minio = MagicMock(spec=MinioKeyframeClient)
    mock_minio.enabled = True
    mock_minio.discover_video_frames.return_value = {
        4: "keyframes/batch_1/L21_V001/scene_0001_frame_000004_0.jpg"
    }
    mock_minio.get_image.return_value = Image.new("RGB", (24, 24), color="blue")
    index = KeyframeIndex(
        root=tmp_path / "keyframes",
        minio=mock_minio,
        cache_dir=tmp_path / "cache",
    )

    image = index.get_image("L21_V001", 4)

    assert image is not None and image.size == (24, 24)
    mock_minio.get_image.assert_called_once_with(
        "keyframes/batch_1/L21_V001/scene_0001_frame_000004_0.jpg"
    )


def test_blip_reranker_with_keyframe_index(tmp_path):
    kf = MagicMock(spec=KeyframeIndex)
    test_img = Image.new("RGB", (224, 224), color="green")
    kf.batch_get_images.return_value = {
        ("L01_V001", 10): test_img,
        ("L01_V001", 20): test_img,
    }

    cfg = Blip2RerankConfig(
        enabled=True,
        model_id="Salesforce/blip-itm-base-coco",
        device="cpu",
        top_n=2,
    )
    reranker = BLIP2Reranker(cfg, kf)

    # Mock processor and model
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_torch = MagicMock()

    mock_torch.inference_mode.return_value.__enter__ = MagicMock()
    mock_torch.inference_mode.return_value.__exit__ = MagicMock()
    mock_logits = MagicMock()
    mock_logits.softmax.return_value = mock_torch.tensor([[0.2, 0.8], [0.1, 0.9]])
    mock_torch.softmax.return_value.cpu.return_value.tolist.return_value = [0.85, 0.95]

    mock_out = MagicMock()
    mock_out.itm_score = mock_logits
    mock_model.return_value = mock_out

    reranker._model = mock_model
    reranker._processor = mock_processor
    reranker._torch = mock_torch

    candidates = [
        Candidate(video_id="L01_V001", frame_id=10, score=0.5, rank=1, source="visual"),
        Candidate(video_id="L01_V001", frame_id=20, score=0.4, rank=2, source="visual"),
    ]

    reranked = reranker.rerank("a green background", candidates)
    assert len(reranked) == 2
    assert "blip2_itm_score" in reranked[0].extra
