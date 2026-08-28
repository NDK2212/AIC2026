"""RRF, weighted RRF and weight normalisation."""

from __future__ import annotations

import pytest

from src.retrieval.fusion import fuse, normalize_weights, rrf, weighted_norm, weighted_rrf
from tests.conftest import make_candidate


def test_rrf_matches_the_formula():
    paths = {
        "ocr": [make_candidate("L01_V001", 10, 9.0), make_candidate("L01_V001", 20, 8.0)],
        "asr": [make_candidate("L01_V001", 20, 5.0), make_candidate("L01_V001", 10, 4.0)],
    }
    fused = {c.key: c.score for c in rrf(paths, k=60)}

    # frame 10: rank 1 in ocr, rank 2 in asr; frame 20 is the mirror image.
    assert fused[("L01_V001", 10)] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[("L01_V001", 20)] == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_rrf_applies_per_path_weights():
    paths = {
        "ocr": [make_candidate("L01_V001", 10, 9.0)],
        "visual": [make_candidate("L01_V001", 20, 9.0)],
    }
    weights = {"ocr": 0.2, "visual": 0.8}
    fused = weighted_rrf(paths, weights, k=60)

    assert fused[0].key == ("L01_V001", 20)          # the heavier path wins
    assert fused[0].score == pytest.approx(0.8 / 61)
    assert fused[1].score == pytest.approx(0.2 / 61)


def test_candidate_present_in_one_path_only_is_not_penalised():
    paths = {
        "ocr": [make_candidate("L01_V001", 10, 9.0)],
        "asr": [make_candidate("L01_V002", 30, 9.0)],
    }
    fused = rrf(paths, k=60)

    assert len(fused) == 2
    assert all(c.score == pytest.approx(1 / 61) for c in fused)
    # Both appear exactly once, each crediting only its own path.
    for candidate in fused:
        assert len(candidate.extra["per_path_rank"]) == 1


def test_duplicate_key_inside_one_path_keeps_the_best_rank():
    paths = {
        "visual": [
            make_candidate("L01_V001", 10, 9.0),
            make_candidate("L01_V001", 20, 8.0),
            make_candidate("L01_V001", 10, 7.0),   # same key again, worse rank
        ]
    }
    fused = {c.key: c for c in rrf(paths, k=60)}
    assert fused[("L01_V001", 10)].extra["per_path_rank"]["visual"] == 1


def test_fusion_is_deterministic_on_ties():
    paths = {
        "ocr": [
            make_candidate("L02_V003", 5, 1.0),
            make_candidate("L01_V001", 5, 1.0),
            make_candidate("L01_V001", 4, 1.0),
        ]
    }
    first = [c.key for c in rrf(paths)]
    second = [c.key for c in rrf(paths)]
    assert first == second

    tied = {"a": [make_candidate("L02_V003", 5, 1.0)],
            "b": [make_candidate("L01_V001", 9, 1.0)]}
    # Equal scores must break on (video_id, frame_id).
    assert [c.key for c in rrf(tied)] == [("L01_V001", 9), ("L02_V003", 5)]


def test_ranks_are_contiguous_and_one_based():
    paths = {"ocr": [make_candidate("L01_V001", f, 10 - f) for f in range(5)]}
    fused = rrf(paths)
    assert [c.rank for c in fused] == [1, 2, 3, 4, 5]


def test_weighted_norm_preserves_score_gaps():
    paths = {
        "visual": [
            make_candidate("L01_V001", 10, 10.0),
            make_candidate("L01_V001", 20, 9.9),
            make_candidate("L01_V001", 30, 1.0),
        ]
    }
    fused = {c.key: c.score for c in weighted_norm(paths, {"visual": 1.0})}
    assert fused[("L01_V001", 10)] == pytest.approx(1.0)
    assert fused[("L01_V001", 20)] == pytest.approx(0.98888, abs=1e-4)
    assert fused[("L01_V001", 30)] == pytest.approx(0.0)


def test_weighted_norm_handles_a_flat_path():
    paths = {"ocr": [make_candidate("L01_V001", f, 3.0) for f in (10, 20)]}
    fused = weighted_norm(paths, {"ocr": 1.0})
    assert all(c.score == pytest.approx(1.0) for c in fused)


def test_empty_paths_fuse_to_nothing():
    assert rrf({}) == []
    assert fuse({"ocr": [], "asr": []}) == []


def test_fuse_dispatches_on_method():
    paths = {"ocr": [make_candidate("L01_V001", 10, 1.0)],
             "visual": [make_candidate("L01_V001", 20, 1.0)]}
    weights = {"ocr": 0.1, "visual": 0.9}

    assert fuse(paths, "rrf", weights)[0].score == pytest.approx(1 / 61)
    assert fuse(paths, "weighted_rrf", weights)[0].key == ("L01_V001", 20)
    assert fuse(paths, "weighted_norm", weights)[0].key == ("L01_V001", 20)


def test_normalize_weights_sums_to_one_and_applies_the_floor():
    weights = normalize_weights({"ocr": 0.0, "asr": 0.1, "visual": 0.9},
                                ["ocr", "asr", "visual"], floor=0.2)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) > 0.0
    # The floor lifts the starved paths before renormalising.
    assert weights["ocr"] == pytest.approx(0.2 / 1.3)


def test_normalize_weights_ignores_inactive_paths():
    weights = normalize_weights({"ocr": 0.5, "asr": 0.5, "visual": 0.0},
                                ["visual"], floor=0.2)
    assert weights == {"visual": pytest.approx(1.0)}


def test_normalize_weights_falls_back_to_uniform():
    weights = normalize_weights({}, ["ocr", "visual"], floor=0.0)
    assert weights == {"ocr": pytest.approx(0.5), "visual": pytest.approx(0.5)}
