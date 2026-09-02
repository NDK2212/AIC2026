"""Text normalisation, JSON extraction from noisy LLM output, keyframe mapping."""

from __future__ import annotations

import json

import pytest

from src.clients.llm import extract_json_object
from src.schemas import LLMParseError
from src.utils.cache import DiskCache, sha256_key
from src.utils.keyframe_index import KeyframeIndex
from src.utils.text_norm import (
    is_unknown,
    looks_like_exact_text,
    majority_vote,
    normalize_text,
    sanitize_answer,
    strip_think_blocks,
)


# ---------------------------------------------------------------------------
# defensive JSON extraction
# ---------------------------------------------------------------------------
def test_json_survives_a_think_block_and_a_code_fence():
    raw = '<think>Let me consider {"decoy": 1} first.</think>\n```json\n{"a": 1}\n```'
    assert extract_json_object(raw) == {"a": 1}


def test_json_survives_chatter_around_the_object():
    raw = 'Sure, here it is: {"a": 1, "b": [1, 2]} — hope that helps!'
    assert extract_json_object(raw) == {"a": 1, "b": [1, 2]}


def test_braces_inside_strings_do_not_break_the_scan():
    raw = '{"text": "a } brace { inside", "n": 2}'
    assert extract_json_object(raw) == {"text": "a } brace { inside", "n": 2}


def test_escaped_quotes_do_not_break_the_scan():
    raw = r'prefix {"q": "he said \"hi\" }", "n": 1} suffix'
    assert extract_json_object(raw) == {"q": 'he said "hi" }', "n": 1}


def test_the_outermost_object_wins():
    raw = '{"outer": {"inner": 1}, "n": 2}'
    assert extract_json_object(raw) == {"outer": {"inner": 1}, "n": 2}


def test_a_realistic_decomposition_response_parses():
    payload = {
        "original_query": "tìm người áo đỏ",
        "modalities": ["image"],
        "ocr_query": None,
        "asr_query": None,
        "image_query": "a person wearing a red shirt",
        "ocr_terms": [], "asr_terms": [], "image_terms": ["red shirt"],
        "modality_weights": {"ocr": 0.0, "asr": 0.0, "image": 1.0},
    }
    raw = f"<think>reasoning</think>\n```json\n{json.dumps(payload)}\n```"
    assert extract_json_object(raw) == payload


def test_unparsable_output_raises():
    with pytest.raises(LLMParseError):
        extract_json_object("I am afraid I cannot do that.")
    with pytest.raises(LLMParseError):
        extract_json_object("")


def test_strip_think_handles_an_unterminated_block():
    assert strip_think_blocks("<think>never closed") == ""
    assert strip_think_blocks("done</think>after") == "after"


# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------
def test_normalisation_keeps_vietnamese_diacritics():
    assert normalize_text("  MÀU  XANH \n") == "màu xanh"


@pytest.mark.parametrize("raw,expected", [
    ("Answer: 5", "5"),
    ("Đáp án: màu xanh.", "màu xanh"),
    ('**"Năm người"**', "Năm người"),
    ("Trong ảnh, có 3 người", "có 3 người"),
    ("5\nplus an explanation line", "5"),
    ("   ", ""),
])
def test_answers_are_reduced_to_atomic_form(raw, expected):
    assert sanitize_answer(raw) == expected


def test_answers_are_truncated_to_the_limit():
    assert len(sanitize_answer("x" * 500, 100)) == 100


def test_unknown_detection():
    assert is_unknown("UNKNOWN")
    assert is_unknown("")
    assert not is_unknown("5")


def test_majority_vote_ignores_unknowns_and_is_deterministic():
    assert majority_vote(["5", "UNKNOWN", "5", "6"]) == "5"
    assert majority_vote(["UNKNOWN", ""]) is None
    # Case-insensitive grouping, first-seen surface form wins.
    assert majority_vote(["Màu xanh", "màu xanh", "đỏ"]) == "Màu xanh"


@pytest.mark.parametrize("query,expected", [
    ('biển ghi "VinFast"', True),
    ("biển quảng cáo có chữ VinFast", True),
    ("một người đang đi bộ", False),
])
def test_exact_text_heuristic(query, expected):
    assert looks_like_exact_text(query) is expected


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_cache_round_trips_json(tmp_path):
    cache = DiskCache(tmp_path)
    key = sha256_key("model", "prompt")
    assert cache.get_json("llm", key) is None
    cache.set_json("llm", key, {"content": "hi"})
    assert cache.get_json("llm", key) == {"content": "hi"}


def test_a_disabled_cache_stores_nothing(tmp_path):
    cache = DiskCache(tmp_path, enabled=False)
    cache.set_json("llm", "abc", {"content": "hi"})
    assert cache.get_json("llm", "abc") is None


def test_cache_keys_depend_on_every_part():
    assert sha256_key("a", "b") != sha256_key("a", "c")
    assert sha256_key("a", "b") == sha256_key("a", "b")


# ---------------------------------------------------------------------------
# keyframe index
# ---------------------------------------------------------------------------
def build_dataset(tmp_path, mapping: dict[str, list[int]] | None = None):
    """Create a keyframe tree, optionally with map-keyframes CSVs."""
    root = tmp_path / "keyframes"
    map_dir = tmp_path / "map-keyframes"
    for video, frames in (mapping or {"L01_V001": [0, 1, 2]}).items():
        directory = root / video
        directory.mkdir(parents=True)
        for i in range(len(frames)):
            (directory / f"{i:04d}.jpg").write_bytes(b"x")
        map_dir.mkdir(exist_ok=True)
        rows = ["n,pts_time,fps,frame_idx"]
        rows += [f"{i + 1},{i * 0.1},25,{frame}" for i, frame in enumerate(frames)]
        (map_dir / f"{video}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root, map_dir


def test_the_map_csv_drives_the_frame_ids(tmp_path):
    root, map_dir = build_dataset(tmp_path, {"L01_V001": [100, 250, 400]})
    index = KeyframeIndex(root, map_dir, None)

    assert index.all_frames("L01_V001") == [100, 250, 400]
    assert index.path_of("L01_V001", 250).name == "0001.jpg"
    assert index.path_of("L01_V001", 251) is None


def test_nearest_snaps_to_a_real_keyframe(tmp_path):
    root, map_dir = build_dataset(tmp_path, {"L01_V001": [100, 250, 400]})
    index = KeyframeIndex(root, map_dir, None)

    assert index.nearest("L01_V001", 260)[0] == 250
    assert index.nearest("L01_V001", 390)[0] == 400
    assert index.nearest("L01_V001", 0)[0] == 100
    assert index.nearest("L01_V001", 9999)[0] == 400
    assert index.nearest("L99_V999", 10) is None


def test_neighbours_are_real_frames_and_never_the_source(tmp_path):
    root, map_dir = build_dataset(tmp_path, {"L01_V001": list(range(0, 500, 10))})
    index = KeyframeIndex(root, map_dir, None)

    neighbours = index.neighbors("L01_V001", 200, [-15, 15, -30, 30])
    assert 200 not in neighbours
    assert len(neighbours) == len(set(neighbours))
    assert all(f in index.all_frames("L01_V001") for f in neighbours)


def test_missing_map_falls_back_to_file_order(tmp_path):
    root = tmp_path / "keyframes" / "L01_V001"
    root.mkdir(parents=True)
    for i in range(3):
        (root / f"{i:04d}.jpg").write_bytes(b"x")

    index = KeyframeIndex(tmp_path / "keyframes", None, None)
    assert index.all_frames("L01_V001") == [0, 1, 2]
    assert index.videos["L01_V001"].exact is False


def test_scene_sample_filename_is_an_exact_frame_map(tmp_path):
    root = tmp_path / "keyframes" / "L01_V001"
    root.mkdir(parents=True)
    (root / "scene_0002_frame_000754_2.jpg").write_bytes(b"x")
    (root / "scene_0001_frame_000117_0.jpg").write_bytes(b"x")
    # Duplicate variants of one sampled frame are collapsed deterministically.
    (root / "scene_0002_frame_000754_0.jpg").write_bytes(b"x")

    index = KeyframeIndex(tmp_path / "keyframes", None, None)

    assert index.all_frames("L01_V001") == [117, 754]
    assert index.path_of("L01_V001", 117).name == "scene_0001_frame_000117_0.jpg"
    assert index.path_of("L01_V001", 754).name == "scene_0002_frame_000754_0.jpg"
    assert index.videos["L01_V001"].exact is True


def test_metadata_json_is_used_when_no_csv_exists(tmp_path):
    root = tmp_path / "keyframes" / "L01_V001"
    root.mkdir(parents=True)
    for i in range(3):
        (root / f"{i:04d}.jpg").write_bytes(b"x")
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "L01_V001.json").write_text(
        json.dumps({"keyframes": [7, 21, 35]}), encoding="utf-8"
    )

    index = KeyframeIndex(tmp_path / "keyframes", None, meta_dir)
    assert index.all_frames("L01_V001") == [7, 21, 35]


def test_median_gap(tmp_path):
    root, map_dir = build_dataset(tmp_path, {"L01_V001": [0, 10, 20, 30]})
    index = KeyframeIndex(root, map_dir, None)
    assert index.median_gap("L01_V001") == 10
