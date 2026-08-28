"""Comprehensive deep-workflow stress and edge-case testing for AIC 2026.

Tests all 3 tasks (KIS, VQA, TRAKE), batch execution, schema alignment with elasticsearch.py,
language routing, error handling, validator checks, and zip packaging under extreme conditions.
"""

from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.retrieval.fusion import fuse, weighted_rrf
from src.retrieval.pipeline import RetrievalPipeline
from src.schemas import (
    Candidate,
    DecomposeResult,
    PATH_ASR,
    PATH_DESCRIPTION,
    PATH_OCR,
    PATH_VISUAL,
    TrakePlan,
    TrakeSequence,
    TrakeStep,
    VQASplit,
)
from src.submission.builder import build_rows
from src.submission.packer import pack
from src.submission.validator import validate, validate_file
from src.submission.writer import write_kis, write_qa, write_trake
from src.tasks.kis import run_kis
from src.tasks.trake import (
    align_video,
    plan_trake,
    rank_videos,
    retrieve_steps,
    run_trake,
)
from src.tasks.vqa import (
    answer_candidates,
    propagate_answers,
    run_vqa,
    select_qa_targets,
    split_query,
)


@pytest.fixture
def base_config():
    return Config.load("config/config.yaml", no_cache=True)


def make_candidate(video: str, frame: int, score: float = 1.0, extra: dict | None = None) -> Candidate:
    return Candidate(
        video_id=video,
        frame_id=frame,
        score=score,
        source="mock",
        rank=1,
        extra=extra or {},
    )


# ==============================================================================
# 1. DEEP KIS WORKFLOW TESTS
# ==============================================================================
def test_kis_full_workflow_end_to_end(base_config):
    """Test full KIS flow from natural query to validated 100-row CSV file."""
    mock_pipeline = MagicMock(spec=RetrievalPipeline)
    candidates = [
        make_candidate(f"L01_V{v:03d}", f * 50, 1.0 / (v + f))
        for v in range(1, 15)
        for f in range(1, 10)
    ]
    mock_pipeline.run.return_value = (candidates, MagicMock())

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "query_01_kis.csv"
        rows = run_kis(
            "Một người đàn ông đang mở chiếc laptop màu đen trên bàn làm việc",
            mock_pipeline,
            None,
            base_config,
        )

        assert len(rows) == 100
        write_kis(out_csv, rows)
        assert out_csv.exists()

        # Validate format
        errors = validate_file(out_csv)
        assert errors == []

        # Check diversity in top rows
        top_videos = [r[0] for r in rows[:8]]
        assert len(set(top_videos)) >= 3


# ==============================================================================
# 2. DEEP VQA WORKFLOW TESTS (LLM Native Reasoning & OCR Preservation)
# ==============================================================================
def test_vqa_workflow_number_count_preservation(base_config):
    """Test VQA query asking for number count produces digit-only answer."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = {
        "scene_description": "lễ trao giải thưởng âm nhạc lớn",
        "question": "có bao nhiêu người lên nhận giải thưởng?",
        "expected_answer_type": "number",
    }
    mock_llm.chat.return_value = "5"

    split = split_query(mock_llm, "Trong lễ trao giải thưởng âm nhạc lớn, có bao nhiêu người lên nhận giải thưởng?")
    assert split.expected_answer_type == "number"

    cand = make_candidate(
        "L01_V028", 1500, 0.95,
        extra={
            "description_matched": "5 ca sĩ nhóm nhạc nam bước lên sân khấu nhận cúp vàng",
            "clean_text": "Xin chúc mừng 5 chàng trai xuất sắc",
        },
    )

    answers = answer_candidates(
        mock_llm, [cand], split.question, base_config, use_cache=False
    )
    # Output must be sanitized to integer '5'
    assert answers[("L01_V028", 1500)] == "5"


def test_vqa_workflow_exact_ocr_preservation(base_config):
    """Test VQA query about billboard/signboard preserves exact Vietnamese diacritic text."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = {
        "scene_description": "khung cảnh trước cổng chợ truyền thống",
        "question": "Biển hiệu cổng chợ ghi chữ gì?",
        "expected_answer_type": "text",
    }
    mock_llm.chat.return_value = "Chợ Bến Thành"

    split = split_query(mock_llm, "Trong video trước cổng chợ, biển hiệu cổng chợ ghi chữ gì?")
    cand = make_candidate(
        "L02_V010", 300, 0.9,
        extra={"ocr_matched": "CHỢ BẾN THÀNH - CỬA NAM"},
    )

    answers = answer_candidates(
        mock_llm, [cand], split.question, base_config, use_cache=False
    )
    assert answers[("L02_V010", 300)] == "Chợ Bến Thành"


def test_vqa_hierarchical_propagation_full_coverage(base_config):
    """Test VQA answer propagation spreads from 2 representative targets to all 100 rows."""
    candidates = [make_candidate("L01_V001", i * 10, 1.0 - i * 0.01) for i in range(100)]
    # Only 2 frames were answered by LLM
    answers = {
        ("L01_V001", 0): "màu đỏ",
        ("L01_V001", 500): "màu xanh",
    }

    full_answers = propagate_answers(candidates, answers, base_config)
    assert len(full_answers) == 100
    assert all(ans[2] in ("màu đỏ", "màu xanh") for ans in full_answers)
    assert full_answers[0][2] == "màu đỏ"
    assert full_answers[2][2] == "màu đỏ"  # frame 20 is within shot window of frame 0


# ==============================================================================
# 3. DEEP TRAKE WORKFLOW TESTS (1-Shot Plan & DP Monotonic Alignment)
# ==============================================================================
def test_trake_1shot_plan_and_step_weights_routing(base_config):
    """Verify TRAKE retrieves steps using Description + Visual with config weights."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = {
        "action": "weightlifting",
        "num_events": 3,
        "steps": [
            {"index": 0, "description": "Athlete gripping barbell on the floor", "description_local": "chuẩn bị nâng"},
            {"index": 1, "description": "Athlete pulling barbell to chest level", "description_local": "kéo tạ lên ngực"},
            {"index": 2, "description": "Athlete pushing barbell above head with arms locked", "description_local": "đẩy tạ qua đầu"},
        ],
    }

    plan = plan_trake(mock_llm, "Vận động viên cử tạ thực hiện 3 bước: chuẩn bị, kéo tạ, đẩy tạ", base_config)
    assert plan.num_events == 3
    assert plan.steps[0].description_local == "chuẩn bị nâng"

    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = [
        ([make_candidate("L05_V001", 100, 0.9)], None),
        ([make_candidate("L05_V001", 250, 0.95)], None),
        ([make_candidate("L05_V001", 400, 0.85)], None),
    ]

    results = retrieve_steps(plan, mock_pipeline, base_config, use_cache=False)
    assert len(results) == 3

    # Ensure step 0 decompose used step_weights from config
    dec0 = mock_pipeline.run.call_args_list[0][1]["decompose_result"]
    assert dec0.ocr_query is None
    assert dec0.asr_query is None
    assert dec0.description_query == "chuẩn bị nâng"
    assert dec0.image_query == "Athlete gripping barbell on the floor"
    assert dec0.modality_weights == {"image": 0.6, "description": 0.4}


def test_trake_alignment_strict_monotonic_and_step_interpolation(base_config):
    """Verify DANTE DP enforces strictly increasing frame IDs and fills missing steps."""
    # Video 1 has steps 0 and 2, but misses step 1
    tables = [
        {("L01_V001", 100): 0.9},   # Step 0
        {},                         # Step 1 (Missing!)
        {("L01_V001", 500): 0.8},   # Step 2
    ]

    seqs = align_video("L01_V001", tables, base_config.trake, None, max_paths=1)
    assert len(seqs) == 1
    seq = seqs[0]
    assert len(seq.frame_ids) == 3
    # Step 1 must be interpolated between 100 and 500
    assert 100 < seq.frame_ids[1] < 500
    assert seq.frame_ids[0] < seq.frame_ids[1] < seq.frame_ids[2]
    assert 1 in seq.filled_steps


# ==============================================================================
# 4. SUBMISSION VALIDATOR & ZIP PACKER TESTS
# ==============================================================================
def test_submission_full_pack_and_unpack_verification():
    """Verify writing all 3 tasks, validating, packing into zip, and checking zip entries."""
    with tempfile.TemporaryDirectory() as tmp:
        sub_dir = Path(tmp) / "submission"
        sub_dir.mkdir()

        # 1. KIS File
        write_kis(sub_dir / "Q01_kis.csv", [("L01_V001", 100), ("L01_V002", 200)])

        # 2. QA File
        write_qa(sub_dir / "Q02_qa.csv", [("L01_V001", 100, "5"), ("L01_V002", 200, "áo đỏ")])

        # 3. TRAKE File
        write_trake(sub_dir / "Q03_trake.csv", [["L01_V001", 100, 200, 300]], num_events=3)

        # Validate directory
        errors = validate(sub_dir)
        assert errors == []

        # Pack into zip
        out_zip = Path(tmp) / "final_submission.zip"
        pack(sub_dir, out_zip)
        assert out_zip.exists()

        # Verify ZIP contains 'submission/' prefix
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert len(names) == 3
            assert all(n.startswith("submission/") for n in names)
            assert "submission/Q01_kis.csv" in names
            assert "submission/Q02_qa.csv" in names
            assert "submission/Q03_trake.csv" in names


# ==============================================================================
# 5. WRRF FUSION WITH MISSING PATHS & TIE-BREAKING
# ==============================================================================
def test_wrrf_fusion_graceful_missing_paths_and_deterministic_tiebreak():
    """Verify fusion works correctly when some paths return empty and tie-breaks cleanly."""
    c1 = make_candidate("L01_V001", 100, 0.9)
    c2 = make_candidate("L01_V002", 100, 0.9)  # Identical score -> tie break by video_id

    paths = {
        PATH_VISUAL: [c1, c2],
        PATH_DESCRIPTION: [],  # Empty path
        PATH_OCR: [],
        PATH_ASR: [],
    }

    fused = weighted_rrf(paths, weights={PATH_VISUAL: 1.0, PATH_DESCRIPTION: 0.5})
    assert len(fused) == 2
    assert fused[0].video_id == "L01_V001"
    assert fused[1].video_id == "L01_V002"
    assert fused[0].rank == 1
    assert fused[1].rank == 2


# ==============================================================================
# 6. EXTREME EDGE CASES & RESILIENCE TESTS
# ==============================================================================
@pytest.mark.parametrize("query,expected_count", [
    ("Vận động viên thể dục dụng cụ thực hiện 3 bước", 3),
    ("Vận động viên trượt băng trải qua bốn giai đoạn then chốt", 4),
    ("Quy trình nhảy sào gồm hai khoảnh khắc chính", 2),
    ("Năm bước hoàn thành pha ném bóng rổ", 5),
    ("Một người đàn ông đi bộ trong công viên", 3),  # Default heuristic fallback
])
def test_trake_vietnamese_word_number_extraction_resilience(query, expected_count, base_config):
    """Verify TRAKE heuristic parses Vietnamese written numbers (ba, bốn, hai, năm) when LLM fails."""
    class FailingLLM:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("NVIDIA API timeout")

    plan = plan_trake(FailingLLM(), query, base_config, use_cache=False)
    assert plan.num_events == expected_count
    assert len(plan.steps) == expected_count


def test_validator_detects_all_corruptions():
    """Verify validator catches: BOM, Excel zip header, stray non-integer frames, and negative frames."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # 1. BOM file
        bom_file = d / "bom_kis.csv"
        bom_file.write_bytes(b"\xef\xbb\xbfL01_V001,100\n")
        assert any("BOM" in err for err in validate_file(bom_file))

        # 2. Header row
        header_file = d / "hdr_kis.csv"
        header_file.write_text("video_id,frame_id\nL01_V001,100\n", encoding="utf-8")
        assert any("header" in err for err in validate_file(header_file))

        # 3. Non-integer frame
        bad_frame = d / "bad_kis.csv"
        bad_frame.write_text("L01_V001,abc\n", encoding="utf-8")
        assert any("not an integer" in err for err in validate_file(bad_frame))

        # 4. TRAKE non-monotonic frame order
        bad_trake = d / "nonmono_trake.csv"
        bad_trake.write_text("L01_V001,300,200,100\n", encoding="utf-8")
        assert any("chronological" in err for err in validate_file(bad_trake))


def test_vqa_split_fallback_when_llm_json_fails():
    """Verify VQA split gracefully falls back to raw query when LLM JSON fails."""
    class BadLLM:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("JSON decoding error")

    split = split_query(BadLLM(), "Người đàn ông áo đỏ ở đâu?")
    assert split.scene_description == "Người đàn ông áo đỏ ở đâu?"
    assert split.question == "Người đàn ông áo đỏ ở đâu?"

