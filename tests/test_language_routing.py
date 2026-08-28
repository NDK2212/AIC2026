import json
import pytest
from unittest.mock import MagicMock

from src.clients.llm import LLMClient
from src.config import Config
from src.retrieval.decompose import (
    _is_vietnamese,
    decompose,
    translate_to_english_visual,
)
from src.schemas import (
    Candidate,
    DecomposeResult,
    TrakePlan,
    TrakeStep,
    VQASplit,
)
from src.submission.validator import validate_file
from src.submission.writer import write_qa, write_trake
from src.tasks.trake import retrieve_steps
from src.tasks.vqa import propagate_answers, run_vqa, split_query
from src.utils.text_norm import (
    clean_query_text,
    is_unknown,
    majority_vote,
    sanitize_answer,
)
from tests.conftest import make_candidate


# ==============================================================================
# 1. DEEP TESTS: VIETNAMESE DIACRITICS DETECTOR & EDGE CASES
# ==============================================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        # Standard Vietnamese sentences
        ("người đàn ông đang nấu ăn trong bếp", True),
        ("xe buýt tuyến số 08 có chữ Bến Thành", True),
        ("vận động viên nhảy sào tiếp đất", True),
        # Tone marks and vowels: sắc, huyền, hỏi, ngã, nặng, ư, ơ, â, ê, đ
        ("Hà Nội", True),
        ("Đà Nẵng", True),
        ("TP. Hồ Chí Minh", True),
        ("ỦY BAN NHÂN DÂN", True),
        ("KỶ LỤC GUINNESS", True),
        ("LỄ HỘI HOA", True),
        ("đường hoa Nguyễn Huệ", True),
        ("ĐỒNG HỒ ĐEO TAY", True),
        # Pure English sentences
        ("a person cooking in the kitchen", False),
        ("a green public transit bus on the road", False),
        ("the athlete takes off from the ground", False),
        ("an elephant standing in a zoo", False),
        ("on the desk in front of the window", False),
        ("son and daughter playing soccer", False),
        # Pure numbers, brand names, and symbols
        ("VinFast VF8 2026", False),
        ("iPhone 15 Pro Max", False),
        ("1234567890", False),
        ("!@#$%^&*()_+", False),
        ("", False),
        (None, False),
    ],
)
def test_vietnamese_diacritics_detector_exhaustive(text, expected):
    assert _is_vietnamese(text) is expected


# ==============================================================================
# 2. DEEP TESTS: DECOMPOSITION, PROMPT ROUTING & FALLBACK ROBUSTNESS
# ==============================================================================

def test_decompose_bilingual_full_payload():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json.return_value = {
        "original_query": "xe cứu thương chạy trên đường có chữ CẤP CỨU và tiếng còi hú",
        "modalities": ["ocr", "asr", "image"],
        "ocr_query": "CẤP CỨU",
        "asr_query": "tiếng còi hú",
        "image_query": "an ambulance vehicle driving on the road with flashing lights",
        "ocr_terms": ["CẤP CỨU", "cấp cứu", "null", ""],
        "asr_terms": ["tiếng còi hú", "còi cấp cứu"],
        "image_terms": ["ambulance", "emergency vehicle"],
        "modality_weights": {"ocr": 0.4, "asr": 0.2, "image": 0.4},
    }

    res = decompose(mock_llm, "xe cứu thương chạy trên đường có chữ CẤP CỨU và tiếng còi hú")
    # OCR and ASR must preserve original Vietnamese
    assert res.ocr_query == "CẤP CỨU"
    assert res.asr_query == "tiếng còi hú"
    # Terms must be cleaned and deduplicated
    assert res.ocr_terms == ["CẤP CỨU"]
    assert res.asr_terms == ["tiếng còi hú", "còi cấp cứu"]
    # Image query must be rich English
    assert res.image_query == "an ambulance vehicle driving on the road with flashing lights"
    assert res.image_terms == ["ambulance", "emergency vehicle"]
    # Modality weights must sum to 1.0
    assert pytest.approx(sum(res.modality_weights.values()), 0.001) == 1.0
    assert res.modality_weights["ocr"] == pytest.approx(0.4)
    assert res.modality_weights["asr"] == pytest.approx(0.2)
    assert res.modality_weights["image"] == pytest.approx(0.4)


def test_decompose_guardrail_triggers_when_image_query_has_vietnamese():
    mock_llm = MagicMock(spec=LLMClient)
    # LLM fails instruction and outputs Vietnamese in image_query
    mock_llm.chat_json.return_value = {
        "original_query": "cô gái mặc váy trắng chụp ảnh",
        "modalities": ["image"],
        "ocr_query": None,
        "asr_query": None,
        "image_query": "cô gái mặc chiếc váy trắng đang tạo dáng chụp ảnh",
        "ocr_terms": [],
        "asr_terms": [],
        "image_terms": [],
        "modality_weights": {"ocr": 0.0, "asr": 0.0, "image": 1.0},
    }
    # Visual translator is invoked to fix it
    mock_llm.chat.return_value = "a girl in a white dress posing for photography"

    res = decompose(mock_llm, "cô gái mặc váy trắng chụp ảnh")
    assert res.image_query == "a girl in a white dress posing for photography"
    mock_llm.chat.assert_called_once()


def test_decompose_fallback_on_network_or_parse_exception():
    mock_llm = MagicMock(spec=LLMClient)
    # LLM raises an unexpected runtime exception
    mock_llm.chat_json.side_effect = TimeoutError("Connection to LLM timed out")
    # Translator fallback also fails
    mock_llm.chat.side_effect = RuntimeError("Service unavailable")

    # Decompose must NEVER crash the entire retrieval pipeline
    res = decompose(mock_llm, "vận động viên chạy tiếp sức")
    assert res.ocr_query == "vận động viên chạy tiếp sức"
    assert res.asr_query == "vận động viên chạy tiếp sức"
    assert res.description_query == "vận động viên chạy tiếp sức"
    assert res.image_query == "vận động viên chạy tiếp sức"
    assert set(res.modalities) == {"ocr", "asr", "description", "image"}
    assert pytest.approx(sum(res.modality_weights.values()), 0.001) == 1.0


def test_translate_to_english_visual_strips_quotes_and_handles_bad_output():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = '  "a red sports car speeding on track"  '

    translated = translate_to_english_visual(mock_llm, "xe thể thao màu đỏ chạy nhanh")
    assert translated == "a red sports car speeding on track"

    # If translator returns empty, fallback to original query
    mock_llm.chat.return_value = ""
    assert translate_to_english_visual(mock_llm, "người đi xe máy") == "người đi xe máy"


# ==============================================================================
# 3. DEEP TESTS: VQA PROMPTING, SPLITTING, SANITIZATION & ANSWER PROPAGATION
# ==============================================================================

def test_vqa_split_handles_complex_bilingual_and_edge_cases():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json.return_value = {
        "scene_description": "buổi họp báo có logo VinFast",
        "question": "Người đàn ông đứng ở giữa mặc cà vạt màu gì?",
        "question_en": "What color is the tie of the man standing in the center?",
        "expected_answer_type": "color",
    }

    split = split_query(mock_llm, "buổi họp báo có logo VinFast, người đàn ông đứng ở giữa mặc cà vạt màu gì?")
    assert split.scene_description == "buổi họp báo có logo VinFast"
    assert split.question == "Người đàn ông đứng ở giữa mặc cà vạt màu gì?"
    assert split.question_en == "What color is the tie of the man standing in the center?"
    assert split.expected_answer_type == "color"


def test_vqa_split_fallback_when_llm_fails():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_json.side_effect = ValueError("Corrupted JSON")

    query = "đếm số lượng quả táo trên bàn"
    split = split_query(mock_llm, query)
    assert split.scene_description == query
    assert split.question == query
    assert split.question_en == query


@pytest.mark.parametrize(
    "raw_vlm_output,expected_clean",
    [
        # Direct atomic answers
        ("5", "5"),
        ("red", "red"),
        ("yes", "yes"),
        # Metadata / prompt tags
        ("Answer: 5", "5"),
        ("Final Answer - 12", "12"),
        ("Đáp án: Màu đỏ", "Màu đỏ"),
        ("Đáp án là: 3 người", "3 người"),
        ("Output: police car", "police car"),
        # Arbitrary prepositional / participial lead-in clauses
        ("Based on the provided video frame, the vehicle is a red sports car", "a red sports car"),
        ("According to the scene shown in this video, there are 4 people", "4 people"),
        ("Looking at the picture carefully: I can see a helmet on the table", "a helmet on the table"),
        ("From the given image, the answer is: cooking", "cooking"),
        ("As seen in the frame, the color appears to be yellow", "yellow"),
        ("Inside the room, there is a large bookshelf", "a large bookshelf"),
        # Vietnamese natural language lead-ins
        ("Dựa vào hình ảnh được cung cấp, kết quả là: màu vàng", "màu vàng"),
        ("Trong video này có thể thấy: 3 chiếc xe", "3 chiếc xe"),
        ("Nhìn vào bức hình: chiếc áo màu xanh", "chiếc áo màu xanh"),
        # Markdown & quotes wrapping
        ('"red"', "red"),
        ("**5**", "5"),
        ("`yes`", "yes"),
        ("'''guitar'''", "guitar"),
        # Rejection / Unknown variations
        ("UNKNOWN", "UNKNOWN"),
        ("unknown", "unknown"),
        ("N/A", "N/A"),
        ("None", "None"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_vqa_sanitize_answer_exhaustive(raw_vlm_output, expected_clean):
    assert sanitize_answer(raw_vlm_output) == expected_clean


def test_vqa_answer_propagation_and_majority_voting():
    cfg = Config.load("config/config.yaml")

    # Create candidate frames from 2 different videos and multiple shots
    # Video 1: Shot 1 (frames 10, 15, 20), Shot 2 (frames 200, 210)
    # Video 2: Shot 1 (frames 50, 60)
    c1 = make_candidate("L01_V001", 10, 0.95)
    c2 = make_candidate("L01_V001", 15, 0.90)
    c3 = make_candidate("L01_V001", 20, 0.85)
    c4 = make_candidate("L01_V001", 200, 0.80)
    c5 = make_candidate("L01_V001", 210, 0.75)
    c6 = make_candidate("L02_V002", 50, 0.70)
    c7 = make_candidate("L02_V002", 60, 0.65)
    rows = [c1, c2, c3, c4, c5, c6, c7]

    # VLM answers representative frames:
    # Frame 10 -> "5"
    # Frame 200 -> "5"
    # Frame 50 -> "UNKNOWN"
    answers = {
        ("L01_V001", 10): "5",
        ("L01_V001", 200): "5",
        ("L02_V002", 50): "UNKNOWN",
    }

    out = propagate_answers(rows, answers, cfg)
    assert len(out) == 7

    # Video 1 Shot 1: frame 10 (answered "5"), frames 15 & 20 inherit "5" (same shot)
    assert out[0] == ("L01_V001", 10, "5")
    assert out[1] == ("L01_V001", 15, "5")
    assert out[2] == ("L01_V001", 20, "5")
    # Video 1 Shot 2: frame 200 ("5"), frame 210 inherits "5"
    assert out[3] == ("L01_V001", 200, "5")
    assert out[4] == ("L01_V001", 210, "5")
    # Video 2: frame 50 was UNKNOWN -> inherits majority vote "5" from top candidates
    assert out[5] == ("L02_V002", 50, "5")
    assert out[6] == ("L02_V002", 60, "5")


def test_vqa_all_unknown_answers_revert_to_config_fallback(tmp_path):
    cfg = Config.load("config/config.yaml")
    assert cfg.vqa.fallback_answer == "unknown"

    c1 = make_candidate("L01_V001", 10, 0.9)
    c2 = make_candidate("L01_V001", 20, 0.8)
    rows = [c1, c2]

    answers = {
        ("L01_V001", 10): "UNKNOWN",
        ("L01_V001", 20): "N/A",
    }

    out = propagate_answers(rows, answers, cfg)
    assert out[0] == ("L01_V001", 10, "unknown")
    assert out[1] == ("L01_V001", 20, "unknown")

    # Verify formatted submission file passes competition validation
    csv_file = tmp_path / "query-3-qa.csv"
    write_qa(csv_file, out)
    errors = validate_file(csv_file)
    assert errors == []


# ==============================================================================
# 4. DEEP TESTS: TRAKE DUAL-LANGUAGE RETRIEVAL & ISOLATION ROBUSTNESS
# ==============================================================================

def test_trake_multistep_retrieval_and_failure_isolation():
    cfg = Config.load("config/config.yaml")

    mock_llm = MagicMock(spec=LLMClient)

    def fake_chat_json(prompt, user, **kwargs):
        if "chuẩn bị" in user:
            return {
                "original_query": user,
                "modalities": ["image"],
                "ocr_query": None,
                "asr_query": None,
                "image_query": "an athlete preparing to jump",
                "ocr_terms": [],
                "asr_terms": [],
                "image_terms": ["athlete jump"],
                "modality_weights": {"ocr": 0.0, "asr": 0.0, "image": 1.0},
            }
        elif "bay qua xà" in user:
            raise RuntimeError("Step 1 decomposed failed unexpectedly")
        else:
            return {
                "original_query": user,
                "modalities": ["image"],
                "ocr_query": None,
                "asr_query": None,
                "image_query": "an athlete landing safely on the mat",
                "ocr_terms": [],
                "asr_terms": [],
                "image_terms": ["landing mat"],
                "modality_weights": {"ocr": 0.0, "asr": 0.0, "image": 1.0},
            }

    mock_llm.chat_json.side_effect = fake_chat_json

    mock_pipeline = MagicMock()
    mock_pipeline.llm = mock_llm

    def fake_run(query, **kwargs):
        if "chuẩn bị" in query:
            return ([make_candidate("L10_V001", 100, 0.9)], DecomposeResult("step 0"))
        elif "bay qua xà" in query:
            raise RuntimeError("Network failure on step 1 search")
        else:
            return ([make_candidate("L10_V001", 300, 0.85)], DecomposeResult("step 2"))

    mock_pipeline.run.side_effect = fake_run

    plan = TrakePlan(
        original_query="nhảy cao 3 bước",
        action="high jump",
        num_events=3,
        steps=[
            TrakeStep(0, "an athlete takes off", "vận động viên chuẩn bị giậm nhảy"),
            TrakeStep(1, "an athlete flies over bar", "vận động viên bay qua xà"),
            TrakeStep(2, "an athlete lands on mat", "vận động viên tiếp đất an toàn"),
        ],
    )

    results = retrieve_steps(plan, mock_pipeline, cfg)
    # Verify Step 1 failure did NOT crash the other steps
    assert len(results) == 3
    assert len(results[0]) == 1
    assert results[0][0].frame_id == 100
    assert len(results[1]) == 0  # Failed gracefully
    assert len(results[2]) == 1
    assert results[2][0].frame_id == 300


# ==============================================================================
# 5. DEEP TESTS: COMPETITION CSV FORMATTING & SEMANTIC VALIDATION
# ==============================================================================

def test_submission_formatting_special_characters_and_quoting(tmp_path):
    # Test CSV generation with special characters in VQA answers (commas, quotes)
    qa_rows = [
        ("L01_V028", 3450, "5"),
        ("L02_V011", 1200, "Năm người"),
        ("L03_V005", 2800, "Màu đỏ, rất đẹp"),
        ("L04_V012", 4100, 'Anh ấy nói "Tuyệt vời"'),
        ("L05_V099", 5500, "unknown"),
    ]

    csv_file = tmp_path / "query-2-qa.csv"
    write_qa(csv_file, qa_rows)

    # Verify CSV file content conforms to standard
    content = csv_file.read_text(encoding="utf-8").splitlines()
    assert content[0] == "L01_V028,3450,5"
    assert content[1] == "L02_V011,1200,Năm người"
    assert content[2] == 'L03_V005,2800,"Màu đỏ, rất đẹp"'
    assert content[3] == 'L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""'
    assert content[4] == "L05_V099,5500,unknown"

    # Run competition validator
    errors = validate_file(csv_file)
    assert errors == []


def test_vqa_native_language_reasoning_and_ocr_preservation():
    """Verify LLM VQA preserves Vietnamese diacritics and OCR entity strings."""
    from src.tasks.vqa import answer_candidates

    mock_llm = MagicMock()
    # 1. Test OCR text answer preserves Vietnamese exact string
    mock_llm.chat.return_value = "Bến Thành"
    c_ocr = make_candidate("L01_V001", 100, 0.9)
    c_ocr = c_ocr.replace(extra={"ocr_matched": "Cửa hàng Bến Thành"})

    cfg = Config.load("config/config.yaml", no_cache=True)
    answers = answer_candidates(mock_llm, [c_ocr], "Biển hiệu ghi chữ gì?", cfg, use_cache=False)

    assert answers[("L01_V001", 100)] == "Bến Thành"
    called_prompt = mock_llm.chat.call_args[0][1]
    assert "Chữ trên màn hình (OCR): Cửa hàng Bến Thành" in called_prompt
    assert "Câu hỏi: Biển hiệu ghi chữ gì?" in called_prompt

    # 2. Test English question produces English answer
    mock_llm.chat.return_value = "police officer"
    c_en = make_candidate("L01_V002", 200, 0.8)
    c_en = c_en.replace(extra={"description_matched": "A police officer directing traffic"})
    answers_en = answer_candidates(mock_llm, [c_en], "Who is directing traffic?", cfg, use_cache=False)

    assert answers_en[("L01_V002", 200)] == "police officer"

