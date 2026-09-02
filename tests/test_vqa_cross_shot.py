"""VQA evidence can live far away from the action shot in the same video."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from src.config import Config
from src.schemas import Candidate, VQASplit
from src.tasks.vqa import (
    answer_candidates,
    answer_candidates_vlm,
    propagate_answers,
    retrieve_cross_shot_evidence,
    split_query,
)


def _candidate(video_id: str, frame_id: int, score: float = 1.0) -> Candidate:
    return Candidate(video_id=video_id, frame_id=frame_id, score=score)


def test_split_query_requests_a_separate_answer_evidence_scene():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "scene_description": "đầu bếp nhét tiêu xanh lá chanh sả vào bốn con cá",
        "question": "Đây là loài cá gì?",
        "question_en": "What species of fish is this?",
        "expected_answer_type": "name",
        "evidence_query": "cận cảnh cá nguyên con, tiêu đề món ăn hoặc lời giới thiệu tên cá",
    }

    split = split_query(llm, "Đầu bếp nhét gia vị vào bốn con cá. Đây là cá gì?")

    assert split.evidence_query.startswith("cận cảnh cá nguyên con")
    assert "evidence_query" in llm.chat_json.call_args.kwargs["schema_hint"]


def test_split_query_builds_evidence_fallback_when_model_omits_it():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "scene_description": "đầu bếp nhét gia vị vào bốn con cá",
        "question": "Đây là loài cá gì?",
        "expected_answer_type": "name",
    }

    split = split_query(llm, "Đầu bếp nhét gia vị vào cá. Đây là cá gì?")

    assert "cùng video" in split.evidence_query
    assert "Đây là loài cá gì" in split.evidence_query


def test_cross_shot_retrieval_is_scoped_to_scene_matched_videos():
    cfg = Config.load("config/config.yaml", no_cache=True)
    cfg.vqa.evidence_video_top_n = 2
    cfg.vqa.evidence_top_n = 4
    pipeline = MagicMock()
    pipeline.run.return_value = (
        [
            _candidate("L01_V001", 900, 0.9),
            _candidate("WRONG_VIDEO", 100, 0.8),
            _candidate("L01_V002", 700, 0.7),
        ],
        None,
    )
    split = VQASplit(
        scene_description="nhét gia vị vào cá",
        question="Đây là cá gì?",
        evidence_query="cận cảnh cá nguyên con và tên món ăn",
    )
    action_rows = [_candidate("L01_V001", 100), _candidate("L01_V002", 200)]

    evidence = retrieve_cross_shot_evidence(pipeline, split, action_rows, cfg)

    assert {candidate.video_id for candidate in evidence} == {"L01_V001", "L01_V002"}
    assert pipeline.run.call_args.kwargs["video_ids"] == ["L01_V001", "L01_V002"]


def test_answerer_combines_distant_shots_from_the_same_video():
    cfg = Config.load("config/config.yaml", no_cache=True)
    llm = MagicMock()
    llm.chat.return_value = "cá chẽm"
    action = Candidate(
        video_id="L01_V001",
        frame_id=100,
        score=1.0,
        extra={"description_matched": "đầu bếp nhét gia vị vào bụng bốn con cá"},
    )
    identity = Candidate(
        video_id="L01_V001",
        frame_id=1900,
        score=0.9,
        extra={"description_matched": "cận cảnh cá chẽm nguyên con trên thớt"},
    )

    answers = answer_candidates(llm, [action, identity], "Đây là loài cá gì?", cfg)

    assert answers[action.key] == answers[identity.key] == "cá chẽm"
    llm.chat.assert_called_once()
    prompt = llm.chat.call_args.args[1]
    assert "Khung hình 100" in prompt
    assert "Khung hình 1900" in prompt
    assert "cận cảnh cá chẽm" in prompt


def test_evidence_answer_propagates_only_inside_its_video():
    cfg = Config.load("config/config.yaml", no_cache=True)
    rows = [_candidate("L01_V001", 100), _candidate("L01_V002", 100)]
    evidence_answers = {
        ("L01_V001", 1900): "cá chẽm",
        ("L01_V002", 1700): "cá thu",
    }

    output = propagate_answers(rows, evidence_answers, cfg)

    assert output == [
        ("L01_V001", 100, "cá chẽm"),
        ("L01_V002", 100, "cá thu"),
    ]


def test_vlm_receives_multiple_images_and_text_metadata_per_video(tmp_path):
    cfg = Config.load("config/config.yaml", no_cache=True)
    cfg.vqa.vlm_images_per_video = 4
    action_path = tmp_path / "action.jpg"
    evidence_path = tmp_path / "evidence.jpg"
    Image.new("RGB", (16, 16), "white").save(action_path)
    Image.new("RGB", (16, 16), "silver").save(evidence_path)

    action = Candidate(
        video_id="L01_V001",
        frame_id=100,
        score=1.0,
        extra={"description_matched": "đầu bếp nhét gia vị vào bốn con cá"},
    )
    evidence = Candidate(
        video_id="L01_V001",
        frame_id=1900,
        score=0.9,
        extra={"ocr_matched": "Cá chẽm nướng sả"},
    )
    paths = {action.key: action_path, evidence.key: evidence_path}
    keyframes = MagicMock()
    keyframes.path_of.side_effect = lambda video_id, frame_id: paths[(video_id, frame_id)]
    vlm = MagicMock()
    vlm.cfg = SimpleNamespace(model="moonshotai/kimi-k3")
    vlm.ask_many.return_value = "cá chẽm"

    answers = answer_candidates_vlm(
        vlm,
        [action, evidence],
        "Đây là loài cá gì?",
        cfg,
        keyframes,
        use_cache=False,
    )

    assert answers == {action.key: "cá chẽm", evidence.key: "cá chẽm"}
    image_paths, _, prompt = vlm.ask_many.call_args.args
    assert image_paths == [action_path, evidence_path]
    assert "Đây là loài cá gì?" in prompt
    contexts = vlm.ask_many.call_args.kwargs["contexts"]
    assert "Description: đầu bếp nhét gia vị" in contexts[0]
    assert "OCR: Cá chẽm nướng sả" in contexts[1]
