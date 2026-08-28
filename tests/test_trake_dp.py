"""The anchored bidirectional DP: generality in N, constraints, and step filling."""

from __future__ import annotations

import pytest

from src.schemas import Candidate
from src.tasks.trake import (
    _clamp_anchor,
    _enforce_monotonic,
    _events_stated_in_query,
    _fit_steps,
    align_video,
    normalize_step_scores,
    rank_sequences,
    rank_videos,
)
from src.schemas import TrakeSequence, TrakeStep


def steps_to_tables(video: str, per_step: list[list[tuple[int, float]]]):
    """Build the normalised score tables the DP consumes."""
    return [
        {(video, frame): score for frame, score in step}
        for step in per_step
    ]


def merge(*tables_lists):
    """Merge several per-video table lists into one list of tables."""
    length = len(tables_lists[0])
    out = []
    for i in range(length):
        merged: dict = {}
        for tables in tables_lists:
            merged.update(tables[i])
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# generality in N
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_dp_handles_any_number_of_steps(n, trake_cfg, fake_kf):
    # Step j has its best candidate at frame 100*j, so the optimum is obvious.
    per_step = [
        [(100 * j, 1.0), (100 * j + 500, 0.2), (max(0, 100 * j - 90), 0.1)]
        for j in range(n)
    ]
    tables = steps_to_tables("L01_V001", per_step)

    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1)

    assert len(sequences) == 1
    sequence = sequences[0]
    assert len(sequence.frame_ids) == n
    assert sequence.frame_ids == [100 * j for j in range(n)]
    assert sequence.filled_steps == []


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8])
def test_output_always_has_exactly_n_frames(n, trake_cfg, fake_kf):
    per_step = [[(10 * j + 5, 0.5)] for j in range(n)]
    tables = steps_to_tables("L01_V001", per_step)
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=3)
    assert sequences
    for sequence in sequences:
        assert len(sequence.frame_ids) == n
        assert len(sequence.per_step_score) == n


# ---------------------------------------------------------------------------
# temporal constraints
# ---------------------------------------------------------------------------
def test_chronological_order_is_never_violated(trake_cfg, fake_kf):
    # The highest-scoring frames are in the *wrong* temporal order, so the DP
    # has to give some of them up to keep the sequence increasing.
    tables = steps_to_tables("L01_V001", [
        [(900, 1.0), (100, 0.6)],
        [(500, 1.0), (400, 0.9)],
        [(50, 1.0), (800, 0.5)],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=5)

    assert sequences
    for sequence in sequences:
        frames = sequence.frame_ids
        assert all(b > a for a, b in zip(frames, frames[1:])), frames


def test_min_gap_is_respected(trake_cfg, fake_kf):
    cfg = trake_cfg
    cfg.min_gap = 50
    cfg.allow_fill = False
    tables = steps_to_tables("L01_V001", [
        [(100, 1.0)],
        [(120, 1.0), (200, 0.4)],    # 120 is only 20 after 100 -> illegal
        [(400, 1.0)],
    ])
    sequences = align_video("L01_V001", tables, cfg, fake_kf, max_paths=1)

    assert sequences
    frames = sequences[0].frame_ids
    assert frames == [100, 200, 400]
    assert all(b - a >= 50 for a, b in zip(frames, frames[1:]))


def test_max_gap_is_respected(trake_cfg, fake_kf):
    cfg = trake_cfg
    cfg.max_gap = 100
    cfg.allow_fill = False
    tables = steps_to_tables("L01_V001", [
        [(100, 1.0)],
        [(900, 1.0), (150, 0.3)],    # 900 is 800 after 100 -> beyond max_gap
        [(200, 1.0)],
    ])
    sequences = align_video("L01_V001", tables, cfg, fake_kf, max_paths=1)

    assert sequences
    frames = sequences[0].frame_ids
    assert frames == [100, 150, 200]
    assert all(0 < b - a <= 100 for a, b in zip(frames, frames[1:]))


def test_a_higher_scoring_but_illegal_chain_loses_to_a_legal_one(trake_cfg, fake_kf):
    trake_cfg.allow_fill = False
    tables = steps_to_tables("L01_V001", [
        [(800, 1.0), (100, 0.7)],
        [(300, 1.0)],
        [(500, 1.0)],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1)
    # Taking frame 800 first would make steps 2 and 3 impossible.
    assert sequences[0].frame_ids == [100, 300, 500]


# ---------------------------------------------------------------------------
# missing steps
# ---------------------------------------------------------------------------
def test_missing_step_is_interpolated_not_dropped(trake_cfg, fake_kf):
    tables = steps_to_tables("L01_V001", [
        [(100, 1.0)],
        [],                # nothing retrieved for the middle moment
        [(300, 1.0)],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1)

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.filled_steps == [1]
    assert len(sequence.frame_ids) == 3
    assert sequence.frame_ids[0] == 100 and sequence.frame_ids[2] == 300
    # Interpolated to ~200 and snapped onto an existing keyframe.
    assert sequence.frame_ids[1] == 200
    assert sequence.frame_ids[1] in fake_kf.all_frames("L01_V001")


def test_leading_and_trailing_missing_steps_are_extrapolated(trake_cfg, fake_kf):
    tables = steps_to_tables("L01_V001", [
        [],
        [(200, 1.0)],
        [(300, 1.0)],
        [],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1)

    sequence = sequences[0]
    assert sequence.filled_steps == [0, 3]
    frames = sequence.frame_ids
    assert len(frames) == 4
    assert all(b > a for a, b in zip(frames, frames[1:])), frames
    assert frames[1] == 200 and frames[2] == 300


def test_filling_is_disabled_when_allow_fill_is_false(trake_cfg, fake_kf):
    trake_cfg.allow_fill = False
    tables = steps_to_tables("L01_V001", [
        [(100, 1.0)],
        [],
        [(300, 1.0)],
    ])
    assert align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1) == []


def test_missing_step_costs_the_miss_penalty(trake_cfg, fake_kf):
    complete = steps_to_tables("L01_V001", [[(100, 1.0)], [(200, 1.0)], [(300, 1.0)]])
    partial = steps_to_tables("L01_V001", [[(100, 1.0)], [], [(300, 1.0)]])

    full = align_video("L01_V001", complete, trake_cfg, fake_kf, max_paths=1)[0]
    gapped = align_video("L01_V001", partial, trake_cfg, fake_kf, max_paths=1)[0]

    assert full.total_score == pytest.approx(3.0)
    assert gapped.total_score == pytest.approx(2.0 + trake_cfg.miss_penalty)
    assert gapped.total_score < full.total_score


def test_a_video_missing_a_step_can_still_beat_a_weak_complete_video(trake_cfg, fake_kf):
    strong = steps_to_tables("L01_V001", [[(100, 1.0)], [], [(300, 1.0)]])
    weak = steps_to_tables("L01_V002", [[(100, 0.2)], [(200, 0.2)], [(300, 0.2)]])
    tables = merge(strong, weak)

    sequences = (
        align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=1)
        + align_video("L01_V002", tables, trake_cfg, fake_kf, max_paths=1)
    )
    ranked = rank_sequences(sequences, trake_cfg, max_rows=10)
    assert ranked[0].video_id == "L01_V001"


# ---------------------------------------------------------------------------
# anchors and multiple paths
# ---------------------------------------------------------------------------
def test_multiple_paths_per_video_are_distinct(trake_cfg, fake_kf):
    tables = steps_to_tables("L01_V001", [
        [(100, 0.9), (110, 0.8)],
        [(200, 0.9), (210, 0.85), (220, 0.8)],
        [(300, 0.9), (310, 0.8)],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf, max_paths=3)

    assert len(sequences) == 3
    markers = {tuple(s.frame_ids) for s in sequences}
    assert len(markers) == 3
    # Distinct paths should differ at the anchor step (index 1 by default).
    assert len({s.frame_ids[1] for s in sequences}) == 3


def test_anchor_index_is_clamped():
    assert _clamp_anchor(1, 4) == 1
    assert _clamp_anchor(9, 3) == 1        # out of range -> the second step
    assert _clamp_anchor(1, 1) == 0        # N == 1 -> the only step
    assert _clamp_anchor(-1, 5) == 1


def test_an_empty_anchor_step_falls_back_to_a_populated_one(trake_cfg, fake_kf):
    tables = steps_to_tables("L01_V001", [
        [(100, 1.0)],
        [],                # the configured anchor has nothing here
        [(300, 1.0)],
    ])
    sequences = align_video("L01_V001", tables, trake_cfg, fake_kf,
                            max_paths=1, anchor_index=1)
    assert sequences and len(sequences[0].frame_ids) == 3


def test_a_video_with_no_candidates_at_all_yields_nothing(trake_cfg, fake_kf):
    assert align_video("L09_V009", [{}, {}, {}], trake_cfg, fake_kf) == []


# ---------------------------------------------------------------------------
# score normalisation, video ranking, head diversity
# ---------------------------------------------------------------------------
def test_step_scores_are_normalised_per_step():
    step_candidates = [
        [Candidate("L01_V001", 10, 100.0), Candidate("L01_V001", 20, 50.0)],
        [Candidate("L01_V001", 30, 0.02), Candidate("L01_V001", 40, 0.01)],
    ]
    tables = normalize_step_scores(step_candidates)
    assert tables[0][("L01_V001", 10)] == pytest.approx(1.0)
    assert tables[0][("L01_V001", 20)] == pytest.approx(0.0)
    # A tiny raw scale must not make a step irrelevant to the DP.
    assert tables[1][("L01_V001", 30)] == pytest.approx(1.0)


def test_video_ranking_rewards_coverage(trake_cfg):
    tables = [
        {("L01_V001", 10): 1.0, ("L01_V002", 10): 1.0},
        {("L01_V002", 20): 0.5},                        # only V002 covers step 2
    ]
    ranked = rank_videos(tables, trake_cfg)
    assert ranked[0] == "L01_V002"


def test_head_diversity_spreads_the_first_rows_over_videos(trake_cfg):
    # Five strong sequences from one video, one weak from each of two others.
    sequences = [TrakeSequence("L01_V001", [i, i + 10, i + 20], 10.0 - i * 0.01)
                 for i in range(5)]
    sequences.append(TrakeSequence("L01_V002", [1, 2, 3], 1.0))
    sequences.append(TrakeSequence("L02_V003", [1, 2, 3], 0.5))

    ranked = rank_sequences(sequences, trake_cfg, max_rows=100)
    head_videos = {s.video_id for s in ranked[:5]}
    assert len(head_videos) >= trake_cfg.head_min_videos
    assert ranked[0].video_id == "L01_V001"     # the best row is still first
    assert len(ranked) == len(sequences)        # nothing is lost


def test_head_diversity_copes_when_only_one_video_exists(trake_cfg):
    sequences = [TrakeSequence("L01_V001", [i, i + 1, i + 2], 1.0 - i * 0.1)
                 for i in range(4)]
    ranked = rank_sequences(sequences, trake_cfg, max_rows=100)
    assert len(ranked) == 4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def test_enforce_monotonic_fixes_order():
    assert _enforce_monotonic([100, 90, 95], 1) == [100, 101, 102]
    assert _enforce_monotonic([10, 20, 30], 1) == [10, 20, 30]
    assert _enforce_monotonic([10, 10], 5) == [10, 15]


@pytest.mark.parametrize("query,expected", [
    ("Tìm 4 khoảnh khắc chính khi vận động viên nhảy cao", 4),
    ("Find the three key moments of the serve", 3),
    ("Chuỗi gồm (1) chạy đà (2) giậm nhảy (3) tiếp đất", 3),
    ("Tìm cảnh một người mở laptop", None),
])
def test_event_count_is_read_from_the_query(query, expected):
    assert _events_stated_in_query(query) == expected


def test_fit_steps_pads_and_trims():
    steps = [TrakeStep(0, "a"), TrakeStep(1, "b")]
    assert len(_fit_steps(steps, 4)) == 4
    assert [s.index for s in _fit_steps(steps, 4)] == [0, 1, 2, 3]
    assert len(_fit_steps(steps, 1)) == 1
    assert _fit_steps(steps, 2) is steps


def test_plan_trake_heuristic_fallback_when_llm_fails():
    """Verify plan_trake does not raise and builds a valid N-step plan when LLM is down."""
    from src.config import Config
    from src.tasks.trake import plan_trake

    cfg = Config.load("config/config.yaml", no_cache=True)

    class FailingLLM:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("API Connection timeout")

    plan = plan_trake(FailingLLM(), "Vận động viên cử tạ thực hiện 3 bước", cfg)
    assert plan.num_events == 3
    assert len(plan.steps) == 3
    assert all(s.description == "Vận động viên cử tạ thực hiện 3 bước" for s in plan.steps)


def test_retrieve_steps_routes_to_description_and_visual_only():
    """Verify retrieve_steps directly constructs DecomposeResult with image and description only."""
    from unittest.mock import MagicMock
    from src.config import Config
    from src.schemas import TrakePlan, TrakeStep
    from src.tasks.trake import retrieve_steps

    cfg = Config.load("config/config.yaml", no_cache=True)
    plan = TrakePlan(
        original_query="Nhảy cao 2 bước",
        num_events=2,
        steps=[
            TrakeStep(0, "athlete running on track", "chạy đà"),
            TrakeStep(1, "athlete jumping over bar", "dậm nhảy"),
        ],
        anchor_index=0,
        action="high jump",
    )

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ([Candidate("L01_V001", 100, 1.0)], None)

    results = retrieve_steps(plan, mock_pipeline, cfg, use_cache=False)

    assert len(results) == 2
    assert mock_pipeline.run.call_count == 2

    # Verify Step 0 decompose result
    call_0_decompose = mock_pipeline.run.call_args_list[0][1]["decompose_result"]
    assert call_0_decompose.ocr_query is None
    assert call_0_decompose.asr_query is None
    assert call_0_decompose.description_query == "chạy đà"
    assert call_0_decompose.image_query == "athlete running on track"
    assert call_0_decompose.modality_weights == {"image": 0.6, "description": 0.4}


