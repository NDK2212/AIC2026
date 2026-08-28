"""Submission row ordering: diversity head, dense body, neighbour expansion."""

from __future__ import annotations

from src.submission.builder import build_rows, head_video_diversity, same_shot
from tests.conftest import make_candidate


def test_diversity_head_has_no_two_frames_from_one_shot(submission_cfg, fake_kf):
    # Ten near-identical frames of one shot, then genuinely different scenes.
    candidates = [make_candidate("L01_V001", 100 + i, 10.0 - i * 0.01) for i in range(10)]
    candidates += [make_candidate("L01_V001", 500, 5.0),
                   make_candidate("L01_V002", 100, 4.9),
                   make_candidate("L01_V002", 800, 4.8),
                   make_candidate("L02_V003", 10, 4.7),
                   make_candidate("L02_V003", 400, 4.6),
                   make_candidate("L02_V003", 900, 4.5),
                   make_candidate("L01_V001", 900, 4.4)]

    rows = build_rows(candidates, submission_cfg, fake_kf)
    head = rows[: submission_cfg.top_diverse]

    for i, a in enumerate(head):
        for b in head[i + 1:]:
            assert not same_shot(a, b, submission_cfg.shot_window), (a, b)


def test_the_best_candidate_stays_at_rank_one(submission_cfg, fake_kf):
    candidates = [make_candidate("L01_V001", 100, 10.0),
                  make_candidate("L01_V001", 105, 9.9),
                  make_candidate("L01_V002", 300, 9.8)]
    rows = build_rows(candidates, submission_cfg, fake_kf)
    assert rows[0].key == ("L01_V001", 100)


def test_the_head_still_fills_when_shots_run_out(submission_cfg, fake_kf):
    # Everything is one shot, so the head cannot be diversified.
    candidates = [make_candidate("L01_V001", 100 + i, 10.0 - i) for i in range(12)]
    rows = build_rows(candidates, submission_cfg, fake_kf)

    assert len(rows) >= submission_cfg.top_diverse
    assert rows[0].key == ("L01_V001", 100)


def test_no_duplicate_rows(submission_cfg, fake_kf):
    candidates = [make_candidate("L01_V001", 100, 10.0),
                  make_candidate("L01_V001", 100, 9.0),      # exact duplicate key
                  make_candidate("L01_V002", 200, 8.0)]
    rows = build_rows(candidates, submission_cfg, fake_kf)

    keys = [c.key for c in rows]
    assert len(keys) == len(set(keys))


def test_ranks_are_contiguous(submission_cfg, fake_kf):
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i) for i in range(9)]
    rows = build_rows(candidates, submission_cfg, fake_kf)
    assert [c.rank for c in rows] == list(range(1, len(rows) + 1))


def test_neighbour_expansion_fills_the_tail(submission_cfg, fake_kf):
    # Only 20 real candidates, but 100 rows are always worth submitting.
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i * 0.1) for i in range(10)]
    candidates += [make_candidate("L01_V002", i * 100, 5.0 - i * 0.1) for i in range(10)]

    rows = build_rows(candidates, submission_cfg, fake_kf)

    assert len(rows) > 20
    expanded = [c for c in rows if c.source == "expanded"]
    assert expanded
    for candidate in expanded:
        assert candidate.frame_id in fake_kf.all_frames(candidate.video_id)


def test_expansion_never_displaces_a_real_candidate_from_the_head(submission_cfg, fake_kf):
    # Plenty of real candidates: the whole protected prefix must stay real.
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i * 0.1) for i in range(5)]
    candidates += [make_candidate("L01_V002", i * 100, 5.0 - i * 0.1) for i in range(30)]

    rows = build_rows(candidates, submission_cfg, fake_kf)

    head = rows[: submission_cfg.neighbor_expansion.start_rank]
    assert len(head) == submission_cfg.neighbor_expansion.start_rank
    assert all(c.source != "expanded" for c in head)


def test_expansion_starts_early_when_real_candidates_run_out(submission_cfg, fake_kf):
    # Only 10 real candidates but 100 rows are worth submitting, so expansion
    # begins right after them rather than leaving ranks 11-14 empty.
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i * 0.1) for i in range(10)]
    rows = build_rows(candidates, submission_cfg, fake_kf)

    real = [c for c in rows if c.source != "expanded"]
    assert len(real) == 10
    # Every real candidate still outranks every expanded one.
    assert all(c.source != "expanded" for c in rows[:10])
    assert rows[10].source == "expanded"
    # The diversity head is untouched either way.
    assert all(c.source != "expanded" for c in rows[: submission_cfg.top_diverse])


def test_expansion_can_be_switched_off(submission_cfg, fake_kf):
    submission_cfg.neighbor_expansion.enabled = False
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i) for i in range(5)]

    rows = build_rows(candidates, submission_cfg, fake_kf)
    assert len(rows) == 5
    assert all(c.source != "expanded" for c in rows)


def test_never_more_than_max_rows(submission_cfg, fake_kf):
    candidates = [make_candidate("L01_V001", i * 10, 10.0 - i * 0.001) for i in range(400)]
    rows = build_rows(candidates, submission_cfg, fake_kf)
    assert len(rows) == submission_cfg.max_rows


def test_no_candidates_produces_no_rows(submission_cfg, fake_kf):
    assert build_rows([], submission_cfg, fake_kf) == []


def test_works_without_a_keyframe_index(submission_cfg):
    candidates = [make_candidate("L01_V001", i * 100, 10.0 - i) for i in range(5)]
    rows = build_rows(candidates, submission_cfg, None)
    assert len(rows) == 5


def test_same_shot_needs_the_same_video():
    a = make_candidate("L01_V001", 100, 1.0)
    b = make_candidate("L01_V002", 101, 1.0)
    c = make_candidate("L01_V001", 101, 1.0)
    assert not same_shot(a, b, 60)
    assert same_shot(a, c, 60)


def test_head_video_diversity_counts_distinct_videos():
    rows = [make_candidate("L01_V001", 1, 1.0),
            make_candidate("L01_V001", 2, 1.0),
            make_candidate("L01_V002", 3, 1.0)]
    assert head_video_diversity(rows, 5) == 2


def test_the_head_is_capped_per_video(submission_cfg, fake_kf):
    # One video dominates the ranking with ten separate, high-scoring shots.
    candidates = [make_candidate("L01_V001", i * 200, 10.0 - i * 0.01) for i in range(10)]
    candidates += [make_candidate("L01_V002", i * 200, 5.0 - i * 0.01) for i in range(5)]
    candidates += [make_candidate("L02_V003", i * 200, 4.0 - i * 0.01) for i in range(5)]

    rows = build_rows(candidates, submission_cfg, fake_kf)
    head = rows[: submission_cfg.top_diverse]

    counts: dict[str, int] = {}
    for candidate in head:
        counts[candidate.video_id] = counts.get(candidate.video_id, 0) + 1
    assert max(counts.values()) <= submission_cfg.head_max_per_video
    assert len(counts) >= 3
    # The single best candidate is still row one.
    assert rows[0].key == ("L01_V001", 0)


def test_the_head_cap_relaxes_when_only_one_video_exists(submission_cfg, fake_kf):
    candidates = [make_candidate("L01_V001", i * 200, 10.0 - i) for i in range(10)]
    rows = build_rows(candidates, submission_cfg, fake_kf)

    # Nothing else to diversify with, so the head must still fill up.
    assert len(rows[: submission_cfg.top_diverse]) == submission_cfg.top_diverse
