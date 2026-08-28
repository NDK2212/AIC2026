"""CSV writing: encoding, quoting, row caps and TRAKE width assertions."""

from __future__ import annotations

import csv

import pytest

from src.schemas import SubmissionError
from src.submission.writer import write_kis, write_qa, write_trake


def read(path):
    return path.read_text(encoding="utf-8")


def test_kis_has_no_header_and_no_mp4_suffix(tmp_path):
    out = tmp_path / "query-1-kis.csv"
    write_kis(out, [("L01_V001.mp4", 1500), ("L02_V003", 42)])

    text = read(out)
    assert text == "L01_V001,1500\nL02_V003,42\n"
    assert "video_id" not in text


def test_kis_caps_at_max_rows(tmp_path):
    out = tmp_path / "query-1-kis.csv"
    written = write_kis(out, [("L01_V001", i) for i in range(250)], max_rows=100)

    assert written == 100
    assert len(read(out).strip().splitlines()) == 100


def test_answers_with_commas_and_quotes_are_escaped(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [
        ("L01_V028", 3450, "5"),
        ("L01_V028", 3451, "Có 3 người, bao gồm nam và nữ"),
        ("L01_V028", 3452, 'Anh ấy nói "Xin chào"'),
    ])

    text = read(out)
    assert "L01_V028,3450,5\n" in text
    assert '"Có 3 người, bao gồm nam và nữ"' in text
    assert '"Anh ấy nói ""Xin chào"""' in text

    # And it round-trips through a standard CSV reader.
    rows = list(csv.reader(text.splitlines()))
    assert rows[1][2] == "Có 3 người, bao gồm nam và nữ"
    assert rows[2][2] == 'Anh ấy nói "Xin chào"'


def test_simple_answers_are_not_quoted(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [("L01_V001", 10, "Năm người")])
    assert read(out) == "L01_V001,10,Năm người\n"


def test_empty_answers_become_the_fallback(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [("L01_V001", 10, ""), ("L01_V001", 20, "   ")],
             fallback_answer="không rõ")

    rows = list(csv.reader(read(out).splitlines()))
    assert all(row[2].strip() for row in rows)
    assert rows[0][2] == "không rõ"


def test_answers_are_truncated_to_the_limit(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [("L01_V001", 10, "x" * 250)], answer_max_chars=100)

    rows = list(csv.reader(read(out).splitlines()))
    assert len(rows[0][2]) == 100


def test_answers_never_contain_a_newline(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [("L01_V001", 10, "dòng 1\ndòng 2")])

    assert len(read(out).strip().splitlines()) == 1


def test_output_is_utf8_without_a_bom(tmp_path):
    out = tmp_path / "query-3-qa.csv"
    write_qa(out, [("L01_V001", 10, "Màu đỏ")])

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8").startswith("L01_V001,10,Màu đỏ")
    assert b"\r\n" not in raw


def test_trake_writes_one_row_per_sequence(tmp_path):
    out = tmp_path / "query-4-trake.csv"
    write_trake(out, [["L10_V001", 1200, 1850, 2100, 2450]], num_events=4)
    assert read(out) == "L10_V001,1200,1850,2100,2450\n"


def test_trake_rejects_the_wrong_number_of_frames(tmp_path):
    out = tmp_path / "query-4-trake.csv"
    with pytest.raises(SubmissionError, match="3 frames but the query asks for 4"):
        write_trake(out, [["L10_V001", 1, 2, 3]], num_events=4)


def test_trake_rejects_non_increasing_frames(tmp_path):
    out = tmp_path / "query-4-trake.csv"
    with pytest.raises(SubmissionError, match="not strictly increasing"):
        write_trake(out, [["L10_V001", 100, 90, 300]], num_events=3)


def test_trake_caps_at_max_rows(tmp_path):
    out = tmp_path / "query-4-trake.csv"
    rows = [["L10_V001", i, i + 1, i + 2] for i in range(1, 300)]
    assert write_trake(out, rows, num_events=3, max_rows=100) == 100
