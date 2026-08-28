"""Every error class the submission validator has to catch."""

from __future__ import annotations

import zipfile

import pytest

from src.schemas import SubmissionError
from src.submission.packer import pack, verify_zip
from src.submission.validator import task_of, validate, validate_file


def write(path, text: str, encoding: str = "utf-8"):
    with path.open("w", encoding=encoding, newline="") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_valid_files_report_nothing(tmp_path):
    write(tmp_path / "query-1-kis.csv", "L01_V001,1500\nL02_V003,42\n")
    write(tmp_path / "query-3-qa.csv", 'L01_V028,3450,5\nL02_V011,1200,"Năm, người"\n')
    write(tmp_path / "query-4-trake.csv", "L10_V001,100,200,300\nL11_V003,50,60,70\n")
    assert validate(tmp_path) == []


def test_task_detection_from_the_filename():
    assert task_of("query-1-kis.csv") == "kis"
    assert task_of("query-3-qa.csv") == "qa"
    assert task_of("query-4-trake.csv") == "trake"
    assert task_of("results.csv") is None


# ---------------------------------------------------------------------------
# error detection
# ---------------------------------------------------------------------------
def test_a_header_row_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "video_id,frame_id\nL01_V001,10\n")
    assert any("header" in e for e in validate_file(path))


def test_more_than_100_rows_is_rejected(tmp_path):
    rows = "".join(f"L01_V001,{i}\n" for i in range(101))
    path = write(tmp_path / "query-1-kis.csv", rows)
    assert any("maximum is 100" in e for e in validate_file(path))


def test_an_empty_file_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "")
    assert any("empty" in e for e in validate_file(path))


def test_the_mp4_suffix_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "L01_V001.mp4,1500\n")
    assert any(".mp4" in e for e in validate_file(path))


def test_a_malformed_video_id_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "video_abc,1500\n")
    assert any("does not match" in e for e in validate_file(path))


def test_a_non_integer_frame_id_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "L01_V001,25 300\n")
    assert any("not an integer" in e for e in validate_file(path))


def test_duplicate_rows_are_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "L01_V001,10\nL01_V001,10\n")
    assert any("duplicate" in e for e in validate_file(path))


def test_a_bom_is_rejected(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "L01_V001,10\n", encoding="utf-8-sig")
    assert any("BOM" in e for e in validate_file(path))


def test_non_utf8_bytes_are_rejected(tmp_path):
    path = tmp_path / "query-1-kis.csv"
    path.write_bytes("L01_V001,10,Màu đỏ\n".encode("latin-1", errors="replace"))
    assert any("UTF-8" in e for e in validate_file(path))


def test_an_excel_file_is_rejected(tmp_path):
    path = tmp_path / "query-1-kis.csv"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sheet.xml", "<x/>")
    assert any("Excel" in e for e in validate_file(path))


# -- Q&A specific -----------------------------------------------------------
def test_qa_needs_exactly_three_columns(tmp_path):
    path = write(tmp_path / "query-3-qa.csv", "L01_V001,10\n")
    assert any("exactly 3 columns" in e for e in validate_file(path))


def test_an_empty_qa_answer_is_rejected(tmp_path):
    path = write(tmp_path / "query-3-qa.csv", "L01_V001,10,\n")
    assert any("answer is empty" in e for e in validate_file(path))


def test_an_over_long_qa_answer_is_rejected(tmp_path):
    path = write(tmp_path / "query-3-qa.csv", f"L01_V001,10,{'x' * 101}\n")
    assert any("maximum is 100" in e for e in validate_file(path))


# -- TRAKE specific ---------------------------------------------------------
def test_trake_rows_must_all_have_the_same_width(tmp_path):
    path = write(tmp_path / "query-4-trake.csv", "L10_V001,1,2,3\nL10_V001,4,5\n")
    assert any("differing frame counts" in e for e in validate_file(path))


def test_trake_rows_must_be_chronological(tmp_path):
    path = write(tmp_path / "query-4-trake.csv", "L10_V001,300,200,100\n")
    assert any("chronological" in e for e in validate_file(path))


def test_trake_width_is_checked_against_the_plan(tmp_path):
    path = write(tmp_path / "query-4-trake.csv", "L10_V001,1,2,3\n")
    errors = validate_file(path, {"query-4-trake": 4})
    assert any("num_events=4" in e for e in errors)


def test_kis_rows_must_have_exactly_two_columns(tmp_path):
    path = write(tmp_path / "query-1-kis.csv", "L01_V001,10,20\n")
    assert any("exactly 2 columns" in e for e in validate_file(path))


# -- directory level --------------------------------------------------------
def test_a_missing_directory_is_reported(tmp_path):
    assert validate(tmp_path / "nope") != []


def test_a_directory_without_csvs_is_reported(tmp_path):
    assert any("no .csv" in e for e in validate(tmp_path))


def test_stray_non_csv_files_are_reported(tmp_path):
    write(tmp_path / "query-1-kis.csv", "L01_V001,10\n")
    write(tmp_path / "notes.txt", "hello")
    assert any("unexpected non-CSV" in e for e in validate(tmp_path))


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------
def test_pack_builds_a_submission_folder_inside_the_zip(tmp_path):
    source = tmp_path / "submission"
    source.mkdir()
    write(source / "query-1-kis.csv", "L01_V001,10\n")
    write(source / "query-3-qa.csv", "L01_V001,10,5\n")

    archive = pack(source, tmp_path / "teamABC.zip")

    with zipfile.ZipFile(archive) as zf:
        names = sorted(zf.namelist())
    assert names == ["submission/query-1-kis.csv", "submission/query-3-qa.csv"]
    assert verify_zip(archive) == []


def test_pack_refuses_an_invalid_submission(tmp_path):
    source = tmp_path / "submission"
    source.mkdir()
    write(source / "query-1-kis.csv", "video_id,frame_id\nL01_V001,10\n")

    with pytest.raises(SubmissionError, match="Refusing to package"):
        pack(source, tmp_path / "out.zip")


def test_verify_zip_catches_bare_csvs_at_the_root(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("query-1-kis.csv", "L01_V001,10\n")
    assert any("not inside the submission/ folder" in e for e in verify_zip(archive))
