"""CLI argument handling and the offline subcommands."""

from __future__ import annotations

import pytest

from src.cli import GLOBAL_DEFAULTS, main, parse_args, task_of


@pytest.mark.parametrize("argv", [
    ["--no-cache", "-v", "--dry-run", "validate"],
    ["validate", "--no-cache", "-v", "--dry-run"],
    ["--no-cache", "validate", "-v", "--dry-run"],
])
def test_global_flags_work_on_either_side_of_the_subcommand(argv):
    args = parse_args(argv)
    assert args.no_cache is True
    assert args.verbose is True
    assert args.dry_run is True


@pytest.mark.parametrize("argv,expected", [
    (["--config", "a.yaml", "validate"], "a.yaml"),
    (["validate", "--config", "b.yaml"], "b.yaml"),
    (["validate"], GLOBAL_DEFAULTS["config"]),
])
def test_config_path_resolution(argv, expected):
    assert parse_args(argv).config == expected


def test_defaults_apply_when_no_flag_is_given():
    args = parse_args(["validate"])
    assert args.no_cache is False and args.verbose is False and args.dry_run is False


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_task_detection_from_query_filenames():
    from pathlib import Path

    assert task_of(Path("query-1-kis.txt")) == "kis"
    assert task_of(Path("query-3-qa.txt")) == "qa"
    assert task_of(Path("query-4-trake.txt")) == "trake"
    assert task_of(Path("notes.txt")) is None


# ---------------------------------------------------------------------------
# offline subcommands run for real (no backend needed)
# ---------------------------------------------------------------------------
def test_validate_returns_zero_on_a_good_directory(tmp_path, capsys):
    (tmp_path / "query-1-kis.csv").write_text("L01_V001,10\n", encoding="utf-8")
    assert main(["validate", "--dir", str(tmp_path)]) == 0
    assert "1 file(s) valid" in capsys.readouterr().out


def test_validate_returns_one_on_a_bad_directory(tmp_path, capsys):
    (tmp_path / "query-1-kis.csv").write_text("video_id,frame\nL01_V001,10\n", encoding="utf-8")
    assert main(["validate", "--dir", str(tmp_path)]) == 1
    assert "header" in capsys.readouterr().out


def test_pack_builds_a_zip(tmp_path, capsys):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "query-1-kis.csv").write_text("L01_V001,10\n", encoding="utf-8")
    archive = tmp_path / "teamABC1.zip"

    assert main(["pack", "--dir", str(source), "--out", str(archive)]) == 0
    assert archive.is_file()
    assert "submission/<query>.csv" in capsys.readouterr().out


def test_pack_refuses_an_invalid_submission(tmp_path, capsys):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "query-1-kis.csv").write_text("bad_id,10\n", encoding="utf-8")

    assert main(["pack", "--dir", str(source), "--out", str(tmp_path / "x.zip")]) == 1
    assert "Refusing to package" in capsys.readouterr().err


def test_a_broken_config_path_exits_two(capsys):
    assert main(["--config", "nope.yaml", "validate"]) == 2
    assert "Configuration error" in capsys.readouterr().err
