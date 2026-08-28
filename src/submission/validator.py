"""Pre-flight validation of a submission directory.

``cli pack`` refuses to build the zip when this reports anything, because a
malformed file still burns one of the three allowed submissions.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from ..logging_utils import get_logger

log = get_logger(__name__)

VIDEO_ID_RE = re.compile(r"^L\d+_V\d+$")
MAX_ROWS = 100
MAX_ANSWER_CHARS = 100

_TASK_SUFFIXES = ("kis", "qa", "trake")


def task_of(filename: str) -> str | None:
    """Infer the task from a query/result filename suffix."""
    stem = Path(filename).stem.lower()
    for suffix in _TASK_SUFFIXES:
        if stem.endswith(f"-{suffix}") or stem.endswith(f"_{suffix}"):
            return suffix
    return None


def load_expected_events(runs_dir: Path | str | None) -> dict[str, int]:
    """Read ``num_events`` per TRAKE query from the run traces.

    Returns ``{query_stem: num_events}``.  Missing traces simply mean the
    per-row event count is only checked for internal consistency.
    """
    out: dict[str, int] = {}
    if not runs_dir:
        return out
    directory = Path(runs_dir)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*_plan.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                plan = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        stem = plan.get("query_stem")
        events = plan.get("num_events")
        if stem and isinstance(events, int) and events > 0:
            out[str(stem)] = events
    return out


def validate_file(path: Path, expected_events: dict[str, int] | None = None) -> list[str]:
    """Validate one CSV file; returns a list of human-readable errors."""
    errors: list[str] = []
    name = path.name
    expected_events = expected_events or {}

    if path.suffix.lower() != ".csv":
        errors.append(f"{name}: not a .csv file")
        return errors

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{name}: cannot read file ({exc})"]

    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"{name}: file starts with a UTF-8 BOM")
    if raw.startswith(b"PK\x03\x04"):
        errors.append(f"{name}: this is an Excel/zip file, not a plain CSV")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return errors + [f"{name}: not valid UTF-8 ({exc})"]

    rows = [row for row in csv.reader(text.splitlines()) if row and any(f.strip() for f in row)]
    if not rows:
        return errors + [f"{name}: file is empty"]
    if len(rows) > MAX_ROWS:
        errors.append(f"{name}: {len(rows)} rows, maximum is {MAX_ROWS}")

    first_cell = rows[0][0].strip().lower()
    if first_cell in {"video_id", "video", "video_name", "tên file video"}:
        errors.append(f"{name}: first row looks like a header - headers are not allowed")

    task = task_of(name)
    seen: set[tuple[str, ...]] = set()
    trake_widths: set[int] = set()

    for i, row in enumerate(rows, start=1):
        cells = [c.strip() for c in row]
        marker = tuple(cells)
        if marker in seen:
            errors.append(f"{name}:{i}: duplicate row {cells}")
        seen.add(marker)

        video_id = cells[0]
        if not VIDEO_ID_RE.match(video_id):
            if video_id.endswith(".mp4"):
                errors.append(f"{name}:{i}: video_id must not carry .mp4 ({video_id!r})")
            else:
                errors.append(f"{name}:{i}: video_id {video_id!r} does not match ^L\\d+_V\\d+$")

        if len(cells) < 2:
            errors.append(f"{name}:{i}: needs at least a video_id and one frame_id")
            continue

        if task == "qa":
            if len(cells) != 3:
                errors.append(f"{name}:{i}: Q&A rows need exactly 3 columns, got {len(cells)}")
            frame_cells = cells[1:2]
        elif task == "trake":
            frame_cells = cells[1:]
            trake_widths.add(len(frame_cells))
        else:
            if len(cells) != 2:
                errors.append(f"{name}:{i}: KIS rows need exactly 2 columns, got {len(cells)}")
            frame_cells = cells[1:2]

        frames: list[int] = []
        for cell in frame_cells:
            try:
                frames.append(int(cell))
            except ValueError:
                errors.append(f"{name}:{i}: frame_id {cell!r} is not an integer")
        if task == "trake" and len(frames) == len(frame_cells):
            if any(b <= a for a, b in zip(frames, frames[1:])):
                errors.append(f"{name}:{i}: TRAKE frames are not in chronological order: {frames}")

        if task == "qa" and len(cells) >= 3:
            answer = row[2]           # raw, so trailing spaces count
            if not answer.strip():
                errors.append(f"{name}:{i}: Q&A answer is empty")
            if len(answer) > MAX_ANSWER_CHARS:
                errors.append(
                    f"{name}:{i}: answer is {len(answer)} characters, maximum is "
                    f"{MAX_ANSWER_CHARS}"
                )

    if task == "trake":
        if len(trake_widths) > 1:
            errors.append(
                f"{name}: rows have differing frame counts {sorted(trake_widths)} - "
                "every row must have the same number of events"
            )
        expected = expected_events.get(Path(name).stem)
        if expected and trake_widths and trake_widths != {expected}:
            errors.append(
                f"{name}: rows carry {sorted(trake_widths)} frames but the query plan "
                f"says num_events={expected}"
            )

    return errors


def validate(
    submission_dir: Path | str,
    expected_events: dict[str, int] | None = None,
) -> list[str]:
    """Validate every CSV in a directory; returns all errors found."""
    directory = Path(submission_dir)
    if not directory.is_dir():
        return [f"{directory}: submission directory does not exist"]

    files = sorted(directory.glob("*.csv"))
    if not files:
        return [f"{directory}: contains no .csv files"]

    stray = [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() != ".csv"]
    errors: list[str] = [
        f"{name}: unexpected non-CSV file in the submission directory" for name in stray
    ]
    for path in files:
        errors.extend(validate_file(path, expected_events))
    return errors
