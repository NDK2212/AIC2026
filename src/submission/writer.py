"""CSV writing that follows the competition format to the letter.

UTF-8 without BOM, comma delimiter, ``\\n`` line endings, no header row, minimal
quoting (so ``5`` stays ``5`` but ``Có 3 người, gồm nam và nữ`` gets quoted and
embedded ``"`` becomes ``""``), and never more than ``max_rows`` rows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from ..logging_utils import get_logger
from ..schemas import SubmissionError
from ..utils.text_norm import sanitize_answer

log = get_logger(__name__)


def _open(path: Path):
    """Open a CSV for writing with the exact encoding the judges expect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def _writer(handle):
    return csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def _clean_video_id(video_id: str) -> str:
    """Strip the ``.mp4`` suffix and stray whitespace."""
    return str(video_id).strip().removesuffix(".mp4")


def write_kis(
    path: Path | str,
    rows: Iterable[tuple[str, int]],
    max_rows: int = 100,
) -> int:
    """Write a Textual KIS submission: ``video_id,frame_id``."""
    target = Path(path)
    written = 0
    with _open(target) as handle:
        out = _writer(handle)
        for video_id, frame_id in rows:
            if written >= max_rows:
                break
            out.writerow([_clean_video_id(video_id), int(frame_id)])
            written += 1
    log.info("Wrote %d KIS rows to %s", written, target)
    return written


def write_qa(
    path: Path | str,
    rows: Iterable[tuple[str, int, str]],
    max_rows: int = 100,
    answer_max_chars: int = 100,
    fallback_answer: str = "unknown",
) -> int:
    """Write a Q&A submission: ``video_id,frame_id,answer``.

    An empty answer would make the row unparsable, so ``fallback_answer`` is
    substituted rather than emitting a blank field.
    """
    target = Path(path)
    written = 0
    with _open(target) as handle:
        out = _writer(handle)
        for video_id, frame_id, answer in rows:
            if written >= max_rows:
                break
            cleaned = sanitize_answer(answer, answer_max_chars)
            if not cleaned:
                cleaned = sanitize_answer(fallback_answer, answer_max_chars) or "n/a"
            out.writerow([_clean_video_id(video_id), int(frame_id), cleaned])
            written += 1
    log.info("Wrote %d Q&A rows to %s", written, target)
    return written


def write_trake(
    path: Path | str,
    rows: Iterable[Sequence[object]],
    num_events: int,
    max_rows: int = 100,
) -> int:
    """Write a TRAKE submission: ``video_id,f_1,...,f_N``.

    Every row must carry exactly ``num_events`` frames; anything else is a
    scoring-time parse failure, so it raises here instead.
    """
    if num_events < 1:
        raise SubmissionError(f"TRAKE num_events must be >= 1, got {num_events}")

    target = Path(path)
    written = 0
    with _open(target) as handle:
        out = _writer(handle)
        for index, row in enumerate(rows):
            if written >= max_rows:
                break
            if len(row) != num_events + 1:
                raise SubmissionError(
                    f"TRAKE row {index} has {len(row) - 1} frames but the query asks "
                    f"for {num_events}: {row!r}"
                )
            video_id = _clean_video_id(str(row[0]))
            frames = [int(f) for f in row[1:]]
            if any(b <= a for a, b in zip(frames, frames[1:])):
                raise SubmissionError(
                    f"TRAKE row {index} frames are not strictly increasing: {frames}"
                )
            out.writerow([video_id, *frames])
            written += 1
    log.info("Wrote %d TRAKE rows (%d events each) to %s", written, num_events, target)
    return written
