"""Zip packaging: the archive must contain a ``submission/`` folder inside it."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from ..logging_utils import get_logger
from ..schemas import SubmissionError
from .validator import validate

log = get_logger(__name__)

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9]+$")


def pack(
    submission_dir: Path | str,
    out_zip: Path | str,
    expected_events: dict[str, int] | None = None,
    skip_validation: bool = False,
) -> Path:
    """Validate then zip a submission directory.

    The archive always contains ``submission/<file>.csv`` entries, never bare
    CSVs at the root - the judging system rejects the latter.
    """
    directory = Path(submission_dir)
    target = Path(out_zip)
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")

    if not skip_validation:
        errors = validate(directory, expected_events)
        if errors:
            raise SubmissionError(
                "Refusing to package an invalid submission:\n  - "
                + "\n  - ".join(errors)
            )

    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SubmissionError(f"{directory} contains no .csv files to package")

    if not _SAFE_NAME_RE.match(target.stem):
        log.warning(
            "Zip name %r contains characters other than letters and digits - the "
            "organisers recommend alphanumeric names only",
            target.stem,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"submission/{path.name}")

    log.info("Packed %d CSV files into %s", len(files), target)
    return target


def verify_zip(path: Path | str) -> list[str]:
    """Check a built archive really has the required ``submission/`` layout."""
    archive_path = Path(path)
    errors: list[str] = []
    if not archive_path.is_file():
        return [f"{archive_path}: zip file does not exist"]

    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if not names:
            return [f"{archive_path}: archive is empty"]
        for name in names:
            if not name.startswith("submission/"):
                errors.append(
                    f"{archive_path}: {name!r} is not inside the submission/ folder"
                )
            elif not name.lower().endswith(".csv"):
                errors.append(f"{archive_path}: {name!r} is not a .csv file")
    return errors
