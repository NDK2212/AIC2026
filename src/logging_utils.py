"""Logging setup shared by the CLI and every module.

Libraries only ever call ``logging.getLogger(__name__)``; the CLI is the single
place that configures handlers.  Output goes to the console and to
``outputs/runs/run.log`` at the same time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)s %(funcName)s:%(lineno)d %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logging(
    verbose: bool = False,
    log_file: Path | str | None = None,
    quiet_libraries: bool = True,
) -> None:
    """Configure root logging once.  Safe to call repeatedly."""
    global _CONFIGURED

    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if _CONFIGURED:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(level)
        return

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FMT))
        root.addHandler(file_handler)

    if quiet_libraries:
        for noisy in (
            "httpx",
            "httpcore",
            "urllib3",
            "elastic_transport.transport",
            "elasticsearch",
            "qdrant_client",
            "PIL",
            "matplotlib",
            "transformers",
            "filelock",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger helper."""
    return logging.getLogger(name)
