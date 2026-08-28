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


from collections import deque
import threading
import time

class WebLogBuffer(logging.Handler):
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._buf_lock = threading.RLock()
        self.subscribers: list[tuple[Any, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            with self._buf_lock:
                self.records.append(entry)
                for q, loop in list(self.subscribers):
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, entry)
                    except Exception:
                        pass
        except Exception:
            self.handleError(record)

    def subscribe(self, q: Any, loop: Any) -> None:
        with self._buf_lock:
            self.subscribers.append((q, loop))

    def unsubscribe(self, q: Any) -> None:
        with self._buf_lock:
            self.subscribers = [(k, l) for k, l in self.subscribers if k is not q]

    def get_recent(self, n: int = 100) -> list[dict[str, Any]]:
        with self._buf_lock:
            return list(self.records)[-n:]


GLOBAL_LOG_BUFFER = WebLogBuffer()


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

    # Attach in-memory ring buffer for Web UI real-time streaming
    GLOBAL_LOG_BUFFER.setLevel(logging.DEBUG)
    root.addHandler(GLOBAL_LOG_BUFFER)

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
