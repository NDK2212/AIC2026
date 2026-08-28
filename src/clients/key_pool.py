"""Multi-Key API Pool for NVIDIA NIM endpoints.

Supports round-robin selection and automatic instant failover when an API key
encounters rate limiting (429), quota limits, or server overload (503).
"""

from __future__ import annotations

import os
import threading
from itertools import cycle
from typing import Iterator

from ..logging_utils import get_logger

log = get_logger(__name__)


class APIKeyPool:
    """Thread-safe key pool with round-robin rotation and failover."""

    def __init__(self, initial_keys: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._cycler: Iterator[str] | None = None
        if initial_keys:
            self.set_keys(initial_keys)
        else:
            # Load from env
            env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
            vlm_key = os.environ.get("VLM_API_KEY", "").strip()
            keys = [k for k in {env_key, vlm_key} if k]
            if keys:
                self.set_keys(keys)

    def set_keys(self, keys: list[str]) -> None:
        with self._lock:
            # Clean and deduplicate while preserving order
            cleaned = []
            seen = set()
            for k in keys:
                k_clean = k.strip().strip(",;\"'")
                if k_clean and k_clean not in seen:
                    cleaned.append(k_clean)
                    seen.add(k_clean)
            self._keys = cleaned
            self._cycler = cycle(self._keys) if self._keys else None
            log.info("APIKeyPool configured with %d active key(s)", len(self._keys))

    def add_key(self, key: str) -> None:
        with self._lock:
            k_clean = key.strip().strip(",;\"'")
            if k_clean and k_clean not in self._keys:
                self._keys.append(k_clean)
                self._cycler = cycle(self._keys)
                log.info("Added new API key to pool. Total keys: %d", len(self._keys))

    def get_keys(self) -> list[str]:
        with self._lock:
            return list(self._keys)

    def next_key(self) -> str:
        """Get the next API key in round-robin order."""
        with self._lock:
            if not self._keys or self._cycler is None:
                env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
                if env_key:
                    return env_key
                raise ValueError("No API keys available in APIKeyPool")
            return next(self._cycler)

    def get_failover_order(self, failed_key: str | None = None) -> list[str]:
        """Return list of keys prioritizing alternative keys after a failure."""
        with self._lock:
            if not self._keys:
                env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
                return [env_key] if env_key else []
            if failed_key and failed_key in self._keys:
                # Put remaining keys first, failed key at the back
                others = [k for k in self._keys if k != failed_key]
                return others + [failed_key]
            return list(self._keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


# Global singleton instance
GLOBAL_KEY_POOL = APIKeyPool()
