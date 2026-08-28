"""High-performance MinIO Keyframe Client.

Provides streaming downloads, connection pooling, exponential backoff retries,
and batch multi-threaded fetching for keyframes stored in MinIO S3 object storage.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..config import MinioConfig
from ..logging_utils import get_logger

log = get_logger(__name__)


class MinioKeyframeClient:
    """Wrapper around MinIO client for resilient, parallel image loading."""

    def __init__(self, cfg: MinioConfig) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._failed = False
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and not self._failed

    def _get_client(self) -> Any:
        if self._client is not None or self._failed:
            return self._client
        try:
            from minio import Minio

            self._client = Minio(
                self.cfg.endpoint,
                access_key=self.cfg.access_key,
                secret_key=self.cfg.secret_key,
                secure=self.cfg.secure,
            )
            log.info(
                "MinIO Client connected to %s (bucket: %s, prefix: %s)",
                self.cfg.endpoint,
                self.cfg.bucket,
                self.cfg.prefix,
            )
            return self._client
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            log.warning("MinIO initialization failed: %s - continuing without MinIO", exc)
            return None

    def get_image(self, object_name: str) -> Image.Image | None:
        """Fetch and decode a single image from MinIO with chunk streaming and retry."""
        client = self._get_client()
        if client is None:
            return None

        max_retries = self.cfg.max_retries
        for attempt in range(max_retries):
            response = None
            try:
                response = client.get_object(self.cfg.bucket, object_name)
                buffer = io.BytesIO()
                for chunk in response.stream(amt=32768):
                    buffer.write(chunk)
                buffer.seek(0)
                img = Image.open(buffer).convert("RGB")
                img.load()
                return img
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    log.warning("Failed to fetch MinIO image %s after %d retries: %s", object_name, max_retries, exc)
                    return None
                time.sleep(2**attempt * 0.25)
            finally:
                if response is not None:
                    try:
                        response.close()
                        response.release_conn()
                    except Exception:
                        pass
        return None

    def batch_get_images(
        self, object_names: Sequence[str], max_workers: int | None = None
    ) -> dict[str, Image.Image]:
        """Fetch multiple images in parallel."""
        workers = max_workers or self.cfg.max_workers
        results: dict[str, Image.Image] = {}
        if not object_names or not self.enabled:
            return results

        def _fetch_one(name: str) -> tuple[str, Image.Image | None]:
            return name, self.get_image(name)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for name, img in executor.map(_fetch_one, object_names):
                if img is not None:
                    results[name] = img
        return results

    def discover_video_frames(self, video_id: str) -> dict[int, str]:
        """List all keyframes of a video and return a mapping of frame_id -> object_name."""
        index_dir = self._cache_dir.parent / "minio_index"
        index_file = index_dir / f"{video_id}.json"
        if index_file.is_file():
            try:
                import json

                with index_file.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                    return {int(k): str(v) for k, v in raw.items()}
            except Exception as exc:
                log.debug("Index cache read failed for %s: %s", video_id, exc)

        client = self._get_client()
        if client is None:
            return {}

        prefix = f"{self.cfg.prefix.rstrip('/')}/{video_id}/"
        frame_map: dict[int, str] = {}
        try:
            import re

            for obj in client.list_objects(self.cfg.bucket, prefix=prefix, recursive=True):
                name = obj.object_name
                fname = name.split("/")[-1]
                match = re.search(r"frame_(\d+)", fname)
                if match:
                    fid = int(match.group(1))
                    frame_map[fid] = name
                else:
                    parts = fname.replace(".jpg", "").split("_")
                    for p in reversed(parts):
                        if p.isdigit():
                            frame_map[int(p)] = name
                            break

            if frame_map:
                try:
                    import json

                    index_dir.mkdir(parents=True, exist_ok=True)
                    with index_file.open("w", encoding="utf-8") as fh:
                        json.dump({str(k): v for k, v in frame_map.items()}, fh)
                except Exception as exc:
                    log.debug("Index cache write failed for %s: %s", video_id, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("MinIO frame discovery failed for %s: %s", video_id, exc)
        return frame_map
