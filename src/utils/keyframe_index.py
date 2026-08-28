"""Two-way mapping between original ``frame_id`` values and keyframe images.

Keyframe files on disk are numbered sequentially (``0000.jpg``, ``0001.jpg``,
...) while Qdrant / Elasticsearch store the *original* frame index inside the
video.  The mapping between the two lives either in ``map-keyframes/<vid>.csv``
(columns ``n,pts_time,fps,frame_idx``) or in the per-video metadata JSON.  This
module auto-detects whichever source exists and falls back - loudly - to
"file order == frame id" when neither does.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..logging_utils import get_logger
from .cache import DiskCache

log = get_logger(__name__)

_CACHE_VERSION = 2
_NUM_RE = re.compile(r"(\d+)")

# Keys that may hold the original frame index inside a metadata JSON blob.
_FRAME_KEYS = ("frame_idx", "frame_id", "frame_index", "frameIdx", "idx", "frame")


@dataclass
class VideoKeyframes:
    """Sorted keyframe table for a single video."""

    video_id: str
    frame_ids: list[int]          # ascending, original frame indices
    paths: list[str]              # parallel to frame_ids
    exact: bool = True            # False when the mapping was guessed

    def position(self, frame_id: int) -> int | None:
        """Index of an exact ``frame_id`` match, or ``None``."""
        i = bisect.bisect_left(self.frame_ids, frame_id)
        if i < len(self.frame_ids) and self.frame_ids[i] == frame_id:
            return i
        return None

    def nearest_position(self, frame_id: int) -> int:
        """Index of the closest keyframe; ties go to the earlier frame."""
        if not self.frame_ids:
            raise ValueError(f"video {self.video_id} has no keyframes")
        i = bisect.bisect_left(self.frame_ids, frame_id)
        if i == 0:
            return 0
        if i >= len(self.frame_ids):
            return len(self.frame_ids) - 1
        before, after = self.frame_ids[i - 1], self.frame_ids[i]
        return i - 1 if (frame_id - before) <= (after - frame_id) else i


class KeyframeIndex:
    """Loads, caches and queries the keyframe tables of every video."""

    def __init__(
        self,
        root: Path | str,
        map_dir: Path | str | None = None,
        metadata_dir: Path | str | None = None,
        image_glob: str = "*.jpg",
        cache: DiskCache | None = None,
        minio: Any = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.map_dir = Path(map_dir) if map_dir else None
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.image_glob = image_glob
        self.cache = cache
        self.minio = minio
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./outputs/cache/keyframes")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.videos: dict[str, VideoKeyframes] = {}
        self._minio_video_maps: dict[str, dict[int, str]] = {}
        self._warned_missing: set[str] = set()
        self._load()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _signature(self) -> str:
        """Cheap fingerprint used to invalidate the on-disk cache."""
        parts: list[str] = [str(_CACHE_VERSION), str(self.root), self.image_glob]
        for directory in (self.root, self.map_dir, self.metadata_dir):
            if directory and directory.exists():
                stat = directory.stat()
                parts.append(f"{directory}:{stat.st_mtime_ns}")
                # one level down catches added/removed videos
                try:
                    children = sorted(p.name for p in directory.iterdir())
                except OSError:  # pragma: no cover
                    children = []
                # hashlib, not hash(): str hashing is salted per process, which
                # would change the cache key on every run and never hit.
                digest = hashlib.sha1("\x00".join(children).encode("utf-8")).hexdigest()
                parts.append(f"{len(children)}:{digest}")
            else:
                parts.append(f"{directory}:missing")
        return "|".join(parts)

    def _load(self) -> None:
        from .cache import sha256_key

        key = sha256_key(self._signature())
        if self.cache is not None:
            cached = self.cache.get_json("keyframe_index", key)
            if cached:
                self.videos = {
                    vid: VideoKeyframes(
                        video_id=vid,
                        frame_ids=[int(f) for f in entry["frame_ids"]],
                        paths=list(entry["paths"]),
                        exact=bool(entry.get("exact", True)),
                    )
                    for vid, entry in cached.items()
                }
                log.info("Keyframe index loaded from cache: %d videos", len(self.videos))
                return

        self.videos = self._build()
        if self.cache is not None and self.videos:
            self.cache.set_json(
                "keyframe_index",
                key,
                {
                    vid: {
                        "frame_ids": v.frame_ids,
                        "paths": v.paths,
                        "exact": v.exact,
                    }
                    for vid, v in self.videos.items()
                },
            )

    def _build(self) -> dict[str, VideoKeyframes]:
        if not self.root.exists():
            log.warning(
                "Keyframe root %s does not exist - image lookups and neighbour "
                "expansion will be disabled",
                self.root,
            )
            return {}

        videos: dict[str, VideoKeyframes] = {}
        guessed = 0
        for video_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            video_id = video_dir.name
            images = sorted(video_dir.glob(self.image_glob), key=_natural_key)
            if not images:
                continue
            mapping = self._read_map(video_id, len(images))
            if mapping is None:
                mapping = list(range(len(images)))
                guessed += 1
                exact = False
            else:
                exact = True
            if len(mapping) != len(images):
                log.warning(
                    "%s: map has %d rows but %d keyframe images - truncating to the "
                    "shorter of the two",
                    video_id,
                    len(mapping),
                    len(images),
                )
                n = min(len(mapping), len(images))
                mapping, images = mapping[:n], images[:n]

            pairs = sorted(zip(mapping, images), key=lambda pair: pair[0])
            videos[video_id] = VideoKeyframes(
                video_id=video_id,
                frame_ids=[int(f) for f, _ in pairs],
                paths=[str(p) for _, p in pairs],
                exact=exact,
            )

        if guessed:
            log.warning(
                "No frame-index map found for %d/%d videos - assuming "
                "frame_id == keyframe file order for those. Submitted frame ids "
                "will be wrong if that assumption does not hold.",
                guessed,
                len(videos),
            )
        log.info("Keyframe index built: %d videos, %d keyframes",
                 len(videos), sum(len(v.frame_ids) for v in videos.values()))
        return videos

    def _read_map(self, video_id: str, n_images: int) -> list[int] | None:
        """Try every known map source; return ``None`` if none applies."""
        if self.map_dir and self.map_dir.is_dir():
            for name in (f"{video_id}.csv", f"{video_id}.CSV"):
                path = self.map_dir / name
                if path.is_file():
                    frames = _read_map_csv(path)
                    if frames:
                        return frames
        if self.metadata_dir and self.metadata_dir.is_dir():
            path = self.metadata_dir / f"{video_id}.json"
            if path.is_file():
                frames = _read_map_json(path, n_images)
                if frames:
                    return frames
        return None

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def _ensure_video(self, video_id: str) -> VideoKeyframes | None:
        """Ensure a video is in self.videos, discovering from MinIO if missing locally."""
        if video_id in self.videos:
            return self.videos[video_id]

        if self.minio is not None and getattr(self.minio, "enabled", False):
            if video_id in self._minio_video_maps:
                fmap = self._minio_video_maps[video_id]
            else:
                fmap = self.minio.discover_video_frames(video_id)
                self._minio_video_maps[video_id] = fmap

            if fmap:
                fids = sorted(fmap.keys())
                vk = VideoKeyframes(
                    video_id=video_id,
                    frame_ids=fids,
                    paths=[fmap[fid] for fid in fids],
                    exact=True,
                )
                self.videos[video_id] = vk
                return vk
        return None

    def has_video(self, video_id: str) -> bool:
        """True when this video has a keyframe table."""
        return self._ensure_video(video_id) is not None

    def path_of(self, video_id: str, frame_id: int) -> Path | None:
        """Image path for an exact ``(video_id, frame_id)``, else ``None``."""
        video = self._ensure_video(video_id)
        if video is None:
            return None
        pos = video.position(int(frame_id))
        return Path(video.paths[pos]) if pos is not None else None

    def nearest(self, video_id: str, frame_id: int) -> tuple[int, Path] | None:
        """Closest existing keyframe to ``frame_id``.

        Returns ``(frame_id, path)``.  Logs a warning when the match is not
        exact so silent frame drift is visible in the run log.
        """
        video = self._ensure_video(video_id)
        if video is None or not video.frame_ids:
            if video_id not in self._warned_missing:
                self._warned_missing.add(video_id)
                log.warning("No keyframes found for video %s", video_id)
            return None
        pos = video.nearest_position(int(frame_id))
        actual = video.frame_ids[pos]
        if actual != int(frame_id):
            log.debug(
                "%s: frame %s not a keyframe, snapping to %s (delta %+d)",
                video_id, frame_id, actual, actual - int(frame_id),
            )
        return actual, Path(video.paths[pos])

    def nearest_frame(self, video_id: str, frame_id: int) -> int:
        """Closest existing keyframe id, or ``frame_id`` itself if unknown."""
        found = self.nearest(video_id, frame_id)
        return found[0] if found else int(frame_id)

    def resolve_image(self, video_id: str, frame_id: int) -> Path | None:
        """Exact image if present, otherwise the nearest keyframe's image."""
        exact = self.path_of(video_id, frame_id)
        if exact is not None:
            return exact
        found = self.nearest(video_id, frame_id)
        return found[1] if found else None

    def get_image(self, video_id: str, frame_id: int) -> Any:
        """Return PIL.Image (RGB) for (video_id, frame_id), checking Disk cache then MinIO."""
        from PIL import Image

        # 1. Check local cache directory first
        cached_file = self.cache_dir / video_id / f"{int(frame_id)}.jpg"
        if cached_file.is_file():
            try:
                with Image.open(cached_file) as img:
                    return img.convert("RGB")
            except Exception:
                pass

        # 2. Check local dataset root
        local_path = self.resolve_image(video_id, frame_id)
        if local_path and local_path.is_file():
            try:
                with Image.open(local_path) as img:
                    return img.convert("RGB")
            except Exception:
                pass

        # 3. Fetch from MinIO if available
        if self.minio is not None and getattr(self.minio, "enabled", False):
            video = self._ensure_video(video_id)
            if video:
                pos = video.position(int(frame_id))
                if pos is None:
                    pos = video.nearest_position(int(frame_id))
                obj_name = video.paths[pos]
                img = self.minio.get_image(obj_name)
                if img is not None:
                    # Save to local disk cache for fast future reuse
                    try:
                        cached_file.parent.mkdir(parents=True, exist_ok=True)
                        img.save(cached_file, format="JPEG", quality=90)
                    except Exception as exc:
                        log.debug("Failed to cache image to %s: %s", cached_file, exc)
                    return img
        return None

    def batch_get_images(
        self, items: Sequence[tuple[str, int]], max_workers: int = 10
    ) -> dict[tuple[str, int], Any]:
        """Fetch multiple images in parallel using local cache and MinIO streaming."""
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image

        results: dict[tuple[str, int], Any] = {}
        missing: list[tuple[str, int, str]] = []

        # Pre-ensure videos concurrently if not loaded
        unique_vids = {vid for vid, _ in items if vid not in self.videos}
        if unique_vids and self.minio is not None and getattr(self.minio, "enabled", False):
            with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_vids))) as pool:
                list(pool.map(self._ensure_video, unique_vids))

        # First pass: check local files and disk cache
        for vid, fid in items:
            key = (vid, int(fid))
            cached_file = self.cache_dir / vid / f"{int(fid)}.jpg"
            if cached_file.is_file():
                try:
                    with Image.open(cached_file) as img:
                        results[key] = img.convert("RGB")
                        continue
                except Exception:
                    pass
            local_path = self.resolve_image(vid, fid)
            if local_path and local_path.is_file():
                try:
                    with Image.open(local_path) as img:
                        results[key] = img.convert("RGB")
                        continue
                except Exception:
                    pass

            if self.minio is not None and getattr(self.minio, "enabled", False):
                video = self._ensure_video(vid)
                if video:
                    pos = video.position(int(fid))
                    if pos is None:
                        pos = video.nearest_position(int(fid))
                    obj_name = video.paths[pos]
                    missing.append((vid, int(fid), obj_name))

        if not missing or self.minio is None:
            return results

        def _fetch_save(item: tuple[str, int, str]) -> tuple[tuple[str, int], Any]:
            vid, fid, obj_name = item
            img = self.minio.get_image(obj_name)
            if img is not None:
                cached_file = self.cache_dir / vid / f"{fid}.jpg"
                try:
                    cached_file.parent.mkdir(parents=True, exist_ok=True)
                    img.save(cached_file, format="JPEG", quality=90)
                except Exception:
                    pass
            return (vid, fid), img

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key, img in executor.map(_fetch_save, missing):
                if img is not None:
                    results[key] = img

        return results

    def neighbors(
        self, video_id: str, frame_id: int, offsets: Iterable[int]
    ) -> list[int]:
        """Existing keyframes closest to ``frame_id + offset`` for each offset.

        The source frame itself and duplicates are filtered out, and the result
        keeps the caller's offset order so the expansion stays deterministic.
        """
        video = self.videos.get(video_id)
        if video is None or not video.frame_ids:
            return []
        base = int(frame_id)
        out: list[int] = []
        seen = {base}
        for offset in offsets:
            pos = video.nearest_position(base + int(offset))
            candidate = video.frame_ids[pos]
            if candidate in seen:
                # step outward to the next unused keyframe in the same direction
                step = 1 if offset > 0 else -1
                probe = pos + step
                while 0 <= probe < len(video.frame_ids):
                    candidate = video.frame_ids[probe]
                    if candidate not in seen:
                        break
                    probe += step
                else:
                    continue
                if candidate in seen:
                    continue
            seen.add(candidate)
            out.append(candidate)
        return out

    def all_frames(self, video_id: str) -> list[int]:
        """Every keyframe id of a video, ascending."""
        video = self.videos.get(video_id)
        return list(video.frame_ids) if video else []

    def median_gap(self, video_id: str) -> int:
        """Typical spacing between consecutive keyframes (for extrapolation)."""
        frames = self.all_frames(video_id)
        if len(frames) < 2:
            return 25
        gaps = sorted(b - a for a, b in zip(frames, frames[1:]) if b > a)
        return gaps[len(gaps) // 2] if gaps else 25

    def __len__(self) -> int:
        return len(self.videos)


# ---------------------------------------------------------------------------
# map parsers
# ---------------------------------------------------------------------------
def _natural_key(path: Path) -> tuple[Any, ...]:
    """Sort ``9.jpg`` before ``10.jpg``."""
    parts = _NUM_RE.split(path.stem)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _read_map_csv(path: Path) -> list[int] | None:
    """Parse a ``map-keyframes`` CSV into an ordered list of frame indices."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return None
            headers = {h.strip().lower(): h for h in reader.fieldnames}
            column = next(
                (headers[k] for k in ("frame_idx", "frame_id", "frame_index", "frame")
                 if k in headers),
                None,
            )
            order_col = next((headers[k] for k in ("n", "idx", "index") if k in headers), None)
            if column is None:
                return None
            rows: list[tuple[int, int]] = []
            for i, row in enumerate(reader):
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    frame = int(float(raw))
                except ValueError:
                    continue
                order = i
                if order_col:
                    try:
                        order = int(float((row.get(order_col) or i)))
                    except ValueError:
                        order = i
                rows.append((order, frame))
        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        return [frame for _, frame in rows]
    except OSError as exc:  # pragma: no cover
        log.warning("Could not read keyframe map %s: %s", path, exc)
        return None


def _read_map_json(path: Path, n_images: int) -> list[int] | None:
    """Pull an ordered frame-index list out of a metadata JSON file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
        log.warning("Could not read metadata %s: %s", path, exc)
        return None

    frames = _extract_frames(data, n_images)
    return frames


def _extract_frames(node: Any, n_images: int, depth: int = 0) -> list[int] | None:
    """Recursively hunt for a frame-index list inside a metadata blob."""
    if depth > 4:
        return None
    if isinstance(node, list):
        if node and all(isinstance(x, (int, float)) for x in node):
            return [int(x) for x in node]
        if node and all(isinstance(x, dict) for x in node):
            for key in _FRAME_KEYS:
                if all(key in x for x in node):
                    try:
                        return [int(float(x[key])) for x in node]
                    except (TypeError, ValueError):
                        continue
        return None
    if isinstance(node, dict):
        # direct hit: a list keyed by something frame-ish
        for key in ("keyframes", "frames", "frame_idx", "frame_indices", "map"):
            if key in node:
                found = _extract_frames(node[key], n_images, depth + 1)
                if found:
                    return found
        # a dict keyed by keyframe number
        keys = list(node.keys())
        if keys and all(str(k).strip().isdigit() for k in keys):
            try:
                ordered = sorted(node.items(), key=lambda kv: int(str(kv[0])))
                values = [kv[1] for kv in ordered]
                if all(isinstance(v, (int, float)) for v in values):
                    return [int(v) for v in values]
                if all(isinstance(v, dict) for v in values):
                    for fk in _FRAME_KEYS:
                        if all(fk in v for v in values):
                            return [int(float(v[fk])) for v in values]
            except (TypeError, ValueError):
                pass
        for value in node.values():
            found = _extract_frames(value, n_images, depth + 1)
            if found:
                return found
    return None
