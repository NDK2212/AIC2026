"""Elasticsearch retrieval for the OCR and ASR paths."""

from __future__ import annotations

from typing import Any

from ..clients.elastic import ElasticWrapper
from ..config import ElasticConfig
from ..logging_utils import get_logger
from ..schemas import Candidate, PATH_ASR, PATH_DESCRIPTION, PATH_OCR
from ..utils.text_norm import normalize_query

log = get_logger(__name__)


class TextSearcher:
    """Runs the OCR, ASR and Description BM25 queries against the keyframe index."""

    def __init__(self, client: ElasticWrapper, cfg: ElasticConfig | None = None) -> None:
        self.client = client
        self.cfg = cfg or client.cfg

    # ------------------------------------------------------------------
    def search_ocr(
        self,
        query: str | None,
        terms: list[str] | None = None,
        size: int | None = None,
        exact_text: bool = False,
    ) -> list[Candidate]:
        """Search the OCR field.  ``exact_text`` boosts the phrase clause hard."""
        return self._search(
            field=self.cfg.fields.get("ocr", "ocr_text"),
            source=PATH_OCR,
            query=query,
            terms=terms or [],
            size=size,
            exact_text=exact_text,
        )

    def search_asr(
        self,
        query: str | None,
        terms: list[str] | None = None,
        size: int | None = None,
    ) -> list[Candidate]:
        """Search the ASR field."""
        return self._search(
            field=self.cfg.fields.get("asr", "clean_text"),
            source=PATH_ASR,
            query=query,
            terms=terms or [],
            size=size,
            exact_text=False,
        )

    def search_description(
        self,
        query: str | None,
        terms: list[str] | None = None,
        size: int | None = None,
    ) -> list[Candidate]:
        """Search the dense natural language description field."""
        return self._search(
            field=self.cfg.fields.get("description", "description_text"),
            source=PATH_DESCRIPTION,
            query=query,
            terms=terms or [],
            size=size,
            exact_text=False,
        )

    # ------------------------------------------------------------------
    def _search(
        self,
        field: str,
        source: str,
        query: str | None,
        terms: list[str],
        size: int | None,
        exact_text: bool,
    ) -> list[Candidate]:
        """Build, run and parse one field-scoped BM25 query."""
        main = normalize_query(query)
        term_blob = normalize_query(" ".join(terms))
        if not main and not term_blob:
            log.debug("%s path disabled (no sub-query)", source)
            return []

        size = size or self.cfg.size
        body = self.build_body(field, main, term_blob, size, exact_text)
        try:
            hits = self.client.search(body, size=size)
        except Exception as exc:  # noqa: BLE001 - a dead path must not kill the run
            log.error("%s search failed: %s", source, exc)
            return []

        return self._to_candidates(hits, field, source)

    def build_body(
        self,
        field: str,
        main: str,
        term_blob: str,
        size: int,
        exact_text: bool,
    ) -> dict[str, Any]:
        """Assemble the ES query body (kept public so tests can inspect it)."""
        boosts = self.cfg.boosts
        should: list[dict[str, Any]] = []

        if main:
            should.append(
                {"match": {field: {"query": main, "operator": "or",
                                   "boost": boosts["match"]}}}
            )
            phrase_boost = boosts["exact_phrase"] if exact_text else boosts["phrase"]
            should.append(
                {"match_phrase": {field: {"query": main,
                                          "slop": self.cfg.phrase_slop,
                                          "boost": phrase_boost}}}
            )
        if term_blob:
            should.append({"match": {field: {"query": term_blob, "boost": boosts["terms"]}}})

        source_fields = [
            self.cfg.fields["video_id"],
            self.cfg.fields["frame_id"],
            field,
        ]
        if field == self.cfg.fields.get("description", "frame_description") or "description" in field:
            for alt in ("frame_description", "video_description", "title", "description", "caption", "dense_caption"):
                if alt not in source_fields:
                    source_fields.append(alt)

        body: dict[str, Any] = {
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "_source": source_fields,
        }
        if self.cfg.min_score > 0:
            body["min_score"] = self.cfg.min_score
        return body

    def _to_candidates(
        self, hits: list[dict[str, Any]], field: str, source: str
    ) -> list[Candidate]:
        """Convert ES hits into ranked candidates, skipping malformed docs."""
        video_field = self.cfg.fields["video_id"]
        frame_field = self.cfg.fields["frame_id"]

        out: list[Candidate] = []
        skipped = 0
        for rank, hit in enumerate(hits, start=1):
            doc = hit.get("_source", {}) or {}
            video_id = doc.get(video_field)
            if video_id is None:
                video_id = doc.get("video_id") or doc.get("video_name") or doc.get("video")

            frame_raw = doc.get(frame_field)
            if frame_raw is None:
                frame_raw = (
                    doc.get("frame_id_ocr")
                    or doc.get("frame_id")
                    or doc.get("frame_id_start")
                    or doc.get("frame_idx")
                    or doc.get("frame")
                )

            if video_id is None or frame_raw is None:
                skipped += 1
                continue
            try:
                frame_id = int(float(frame_raw))
            except (TypeError, ValueError):
                skipped += 1
                continue

            matched = doc.get(field)
            if not matched and source == PATH_DESCRIPTION:
                matched = (
                    doc.get("frame_description")
                    or doc.get("description_text")
                    or doc.get("video_description")
                    or doc.get("title")
                    or doc.get("description")
                    or doc.get("caption")
                    or doc.get("dense_caption")
                    or ""
                )
            elif not matched:
                matched = ""

            out.append(
                Candidate(
                    video_id=str(video_id).removesuffix(".mp4"),
                    frame_id=frame_id,
                    score=float(hit.get("_score") or 0.0),
                    source=source,
                    rank=len(out) + 1,
                    extra={
                        "matched_text": str(matched)[:400],
                        f"{source}_matched": str(matched)[:400],
                        f"{source}_rank": rank,
                    },
                )
            )
        if skipped:
            log.warning(
                "%s: skipped %d ES hits missing %s/%s",
                source, skipped, video_field, frame_field,
            )
        log.info("%s path -> %d candidates", source, len(out))
        return out

    def fetch_metadata(
        self, targets: Sequence[Candidate], max_shot_gap: int = 60
    ) -> dict[tuple[str, int], dict[str, str]]:
        """Fetch rich Description, OCR, and ASR metadata for target frames.

        Performs a single batch query to Elasticsearch and associates each
        target candidate with the closest matching video segment metadata.
        """
        if not targets:
            return {}

        video_ids = list({c.video_id for c in targets})
        if not video_ids:
            return {}

        # Construct batch query across target video IDs
        body = {
            "size": min(1000, len(targets) * 10),
            "query": {
                "bool": {
                    "should": [
                        {"terms": {"video_id.keyword": video_ids}},
                        {"terms": {"video_id": video_ids}},
                        {"terms": {"video_name.keyword": video_ids}},
                        {"terms": {"video_name": video_ids}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

        try:
            hits = self.client.search(body, size=min(1000, len(targets) * 10))
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_metadata failed against Elasticsearch: %s", exc)
            return {}

        # Parse documents from ES hits
        docs: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {}) or {}
            v_id = source.get("video_id") or source.get("video_name") or source.get("video")
            if not v_id:
                continue
            v_clean = str(v_id).removesuffix(".mp4")

            f_raw = (
                source.get("frame_id_ocr")
                or source.get("frame_id")
                or source.get("frame_id_start")
                or source.get("frame_idx")
                or source.get("frame")
            )
            try:
                f_id = int(float(f_raw)) if f_raw is not None else 0
            except (TypeError, ValueError):
                f_id = 0

            desc = str(
                source.get("frame_description")
                or source.get("description_text")
                or source.get("video_description")
                or source.get("title")
                or source.get("description")
                or source.get("caption")
                or source.get("dense_caption")
                or ""
            ).strip()

            ocr = str(
                source.get("ocr_text")
                or source.get("ocr")
                or source.get("text")
                or ""
            ).strip()

            asr = str(
                source.get("clean_text")
                or source.get("asr_text")
                or source.get("asr")
                or ""
            ).strip()

            docs.append({
                "video_id": v_clean,
                "frame_id": f_id,
                "description": desc,
                "ocr": ocr,
                "asr": asr,
            })

        # Match each target candidate to the closest document in the same video
        result: dict[tuple[str, int], dict[str, str]] = {}
        for target in targets:
            matching_docs = [d for d in docs if d["video_id"] == target.video_id]
            if not matching_docs:
                continue

            # Sort by frame distance
            matching_docs.sort(key=lambda d: abs(d["frame_id"] - target.frame_id))
            closest = matching_docs[0]
            if abs(closest["frame_id"] - target.frame_id) <= max_shot_gap:
                result[target.key] = {
                    "description": closest["description"],
                    "ocr": closest["ocr"],
                    "asr": closest["asr"],
                }

        log.info(
            "Enriched metadata for %d/%d target candidates from Elasticsearch",
            len(result), len(targets),
        )
        return result

