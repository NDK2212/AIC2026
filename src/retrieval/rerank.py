"""Rerankers: BLIP-2 ITM (Visual Path) and BGE Cross-Encoder (Fused Candidates).

- BLIP-2 ITM: Rescores visual candidates right after Qdrant using image-text matching.
- BGE Reranker: Rescores fused candidates right after WRRF using deep cross-encoding.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

from ..config import BGERerankConfig, Blip2RerankConfig, RerankConfig
from ..logging_utils import get_logger
from ..schemas import Candidate
from ..utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# BLIP-2 Image-Text Matching Reranker (Visual Path)
# ---------------------------------------------------------------------------
class BLIP2Reranker:
    """Rescore visual candidates from Qdrant with an image-text matching model."""

    def __init__(self, cfg: Blip2RerankConfig, kf: KeyframeIndex | None = None) -> None:
        self.cfg = cfg
        self.kf = kf
        self._model: Any = None
        self._processor: Any = None
        self._failed = False
        self._torch: Any = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """True when BLIP-2 reranking is configured on and still functional."""
        return self.cfg.enabled and not self._failed

    def _load(self) -> None:
        """Load the BLIP-2 ITM model lazily; failure disables it gracefully."""
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is not None or self._failed:
                return
            for attempt in range(3):
                try:
                    import torch
                    from transformers import AutoProcessor

                    if "blip2" in str(self.cfg.model_id).lower() or "blip-2" in str(self.cfg.model_id).lower():
                        from transformers import Blip2ForImageTextRetrieval as ModelCls
                    else:
                        from transformers import BlipForImageTextRetrieval as ModelCls

                    self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
                    model = ModelCls.from_pretrained(self.cfg.model_id)
                    if str(self.cfg.device).lower() not in ("cpu", "auto"):
                        try:
                            model.to(self.cfg.device)
                        except Exception:
                            pass
                    model.eval()
                    self._model = model
                    self._torch = torch
                    log.info("BLIP Reranker %s loaded on %s", self.cfg.model_id, self.cfg.device)
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt < 2:
                        import time
                        time.sleep(0.5)
                        continue
                    self._failed = True
                    log.warning("BLIP Reranker unavailable (%s) - continuing without it", exc)

    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[Candidate]:
        """Return visual candidates reordered by ITM score; identity when disabled."""
        ordered = list(candidates)
        if not self.enabled or not ordered or not query or self.kf is None:
            return ordered

        self._load()
        if self._model is None:
            return ordered

        head, tail = ordered[: self.cfg.top_n], ordered[self.cfg.top_n:]
        # 1. Parallel image pre-fetching via KeyframeIndex (supporting Local SSD + MinIO)
        items = [(cand.video_id, cand.frame_id) for cand in head]
        if hasattr(self.kf, "batch_get_images"):
            image_map = self.kf.batch_get_images(items, max_workers=min(10, max(1, len(head))))
            loaded_images = [image_map.get((cand.video_id, cand.frame_id)) for cand in head]
        else:
            def _load_image(cand: Candidate) -> Any:
                if hasattr(self.kf, "get_image"):
                    return self.kf.get_image(cand.video_id, cand.frame_id)
                path = self.kf.resolve_image(cand.video_id, cand.frame_id)
                if path is None or not Path(path).is_file():
                    return None
                try:
                    with Image.open(path) as img:
                        return img.convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    log.debug("Image load failed for %s: %s", path, exc)
                    return None

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(head))), thread_name_prefix="blip_io") as pool:
                loaded_images = list(pool.map(_load_image, head))

        # 2. Batched model inference
        valid_pairs = [(idx, img) for idx, img in enumerate(loaded_images) if img is not None]
        itm_scores: dict[int, float] = {}

        if valid_pairs:
            batch_size = max(1, getattr(self.cfg, "batch_size", 16))
            for i in range(0, len(valid_pairs), batch_size):
                chunk = valid_pairs[i : i + batch_size]
                chunk_imgs = [img for _, img in chunk]
                chunk_texts = [query] * len(chunk_imgs)
                try:
                    inputs = self._processor(
                        images=chunk_imgs, text=chunk_texts, padding=True, return_tensors="pt"
                    )
                    inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
                    is_blip2 = "blip2" in str(self.cfg.model_id).lower() or "blip-2" in str(self.cfg.model_id).lower()
                    forward_kwargs = {"use_image_text_matching_head": True} if is_blip2 else {"use_itm_head": True}
                    with self._torch.inference_mode():
                        if "cuda" in str(self.cfg.device):
                            with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                                out = self._model(**inputs, **forward_kwargs)
                        else:
                            out = self._model(**inputs, **forward_kwargs)
                    logits = getattr(out, "itm_score", None)
                    if logits is None:
                        logits = getattr(out, "logits_per_image", None)
                    if logits is None:
                        logits = out[0]
                    probs = self._torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                    if isinstance(probs, float):
                        probs = [probs]
                    for (orig_idx, _), prob in zip(chunk, probs):
                        itm_scores[orig_idx] = float(prob)
                except Exception as exc:  # noqa: BLE001
                    log.warning("BLIP-2 batch inference failed: %s", exc)

        scored: list[tuple[float, Candidate]] = []
        for idx, candidate in enumerate(head):
            itm = itm_scores.get(idx, 0.0)
            blended = candidate.score + self.cfg.weight * itm
            scored.append((
                blended,
                candidate.replace(
                    score=blended,
                    extra={
                        **candidate.extra,
                        "blip2_itm_score": itm,
                        "pre_blip2_score": candidate.score,
                    },
                ),
            ))

        scored.sort(key=lambda pair: (-pair[0], pair[1].video_id, pair[1].frame_id))
        merged = [c for _, c in scored] + tail
        return [c.replace(rank=i) for i, c in enumerate(merged, start=1)]

    def _score(self, query: str, image_path: Any) -> float:
        """Raw image-text matching score for one pair (kept for single-item tests/callers)."""
        from PIL import Image

        with Image.open(image_path) as img:
            inputs = self._processor(
                images=img.convert("RGB"), text=query, return_tensors="pt"
            )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            out = self._model(**inputs, use_image_text_matching_head=True)
        logits = getattr(out, "logits_per_image", None)
        if logits is None:
            logits = out[0]
        return float(self._torch.softmax(logits, dim=-1)[0, 1].item())


# ---------------------------------------------------------------------------
# BGE Cross-Encoder Reranker (Text Modality Paths)
# ---------------------------------------------------------------------------
class BGEReranker:
    """Rescore text candidates using a BGE Cross-Encoder."""

    def __init__(self, cfg: BGERerankConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._tokenizer: Any = None
        self._failed = False
        self._torch: Any = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """True when BGE reranking is configured on and still functional."""
        return self.cfg.enabled and not self._failed

    def _load(self) -> None:
        """Load the BGE Cross-Encoder lazily; failure disables it gracefully."""
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is not None or self._failed:
                return
            for attempt in range(3):
                try:
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_id)
                    model.eval().to(self.cfg.device)
                    self._model = model
                    self._torch = torch
                    log.info("BGE Reranker %s loaded on %s", self.cfg.model_id, self.cfg.device)
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt < 2:
                        import time
                        time.sleep(0.5)
                        continue
                    self._failed = True
                    log.warning("BGE Reranker unavailable (%s) - continuing without it", exc)

    def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[Candidate]:
        """Return text candidates reordered by selective BGE score; identity when disabled."""
        ordered = list(candidates)
        if not self.enabled or not ordered or not query:
            return ordered

        self._load()
        if self._model is None:
            return ordered

        head, tail = ordered[: self.cfg.top_n], ordered[self.cfg.top_n:]
        text_pairs: list[tuple[int, str, str]] = []
        for idx, candidate in enumerate(head):
            doc_text = self._extract_candidate_text(candidate)
            if doc_text and doc_text.strip():
                text_pairs.append((idx, query, doc_text.strip()))

        bge_score_map: dict[int, float] = {}
        if text_pairs:
            try:
                pairs_only = [(q, t) for _, q, t in text_pairs]
                scores = self._compute_scores(pairs_only)
                for (idx, _, _), score in zip(text_pairs, scores):
                    bge_score_map[idx] = float(score)
            except Exception as exc:  # noqa: BLE001
                log.warning("BGE rerank batch failed: %s - skipping BGE rerank", exc)
                return ordered

        scored: list[tuple[float, Candidate]] = []
        for idx, candidate in enumerate(head):
            if idx in bge_score_map:
                bge_score = bge_score_map[idx]
                blended = candidate.score + self.cfg.weight * bge_score
                scored.append((
                    blended,
                    candidate.replace(
                        score=blended,
                        extra={
                            **(candidate.extra or {}),
                            "bge_score": bge_score,
                            "pre_bge_score": candidate.score,
                        },
                    ),
                ))
            else:
                # Retains original candidate score if text extraction is empty
                scored.append((candidate.score, candidate))

        scored.sort(key=lambda pair: (-pair[0], pair[1].video_id, pair[1].frame_id))
        merged = [c for _, c in scored] + tail
        return [c.replace(rank=i) for i, c in enumerate(merged, start=1)]

    def _extract_candidate_text(self, candidate: Candidate) -> str:
        """Extract text representation of candidate from its metadata."""
        extra = candidate.extra or {}
        matched = extra.get("matched_text")
        if matched and str(matched).strip():
            return str(matched).strip()
        parts = []
        if "description_matched" in extra and str(extra["description_matched"]).strip():
            parts.append(str(extra["description_matched"]))
        if "ocr_matched" in extra and str(extra["ocr_matched"]).strip():
            parts.append(f"OCR: {extra['ocr_matched']}")
        if "asr_matched" in extra and str(extra["asr_matched"]).strip():
            parts.append(f"ASR: {extra['asr_matched']}")
        if parts:
            return " | ".join(parts)
        return ""

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-encoder relevance scores for query-text pairs."""
        all_scores: list[float] = []
        batch_size = max(1, self.cfg.batch_size)

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            with self._torch.inference_mode():
                out = self._model(**inputs, return_dict=True)
                logits = out.logits.view(-1).float()
                # Apply sigmoid if unbounded logits
                probs = self._torch.sigmoid(logits).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                all_scores.extend(probs)
        return all_scores


# Backward compatibility alias
Reranker = BLIP2Reranker
