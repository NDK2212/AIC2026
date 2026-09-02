"""Multimodal and legacy rerankers.

Qwen3-VL is the active post-fusion image+text reranker. BLIP-2 and BGE are kept
for backward-compatible configurations and A/B tests.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

from ..config import (
    BGERerankConfig,
    Blip2RerankConfig,
    Qwen3VLRerankConfig,
    RerankConfig,
)
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


# ---------------------------------------------------------------------------
# Qwen3-VL unified multimodal reranker (post-fusion)
# ---------------------------------------------------------------------------
class Qwen3VLReranker:
    """Rerank fused keyframes with Qwen3-VL using image and text together.

    The implementation follows Qwen's official yes/no likelihood scoring:
    every query-document pair is formatted as a multimodal chat and the final
    hidden state is projected with ``lm_head[yes] - lm_head[no]``.
    """

    def __init__(
        self,
        cfg: Qwen3VLRerankConfig,
        kf: KeyframeIndex | None = None,
    ) -> None:
        self.cfg = cfg
        self.kf = kf
        self._model: Any = None
        self._processor: Any = None
        self._score_linear: Any = None
        self._torch: Any = None
        self._process_vision_info: Any = None
        self._failed = False
        self._load_lock = threading.Lock()
        # TRAKE retrieves steps concurrently; one shared 2B model must not run
        # overlapping forward passes and unexpectedly exhaust VRAM.
        self._inference_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and not self._failed

    def _load(self) -> None:
        if self._model is not None or self._failed:
            return
        with self._load_lock:
            if self._model is not None or self._failed:
                return
            try:
                import torch
                from qwen_vl_utils import process_vision_info
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                if str(self.cfg.device).startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError(
                        "rerank.qwen3_vl.device is CUDA but torch.cuda.is_available() is false"
                    )
                dtype = getattr(torch, self.cfg.torch_dtype, None)
                if dtype is None:
                    raise ValueError(f"Unsupported torch dtype: {self.cfg.torch_dtype}")

                lm = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.cfg.model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(self.cfg.device)
                processor = AutoProcessor.from_pretrained(
                    self.cfg.model_id,
                    trust_remote_code=True,
                    padding_side="left",
                )
                token_yes = processor.tokenizer.get_vocab()["yes"]
                token_no = processor.tokenizer.get_vocab()["no"]
                weight = lm.lm_head.weight.data[token_yes] - lm.lm_head.weight.data[token_no]
                score_linear = torch.nn.Linear(weight.shape[0], 1, bias=False)
                with torch.no_grad():
                    score_linear.weight[0].copy_(weight)

                self._model = lm.model.eval()
                self._processor = processor
                self._score_linear = score_linear.to(self.cfg.device).to(self._model.dtype).eval()
                self._torch = torch
                self._process_vision_info = process_vision_info
                log.info(
                    "Qwen3-VL reranker %s loaded on %s (%s)",
                    self.cfg.model_id,
                    self.cfg.device,
                    self.cfg.torch_dtype,
                    extra={"progress": {
                        "phase": "rerank",
                        "status": "running",
                        "title": "Qwen3-VL reranker đã sẵn sàng",
                        "detail": (
                            f"{self.cfg.model_id} trên {self.cfg.device} "
                            f"({self.cfg.torch_dtype})"
                        ),
                    }},
                )
            except Exception as exc:  # noqa: BLE001 - retrieval must degrade gracefully
                self._failed = True
                log.warning(
                    "Qwen3-VL reranker unavailable (%s) - continuing without reranking",
                    exc,
                    extra={"progress": {
                        "phase": "rerank",
                        "status": "warning",
                        "title": "Reranker không khả dụng",
                        "detail": str(exc)[:240],
                    }},
                )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        top_n: int | None = None,
        weight: float | None = None,
    ) -> list[Candidate]:
        ordered = list(candidates)
        if not self.enabled or not query or not ordered or self.kf is None:
            return ordered
        self._load()
        if self._model is None:
            return ordered

        effective_top_n = max(1, int(top_n or self.cfg.top_n))
        effective_weight = max(
            0.0, float(self.cfg.weight if weight is None else weight)
        )
        head, tail = ordered[:effective_top_n], ordered[effective_top_n:]
        items = [(candidate.video_id, candidate.frame_id) for candidate in head]
        image_map = self.kf.batch_get_images(
            items, max_workers=min(10, max(1, len(items)))
        )

        scored: list[tuple[float, Candidate]] = []
        with self._inference_lock:
            for candidate in head:
                image = image_map.get(candidate.key)
                text = self._extract_candidate_text(candidate)
                if image is None and not text:
                    scored.append((candidate.score, candidate))
                    continue
                try:
                    relevance = self._score_pair(query, text=text, image=image)
                except Exception as exc:  # noqa: BLE001 - keep this candidate's recall score
                    log.warning("Qwen3-VL rerank failed on %s: %s", candidate.key, exc)
                    scored.append((candidate.score, candidate))
                    continue
                blended = candidate.score + effective_weight * relevance
                scored.append((
                    blended,
                    candidate.replace(
                        score=blended,
                        extra={
                            **(candidate.extra or {}),
                            "qwen3_vl_rerank_score": relevance,
                            "pre_qwen3_vl_score": candidate.score,
                        },
                    ),
                ))

        scored.sort(key=lambda pair: (-pair[0], pair[1].video_id, pair[1].frame_id))
        merged = [candidate for _, candidate in scored] + tail
        return [candidate.replace(rank=i) for i, candidate in enumerate(merged, start=1)]

    @staticmethod
    def _extract_candidate_text(candidate: Candidate) -> str:
        extra = candidate.extra or {}
        parts: list[str] = []
        for key, label in (
            ("description_matched", "Description"),
            ("ocr_matched", "OCR"),
            ("asr_matched", "ASR"),
        ):
            value = str(extra.get(key) or "").strip()
            if value:
                parts.append(f"{label}: {value}")
        if not parts:
            matched = str(extra.get("matched_text") or "").strip()
            if matched:
                parts.append(matched)
        return " | ".join(parts)

    def _score_pair(self, query: str, *, text: str, image: Any) -> float:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"<Instruct>: {self.cfg.instruction}"},
            {"type": "text", "text": f"<Query>: {query}"},
            {"type": "text", "text": "\n<Document>:"},
        ]
        if image is not None:
            content.append({
                "type": "image",
                "image": image,
                "min_pixels": self.cfg.min_pixels,
                "max_pixels": self.cfg.max_pixels,
            })
        if text:
            content.append({"type": "text", "text": text})
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "Judge whether the Document meets the requirements based "
                        "on the Query and the Instruct provided. The answer can "
                        "only be yes or no."
                    ),
                }],
            },
            {"role": "user", "content": content},
        ]
        inputs = self._tokenize(messages).to(self.cfg.device)
        with self._torch.inference_mode():
            hidden = self._model(**inputs).last_hidden_state[:, -1]
            score = self._torch.sigmoid(self._score_linear(hidden)).squeeze(-1)
        return float(score[0].detach().cpu().item())

    def _tokenize(self, messages: list[dict[str, Any]]) -> Any:
        pairs = [messages]
        rendered = self._processor.apply_chat_template(
            pairs, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = self._process_vision_info(
            pairs,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None
        inputs = self._processor(
            text=rendered,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )
        special_ids = set(self._processor.tokenizer.all_special_ids)
        input_rows = [
            row.tolist() if hasattr(row, "tolist") else list(row)
            for row in inputs["input_ids"]
        ]
        mm_rows_raw = inputs.get("mm_token_type_ids")
        mm_rows = None
        if mm_rows_raw is not None:
            mm_rows = [
                row.tolist() if hasattr(row, "tolist") else list(row)
                for row in mm_rows_raw
            ]

        # Transformers 5.x returns mm_token_type_ids as nested Python lists,
        # while Qwen3VLModel requires a torch.IntTensor. Keep it aligned with
        # input_ids through the same special-token-preserving truncation.
        truncated_ids: list[list[int]] = []
        truncated_mm: list[list[int]] | None = [] if mm_rows is not None else None
        for index, raw_ids in enumerate(input_rows):
            token_ids = [int(token) for token in raw_ids]
            token_types = (
                [int(value) for value in mm_rows[index]] if mm_rows is not None else None
            )
            if len(token_ids) <= self.cfg.max_length:
                truncated_ids.append(token_ids)
                if truncated_mm is not None and token_types is not None:
                    truncated_mm.append(token_types)
                continue

            suffix = token_ids[-5:]
            suffix_types = token_types[-5:] if token_types is not None else None
            body = token_ids[:-5]
            body_types = token_types[:-5] if token_types is not None else None
            special_count = sum(token in special_ids for token in body)
            budget = max(0, self.cfg.max_length - len(suffix) - special_count)
            kept: list[int] = []
            kept_types: list[int] = []
            normal_count = 0
            for position, token in enumerate(body):
                if token in special_ids or normal_count < budget:
                    kept.append(token)
                    if body_types is not None:
                        kept_types.append(body_types[position])
                    if token not in special_ids:
                        normal_count += 1
            truncated_ids.append(kept + suffix)
            if truncated_mm is not None:
                truncated_mm.append(kept_types + (suffix_types or []))

        padded = self._processor.tokenizer.pad(
            {"input_ids": truncated_ids},
            padding=True,
            return_tensors="pt",
        )
        for key, value in padded.items():
            inputs[key] = value
        if truncated_mm is not None:
            width = int(inputs["input_ids"].shape[1])
            padding_side = getattr(self._processor.tokenizer, "padding_side", "right")
            padded_mm: list[list[int]] = []
            for row in truncated_mm:
                padding = [0] * (width - len(row))
                padded_mm.append(padding + row if padding_side == "left" else row + padding)
            inputs["mm_token_type_ids"] = self._torch.tensor(
                padded_mm, dtype=self._torch.long
            )
        return inputs


# Backward compatibility alias
Reranker = BLIP2Reranker
