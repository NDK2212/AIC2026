"""AIC 2026 Interactive Web Server.

High-performance async local server (FastAPI + Uvicorn) providing:
- Web UI SPA static serving
- Real-time 4-path retrieval execution with live telemetry
- On-demand MinIO image streaming & caching
- 25-frame neighbor lookups via Elasticsearch
- Multi-key NVIDIA API management with dynamic failover
- Task-specific flows (KIS, VQA Vision vs Text, TRAKE temporal DP)
- AIC 2026 compliant submission CSV exporter & validator
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from ..clients.elastic import ElasticWrapper
from ..clients.key_pool import GLOBAL_KEY_POOL
from ..clients.llm import LLMClient
from ..clients.minio_client import MinioKeyframeClient
from ..clients.qdrant import QdrantWrapper
from ..clients.vlm import VLMClient
from ..config import Config, load_config
from ..logging_utils import GLOBAL_LOG_BUFFER, get_logger
from ..retrieval.decompose import decompose
from ..retrieval.pipeline import RetrievalPipeline
from ..retrieval.rerank import BGEReranker, BLIP2Reranker, Qwen3VLReranker
from ..retrieval.search_text import TextSearcher
from ..retrieval.search_visual import VisualSearcher
from ..schemas import Candidate, PATH_ASR, PATH_DESCRIPTION, PATH_OCR, PATH_VISUAL
from ..submission.builder import build_rows
from ..tasks.trake import run_trake
from ..tasks.vqa import (
    answer_candidates,
    answer_candidates_vlm,
    propagate_answers,
    retrieve_cross_shot_evidence,
    select_vlm_targets,
    split_query,
)
from ..utils.cache import DiskCache
from ..utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"


def _report_progress(
    phase: str,
    title: str,
    detail: str = "",
    *,
    status: str = "running",
) -> None:
    """Emit machine-readable progress while retaining a useful text log."""
    log.info(
        "%s%s",
        title,
        f": {detail}" if detail else "",
        extra={"progress": {
            "phase": phase,
            "status": status,
            "title": title,
            "detail": detail,
        }},
    )


class WebApp:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path
        self.cfg: Config = load_config(config_path)
        self.cache = DiskCache(self.cfg.cache.dir, enabled=self.cfg.cache.enabled)
        self.minio = MinioKeyframeClient(self.cfg.minio)
        self.es = ElasticWrapper(self.cfg.elasticsearch)
        self.qdrant = QdrantWrapper(self.cfg.qdrant)
        self.llm = LLMClient(self.cfg.llm, self.cache)
        self.vlm = VLMClient(self.cfg.vlm, self.cache)
        self.kf = KeyframeIndex(
            root=self.cfg.keyframes.root,
            map_dir=self.cfg.keyframes.map_dir,
            metadata_dir=self.cfg.keyframes.metadata_dir,
            image_glob=self.cfg.keyframes.image_glob,
            cache=self.cache,
            minio=self.minio,
            cache_dir=self.cfg.minio.cache_dir,
        )
        self.text_searcher = TextSearcher(self.es)
        self.visual_searcher = VisualSearcher(self.cfg, self.qdrant, self.cache)
        self.pipeline = RetrievalPipeline(
            cfg=self.cfg,
            llm=self.llm,
            text_searcher=self.text_searcher,
            visual_searcher=self.visual_searcher,
            blip2_reranker=BLIP2Reranker(self.cfg.rerank.blip2, self.kf) if self.cfg.rerank.blip2.enabled else None,
            bge_reranker=BGEReranker(self.cfg.rerank.bge) if self.cfg.rerank.bge.enabled else None,
            qwen3_vl_reranker=(
                Qwen3VLReranker(self.cfg.rerank.qwen3_vl, self.kf)
                if self.cfg.rerank.qwen3_vl.enabled else None
            ),
        )

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------
    async def handle_index(self) -> Response:
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            return Response(content="index.html not found", status_code=404)
        return FileResponse(index_file)

    async def handle_status(self) -> JSONResponse:
        """Health check for Elasticsearch, Qdrant, MinIO, LLM keys."""
        loop = asyncio.get_running_loop()
        es_ok = await loop.run_in_executor(None, lambda: self.es.client.ping())
        qd_ok = await loop.run_in_executor(None, lambda: self.qdrant.client.collection_exists(self.cfg.qdrant.collection))
        keys = GLOBAL_KEY_POOL.get_keys()
        return JSONResponse({
            "status": "healthy",
            "elasticsearch": {"connected": bool(es_ok), "index": self.cfg.elasticsearch.index},
            "qdrant": {"connected": bool(qd_ok), "collection": self.cfg.qdrant.collection},
            "minio": {"enabled": self.cfg.minio.enabled, "endpoint": self.cfg.minio.endpoint, "bucket": self.cfg.minio.bucket},
            "keys_count": len(keys),
            "keys_masked": [k[:8] + "..." + k[-4:] if len(k) > 12 else "key" for k in keys],
        })

    async def handle_get_config(self) -> JSONResponse:
        """Return current configuration & tunable parameters with defaults."""
        return JSONResponse({
            "retrieval_paths": {
                "ocr": {"enabled": True, "weight": self.cfg.fusion.weights.get("ocr", 1.0)},
                "asr": {"enabled": True, "weight": self.cfg.fusion.weights.get("asr", 1.0)},
                "description": {"enabled": True, "weight": self.cfg.fusion.weights.get("description", 1.2)},
                "visual": {"enabled": True, "weight": self.cfg.fusion.weights.get("visual", 1.5)},
            },
            "fusion": {
                "method": self.cfg.fusion.method,
                "rrf_k": self.cfg.fusion.rrf_k,
                "adaptive": self.cfg.fusion.adaptive,
                "adaptive_floor": self.cfg.fusion.adaptive_floor,
            },
            "rerank": {
                "qwen3_vl": {
                    "enabled": self.cfg.rerank.qwen3_vl.enabled,
                    "top_n": self.cfg.rerank.qwen3_vl.top_n,
                    "weight": self.cfg.rerank.qwen3_vl.weight,
                    "device": self.cfg.rerank.qwen3_vl.device,
                },
                "blip2": {
                    "enabled": self.cfg.rerank.blip2.enabled,
                    "top_n": self.cfg.rerank.blip2.top_n,
                    "weight": self.cfg.rerank.blip2.weight,
                },
                "bge": {
                    "enabled": self.cfg.rerank.bge.enabled,
                    "top_n": self.cfg.rerank.bge.top_n,
                    "weight": self.cfg.rerank.bge.weight,
                },
            },
            "visual_encoders": {
                "siglip": self.cfg.embedding.siglip.enabled,
                "beit3": self.cfg.embedding.beit3.enabled,
                "qwen": self.cfg.embedding.qwen.enabled,
            },
            "submission": {
                "max_rows": self.cfg.submission.max_rows,
                "top_diverse": self.cfg.submission.top_diverse,
                "head_max_per_video": self.cfg.submission.head_max_per_video,
                "shot_window": self.cfg.submission.shot_window,
                "neighbor_expansion": self.cfg.submission.neighbor_expansion.enabled,
            },
            "vqa": {
                "vlm_top_n": self.cfg.vqa.vlm_top_n,
                "vlm_images_per_video": self.cfg.vqa.vlm_images_per_video,
                "propagate": self.cfg.vqa.propagate,
                "vqa_mode": self.cfg.vqa.mode,
            },
            "trake": {
                "per_step_topk": self.cfg.trake.per_step_topk,
                "step_weight_visual": self.cfg.trake.step_weights.get("visual", 0.6),
                "step_weight_desc": self.cfg.trake.step_weights.get("description", 0.4),
                "coverage_bonus": self.cfg.trake.coverage_bonus,
                "miss_penalty": self.cfg.trake.miss_penalty,
                "allow_fill": self.cfg.trake.allow_fill,
            },
        })

    async def handle_get_keys(self) -> JSONResponse:
        keys = GLOBAL_KEY_POOL.get_keys()
        return JSONResponse({
            "count": len(keys),
            "keys": keys,
            "masked": [k[:8] + "..." + k[-4:] if len(k) > 12 else "key" for k in keys],
        })

    async def handle_set_keys(self, request: Request) -> JSONResponse:
        try:
            data = await request.json()
            keys_input = data.get("keys", [])
            if isinstance(keys_input, str):
                # Split by comma or newline
                raw_keys = [k.strip() for line in keys_input.split("\n") for k in line.split(",") if k.strip()]
            elif isinstance(keys_input, list):
                raw_keys = [str(k).strip() for k in keys_input if str(k).strip()]
            else:
                return JSONResponse({"error": "Invalid format for keys"}, status_code=400)

            GLOBAL_KEY_POOL.set_keys(raw_keys)
            return JSONResponse({
                "success": True,
                "count": len(GLOBAL_KEY_POOL),
                "keys": GLOBAL_KEY_POOL.get_keys(),
            })
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def handle_image(self, video_id: str, frame_id: int) -> Response:
        """Stream keyframe image directly (cache -> MinIO -> JPEG bytes)."""
        video_id = video_id.strip()

        loop = asyncio.get_running_loop()
        # Fetch from keyframe index / MinIO
        img = await loop.run_in_executor(
            None, lambda: self.kf.get_image(video_id, frame_id)
        )
        if img is None:
            # Fallback 1x1 gray pixel placeholder
            buf = io.BytesIO()
            placeholder = Image.new("RGB", (320, 180), color=(30, 35, 45))
            placeholder.save(buf, format="JPEG", quality=80)
            return Response(content=buf.getvalue(), media_type="image/jpeg")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    async def handle_neighbors(self, video_id: str, frame_id: int) -> JSONResponse:
        """Lookup 25 neighbor frames centered at frame_id in video_id via Elasticsearch."""
        video_id = video_id.strip()

        loop = asyncio.get_running_loop()

        def _fetch_neighbors() -> list[dict[str, Any]]:
            v_field = self.cfg.elasticsearch.fields["video_id"]
            f_field = self.cfg.elasticsearch.fields["frame_id"]
            desc_field = self.cfg.elasticsearch.fields["description"]
            ocr_field = self.cfg.elasticsearch.fields["ocr"]

            res = self.es.client.search(
                index=self.cfg.elasticsearch.index,
                query={"bool": {"filter": [{"term": {v_field: video_id}}]}},
                sort=[{f_field: "asc"}],
                size=1000,
                source=[f_field, desc_field, ocr_field],
            )
            hits = res["hits"]["hits"]
            all_frames: list[int] = []
            meta_map: dict[int, dict[str, str]] = {}
            for h in hits:
                src = h["_source"]
                if f_field in src:
                    fid = int(src[f_field])
                    all_frames.append(fid)
                    meta_map[fid] = {
                        "description": str(src.get(desc_field) or ""),
                        "ocr": str(src.get(ocr_field) or ""),
                    }

            if not all_frames:
                # Synthetic window fallback
                return [{"frame_id": frame_id, "image_url": f"/api/image/{video_id}/{frame_id}", "is_target": True}]

            if frame_id in all_frames:
                idx = all_frames.index(frame_id)
            else:
                # Closest
                closest = min(all_frames, key=lambda x: abs(x - frame_id))
                idx = all_frames.index(closest)

            start = max(0, idx - 12)
            end = min(len(all_frames), idx + 13)
            slice_frames = all_frames[start:end]

            items = []
            for f in slice_frames:
                meta = meta_map.get(f, {})
                items.append({
                    "video_id": video_id,
                    "frame_id": f,
                    "image_url": f"/api/image/{video_id}/{f}",
                    "is_target": (f == frame_id),
                    "description": meta.get("description", ""),
                    "ocr": meta.get("ocr", ""),
                })
            return items

        try:
            neighbors = await loop.run_in_executor(None, _fetch_neighbors)
            return JSONResponse({
                "video_id": video_id,
                "target_frame_id": frame_id,
                "count": len(neighbors),
                "neighbors": neighbors,
            })
        except Exception as exc:
            log.error("Failed to fetch neighbors: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def handle_search(self, request: Request) -> JSONResponse:
        """Full pipeline execution endpoint for KIS, VQA, TRAKE with custom parameter overrides."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        task = str(body.get("task", "kis")).strip().lower()
        query = str(body.get("query", "")).strip()
        if not query:
            return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

        use_cache = bool(body.get("use_cache", True))
        params = body.get("params", {})
        vqa_mode = str(body.get("vqa_mode", self.cfg.vqa.mode)).lower()

        loop = asyncio.get_running_loop()
        started = time.perf_counter()

        def _execute() -> dict[str, Any]:
            # Per-request copy: task controls must not leak into concurrent runs.
            cfg = copy.deepcopy(self.cfg)

            # Path weights & enables
            paths_cfg = params.get("retrieval_paths", {})
            enabled_paths = []
            active_weights = {}
            for p in (PATH_OCR, PATH_ASR, PATH_DESCRIPTION, PATH_VISUAL):
                p_info = paths_cfg.get(p, {})
                if p_info.get("enabled", True):
                    enabled_paths.append(p)
                active_weights[p] = float(p_info.get("weight", cfg.fusion.weights.get(p, 1.0)))

            fusion_cfg = params.get("fusion", {})
            fusion_method = str(fusion_cfg.get("method", cfg.fusion.method))
            fusion_adaptive = bool(fusion_cfg.get("adaptive", cfg.fusion.adaptive))
            rrf_k = max(1, int(fusion_cfg.get("rrf_k", cfg.fusion.rrf_k)))

            # Reranker settings
            rerank_cfg = params.get("rerank", {})
            qwen_enabled = rerank_cfg.get("qwen3_vl", {}).get("enabled", cfg.rerank.qwen3_vl.enabled)
            qwen_top_n = int(rerank_cfg.get("qwen3_vl", {}).get("top_n", cfg.rerank.qwen3_vl.top_n))
            qwen_weight = float(rerank_cfg.get("qwen3_vl", {}).get("weight", cfg.rerank.qwen3_vl.weight))
            # Task and submission controls use the request-local config copy.
            submission_cfg = params.get("submission", {})
            cfg.submission.shot_window = max(
                1, int(submission_cfg.get("shot_window", cfg.submission.shot_window))
            )
            cfg.submission.top_diverse = max(
                0, int(submission_cfg.get("top_diverse", cfg.submission.top_diverse))
            )
            cfg.submission.head_max_per_video = max(
                1, int(submission_cfg.get(
                    "head_max_per_video", cfg.submission.head_max_per_video
                ))
            )
            cfg.submission.neighbor_expansion.enabled = bool(submission_cfg.get(
                "neighbor_expansion", cfg.submission.neighbor_expansion.enabled
            ))

            vqa_cfg = params.get("vqa", {})
            cfg.vqa.vlm_top_n = max(1, int(vqa_cfg.get("vlm_top_n", cfg.vqa.vlm_top_n)))
            cfg.vqa.vlm_images_per_video = max(1, int(vqa_cfg.get(
                "vlm_images_per_video", cfg.vqa.vlm_images_per_video
            )))
            cfg.vqa.llm_top_n = cfg.vqa.vlm_top_n
            cfg.vqa.propagate = bool(vqa_cfg.get("propagate", cfg.vqa.propagate))

            trake_cfg = params.get("trake", {})
            cfg.trake.per_step_topk = max(
                1, int(trake_cfg.get("per_step_topk", cfg.trake.per_step_topk))
            )
            cfg.trake.coverage_bonus = float(trake_cfg.get(
                "coverage_bonus", cfg.trake.coverage_bonus
            ))
            cfg.trake.miss_penalty = float(trake_cfg.get(
                "miss_penalty", cfg.trake.miss_penalty
            ))

            pipeline_options = {
                "enabled_paths": enabled_paths,
                "fusion_method": fusion_method,
                "fusion_weights": active_weights,
                "fusion_adaptive": fusion_adaptive,
                "rrf_k": rrf_k,
                "qwen_enabled": bool(qwen_enabled),
                "qwen_top_n": max(1, qwen_top_n),
                "qwen_weight": max(0.0, qwen_weight),
            }

            # Execution trace collector
            trace: dict[str, Any] = {
                "task": task,
                "query": query,
                "stages": [],
            }
            _report_progress("start", "Đã nhận truy vấn", query[:180])

            # ==========================================================
            # Task: KIS (Known-Item Search)
            # ==========================================================
            if task == "kis":
                # 1. Query Decomposition
                _report_progress(
                    "decompose", "Đang tách truy vấn", "Visual, Description, OCR, ASR"
                )
                t0 = time.perf_counter()
                dec = decompose(
                    self.llm,
                    query,
                    adaptive_floor=cfg.fusion.adaptive_floor,
                    default_weights=active_weights,
                    use_cache=use_cache,
                )
                t_dec = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "Decomposition",
                    "latency_ms": round(t_dec * 1000, 1),
                    "details": dec.to_dict(),
                })

                # 2. Parallel 4-path retrieval
                _report_progress(
                    "retrieval", "Đang truy hồi song song", "Các path đang bật"
                )
                t0 = time.perf_counter()
                raw_cands, dec_res = self.pipeline.run(
                    query,
                    topk=cfg.submission.max_rows * 3,
                    decompose_result=dec,
                    write_trace=False,
                    use_cache=use_cache,
                    **pipeline_options,
                )
                t_ret = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "Parallel Retrieval & Fusion",
                    "latency_ms": round(t_ret * 1000, 1),
                    "total_fused": len(raw_cands),
                })

                # 3. Submission formatting (shot diversity + neighbor expansion)
                t0 = time.perf_counter()
                rows = build_rows(raw_cands, cfg.submission, self.kf)
                t_sub = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "Postprocessing & Shot Diversity",
                    "latency_ms": round(t_sub * 1000, 1),
                    "output_rows": len(rows),
                })

                # Prepare results
                candidates_out = []
                for rank, c in enumerate(rows[:cfg.submission.max_rows], 1):
                    candidates_out.append({
                        "rank": rank,
                        "video_id": c.video_id,
                        "frame_id": c.frame_id,
                        "score": round(c.score, 4),
                        "source": c.source,
                        "image_url": f"/api/image/{c.video_id}/{c.frame_id}",
                        "extra": c.extra,
                    })

                _report_progress(
                    "complete",
                    "Pipeline hoàn tất",
                    f"{len(candidates_out)} kết quả",
                    status="done",
                )

                return {
                    "task": "kis",
                    "query": query,
                    "total_results": len(candidates_out),
                    "candidates": candidates_out,
                    "decomposition": dec.to_dict(),
                    "trace": trace,
                }

            # ==========================================================
            # Task: VQA (Video Question Answering)
            # ==========================================================
            elif task == "vqa":
                # 1. Split Scene Description vs Question
                _report_progress("split", "Đang tách cảnh và câu hỏi")
                t0 = time.perf_counter()
                split = split_query(self.llm, query, use_cache=use_cache)
                t_split = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "Q&A Query Split",
                    "latency_ms": round(t_split * 1000, 1),
                    "scene_description": split.scene_description,
                    "question": split.question,
                    "question_en": split.question_en,
                    "expected_answer_type": split.expected_answer_type,
                    "evidence_query": split.evidence_query,
                })

                # 2. Retrieve candidates based on scene_description
                _report_progress(
                    "retrieval",
                    "Đang tìm video theo cảnh hành động",
                    split.scene_description[:180],
                )
                t0 = time.perf_counter()
                cands, dec = self.pipeline.run(
                    split.scene_description,
                    topk=cfg.submission.max_rows * 3,
                    write_trace=False,
                    use_cache=use_cache,
                    **pipeline_options,
                )
                rows = build_rows(cands, cfg.submission, self.kf)
                t_ret = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "Scene Retrieval & Fusion",
                    "latency_ms": round(t_ret * 1000, 1),
                    "fused_count": len(cands),
                    "top_rows": len(rows),
                })

                # 3. Representative frame selection
                vlm_targets = select_vlm_targets(
                    rows,
                    cfg.submission.shot_window,
                    params.get("vqa", {}).get("vlm_top_n", cfg.vqa.vlm_top_n),
                )
                _report_progress(
                    "evidence",
                    "Đang tìm evidence trong cùng video",
                    split.evidence_query[:180],
                )
                evidence_targets = retrieve_cross_shot_evidence(
                    self.pipeline, split, rows, cfg, use_cache=use_cache,
                    pipeline_options=pipeline_options,
                )
                seen_target_keys = {candidate.key for candidate in vlm_targets}
                vlm_targets.extend(
                    candidate for candidate in evidence_targets
                    if candidate.key not in seen_target_keys
                )

                # 4. Answering: VLM with images OR LLM with Text Metadata Only
                t0 = time.perf_counter()
                answers: dict[tuple[str, int], str] = {}
                if vqa_mode == "vision":
                    _report_progress(
                        "answer",
                        "Kimi K3 đang đọc ảnh và metadata",
                        f"Tối đa {cfg.vqa.vlm_images_per_video} ảnh mỗi video",
                    )
                    answers = answer_candidates_vlm(
                        self.vlm,
                        vlm_targets,
                        split.question,
                        cfg,
                        self.kf,
                        text_searcher=self.text_searcher,
                        use_cache=use_cache,
                    )
                else:
                    # Text & Metadata Only Mode (LLM + ES dense description, OCR, ASR)
                    _report_progress(
                        "answer", "LLM đang đọc Description, OCR và ASR"
                    )
                    answers = answer_candidates(
                        self.llm,
                        vlm_targets,
                        split.question,
                        cfg,
                        text_searcher=self.text_searcher,
                        use_cache=use_cache,
                    )
                t_ans = time.perf_counter() - t0
                trace["stages"].append({
                    "name": f"Answering ({vqa_mode.upper()} Mode)",
                    "latency_ms": round(t_ans * 1000, 1),
                    "targets_answered": len(answers),
                    "cross_shot_evidence_frames": len(evidence_targets),
                    "evidence_query": split.evidence_query,
                    "evidence_frames": [
                        {
                            "video_id": candidate.video_id,
                            "frame_id": candidate.frame_id,
                            "score": round(candidate.score, 4),
                        }
                        for candidate in evidence_targets
                    ],
                })

                # 5. Propagate answers
                vqa_rows = propagate_answers(rows, answers, cfg)

                candidates_out = []
                evidence_by_video: dict[str, list[dict[str, Any]]] = {}
                for candidate in evidence_targets:
                    evidence_by_video.setdefault(candidate.video_id, []).append({
                        "video_id": candidate.video_id,
                        "frame_id": candidate.frame_id,
                        "score": round(candidate.score, 4),
                        "image_url": f"/api/image/{candidate.video_id}/{candidate.frame_id}",
                    })
                evidence_attached: set[str] = set()
                for rank, (vid, fid, ans) in enumerate(vqa_rows[:cfg.submission.max_rows], 1):
                    video_evidence = []
                    if vid not in evidence_attached:
                        video_evidence = evidence_by_video.get(vid, [])[:4]
                        evidence_attached.add(vid)
                    candidates_out.append({
                        "rank": rank,
                        "video_id": vid,
                        "frame_id": fid,
                        "answer": ans,
                        "is_target": (vid, fid) in answers,
                        "image_url": f"/api/image/{vid}/{fid}",
                        "evidence_frames": video_evidence,
                    })

                _report_progress(
                    "complete",
                    "VQA hoàn tất",
                    f"{len(candidates_out)} kết quả",
                    status="done",
                )

                return {
                    "task": "vqa",
                    "query": query,
                    "vqa_mode": vqa_mode,
                    "split": {
                        "scene": split.scene_description,
                        "question": split.question,
                        "type": split.expected_answer_type,
                        "evidence_query": split.evidence_query,
                    },
                    "evidence_candidates": [
                        item
                        for values in evidence_by_video.values()
                        for item in values
                    ],
                    "total_results": len(candidates_out),
                    "candidates": candidates_out,
                    "trace": trace,
                }

            # ==========================================================
            # Task: TRAKE (Temporal Action & Key Event)
            # ==========================================================
            elif task == "trake":
                # Run complete TRAKE task: plan + step retrieval + DP alignment
                t0 = time.perf_counter()
                trake_rows, plan = run_trake(
                    query, self.pipeline, self.kf, cfg, use_cache=use_cache,
                    pipeline_options=pipeline_options,
                )
                t_total_trake = time.perf_counter() - t0
                trace["stages"].append({
                    "name": "TraKE Pipeline Execution",
                    "latency_ms": round(t_total_trake * 1000, 1),
                    "events_count": plan.num_events,
                    "anchor_index": plan.anchor_index,
                    "steps": [s.to_dict() for s in plan.steps],
                    "sequences_found": len(trake_rows),
                })

                # Prepare TRAKE sequence output
                sequences_out = []
                for rank, row in enumerate(trake_rows[:cfg.submission.max_rows], 1):
                    vid = str(row[0])
                    frames = [int(f) for f in row[1:]]
                    step_items = []
                    for s_idx, fid in enumerate(frames, 1):
                        step_items.append({
                            "step_index": s_idx,
                            "frame_id": fid,
                            "image_url": f"/api/image/{vid}/{fid}",
                        })
                    sequences_out.append({
                        "rank": rank,
                        "video_id": vid,
                        "score": round(1.0 / rank, 4),
                        "frames": frames,
                        "steps": step_items,
                    })

                return {
                    "task": "trake",
                    "query": query,
                    "plan": {
                        "num_events": plan.num_events,
                        "anchor_index": plan.anchor_index,
                        "steps": [s.to_dict() for s in plan.steps],
                    },
                    "total_results": len(sequences_out),
                    "sequences": sequences_out,
                    "trace": trace,
                }

            else:
                return {"error": f"Unknown task {task!r}"}

        try:
            result = await loop.run_in_executor(None, _execute)
            total_duration = time.perf_counter() - started
            result["elapsed_total_s"] = round(total_duration, 2)
            return JSONResponse(result)
        except Exception as exc:
            log.error("Search execution failed: %s", exc, exc_info=True)
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def handle_export_csv(self, request: Request) -> Response:
        """Export and download validated AIC 2026 submission CSV file."""
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        task = str(data.get("task", "kis")).strip().lower()
        items = data.get("items", [])
        output_filename = str(data.get("filename", f"submission_{task}.csv"))

        if not items:
            return JSONResponse({"error": "No items to export"}, status_code=400)

        csv_lines: list[str] = []
        if task == "kis":
            # 2 columns: video_id,frame_id (no header)
            for it in items[:100]:
                vid = str(it.get("video_id", "")).strip()
                fid = int(it.get("frame_id", 0))
                csv_lines.append(f"{vid},{fid}")

        elif task == "vqa":
            # 3 columns: video_id,frame_id,answer (no header)
            for it in items[:100]:
                vid = str(it.get("video_id", "")).strip()
                fid = int(it.get("frame_id", 0))
                ans = str(it.get("answer", "unknown")).replace("\n", " ").strip()
                if "," in ans or '"' in ans:
                    ans = f'"{ans.replace(chr(34), chr(34)+chr(34))}"'
                csv_lines.append(f"{vid},{fid},{ans}")

        elif task == "trake":
            # 1+K columns: video_id,f1,f2,...,fK (no header)
            for it in items[:100]:
                vid = str(it.get("video_id", "")).strip()
                frames = it.get("frames", [])
                f_str = ",".join(str(int(f)) for f in frames)
                csv_lines.append(f"{vid},{f_str}")

        csv_content = "\n".join(csv_lines) + "\n"

        return Response(
            content=csv_content.encode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
            },
        )

    async def handle_get_logs(self) -> JSONResponse:
        """Return recent log entries."""
        return JSONResponse({"logs": GLOBAL_LOG_BUFFER.get_recent(1000)})

    async def handle_stream_logs(self, request: Request) -> StreamingResponse:
        """Server-Sent Events (SSE) stream for real-time console logs."""
        try:
            tail = max(0, min(1000, int(request.query_params.get("tail", "100"))))
        except ValueError:
            tail = 100

        async def event_stream() -> AsyncIterator[str]:
            recent = GLOBAL_LOG_BUFFER.get_recent(tail) if tail else []
            for entry in recent:
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"

            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            GLOBAL_LOG_BUFFER.subscribe(q, loop)
            try:
                while True:
                    entry = await q.get()
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                return
            finally:
                GLOBAL_LOG_BUFFER.unsubscribe(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    app_instance = WebApp(config_path)
    app = FastAPI(
        title="AIC 2026 Video Retrieval API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.services = app_instance
    app.add_api_route("/", app_instance.handle_index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/api/status", app_instance.handle_status, methods=["GET"])
    app.add_api_route("/api/config", app_instance.handle_get_config, methods=["GET"])
    app.add_api_route("/api/keys", app_instance.handle_get_keys, methods=["GET"])
    app.add_api_route("/api/keys", app_instance.handle_set_keys, methods=["POST"])
    app.add_api_route("/api/logs", app_instance.handle_get_logs, methods=["GET"])
    app.add_api_route("/api/logs/stream", app_instance.handle_stream_logs, methods=["GET"])
    app.add_api_route("/api/image/{video_id}/{frame_id}", app_instance.handle_image, methods=["GET"])
    app.add_api_route("/api/neighbors/{video_id}/{frame_id}", app_instance.handle_neighbors, methods=["GET"])
    app.add_api_route("/api/search", app_instance.handle_search, methods=["POST"])
    app.add_api_route("/api/export_csv", app_instance.handle_export_csv, methods=["POST"])

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def run_server(host: str = "127.0.0.1", port: int = 7860, config_path: str = "config/config.yaml") -> None:
    import sys
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    app = create_app(config_path)
    banner = f"""
=======================================================
  AIC 2026 Interactive Web UI running at:
  http://{host}:{port}/
=======================================================
"""
    print(banner, flush=True)
    log.info("Server listening on http://%s:%d/", host, port)
    uvicorn.run(app, host=host, port=port, access_log=False)


if __name__ == "__main__":
    run_server()
