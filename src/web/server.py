"""AIC 2026 Interactive Web Server.

High-performance async local server (aiohttp.web) providing:
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
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image

from ..clients.elastic import ElasticWrapper
from ..clients.key_pool import GLOBAL_KEY_POOL
from ..clients.llm import LLMClient
from ..clients.minio_client import MinioKeyframeClient
from ..clients.qdrant import QdrantWrapper
from ..clients.vlm import VLMClient
from ..config import Config, load_config
from ..logging_utils import GLOBAL_LOG_BUFFER, get_logger
from ..retrieval.decompose import decompose
from ..retrieval.fusion import fuse
from ..retrieval.pipeline import RetrievalPipeline
from ..retrieval.rerank import BGEReranker, BLIP2Reranker
from ..retrieval.search_text import TextSearcher
from ..retrieval.search_visual import VisualSearcher
from ..schemas import Candidate, PATH_ASR, PATH_DESCRIPTION, PATH_OCR, PATH_VISUAL
from ..submission.builder import build_rows
from ..submission.validator import validate_file
from ..submission.writer import write_kis, write_qa, write_trake
from ..tasks.kis import run_kis
from ..tasks.trake import plan_trake, run_trake
from ..tasks.vqa import answer_candidates, propagate_answers, select_vlm_targets, split_query
from ..utils.cache import DiskCache
from ..utils.keyframe_index import KeyframeIndex

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


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
        )

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------
    async def handle_index(self, request: web.Request) -> web.FileResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            return web.Response(text="index.html not found", status=404)
        return web.FileResponse(index_file)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Health check for Elasticsearch, Qdrant, MinIO, LLM keys."""
        loop = asyncio.get_running_loop()
        es_ok = await loop.run_in_executor(None, lambda: self.es.client.ping())
        qd_ok = await loop.run_in_executor(None, lambda: self.qdrant.client.collection_exists(self.cfg.qdrant.collection))
        keys = GLOBAL_KEY_POOL.get_keys()
        return web.json_response({
            "status": "healthy",
            "elasticsearch": {"connected": bool(es_ok), "index": self.cfg.elasticsearch.index},
            "qdrant": {"connected": bool(qd_ok), "collection": self.cfg.qdrant.collection},
            "minio": {"enabled": self.cfg.minio.enabled, "endpoint": self.cfg.minio.endpoint, "bucket": self.cfg.minio.bucket},
            "keys_count": len(keys),
            "keys_masked": [k[:8] + "..." + k[-4:] if len(k) > 12 else "key" for k in keys],
        })

    async def handle_get_config(self, request: web.Request) -> web.Response:
        """Return current configuration & tunable parameters with defaults."""
        return web.json_response({
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
                "propagate": self.cfg.vqa.propagate,
                "vqa_mode": "text_only",  # "vision" | "text_only"
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

    async def handle_get_keys(self, request: web.Request) -> web.Response:
        keys = GLOBAL_KEY_POOL.get_keys()
        return web.json_response({
            "count": len(keys),
            "keys": keys,
            "masked": [k[:8] + "..." + k[-4:] if len(k) > 12 else "key" for k in keys],
        })

    async def handle_set_keys(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            keys_input = data.get("keys", [])
            if isinstance(keys_input, str):
                # Split by comma or newline
                raw_keys = [k.strip() for line in keys_input.split("\n") for k in line.split(",") if k.strip()]
            elif isinstance(keys_input, list):
                raw_keys = [str(k).strip() for k in keys_input if str(k).strip()]
            else:
                return web.json_response({"error": "Invalid format for keys"}, status=400)

            GLOBAL_KEY_POOL.set_keys(raw_keys)
            return web.json_response({
                "success": True,
                "count": len(GLOBAL_KEY_POOL),
                "keys": GLOBAL_KEY_POOL.get_keys(),
            })
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_image(self, request: web.Request) -> web.Response:
        """Stream keyframe image directly (cache -> MinIO -> JPEG bytes)."""
        video_id = request.match_info["video_id"].strip()
        frame_id_str = request.match_info["frame_id"].strip()
        try:
            frame_id = int(frame_id_str)
        except ValueError:
            return web.Response(text="Invalid frame_id", status=400)

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
            return web.Response(body=buf.getvalue(), content_type="image/jpeg")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return web.Response(
            body=buf.getvalue(),
            content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    async def handle_neighbors(self, request: web.Request) -> web.Response:
        """Lookup 25 neighbor frames centered at frame_id in video_id via Elasticsearch."""
        video_id = request.match_info["video_id"].strip()
        try:
            frame_id = int(request.match_info["frame_id"].strip())
        except ValueError:
            return web.Response(text="Invalid frame_id", status=400)

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
            return web.json_response({
                "video_id": video_id,
                "target_frame_id": frame_id,
                "count": len(neighbors),
                "neighbors": neighbors,
            })
        except Exception as exc:
            log.error("Failed to fetch neighbors: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_search(self, request: web.Request) -> web.Response:
        """Full pipeline execution endpoint for KIS, VQA, TRAKE with custom parameter overrides."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        task = str(body.get("task", "kis")).strip().lower()
        query = str(body.get("query", "")).strip()
        if not query:
            return web.json_response({"error": "Query cannot be empty"}, status=400)

        use_cache = bool(body.get("use_cache", True))
        params = body.get("params", {})
        vqa_mode = str(body.get("vqa_mode", "text_only")).lower()  # "vision" | "text_only"

        loop = asyncio.get_running_loop()
        started = time.perf_counter()

        def _execute() -> dict[str, Any]:
            # Apply dynamic parameter overrides to config
            cfg = self.cfg

            # Path weights & enables
            paths_cfg = params.get("retrieval_paths", {})
            enabled_paths = []
            active_weights = {}
            for p in (PATH_OCR, PATH_ASR, PATH_DESCRIPTION, PATH_VISUAL):
                p_info = paths_cfg.get(p, {})
                if p_info.get("enabled", True):
                    enabled_paths.append(p)
                active_weights[p] = float(p_info.get("weight", cfg.fusion.weights.get(p, 1.0)))

            # Reranker settings
            rerank_cfg = params.get("rerank", {})
            blip_enabled = rerank_cfg.get("blip2", {}).get("enabled", cfg.rerank.blip2.enabled)
            blip_top_n = int(rerank_cfg.get("blip2", {}).get("top_n", cfg.rerank.blip2.top_n))
            blip_weight = float(rerank_cfg.get("blip2", {}).get("weight", cfg.rerank.blip2.weight))

            bge_enabled = rerank_cfg.get("bge", {}).get("enabled", cfg.rerank.bge.enabled)
            bge_top_n = int(rerank_cfg.get("bge", {}).get("top_n", cfg.rerank.bge.top_n))
            bge_weight = float(rerank_cfg.get("bge", {}).get("weight", cfg.rerank.bge.weight))

            # Visual encoders in Qdrant
            enc_cfg = params.get("visual_encoders", {})
            siglip_on = enc_cfg.get("siglip", cfg.embedding.siglip.enabled)
            beit3_on = enc_cfg.get("beit3", cfg.embedding.beit3.enabled)
            qwen_on = enc_cfg.get("qwen", cfg.embedding.qwen.enabled)

            # Execution trace collector
            trace: dict[str, Any] = {
                "task": task,
                "query": query,
                "stages": [],
            }

            # ==========================================================
            # Task: KIS (Known-Item Search)
            # ==========================================================
            if task == "kis":
                # 1. Query Decomposition
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
                t0 = time.perf_counter()
                raw_cands, dec_res = self.pipeline.run(
                    query,
                    topk=cfg.submission.max_rows * 3,
                    decompose_result=dec,
                    write_trace=False,
                    use_cache=use_cache,
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
                })

                # 2. Retrieve candidates based on scene_description
                t0 = time.perf_counter()
                cands, dec = self.pipeline.run(
                    split.scene_description,
                    topk=cfg.submission.max_rows * 3,
                    write_trace=False,
                    use_cache=use_cache,
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

                # 4. Answering: VLM with images OR LLM with Text Metadata Only
                t0 = time.perf_counter()
                answers: dict[tuple[str, int], str] = {}
                if vqa_mode == "vision":
                    # Vision-Language Mode (Llama-3.2-Vision / VLM)
                    for target in vlm_targets:
                        # Fetch image
                        img_path = self.kf.cache_dir / target.video_id / f"{target.frame_id}.jpg"
                        if not img_path.is_file():
                            img = self.kf.get(target.video_id, target.frame_id)
                        if img_path.is_file():
                            try:
                                ans = self.vlm.ask(
                                    img_path,
                                    "You are an expert video visual QA assistant. Answer accurately in concise Vietnamese or English matching question.",
                                    split.question,
                                    use_cache=use_cache,
                                )
                                answers[target.key] = ans
                            except Exception as e:
                                log.warning("VLM answering failed on %s: %s", target.key, e)
                else:
                    # Text & Metadata Only Mode (LLM + ES dense description, OCR, ASR)
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
                })

                # 5. Propagate answers
                vqa_rows = propagate_answers(rows, answers, cfg)

                candidates_out = []
                for rank, (vid, fid, ans) in enumerate(vqa_rows[:cfg.submission.max_rows], 1):
                    candidates_out.append({
                        "rank": rank,
                        "video_id": vid,
                        "frame_id": fid,
                        "answer": ans,
                        "is_target": (vid, fid) in answers,
                        "image_url": f"/api/image/{vid}/{fid}",
                    })

                return {
                    "task": "vqa",
                    "query": query,
                    "vqa_mode": vqa_mode,
                    "split": {
                        "scene": split.scene_description,
                        "question": split.question,
                        "type": split.expected_answer_type,
                    },
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
                trake_rows, plan = run_trake(query, self.pipeline, self.kf, cfg, use_cache=use_cache)
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
            return web.json_response(result)
        except Exception as exc:
            log.error("Search execution failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_export_csv(self, request: web.Request) -> web.Response:
        """Export and download validated AIC 2026 submission CSV file."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        task = str(data.get("task", "kis")).strip().lower()
        items = data.get("items", [])
        output_filename = str(data.get("filename", f"submission_{task}.csv"))

        if not items:
            return web.json_response({"error": "No items to export"}, status=400)

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

        return web.Response(
            body=csv_content.encode("utf-8"),
            content_type="text/csv",
            charset="utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
            },
        )

    async def handle_get_logs(self, request: web.Request) -> web.Response:
        """Return recent log entries."""
        with GLOBAL_LOG_BUFFER.lock:
            logs = list(GLOBAL_LOG_BUFFER.records)
        return web.json_response({"logs": logs})

    async def handle_stream_logs(self, request: web.Request) -> web.StreamResponse:
        """Server-Sent Events (SSE) stream for real-time console logs."""
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Send existing recent logs first
        with GLOBAL_LOG_BUFFER.lock:
            recent = list(GLOBAL_LOG_BUFFER.records)[-100:]
        for entry in recent:
            msg = f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            await response.write(msg.encode("utf-8"))

        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        GLOBAL_LOG_BUFFER.subscribe(q, loop)
        try:
            while True:
                entry = await q.get()
                msg = f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                await response.write(msg.encode("utf-8"))
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            GLOBAL_LOG_BUFFER.unsubscribe(q)

        return response


def create_app(config_path: str = "config/config.yaml") -> web.Application:
    app_instance = WebApp(config_path)
    app = web.Application()
    app.router.add_get("/", app_instance.handle_index)
    app.router.add_get("/api/status", app_instance.handle_status)
    app.router.add_get("/api/config", app_instance.handle_get_config)
    app.router.add_get("/api/keys", app_instance.handle_get_keys)
    app.router.add_post("/api/keys", app_instance.handle_set_keys)
    app.router.add_get("/api/logs", app_instance.handle_get_logs)
    app.router.add_get("/api/logs/stream", app_instance.handle_stream_logs)
    app.router.add_get("/api/image/{video_id}/{frame_id}", app_instance.handle_image)
    app.router.add_get("/api/neighbors/{video_id}/{frame_id}", app_instance.handle_neighbors)
    app.router.add_post("/api/search", app_instance.handle_search)
    app.router.add_post("/api/export_csv", app_instance.handle_export_csv)

    if STATIC_DIR.is_dir():
        app.router.add_static("/static/", STATIC_DIR, show_index=False)

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
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(r"D:\AIC2026 (1)\AIC2026\.env")
    run_server()
