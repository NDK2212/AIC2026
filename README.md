# AIC 2026 — Online Search Pipeline

Terminal codebase for the three AIC 2026 preliminary-round tasks: **Textual KIS**,
**Q&A (VQA)** and **TRAKE**. Offline indexing is assumed done — Qdrant holds the
keyframe embeddings and Elasticsearch holds the OCR/ASR text.

```
query
  │
  ├─ LLM decompose ──► {ocr_query, asr_query, image_query, modality_weights}
  │
  ├─ OCR    ─► Elasticsearch (ocr_text)              ─► top-K
  ├─ ASR    ─► Elasticsearch (asr_text)              ─► top-K
  └─ VISUAL ─► SigLIP + BEiT-3 + Qwen3-VL text towers ─► Qdrant ─► top-K
                 (embedding models on CPU; vector fusion inside Qdrant)
                            │
                            ▼
              Weighted RRF (Adaptive Score Fusion)
                            │
                            ▼
          Qwen3-VL-Reranker-2B (top-N fused, image + metadata, GPU)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     KIS                  VQA                 TRAKE
  top-100 → CSV     VLM per shot → answer   N-step DP → CSV
```

---

## 1. Install

Requirements: Python 3.10+ and Node.js 20+ with `npm`.

```bash
make install
```

This creates `.venv`, installs every Python dependency from `requirements.txt`,
and runs `npm ci` for the React frontend. It also creates `.env` from
`.env.example` only when `.env` does not already exist. Make commands invoke
`.venv/bin/python` directly, so manual virtual-environment activation is not
required.

`.env` holds every secret — nothing is ever hard-coded in the source:

| variable | used for |
|---|---|
| `NVIDIA_API_KEY` | the decomposition LLM and (by default) the VLM |
| `QDRANT_API_KEY` | Qdrant, if it needs auth |
| `ES_USER` / `ES_PASSWORD` | Elasticsearch basic auth |
| `ES_API_KEY` | Elasticsearch API-key auth (instead of user/password) |
| `VLM_API_KEY` | a separate vision endpoint; falls back to `NVIDIA_API_KEY` |
| `VLM_MODEL` | primary multimodal model id; defaults to `moonshotai/kimi-k3` |

---

## 2. Point the config at your data

Open `config/config.yaml` and set at minimum:

* `qdrant.url`, `qdrant.collection`, `qdrant.vector_names`, `qdrant.payload`
* `elasticsearch.hosts`, `elasticsearch.index`, `elasticsearch.fields`
* `embedding.siglip.model_id` / `embedding.beit3.*` — **must be the same models
  used at indexing time**, otherwise the query vectors live in a different space
* `keyframes.root`, `keyframes.map_dir`, `keyframes.metadata_dir`

Then verify the wiring before spending any API quota:

```bash
python -m src.cli inspect
```

`inspect` prints the *real* Qdrant vector names and dimensions, the *real* ES
field mapping, and asserts each encoder's width equals its collection vector's
width. A mismatch is a hard error at start-up — never a silently wrong result.
Add `--skip-encoders` to check only the databases.

### BEiT-3 without a checkpoint

BEiT-3 has no first-class `transformers` implementation. If the checkpoint or
the reference code is unavailable, the visual path logs a warning and continues
on SigLIP alone. To make that explicit, set `embedding.beit3.enabled: false`.

---

## 3. Run

For direct CLI commands, activate the environment once with
`source .venv/bin/activate`; the `make` commands below do not require this.

```bash
# debug the decomposition prompt
python -m src.cli decompose --query "Tìm video có người mặc áo đỏ..."

# debug fusion: fused top-k with each path's contribution
python -m src.cli search --query "..." --topk 20

# one query at a time
python -m src.cli kis   --query-file queries/query-1-kis.txt
python -m src.cli qa    --query-file queries/query-3-qa.txt
python -m src.cli trake --query-file queries/query-4-trake.txt

# the whole package (task detected from the -kis / -qa / -trake suffix)
python -m src.cli batch --query-dir queries/ --out-dir outputs/submission/

# check, then package
python -m src.cli validate --dir outputs/submission/
python -m src.cli pack     --dir outputs/submission/ --out teamABC1.zip
```

Global flags: `--config`, `--no-cache`, `-v/--verbose`, `--dry-run`.

`batch` never stops on a single failing query — it logs the error, continues,
and prints a success/failure table at the end.

### React web workbench

Run only the FastAPI backend (including the committed production React bundle):

```bash
make backend
```

Open <http://localhost:7860>. Interactive API documentation is available at
<http://localhost:7860/api/docs> and the OpenAPI schema at
<http://localhost:7860/api/openapi.json>.

Run only the Vite development server with hot reload:

```bash
make frontend
```

Open <http://localhost:5173>. Vite proxies `/api/*` to FastAPI at port `7860`,
so the backend must also be running for search and image requests.

Run both development servers together:

```bash
make full
```

Press `Ctrl+C` once to stop both processes. Host, port and config can be
overridden without editing the Makefile, for example:

```bash
make backend BACKEND_PORT=8000 CONFIG=config/config.yaml
make frontend FRONTEND_PORT=3000
```

`make full BACKEND_PORT=8000` also passes the matching backend URL to Vite's
proxy automatically. To use another backend host, set `BACKEND_URL`, for example
`make frontend BACKEND_URL=http://192.168.1.10:8000`. To rebuild the production
bundle served by FastAPI after frontend changes, run `make build-frontend`.

### Debugging a bad result

Every retrieval writes `outputs/runs/<timestamp>_<query>.json` containing the
decomposition, the top-20 of *each* path, the fusion weights actually used and
the fused top-50. That file is the first place to look when a query misses.
Console and file logs both land in `outputs/runs/run.log`.

---

## 4. Why the ranking looks the way it does

Scoring is `R@k = max` over the first `k` rows, averaged over
`k ∈ {1, 5, 20, 50, 100}`. Three consequences are built into the code:

1. **Always submit 100 rows.** A deep row can only ever help; it is a free lottery
   ticket. `submission.max_rows` is only a cap, never a target to fall short of.
2. **The head is expensive.** `R@1` and `R@5` are two of the five terms, so
   `submission/builder.py` forces the first `top_diverse` rows onto different
   shots *and* caps how many may come from one video — five frames of a single
   wrong shot would throw away four fifths of a term.
3. **The tail is cheap.** The ground-truth window `[s, e]` is narrow (under 10
   frames for TRAKE) while keyframes are sparse, so from `start_rank` onward the
   builder splices in neighbouring keyframes of strong candidates.

---

## 5. TRAKE: the alignment algorithm

1. An LLM reads the query and emits `N` chronologically ordered key moments —
   `N` is taken from the query itself when it states one.
2. Every step is retrieved independently through the normal four-path pipeline
   (steps run in parallel).
3. Step scores are min-max normalised **per step**, so no step dominates the
   objective and `miss_penalty` means the same thing everywhere.
4. Videos are ranked by summed best-per-step score plus a coverage bonus; the
   top `max_videos` go to alignment. The DP always runs *inside* one video, which
   is what guarantees the single-video output the rules require.
5. Inside a video, an **anchored bidirectional DP** picks one frame per step:

   * `prefix[j][i]` — best score covering steps `0..j` with step `j` at frame `i`
   * `suffix[j][i]` — best score covering steps `j..N-1` with step `j` at frame `i`
   * the best sequence through anchor candidate `i` is
     `prefix[a][i] + suffix[a][i] − score[a][i]`

   Transitions use a sliding-window maximum (a monotonic deque), so the temporal
   constraints `min_gap` / `max_gap` cost nothing extra. Complexity is
   `O(N² · K log K)` per video — never `O(K^N)`. Enumerating anchor candidates
   also yields genuinely distinct runner-up paths for free.
6. A step with no candidate is **skipped, not fatal**: it costs `miss_penalty`,
   and its frame is interpolated between its neighbours (extrapolated at the
   ends) then snapped onto a real keyframe. TRAKE scores a *fraction* of matched
   moments — 3 of 4 in the right video is 0.75, while dropping the right video is 0.
7. Sequences are ranked globally and the first five rows are spread across at
   least `head_min_videos` videos, because a wrong video scores exactly zero.

`refine_frame(video_id, frame_id, step_description)` in `tasks/trake.py` is an
identity hook left in place for a future dense local search around a candidate.

Works for any `N ≥ 1`; tested at `N = 2, 3, 4, 6`. `anchor_index` is clamped
into range automatically.

---

## 6. Configuration reference

### `qdrant`
| key | meaning |
|---|---|
| `url`, `api_key`, `timeout` | connection (`api_key` comes from the env) |
| `collection` | collection holding one point per keyframe |
| `vector_names.siglip` / `.beit3` | named vectors inside that collection |
| `payload.video_id` / `.frame_id` | payload fields carrying the identity |
| `prefetch_limit` | candidates each vector branch retrieves before fusing |
| `search_limit` | results returned after fusion |
| `fusion` | `rrf` \| `dbsf` \| `manual` (manual = fuse client-side) |

### `elasticsearch`
| key | meaning |
|---|---|
| `hosts`, `index`, `timeout`, `verify_certs` | connection |
| `fields.ocr` / `.asr` / `.video_id` / `.frame_id` | field names in your index |
| `size` | hits per text path |
| `min_score` | drop weak BM25 hits (0 = keep all) |
| `boosts.match` / `.phrase` / `.terms` | clause weights in the bool query |
| `boosts.exact_phrase` | phrase boost when the query wants literal on-screen text |
| `phrase_slop` | `match_phrase` slop |

### `embedding.siglip` / `embedding.beit3`
| key | meaning |
|---|---|
| `enabled` | turn the encoder — and its Qdrant branch — off |
| `backend` | `transformers` \| `open_clip` (SigLIP), `torchscale` \| `transformers` (BEiT-3) |
| `model_id` | **must match the model used at indexing time** |
| `dim` | asserted against the collection's vector width at start-up |
| `device` | `auto` \| `cuda` \| `mps` \| `cpu` |
| `checkpoint_path`, `tokenizer_path` | BEiT-3 weights and `beit3.spm` |
| `max_length`, `batch_size` | tokenisation and batching |

### `llm` / `vlm`
| key | meaning |
|---|---|
| `model` | model id (`vlm.model` must be a vision model) |
| `temperature`, `top_p`, `max_tokens` | sampling |
| `enable_thinking` | `<think>` blocks; the JSON parser strips them |
| `max_retries`, `retry_backoff` | transport retries on timeout, 429 and 5xx |
| `llm.json_retries` | retries when a completion is not valid JSON |
| `timeout` | HTTP timeout in seconds for OpenAI-compatible providers |
| `fallback.*` | secondary provider/model/base URL used after the primary fails |
| `vlm.max_workers` | frames asked in parallel |
| `vlm.image_max_side` | downscale before base64 to save tokens |

VQA defaults to multimodal mode. For each candidate video, Kimi K3 receives up
to `vqa.vlm_images_per_video` action/evidence frames in one request. Each image
is interleaved with its Description, OCR and ASR metadata, so the model can use
both the pixels and the indexed text when answering. The primary model comes
from `VLM_MODEL` in `.env` (the existing `VLM_model` spelling is also accepted).

The default configuration makes one NVIDIA/Kimi attempt and then falls back to
the local Ollama OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1`,
using `qwen3-vl:2b-instruct`. The same multi-image and metadata payload is sent
to the fallback. After a primary timeout, that client stays on the local
fallback until the backend is restarted, avoiding another remote timeout for
every candidate video.

### `fusion`
| key | meaning |
|---|---|
| `method` | `rrf` \| `weighted_rrf` \| `weighted_norm` |
| `rrf_k` | the `k` in `w / (k + rank)` |
| `weights` | static per-path weights, used when `adaptive: false` (ratios, not capped) |
| `adaptive` | use the LLM's `modality_weights` instead |
| `adaptive_floor` | minimum weight per active path, so the LLM cannot kill one |

### `submission`
| key | meaning |
|---|---|
| `max_rows` | hard cap (100) |
| `top_diverse` | rows in the protected diversity head |
| `head_max_per_video` | max head rows from one video |
| `shot_window` | frames closer than this count as the same shot |
| `neighbor_expansion.enabled` | splice neighbouring keyframes into the tail |
| `neighbor_expansion.start_rank` | first rank expansion may occupy |
| `neighbor_expansion.offsets` | frame offsets to try, nearest first |

### `vqa`
| key | meaning |
|---|---|
| `mode` | `vision` (default: images + Description/OCR/ASR) or `text_only` |
| `vlm_top_n` | how many representative action-scene frames reach the answerer |
| `vlm_images_per_video` | maximum action/evidence images combined in one VLM request per video |
| `cross_shot_evidence` | search answer-bearing shots elsewhere in the same candidate videos |
| `evidence_video_top_n` | number of top scene-matched videos searched for secondary evidence |
| `evidence_top_n` | maximum secondary evidence frames sent to the answerer |
| `propagate` | reuse answers within the same shot/video before global fallback |
| `answer_max_chars` | hard limit (100 per the rules) |
| `fallback_answer` | used when everything is UNKNOWN — never left empty |

### `trake`
| key | meaning |
|---|---|
| `anchor_index` | 0-based anchor step (clamped into range) |
| `per_step_topk` | candidates retrieved per step |
| `max_videos` | videos handed to the DP |
| `paths_per_video` | distinct sequences kept per video |
| `coverage_bonus` | reward per step a video actually covers |
| `miss_penalty` | cost of interpolating a step |
| `min_gap` / `max_gap` | temporal constraints between consecutive steps |
| `allow_fill` | permit skipped-and-interpolated steps at all |
| `head_min_videos` / `head_window` | video spread required in the head |

### `rerank`, `cache`, `runs`
`rerank.qwen3_vl` applies `Qwen/Qwen3-VL-Reranker-2B` once to the fused top-N.
It scores the query against each keyframe image plus available Description/OCR/ASR
metadata. The embedding models use CPU while this reranker uses CUDA. Legacy
`rerank.blip2` and `rerank.bge` remain parse-compatible but are disabled. `cache.dir` holds the LLM,
VLM, embedding and keyframe-index caches — reruns cost no quota. `runs.dir`
holds the per-query JSON traces and `run.log`.

---

## 7. Submission format

Enforced by `submission/writer.py` and re-checked by `submission/validator.py`:

* UTF-8 **without** BOM, `,` delimiter, `\n` line endings, **no header row**
* at most 100 rows; `video_id` carries **no** `.mp4`; `frame_id` is an integer
* Q&A answers ≤ 100 characters, never empty; quoted only when they contain
  `,` `"` or a newline, with inner `"` escaped as `""` (`csv.QUOTE_MINIMAL`)
* TRAKE rows all carry exactly `num_events` frames in increasing order —
  the writer raises rather than emit a row that would fail to parse
* the zip contains a `submission/` folder; bare CSVs at the root are rejected

`pack` runs the validator first and refuses to build an invalid archive — a
malformed file still burns one of your three submissions.

---

## 8. Tests

```bash
pytest -q
```

169 tests, no network access required — the LLM, VLM, Qdrant and Elasticsearch
are all faked.

| file | covers |
|---|---|
| `test_fusion.py` | RRF / weighted RRF formulas, single-path candidates, determinism, weight normalisation |
| `test_trake_dp.py` | the DP at `N = 2, 3, 4, 6`; `min_gap`/`max_gap` never violated; step filling and its penalty; anchors; head diversity |
| `test_builder.py` | shot and video diversity in the head, neighbour expansion, row caps |
| `test_writer.py` | CSV escaping, no header, ≤ 100 rows, TRAKE width assertions |
| `test_validator.py` | every rejection class, plus zip layout |
| `test_utils.py` | JSON extraction from `<think>`/fenced/chatty output, answer cleanup, keyframe mapping |
| `test_pipeline.py` | path dispatch, failure isolation, adaptive vs static weights, run traces |
| `test_end_to_end.py` | all three tasks from query to a CSV the validator accepts |
| `test_cli.py` | global flags on either side of the subcommand, `validate`/`pack` exit codes |

---

## 9. Layout

```
src/
├── cli.py                    entrypoint, one subcommand per operation
├── config.py                 YAML + env → validated dataclasses
├── schemas.py                Candidate, DecomposeResult, TrakePlan, ...
├── clients/                  qdrant, elastic, llm (JSON-hardened), vlm
├── embedding/                SigLIP / BEiT-3 text towers, lazy + cached
├── retrieval/                decompose, search_text, search_visual, fusion,
│                             rerank, pipeline
├── tasks/                    kis, vqa, trake
├── submission/               builder, writer, validator, packer
└── utils/                    keyframe_index, cache, text_norm
```
